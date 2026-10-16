"""Ask Gemini to listen to failure cases (analysis only -- never used for training / submission).

    GOOGLE_API_KEY=... python scripts/gemini_listen.py [--model gemini-2.5-flash]

Writes exp/analysis/gemini/{suspect_speakers,trials}.jsonl (resumable: done items are skipped).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from google import genai
from google.genai import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sred.data import PUBLIC_DIR

A = Path(__file__).resolve().parents[1] / "exp/analysis"
OUT = A / "gemini"

CLIP_FIELDS = {
    "gender": {"type": "string", "enum": ["male", "female", "unclear"]},
    "age_group": {"type": "string", "enum": ["child", "young_adult", "adult", "elderly", "unclear"]},
    "speaking_style": {"type": "string", "enum": ["read", "conversational", "emotional", "whisper", "shouting",
                                                   "singing", "imitating_or_acting", "other"]},
    "emotion": {"type": "string", "enum": ["neutral", "happy", "sad", "angry", "excited", "other"]},
    "channel": {"type": "string", "enum": ["studio_or_close_mic", "phone_or_narrowband", "far_field_or_reverberant",
                                            "compressed_or_codec", "other"]},
    "background": {"type": "string", "enum": ["clean", "stationary_noise", "babble_or_crowd", "music", "street_or_vehicle",
                                               "other_speakers_audible", "other"]},
    "num_speakers": {"type": "integer"},
    "speech_amount": {"type": "string", "enum": ["very_little(<1s)", "short(1-3s)", "normal(>3s)"]},
    "language": {"type": "string"},
    "notes": {"type": "string"},
}
CLIP = {"type": "object", "properties": CLIP_FIELDS, "required": list(CLIP_FIELDS)}

TRIAL_SCHEMA = {
    "type": "object",
    "properties": {
        "clip_a": CLIP, "clip_b": CLIP,
        "same_speaker": {"type": "string", "enum": ["yes", "no", "unsure"]},
        "confidence": {"type": "number"},
        "main_difficulty": {"type": "string"},
    },
    "required": ["clip_a", "clip_b", "same_speaker", "confidence", "main_difficulty"],
}
GROUP_SCHEMA = {
    "type": "object",
    "properties": {
        "clips": {"type": "array", "items": {"type": "object", "properties": {"clip": {"type": "string"}, **CLIP_FIELDS},
                                             "required": ["clip", "gender", "age_group", "channel", "background"]}},
        "num_distinct_speakers": {"type": "integer"},
        "grouping": {"type": "array", "items": {"type": "array", "items": {"type": "string"}},
                     "description": "clips grouped by speaker identity"},
        "explanation": {"type": "string"},
    },
    "required": ["clips", "num_distinct_speakers", "grouping", "explanation"],
}

TRIAL_PROMPT = """You are an expert forensic phonetician. Listen to clip A and clip B (Vietnamese speech).
Describe each clip, then judge whether they are spoken by the SAME person. Base the identity judgement on voice
characteristics (timbre, pitch range, accent, articulation), not on channel or content. In main_difficulty, state in
one sentence what makes this pair hard to judge (e.g. very short, pitch/style change, noise, channel mismatch,
similar voices), or "none"."""
GROUP_PROMPT = """You are an expert forensic phonetician. The following {n} clips (Vietnamese speech) are labelled
as ONE speaker in a dataset. Listen carefully and decide how many distinct people actually speak across these clips,
and group the clip names by speaker identity. Base identity on voice characteristics (timbre, pitch range, accent,
articulation), not on channel or content. Also describe each clip."""


def part(path: Path) -> types.Part:
    return types.Part.from_bytes(data=path.read_bytes(), mime_type="audio/wav")


def call(client, model, contents, schema, tries=5):
    for k in range(tries):
        try:
            r = client.models.generate_content(
                model=model, contents=contents,
                config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=schema,
                                                   temperature=0.0))
            return json.loads(r.text)
        except Exception as e:  # rate limits / transient errors
            if k == tries - 1:
                return {"error": str(e)[:300]}
            time.sleep(10 * (k + 1))


def done_keys(path):
    return {json.loads(l)["key"] for l in open(path)} if path.exists() else set()


def run_jobs(jobs, path, workers):
    done = done_keys(path)
    jobs = [j for j in jobs if j[0] not in done]
    print(f"{path.name}: {len(jobs)} to do ({len(done)} done)", flush=True)
    with ThreadPoolExecutor(workers) as ex, open(path, "a") as f:
        for key, meta, res in ex.map(lambda j: (j[0], j[1], j[2]()), jobs):
            f.write(json.dumps({"key": key, **meta, "gemini": res}, ensure_ascii=False) + "\n")
            f.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n-control", type=int, default=40)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    client = genai.Client()

    # 1) suspect speaker folders
    sus = pd.read_csv(A / "listen_suspect_speakers.csv")
    jobs = []
    for s, d in sus.groupby("speaker"):
        names = [f"clip{i + 1}" for i in range(len(d))]
        contents = [GROUP_PROMPT.format(n=len(d))]
        for nm, p in zip(names, d.file):
            contents += [f"{nm}:", part(Path(p))]
        meta = {"speaker": s, "clips": dict(zip(names, d.file.map(lambda p: Path(p).relative_to(PUBLIC_DIR).as_posix()))),
                "embedding_cluster": dict(zip(names, d.cluster.astype(int).tolist()))}
        jobs.append((s, meta, lambda c=contents: call(client, a.model, c, GROUP_SCHEMA)))
    run_jobs(jobs, OUT / "suspect_speakers.jsonl", a.workers)

    # 2) failed trials (all 4 submittable models wrong) + control trials (all 6 systems right)
    t = pd.read_csv(A / "trial_errors.csv")
    errs = [c for c in t if c.startswith("err_")]
    fr = pd.read_csv(A / "worst_false_rejects.csv").assign(group="false_reject")
    fa = pd.read_csv(A / "worst_false_accepts.csv").assign(group="false_accept")
    ok = t[(t[errs].sum(1) == 0)]
    ctl = pd.concat([ok[ok.y == 1].sample(a.n_control // 2, random_state=0).assign(group="correct_target"),
                     ok[ok.y == 0].sample(a.n_control // 2, random_state=0).assign(group="correct_nontarget")])
    jobs = []
    for _, r in pd.concat([fr, fa, ctl]).iterrows():
        contents = [TRIAL_PROMPT, "clip A:", part(PUBLIC_DIR / r.a), "clip B:", part(PUBLIC_DIR / r.b)]
        meta = {"group": r.group, "a": r.a, "b": r.b, "y": int(r.y)}
        jobs.append((f"{r.a}|{r.b}", meta, lambda c=contents: call(client, a.model, c, TRIAL_SCHEMA)))
    run_jobs(jobs, OUT / "trials.jsonl", a.workers)


if __name__ == "__main__":
    main()
