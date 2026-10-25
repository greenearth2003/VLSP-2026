"""Manual-review aid (approved by the organisers, 2026-10-20): Gemini listens to random private pairs and reports
data-quality issues and its own same/different judgement. Our score is NOT shown to Gemini; it is joined afterwards.
Analysis only: nothing here feeds training, scoring or model selection.

    python scripts/gemini_review_private.py [--n 100] [--model gemini-pro-latest] [--seed 20]

Writes exp/analysis/gemini/private_review.jsonl (resumable) and prints a summary.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from google import genai

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gemini_listen import CLIP, call, part, run_jobs  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "exp/analysis/gemini"
PRIV = ROOT / "data/Vi-SRED26/private_test/data"

SCHEMA = {
    "type": "object",
    "properties": {
        "clip_a": CLIP, "clip_b": CLIP,
        "same_speaker": {"type": "string", "enum": ["yes", "no", "unsure"]},
        "confidence": {"type": "number", "description": "0-1"},
        "main_difficulty": {"type": "string"},
        "quality_issues": {"type": "array", "items": {"type": "string", "enum": [
            "loud_background_noise", "music", "other_speakers_audible", "overlapping_speech", "very_low_volume",
            "very_short_speech", "no_speech", "clipping_or_distortion", "narrowband_or_codec", "reverberant",
            "non_speech_sound_effects", "singing", "language_not_vietnamese", "same_recording_session", "none"]}},
        "notes": {"type": "string"},
    },
    "required": ["clip_a", "clip_b", "same_speaker", "confidence", "main_difficulty", "quality_issues", "notes"],
}
CAUSES = ["more_than_one_speaker", "very_short_speech", "background_noise", "music_or_sound_effects", "low_volume",
          "same_recording_session_or_channel", "speaking_style_or_emotion_change", "similar_voices", "channel_mismatch",
          "overlapping_speech", "no_clear_cause"]
VALIDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "clip_a": CLIP, "clip_b": CLIP,
        "audio_quality_a": {"type": "string", "description": "one sentence: noise, music, other voices, level, length"},
        "audio_quality_b": {"type": "string"},
        "own_same_speaker": {"type": "string", "enum": ["yes", "no", "unsure"]},
        "own_confidence": {"type": "number", "description": "0-1"},
        "model_decision_correct": {"type": "string", "enum": ["yes", "no", "unsure"]},
        "likely_causes": {"type": "array", "items": {"type": "string", "enum": CAUSES},
                          "description": "if the model decision looks wrong: what in the audio most likely misled it"},
        "explanation": {"type": "string", "description": "2-3 sentences: why you agree or disagree with the model"},
    },
    "required": ["clip_a", "clip_b", "audio_quality_a", "audio_quality_b", "own_same_speaker", "own_confidence",
                 "model_decision_correct", "likely_causes", "explanation"],
}
VALIDATE_PROMPT = """You are an expert forensic phonetician auditing an automatic speaker-verification system on
Vietnamese speech. STEP 1, before reading the system's output: listen to clip A and clip B, describe the audio quality
of each, and judge from voice characteristics alone (timbre, pitch range, accent, articulation; not channel,
background or content) whether the main speaker of A and the main speaker of B are the same person.
STEP 2: the system decided "{decision}" (score {score:+.2f}; positive = same speaker, the further from 0 the more
confident). Say whether that decision is correct. If you think it is wrong, list what in the audio most likely misled
the system (likely_causes) and explain in 2-3 sentences why the voices are or are not the same. Do not change your
STEP 1 judgement to match the system."""
PROMPT = """You are an expert forensic phonetician reviewing a speaker-verification test set (Vietnamese speech).
Listen to clip A and clip B. Describe each clip, list every data-quality issue you hear (quality_issues), then judge
whether the main speaker of A and the main speaker of B are the SAME person, from voice characteristics (timbre,
pitch range, accent, articulation), not from channel, background or content. If both clips sound like the same
recording session or programme, include same_recording_session. In main_difficulty, one sentence on what makes this
pair hard to judge, or "none"."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--model", default="gemini-pro-latest")
    ap.add_argument("--seed", type=int, default=20)
    ap.add_argument("--scores", default="submissions/final_private/results_result-private-S3-ms025-t0.txt")
    ap.add_argument("--threshold", type=float, default=-0.022, help="the final system's EER threshold on the synthetic stand-in")
    ap.add_argument("--validate", action="store_true", help="show Gemini our decision and ask it to validate / explain")
    ap.add_argument("--public-check", type=int, default=0,
                    help="instead: N labelled pairs (half same-speaker) from the real-noise public stand-in, to measure accuracy")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    client = genai.Client()
    tag = a.model.replace("/", "_")
    if a.public_check:
        from sred.data import read_trials
        pairs, y = read_trials()
        rng = np.random.default_rng(a.seed)
        idx = np.concatenate([rng.choice(np.flatnonzero(y == v), a.public_check // 2, replace=False) for v in (1, 0)])
        jobs = []
        for i in idx:
            pa, pb = pairs[i]
            root = ROOT / "data/public_realnoise"
            contents = [PROMPT, "clip A:", part(root / pa), "clip B:", part(root / pb)]
            jobs.append((f"{pa}|{pb}", {"a": pa, "b": pb, "truth": "yes" if y[i] else "no", "model": a.model},
                         lambda c=contents: call(client, a.model, c, SCHEMA)))
        path = OUT / f"public_check_{tag}.jsonl"
        run_jobs(jobs, path, 4)
        res = [json.loads(l) for l in open(path)]
        res = [r for r in res if "error" not in r["gemini"]]
        g = pd.DataFrame([{"truth": r["truth"], "gem": r["gemini"]["same_speaker"], "conf": r["gemini"]["confidence"]} for r in res])
        sure = g[g.gem != "unsure"]
        print(f"{a.model}: {len(g)} labelled real-noise public pairs; accuracy where not unsure {(sure.gem == sure.truth).mean():.0%} "
              f"of {len(sure)}; says 'yes' on {(g.gem == 'yes').mean():.0%} (truth: 50%)")
        print(pd.crosstab(g.truth, g.gem).to_string())
        return
    rows = [l.split() for l in open(ROOT / a.scores)]
    t = pd.DataFrame(rows, columns=["a", "b", "score"]).astype({"score": float})
    pick = t.iloc[np.sort(np.random.default_rng(a.seed).choice(len(t), a.n, replace=False))]
    if a.validate:
        jobs = []
        for _, r in pick.iterrows():
            dec = "same speaker" if r.score > a.threshold else "different speakers"
            contents = [VALIDATE_PROMPT.format(decision=dec, score=r.score - a.threshold), "clip A:", part(PRIV / r.a), "clip B:", part(PRIV / r.b)]
            meta = {"a": r.a, "b": r.b, "score": r.score, "ours": "same" if r.score > a.threshold else "different", "model": a.model}
            jobs.append((f"{r.a}|{r.b}", meta, lambda c=contents: call(client, a.model, c, VALIDATE_SCHEMA)))
        path = OUT / f"private_validate_{tag}.jsonl"
        run_jobs(jobs, path, 4)
        res = [json.loads(l) for l in open(path)]
        res = [r for r in res if "error" not in r["gemini"]]
        g = pd.DataFrame([{"a": r["a"], "b": r["b"], "score": r["score"], "ours": r["ours"],
                           "own": r["gemini"]["own_same_speaker"], "correct": r["gemini"]["model_decision_correct"],
                           "causes": r["gemini"]["likely_causes"], "qa": r["gemini"]["audio_quality_a"],
                           "qb": r["gemini"]["audio_quality_b"], "why": r["gemini"]["explanation"]} for r in res])
        own_sure = g[g.own != "unsure"]
        print(f"{len(g)} pairs, {a.model}. Gemini's own verdict (step 1): {dict(Counter(g.own))}")
        print(f"step-1 verdict agrees with our decision: {((own_sure.own == 'yes') == (own_sure.ours == 'same')).mean():.0%} of {len(own_sure)}")
        print(f"step 2, our decision judged: {dict(Counter(g.correct))}")
        wrong = g[g.correct == "no"]
        print(f"\nlikely causes where our decision is judged wrong ({len(wrong)} pairs):")
        for k, v in Counter(c for L in wrong.causes for c in L).most_common():
            print(f"  {k:36s} {v}")
        print("\nby our decision:", wrong.groupby("ours").size().to_dict())
        for _, r in wrong.sort_values("score").iterrows():
            print(f"\n{r.a} | {r.b}  ours={r.ours} ({r.score:+.2f})  Gemini own={r.own}\n  A: {r.qa}\n  B: {r.qb}\n  why: {r.why}")
        g.to_csv(OUT / f"private_validate_{tag}.csv", index=False)
        return
    jobs = []
    for _, r in pick.iterrows():
        contents = [PROMPT, "clip A:", part(PRIV / r.a), "clip B:", part(PRIV / r.b)]
        meta = {"a": r.a, "b": r.b, "score": r.score, "ours": "same" if r.score > a.threshold else "different", "model": a.model}
        jobs.append((f"{r.a}|{r.b}", meta, lambda c=contents: call(client, a.model, c, SCHEMA)))
    path = OUT / f"private_review_{tag}.jsonl"
    run_jobs(jobs, path, 4)

    res = [json.loads(l) for l in open(path)]
    res = [r for r in res if "error" not in r["gemini"]]
    print(f"\n{len(res)} pairs reviewed by {a.model}")
    g = pd.DataFrame([{"ours": r["ours"], "score": r["score"], "gem": r["gemini"]["same_speaker"],
                       "conf": r["gemini"]["confidence"], "issues": r["gemini"]["quality_issues"],
                       "spk_a": r["gemini"]["clip_a"]["num_speakers"], "spk_b": r["gemini"]["clip_b"]["num_speakers"],
                       "bg_a": r["gemini"]["clip_a"]["background"], "bg_b": r["gemini"]["clip_b"]["background"],
                       "a": r["a"], "b": r["b"], "difficulty": r["gemini"]["main_difficulty"]} for r in res])
    print("Gemini's judgement:", dict(Counter(g.gem)))
    sure = g[g.gem != "unsure"]
    agree = ((sure.gem == "yes") == (sure.ours == "same"))
    print(f"agreement with our decision where Gemini is not unsure: {agree.mean():.0%} of {len(sure)}")
    print(pd.crosstab(g.ours, g.gem).to_string())
    iss = Counter(i for L in g.issues for i in L if i != "none")
    print("\nquality issues (pairs with the issue in either clip):")
    for k, v in iss.most_common():
        print(f"  {k:28s} {v:3d}  ({v / len(g):.0%})")
    print(f"\nclips with >1 speaker: {((g.spk_a > 1).sum() + (g.spk_b > 1).sum())} of {2 * len(g)}")
    print("backgrounds:", dict(Counter(list(g.bg_a) + list(g.bg_b))))
    dis = sure[~agree].copy()
    dis["gem_conf"] = dis.conf
    print(f"\npairs where Gemini (not unsure) disagrees with us: {len(dis)}")
    print(dis[["a", "b", "score", "ours", "gem", "conf", "difficulty"]].sort_values("conf", ascending=False).to_string(index=False))
    g.to_csv(OUT / f"private_review_summary_{tag}.csv", index=False)


if __name__ == "__main__":
    main()
