"""Manual-review aid (Gemini review approved by the organisers, 2026-10-20): Gemini arbitrates trials where our final system
and an external reference (hynt ResNet293 + Silero VAD, max over speech segments; scripts/ext_vad_max.py) disagree, and
explains which one is right, with evidence and the audio artifact that misled the wrong one. Analysis only: nothing here
feeds training, scoring or the submission.

    python scripts/gemini_mismatch_review.py --model gemini-3.8-flash [--n-priv 40 30 40 --n-silver 40]

The two systems are shown as "System X" / "System Y" in random order (which one is ours is hidden). Strata of private
disagreements (margins = distance from each system's own silver EER threshold, in units of its score spread):
  P1  external confident (>= 0.5 sd), ours unsure (<= 0.25 sd)   -- on silver, ours is wrong 55% here
  P2  both confident (>= 0.5 sd), opposite decisions
  P3  random disagreements
  C   silver disagreements with known labels (calibration: how often Gemini picks the right system)
Writes exp/analysis/gemini/mismatch_<model>.jsonl (resumable) and a summary.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from google import genai
from google.genai import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gemini_listen import CLIP, part, run_jobs  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "exp/analysis/gemini"
PRIV = ROOT / "data/Vi-SRED26/private_test/data"
SILVER = ROOT / "data/public_rumble_v2"

CAUSES = ["very_short_speech", "more_than_one_speaker", "main_speaker_ambiguous", "overlapping_speech", "background_noise",
          "music_or_sound_effects", "low_frequency_rumble", "low_volume", "level_mismatch", "channel_or_codec_mismatch",
          "shared_background_or_channel", "expressive_style_or_pitch_change", "similar_voices",
          "whisper_shout_or_singing", "reverberation", "clipping_or_distortion", "mostly_silence_or_non_speech",
          "no_clear_cause"]
SCHEMA = {
    "type": "object",
    "properties": {
        "clip_a": CLIP, "clip_b": CLIP,
        "net_speech_seconds_a": {"type": "number"}, "net_speech_seconds_b": {"type": "number"},
        "main_speaker_ambiguous": {"type": "boolean", "description": "true if it is unclear which voice is the main speaker of A or B"},
        "own_same_speaker": {"type": "string", "enum": ["yes", "no", "unsure"]},
        "own_confidence": {"type": "number", "description": "0-1"},
        "correct_system": {"type": "string", "enum": ["X", "Y", "unsure"]},
        "evidence": {"type": "string", "description": "2-4 sentences: the concrete voice cues (with times if possible)"},
        "root_causes": {"type": "array", "items": {"type": "string", "enum": CAUSES},
                        "description": "what in this pair most likely pushed the WRONG system to its answer"},
        "root_cause_explanation": {"type": "string", "description": "2-3 sentences: how these artifacts mislead an automatic system"},
    },
    "required": ["clip_a", "clip_b", "net_speech_seconds_a", "net_speech_seconds_b", "main_speaker_ambiguous",
                 "own_same_speaker", "own_confidence", "correct_system", "evidence", "root_causes", "root_cause_explanation"],
}
SYSTEM = """You are an expert forensic phonetician auditing automatic speaker-verification systems on Vietnamese speech.
For each request you hear two clips, A and B. Two automatic systems disagree on whether the MAIN speaker of A and the
MAIN speaker of B are the same person: one says SAME, the other says DIFFERENT. Work in this order:
1. Listen to both clips completely. For each clip, identify all voices and which one is the main (dominant) speaker,
   estimate the net speech duration of the main speaker, and describe gender, age group, speaking style, emotion,
   background (noise, music, sound effects, low-frequency rumble or wave-like sound, other voices), channel (close mic,
   phone or narrowband, codec, reverberant) and level.
2. Judge on your own, before considering the systems, whether the two main speakers are the same person. Use only
   voice identity cues: timbre and resonance, pitch range and habits, accent or dialect, articulation, voice quality
   (breathy, creaky, nasal). Do not use content. A shared background, programme or channel is NOT evidence of the same
   speaker, and different recording conditions are NOT evidence of different speakers.
3. Then say which system is correct (X or Y), or "unsure" when the audio does not allow a confident call. Give the
   concrete evidence. Be calibrated: do not pick a system because you expect it to be the better one.
4. Root cause: name the properties of this pair that would push an automatic system toward the wrong answer (for
   example very little speech, several voices or an unclear main speaker, overlapping speech, music or noise, a
   low-frequency rumble, low volume or a large level difference, channel or codec mismatch, the same background shared
   by different people, a strong change of speaking style or pitch, two genuinely similar voices) and explain how."""


def call(client, model, contents, tries=6):
    for k in range(tries):
        try:
            r = client.models.generate_content(
                model=model, contents=contents,
                config=types.GenerateContentConfig(system_instruction=SYSTEM, response_mime_type="application/json",
                                                   response_schema=SCHEMA, temperature=0.0))
            return json.loads(r.text)
        except Exception as e:  # rate limits / transient errors
            if k == tries - 1:
                return {"error": str(e)[:300]}
            time.sleep(15 * (k + 1))


def eer_thr(s, y):
    o = np.argsort(s)
    yo = y[o]
    frr = np.cumsum(yo) / yo.sum()
    far = 1 - np.cumsum(1 - yo) / (1 - yo).sum()
    return s[o][np.argmin(np.abs(frr - far))]


def build_items(n_priv, n_silver, seed):
    import ext_vad_max as X
    from sred.data import read_trials, unique_files
    pairs, y = read_trials()
    files = unique_files(pairs)
    pos = {f: i for i, f in enumerate(files)}
    ia = np.array([pos[x] for x, _ in pairs]); ib = np.array([pos[x] for _, x in pairs])
    s_ext = X.trial_scores(np.load(X.OUT / "public_rumble_v2_vadseg.npz"), ia, ib)["max"]
    s_our = pd.DataFrame(pairs, columns=["a", "b"]).merge(pd.read_csv(ROOT / "exp/scores/silver/S3_seg_ms025.txt", sep=" ",
                                                                      header=None, names=["a", "b", "s"]), how="left").s.to_numpy()
    to, te = eer_thr(s_our, y), eer_thr(s_ext, y)
    so, se = s_our.std(), s_ext.std()
    rng = np.random.default_rng(seed)
    items = []
    # calibration: silver disagreements, labels known
    dis = np.flatnonzero((s_our > to) != (s_ext > te))
    for i in rng.choice(dis, n_silver, replace=False):
        items.append(dict(stratum="C_silver", a=pairs[i][0], b=pairs[i][1], root=str(SILVER), ours_same=bool(s_our[i] > to),
                          m_ours=float((s_our[i] - to) / so), m_ext=float((s_ext[i] - te) / se), truth="same" if y[i] else "different"))
    d = pd.read_csv(ROOT / "exp/analysis/ext_pseudo_max_private_disagreements.csv")
    d["m_ours"], d["m_ext"] = (d.ours - to) / so, (d.ext - te) / se
    p1 = d[(d.m_ext.abs() >= 0.5) & (d.m_ours.abs() <= 0.25)]
    p2 = d[(d.m_ext.abs() >= 0.5) & (d.m_ours.abs() > 0.5)]
    print(f"private disagreements {len(d):,}: P1 {len(p1):,}, P2 {len(p2):,}")
    taken = set()
    for name, pool, n in (("P1_ext_confident_ours_unsure", p1, n_priv[0]), ("P2_both_confident", p2, n_priv[1]),
                          ("P3_random", d, n_priv[2])):
        pool = pool[[(a, b) not in taken for a, b in zip(pool.a, pool.b)]]
        for _, r in pool.iloc[rng.choice(len(pool), min(n, len(pool)), replace=False)].iterrows():
            taken.add((r.a, r.b))
            items.append(dict(stratum=name, a=r.a, b=r.b, root=str(PRIV), ours_same=bool(r.ours_accept == 1),
                              m_ours=float(r.m_ours), m_ext=float(r.m_ext), truth=""))
    for it in items:  # hide which system is ours
        it["ours_is"] = "X" if rng.random() < 0.5 else "Y"
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-3.8-flash")
    ap.add_argument("--n-priv", type=int, nargs=3, default=[40, 30, 40])
    ap.add_argument("--n-silver", type=int, default=40)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    items_path = OUT / f"mismatch_items_seed{a.seed}.json"
    if items_path.exists():
        items = json.loads(items_path.read_text())
    else:
        items = build_items(a.n_priv, a.n_silver, a.seed)
        items_path.write_text(json.dumps(items, indent=1))
    client = genai.Client()
    jobs = []
    for it in items:
        x_same = it["ours_same"] if it["ours_is"] == "X" else not it["ours_same"]
        verdicts = f"System X says: {'SAME' if x_same else 'DIFFERENT'} speaker. System Y says: {'DIFFERENT' if x_same else 'SAME'} speaker."
        contents = [verdicts, "Clip A:", part(Path(it["root"]) / it["a"]), "Clip B:", part(Path(it["root"]) / it["b"])]
        jobs.append((f"{it['stratum']}|{it['a']}|{it['b']}", {**it, "model": a.model}, lambda c=contents: call(client, a.model, c)))
    path = OUT / f"mismatch_{a.model.replace('/', '_')}.jsonl"
    run_jobs(jobs, path, a.workers)
    summarize(path)


def summarize(path):
    res = [json.loads(l) for l in open(path)]
    ok = [r for r in res if "error" not in r["gemini"]]
    print(f"\n{path.name}: {len(ok)} reviewed, {len(res) - len(ok)} errors")
    rows = []
    for r in ok:
        g = r["gemini"]
        pick = g["correct_system"]
        winner = "unsure" if pick == "unsure" else ("ours" if pick == r["ours_is"] else "external")
        own = g["own_same_speaker"]
        own_side = "unsure" if own == "unsure" else ("ours" if (own == "yes") == r["ours_same"] else "external")
        rows.append(dict(stratum=r["stratum"], a=r["a"], b=r["b"], ours_same=r["ours_same"], truth=r["truth"], winner=winner,
                         own_side=own_side, conf=g["own_confidence"], causes=g["root_causes"], amb=g["main_speaker_ambiguous"],
                         ev=g["evidence"], why=g["root_cause_explanation"], sp_a=g["net_speech_seconds_a"], sp_b=g["net_speech_seconds_b"],
                         m_ours=r["m_ours"], m_ext=r["m_ext"]))
    g = pd.DataFrame(rows)
    g["consistent"] = g.winner == g.own_side
    print("which system Gemini says is correct, by stratum (its step-1 own verdict agrees with its pick in "
          f"{g.consistent.mean():.0%} of cases):")
    print(pd.crosstab(g.stratum, g.winner, margins=True).to_string())
    c = g[g.stratum == "C_silver"].copy()
    if len(c):
        c["truly_ours"] = (c.truth == "same") == c.ours_same
        sure = c[c.winner != "unsure"]
        right = (sure.winner == "ours") == sure.truly_ours
        print(f"\ncalibration on {len(c)} silver disagreements (labels known): ours actually right in {c.truly_ours.mean():.0%}; "
              f"Gemini picks the right system in {right.mean():.0%} of the {len(sure)} it decides "
              f"(when it says 'external': right {right[sure.winner == 'external'].mean():.0%} of {(sure.winner == 'external').sum()}; "
              f"'ours': right {right[sure.winner == 'ours'].mean():.0%} of {(sure.winner == 'ours').sum()})")
    p = g[(g.stratum != "C_silver") & (g.winner == "external")]
    print(f"\nprivate pairs where Gemini says OUR system is wrong: {len(p)} of {(g.stratum != 'C_silver').sum()}")
    print("  root causes:", Counter(x for L in p.causes for x in L).most_common())
    print(f"  our wrong decision was: {dict(Counter(np.where(p.ours_same, 'false accept (we said same)', 'false reject (we said different)')))}")
    print(f"  main speaker ambiguous in {p.amb.mean():.0%}; median net speech of the shorter clip {np.median(np.minimum(p.sp_a, p.sp_b)):.1f} s")
    g.to_csv(path.with_suffix(".csv"), index=False)


if __name__ == "__main__":
    main()
