"""Canonical training manifest (NEW_PLAN §5): one JSON line per utterance of the train list.

    .venv/bin/python experiments/distill_noiseaug_durationaware/prep/build_manifest.py

Fields: row (index in the train list), audio_path (relative to TRAIN_DIR), speaker_id, sample_rate, num_samples,
total_duration, usable_duration (sum of active intervals), active_intervals [[start_s, end_s], ...].
Writes artifacts/manifest.jsonl and artifacts/manifest.sha256. Source audio is never modified.
"""
from __future__ import annotations

import hashlib
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import soundfile as sf

EXP = Path(__file__).resolve().parents[1]
SRED = EXP.parents[1]
sys.path.insert(0, str(SRED))
sys.path.insert(0, str(EXP))

from dnd.vad import active_intervals  # noqa: E402
from sred.data import TRAIN_DIR, read_train_list  # noqa: E402

TRAIN_LIST = "lists/clean/train_clean.txt"


def one(job):
    row, spk, rel = job
    x, sr = sf.read(str(TRAIN_DIR / rel), dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(1)
    iv = active_intervals(x)
    return {"row": row, "audio_path": rel, "speaker_id": spk, "sample_rate": sr, "num_samples": len(x),
            "total_duration": round(len(x) / sr, 4), "usable_duration": round(sum(e - s for s, e in iv), 3),
            "active_intervals": [list(v) for v in iv]}


def main():
    items = read_train_list(SRED / TRAIN_LIST)
    with Pool(32) as pool:
        rows = pool.map(one, [(i, s, r) for i, (s, r) in enumerate(items)], chunksize=64)
    out = EXP / "artifacts"
    out.mkdir(exist_ok=True)
    text = "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows)
    (out / "manifest.jsonl").write_text(text)
    h = hashlib.sha256(text.encode()).hexdigest()
    (out / "manifest.sha256").write_text(f"{h}  manifest.jsonl  (train list {TRAIN_LIST})\n")
    sr = {r["sample_rate"] for r in rows}
    print(f"{len(rows)} utterances, {len({r['speaker_id'] for r in rows})} speakers, sample rates {sr}, "
          f"{sum(r['total_duration'] for r in rows) / 3600:.1f} h audio, "
          f"{sum(r['usable_duration'] for r in rows) / 3600:.1f} h usable; sha256 {h[:12]}")


if __name__ == "__main__":
    main()
