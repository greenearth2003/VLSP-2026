"""Inference for the released voice-embedding model VE v1.0.lite (= E8-b6-lowrank-kd-continue step 5000, WiSE 0.7,
QMF2 calibrator with norm + speech view + AS-norm). CPU or GPU; no training data needed.

    # embeddings: one 192-D L2-normalised vector per file (whole-clip + speech-only view, as the calibrator expects)
    .venv/bin/python experiments/distill_noiseaug_durationaware/ve_infer.py embed --model-dir MODEL_DIR a.wav b.wav --out emb.npz
    # calibrated scores for a trial list ("fileA fileB [label]"), "fileA fileB score" lines in trial order
    .venv/bin/python experiments/distill_noiseaug_durationaware/ve_infer.py score --model-dir MODEL_DIR \\
        --trials trials.txt --audio-dir DIR --out results.txt [--device cpu --threads 4]

MODEL_DIR holds ve_v1.0.lite.pt, calibrator.json, calibrator.cohort.npy and b6_template.pt (the pretrained
ReDimNet2-B6 file, used only to rebuild the architecture; its weights are overwritten). Audio: mono, 16 kHz.
On GPU the model runs in bf16 autocast (as for the submission); on CPU in fp32.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

EXP = Path(__file__).resolve().parent
SRED = EXP.parents[1]
sys.path[:0] = [str(SRED), str(EXP)]

from calibrate_eval import CAP, quality  # noqa: E402
from posttrain_embed import speech_only  # noqa: E402
from dnd.vad import SR  # noqa: E402


def load(model_dir: Path, device: str):
    import dnd.lowrank as lr
    from backend import Calibrator
    lr.B6_PRETRAINED = model_dir / "b6_template.pt"
    model = lr.load_lowrank(model_dir / "ve_v1.0.lite.pt").to(device).eval()
    cal = Calibrator(model_dir / "calibrator.json")
    return model, cal


@torch.inference_mode()
def raw(model, x, device):
    t = torch.from_numpy(np.ascontiguousarray(x[:CAP], dtype=np.float32))[None].to(device)
    if device == "cuda":
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return model(t).float()[0].cpu().numpy()
    return model(t)[0].numpy()


def read(path):
    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if sr != SR:
        raise SystemExit(f"{path}: {sr} Hz, expected {SR} Hz mono")
    return x if x.ndim == 1 else x.mean(1)


def embed_files(model, cal, paths, device):
    """-> (embeddings in the calibrator's view [n, 192], qualities [n, 3]: speech s, SNR dB, raw norm)."""
    W, S, Q = [], [], []
    for i, p in enumerate(paths):
        x = read(p)
        w = raw(model, x, device)
        W.append(w)
        S.append(raw(model, speech_only(x[:CAP]), device) if cal.c["flags"]["speech"] else w)
        Q.append((*quality(x[:CAP]), float(np.linalg.norm(w))))
        if i and i % 5000 == 0:
            print(f"  {i}/{len(paths)} files", flush=True)
    return cal.embed_view(np.stack(W), np.stack(S)), np.array(Q, np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["embed", "score"])
    ap.add_argument("files", nargs="*")
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--trials")
    ap.add_argument("--audio-dir", default=".")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (0 = torch default)")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    model, cal = load(a.model_dir, a.device)

    if a.cmd == "embed":
        emb, q = embed_files(model, cal, a.files, a.device)
        np.savez(a.out, files=np.array(a.files), emb=emb, quality=q)
        print(f"wrote {a.out}: {len(a.files)} embeddings ({emb.shape[1]}-D)")
        return

    lines = [ln.split() for ln in Path(a.trials).read_text().splitlines() if ln.strip()]
    files = sorted({x for ln in lines for x in ln[:2]})
    emb, q = embed_files(model, cal, [Path(a.audio_dir) / f for f in files], a.device)
    pos = {f: i for i, f in enumerate(files)}
    ia, ib = np.array([pos[ln[0]] for ln in lines]), np.array([pos[ln[1]] for ln in lines])
    s = cal.score(emb, q, ia, ib)
    Path(a.out).write_text("".join(f"{ln[0]} {ln[1]} {v:.6f}\n" for ln, v in zip(lines, s)))
    print(f"wrote {a.out} ({len(lines):,} trials)")
    if all(len(ln) > 2 for ln in lines):
        import evaluate as ev
        y = np.array([ln[2] in ("target", "1") for ln in lines], int)
        print(f"EER {ev.eer(s, y):.3f} %")


if __name__ == "__main__":
    main()
