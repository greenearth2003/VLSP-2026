"""Evaluate an external ONNX speaker-embedding model (WeSpeaker-style front-end) for comparison only (not approved, not used
in any submission): hynt/Speaker-Embedding-30M-Low-Quality-Speech (ResNet293 28.6M / ResNet34 6.6M, cc-by-nc-4.0).

    python scripts/eval_external_onnx.py --onnx <dir>/speaker_embedding_resnet293.onnx --splits public public_rumble_v2 private

Front-end per the model card: 16 kHz int16-scale samples -> 80-dim Kaldi fbank (25/10 ms, no dither) -> per-utterance mean
subtraction; L2-normalised 256-d embeddings, cosine scoring. CPU onnxruntime, one thread per worker process.
Embeddings cached in exp/emb/ext_hynt_<model>/<split>_modefull.npy (rows in the split's usual file order).
Labelled splits: EER all / cleaned list for cosine and for the duration QMF fitted on train-cohort trials (as ours).
Private: cosine and QMF scores written to exp/scores/private/ext_hynt_<model>_{cos,qmf}.txt.
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from chunk_embed import split_files  # noqa: E402
from sred.data import PUBLIC_DIR, read_trials, unique_files  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
_sess = None


def _init(onnx_path):
    global _sess
    import onnxruntime as ort
    import torch
    torch.set_num_threads(1)
    o = ort.SessionOptions()
    o.intra_op_num_threads, o.inter_op_num_threads = 1, 1
    _sess = ort.InferenceSession(onnx_path, o, providers=["CPUExecutionProvider"])


def _embed(path):
    import soundfile as sf
    import torch
    import torchaudio.compliance.kaldi as kaldi
    data, sr = sf.read(path, dtype="int16", always_2d=True)
    assert sr == 16000, (path, sr)
    x = torch.from_numpy(data.astype(np.float32).mean(axis=1))
    if len(x) < 400:  # shorter than one 25-ms frame
        x = x.repeat(int(np.ceil(400 / max(1, len(x)))))[:400]
    feat = kaldi.fbank(x[None], num_mel_bins=80, frame_length=25, frame_shift=10, dither=0.0, sample_frequency=16000)
    feat = feat - feat.mean(dim=0)
    e = _sess.run(["embs"], {"feats": feat[None].numpy()})[0][0]
    return e / np.linalg.norm(e)


def files_of(split):
    if split == "public":
        files = unique_files(read_trials()[0])
        return files, PUBLIC_DIR, None
    files, root, dur = split_files(split)
    return files, root, dur


def eer(s, y):
    o = np.argsort(s)
    yo = y[o]
    frr = np.cumsum(yo) / yo.sum()
    far = 1 - np.cumsum(1 - yo) / (1 - yo).sum()
    k = np.argmin(np.abs(frr - far))
    return 100 * (frr[k] + far[k]) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--splits", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=min(28, os.cpu_count()))
    a = ap.parse_args()
    tag = "ext_hynt_" + Path(a.onnx).stem.replace("speaker_embedding_", "")
    out_dir = ROOT / "exp/emb" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    def embed_split(split):
        p = out_dir / f"{split}_modefull.npy"
        if p.exists():
            return np.load(p)
        files, root, _ = files_of(split)
        with Pool(a.workers, initializer=_init, initargs=(a.onnx,)) as pool:
            E = np.stack(pool.map(_embed, [str(root / f) for f in files], chunksize=16))
        np.save(p, E.astype(np.float32))
        print(f"  {split}: {len(files)} embeddings -> {p}", flush=True)
        return E

    # duration QMF fitted on train-cohort trials, as for our models (sred.calibrate.QMF, no AS-norm)
    from sred.calibrate import QMF
    from sred.data import cohort_subset, read_train_list
    ci = cohort_subset(read_train_list(), per_spk=5)
    lengths = json.loads((ROOT / "lists/train_lengths.json").read_text())
    cdur = np.array([lengths[r] / 16000 for _, r in ci])
    q = QMF(use_asnorm=False).fit(embed_split("traincohort5"), [s for s, _ in ci], cdur)

    meta = pd.read_csv("/mnt/tsharp/thucth/dataset/vlsp2026-sred/hf_repo/standins/trial_metadata.csv")
    for split in a.splits:
        E = embed_split(split)
        if split == "private":
            pairs, _ = read_trials(ROOT / "submissions/final_private/trials_2col.txt")
            files = unique_files(pairs)
            dur = np.load(ROOT / "lists/private_durations.npy")
        else:
            pairs, y = read_trials()
            files = unique_files(pairs)
            dur = pd.read_csv(ROOT / "exp/analysis/features_public.csv").set_index("rel").loc[files].dur.to_numpy()
        pos = {f: i for i, f in enumerate(files)}
        ia = np.array([pos[x] for x, _ in pairs]); ib = np.array([pos[x] for _, x in pairs])
        cos = (E[ia] * E[ib]).sum(1)
        qmf = q.score(E, dur, ia, ib)
        if split == "private":
            sd = ROOT / "exp/scores/private"
            sd.mkdir(parents=True, exist_ok=True)
            for nm, s in (("cos", cos), ("qmf", qmf)):
                (sd / f"{tag}_{nm}.txt").write_text("".join(f"{x} {z} {v:.6f}\n" for (x, z), v in zip(pairs, s)))
            print(f"{tag} private: scores written to {sd}/{tag}_{{cos,qmf}}.txt", flush=True)
            continue
        cl = pd.DataFrame(pairs, columns=["file_a", "file_b"]).merge(meta, how="left").in_cleaned.fillna(False).to_numpy(bool)
        print(f"{tag} {split:17s} cosine EER {eer(cos, y):.2f} (cleaned {eer(cos[cl], y[cl]):.2f}) | "
              f"QMF EER {eer(qmf, y):.2f} (cleaned {eer(qmf[cl], y[cl]):.2f})", flush=True)


if __name__ == "__main__":
    main()
