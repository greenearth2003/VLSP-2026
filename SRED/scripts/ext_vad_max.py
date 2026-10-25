"""External reference scorer (analysis only, not part of any submission): hynt ResNet293 ONNX embeddings + Silero VAD.

    python scripts/ext_vad_max.py embed --splits public_rumble_v2 private
    python scripts/ext_vad_max.py score            # silver EER of each variant + pseudo-label comparison with our best

Per clip (CPU, one thread per worker): Silero VAD (ONNX, threshold 0.5, >= 250 ms speech, splits at >= 300 ms silence) ->
  full    whole-clip embedding
  vadcat  embedding of all detected speech joined (whole clip if < 0.5 s detected)
  segs    embeddings of the up to 8 longest speech segments >= 1 s (one per speech turn: the "split speakers")
A trial scores by cosine of full / vadcat, or MAX over every cross pair of {full, vadcat, segs} (best-matched voices).
Cache: exp/emb/ext_hynt_resnet293/<split>_vadseg.npz.
"""
import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from chunk_embed import split_files  # noqa: E402
from sred.data import read_trials, unique_files  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ONNX = "/mnt/tsharp/thucth/dataset/vlsp2026-sred/external/hynt_Speaker-Embedding-30M/speaker_embedding_resnet293.onnx"
OUT = ROOT / "exp/emb/ext_hynt_resnet293"
K, SR = 8, 16000
_s = {}


def _init():
    import onnxruntime as ort
    import torch
    from silero_vad import load_silero_vad
    torch.set_num_threads(1)
    o = ort.SessionOptions()
    o.intra_op_num_threads, o.inter_op_num_threads = 1, 1
    _s["emb"] = ort.InferenceSession(ONNX, o, providers=["CPUExecutionProvider"])
    _s["vad"] = load_silero_vad(onnx=True)


def _emb(x16):
    import torch
    import torchaudio.compliance.kaldi as kaldi
    x = torch.from_numpy(x16)
    if len(x) < 400:
        x = x.repeat(int(np.ceil(400 / max(1, len(x)))))[:400]
    f = kaldi.fbank(x[None], num_mel_bins=80, frame_length=25, frame_shift=10, dither=0.0, sample_frequency=SR)
    e = _s["emb"].run(["embs"], {"feats": (f - f.mean(dim=0))[None].numpy()})[0][0]
    return e / np.linalg.norm(e)


def _process(path):
    import soundfile as sf
    import torch
    from silero_vad import get_speech_timestamps
    d, sr = sf.read(path, dtype="int16")
    x16 = d.astype(np.float32) if d.ndim == 1 else d.astype(np.float32).mean(1)
    ts = get_speech_timestamps(torch.from_numpy(x16 / 32768.0), _s["vad"], sampling_rate=SR, threshold=0.5,
                               min_speech_duration_ms=250, min_silence_duration_ms=300, speech_pad_ms=30)
    full = _emb(x16)
    sp = np.concatenate([x16[t["start"]:t["end"]] for t in ts]) if ts else x16[:0]
    vadcat = _emb(sp) if len(sp) >= SR // 2 else full
    segs = sorted([(t["end"] - t["start"], t["start"], t["end"]) for t in ts if t["end"] - t["start"] >= SR], reverse=True)[:K]
    S = np.zeros((K, 256), np.float32)
    for i, (_, a, b) in enumerate(segs):
        S[i] = _emb(x16[a:b])
    return full, vadcat, S, len(segs), sum(t["end"] - t["start"] for t in ts) / SR


def embed(splits, workers):
    OUT.mkdir(parents=True, exist_ok=True)
    for split in splits:
        p = OUT / f"{split}_vadseg.npz"
        if p.exists():
            print("cached", p)
            continue
        files, root, _ = split_files(split)
        with Pool(workers, initializer=_init) as pool:
            r = pool.map(_process, [str(root / f) for f in files], chunksize=8)
        np.savez(p, full=np.stack([x[0] for x in r]), vadcat=np.stack([x[1] for x in r]), segs=np.stack([x[2] for x in r]),
                 n=np.array([x[3] for x in r]), speech=np.array([x[4] for x in r]))
        print(f"{split}: {len(files)} clips -> {p}", flush=True)


def trial_scores(z, ia, ib, chunk=20000):
    full, vc, segs, n = z["full"], z["vadcat"], z["segs"], z["n"]
    C = np.concatenate([full[:, None], vc[:, None], segs], 1)  # [N, 2 + K, 256]
    valid = np.concatenate([np.ones((len(n), 2), bool), np.arange(K)[None] < n[:, None]], 1)
    out = {"full": (full[ia] * full[ib]).sum(1), "vadcat": (vc[ia] * vc[ib]).sum(1), "max": np.zeros(len(ia), np.float32)}
    for s in range(0, len(ia), chunk):
        a, b = ia[s:s + chunk], ib[s:s + chunk]
        P = np.einsum("pkd,pld->pkl", C[a], C[b])
        m = valid[a][:, :, None] & valid[b][:, None, :]
        out["max"][s:s + chunk] = np.where(m, P, -2).reshape(len(a), -1).max(1)
    return out


def eer_thr(s, y):
    o = np.argsort(s)
    yo = y[o]
    frr = np.cumsum(yo) / yo.sum()
    far = 1 - np.cumsum(1 - yo) / (1 - yo).sum()
    k = np.argmin(np.abs(frr - far))
    return 100 * (frr[k] + far[k]) / 2, s[o][k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["embed", "score"])
    ap.add_argument("--splits", nargs="+", default=["public_rumble_v2", "private"])
    ap.add_argument("--workers", type=int, default=min(30, os.cpu_count()))
    ap.add_argument("--ours-silver", default="exp/scores/silver/S3_seg_ms025.txt")
    ap.add_argument("--ours-private", default="submissions/final_private/results_result-private-S3-seg-ms025-t0.txt")
    a = ap.parse_args()
    if a.phase == "embed":
        return embed(a.splits, a.workers)

    meta = pd.read_csv("/mnt/tsharp/thucth/dataset/vlsp2026-sred/hf_repo/standins/trial_metadata.csv")
    pairs, y = read_trials()
    files = unique_files(pairs)
    pos = {f: i for i, f in enumerate(files)}
    ia = np.array([pos[x] for x, _ in pairs]); ib = np.array([pos[x] for _, x in pairs])
    cl = pd.DataFrame(pairs, columns=["file_a", "file_b"]).merge(meta, how="left").in_cleaned.fillna(False).to_numpy(bool)
    ext = trial_scores(np.load(OUT / "public_rumble_v2_vadseg.npz"), ia, ib)
    ours = pd.DataFrame(pairs, columns=["a", "b"]).merge(
        pd.read_csv(ROOT / a.ours_silver, sep=" ", header=None, names=["a", "b", "s"]), how="left").s.to_numpy()
    e_o, t_o = eer_thr(ours, y)
    print(f"silver, all trials / cleaned list: ours (S3 + gated trimming + multi-speaker) {e_o:.2f} / {eer_thr(ours[cl], y[cl])[0]:.2f}")
    thr = {}
    for k, s in ext.items():
        e, t = eer_thr(s, y)
        thr[k] = t
        print(f"  external ResNet293 {k:7s} {e:.2f} / {eer_thr(s[cl], y[cl])[0]:.2f}")
    # how informative are the external model's decisions about OUR errors? (silver, labels known)
    best = min(ext, key=lambda k: eer_thr(ext[k], y)[0])
    d_o, d_e = ours > t_o, ext[best] > thr[best]
    dis = d_o != d_e
    wrong_o = d_o != (y == 1)
    print(f"silver, external '{best}' at its EER threshold vs ours at ours: disagree on {dis.mean():.1%} of trials; "
          f"when they disagree, ours is wrong {wrong_o[dis].mean():.0%}; of our errors, the external model flags {dis[wrong_o].mean():.0%}")

    # private: pseudo-labels = external decisions at the silver EER threshold; our decisions at our silver EER threshold
    ppairs, _ = read_trials(ROOT / "submissions/final_private/trials_2col.txt")
    pfiles = unique_files(ppairs)
    ppos = {f: i for i, f in enumerate(pfiles)}
    ja = np.array([ppos[x] for x, _ in ppairs]); jb = np.array([ppos[x] for _, x in ppairs])
    pext = trial_scores(np.load(OUT / "private_vadseg.npz"), ja, jb)
    po = pd.read_csv(ROOT / a.ours_private, sep=" ", header=None, names=["a", "b", "s"])
    assert (po.a.to_numpy() == np.array([x for x, _ in ppairs])).all()
    s_o = po.s.to_numpy()
    for k in (best, "max"):
        pl = (pext[k] > thr[k]).astype(int)
        dec = s_o > t_o
        print(f"\nprivate, pseudo-labels from external '{k}' (accept {pl.mean():.1%} of trials) vs ours (accept {dec.mean():.1%}):")
        print(f"  decisions agree on {np.mean(dec == (pl == 1)):.1%}; our 'EER' against these pseudo-labels {eer_thr(s_o, pl)[0]:.2f}; "
              f"score correlation {np.corrcoef(s_o, pext[k])[0, 1]:.3f}")
        print(f"  disagreements: we accept / they reject {np.mean(dec & (pl == 0)):.1%}, we reject / they accept {np.mean(~dec & (pl == 1)):.1%}")
        fv = pd.read_csv(ROOT / "exp/analysis/features_private.csv").set_index("rel")
        A, B = fv.loc[[x for x, _ in ppairs]].reset_index(drop=True), fv.loc[[x for _, x in ppairs]].reset_index(drop=True)
        short = (np.minimum(A.speech_dur, B.speech_dur) < 1).to_numpy()
        dis_p = dec != (pl == 1)
        print(f"  disagreement rate: shorter clip < 1 s speech {dis_p[short].mean():.1%} vs others {dis_p[~short].mean():.1%}")
        out = ROOT / f"exp/analysis/ext_pseudo_{k}_private_disagreements.csv"
        pd.DataFrame({"a": [x for x, _ in ppairs], "b": [x for _, x in ppairs], "ours": s_o, "ext": pext[k], "pseudo": pl,
                      "ours_accept": dec.astype(int)})[dis_p].to_csv(out, index=False)
        print(f"  {dis_p.sum():,} disagreeing private trials -> {out}")


if __name__ == "__main__":
    main()
