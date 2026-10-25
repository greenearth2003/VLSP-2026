"""Evaluate an external SpeechBrain ECAPA-TDNN speaker model on the public test and the noisy stand-ins.

    python scripts/eval_external_ecapa.py --model-dir /path/to/CakeVoiceID-ECAPA-TDNN-v1.0 --splits public public_noisy public_realnoise

Embeddings (raw, and with the model's own global embedding normalisation) are cached in exp/emb/ext_<name>/.
Scored like our systems: cosine, and QMF(dur) fitted on the same train cohort (no test labels). EER % / minDCF.
For information only: this model is not on the organisers' list of approved checkpoints.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sred.calibrate import QMF  # noqa: E402
from sred.data import TRAIN_DIR, cohort_subset, read_train_list, read_trials, unique_files  # noqa: E402
from sred.metrics import compute_eer, compute_min_dcf  # noqa: E402
from sred.scoring import l2n  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def load(model_dir):
    from speechbrain.inference.speaker import EncoderClassifier
    hp = Path(model_dir) / "hyperparams.yaml"
    # the card's yaml points at the HF repo; load the local copy instead
    text = hp.read_text().replace("pretrained_path: CakeByVPBank/CakeVoiceID-ECAPA-TDNN-v1.0", f"pretrained_path: {model_dir}")
    local = Path(model_dir) / "hyperparams_local.yaml"
    local.write_text(text)
    return EncoderClassifier.from_hparams(source=str(model_dir), hparams_file="hyperparams_local.yaml",
                                          savedir=str(Path(model_dir) / "sb_cache"), run_opts={"device": "cuda"})


@torch.inference_mode()
def embed(clf, files, root, cache):
    if cache.exists():
        z = np.load(cache)
        return z["raw"], z["norm"]
    raw, nrm = [], []
    for k, f in enumerate(files):
        x, sr = sf.read(str(root / f), dtype="float32")
        assert sr == 16000
        x = x[: 60 * sr]
        w = torch.from_numpy(x)[None].cuda()
        raw.append(clf.encode_batch(w, normalize=False)[0, 0].cpu().numpy())
        nrm.append(raw[-1])  # the card's normalizer.ckpt is empty (count 0): no normalised variant
        if k % 3000 == 0:
            print(f"  {root.name}: {k}/{len(files)}", flush=True)
    raw, nrm = np.stack(raw), np.stack(nrm)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, raw=raw, norm=nrm)
    return raw, nrm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--splits", nargs="+", default=["public", "public_noisy", "public_realnoise"])
    a = ap.parse_args()
    name = Path(a.model_dir).name
    clf = load(a.model_dir)
    n_params = sum(p.numel() for p in clf.mods.embedding_model.parameters())
    print(f"{name}: embedding model {n_params / 1e6:.2f}M parameters")
    pairs, y = read_trials()
    files = unique_files(pairs)
    pos = {f: i for i, f in enumerate(files)}
    ia = np.array([pos[p] for p, _ in pairs])
    ib = np.array([pos[q] for _, q in pairs])
    dur = pd.read_csv(ROOT / "exp/analysis/features_public.csv").set_index("rel").loc[files].dur.values
    ci = cohort_subset(read_train_list(), per_spk=5)
    cf, cs = [r for _, r in ci], [s for s, _ in ci]
    craw, cnrm = embed(clf, cf, TRAIN_DIR, ROOT / f"exp/emb/ext_{name}/traincohort5.npz")
    cdur = np.array([json.loads((ROOT / "lists/train_lengths.json").read_text())[r] / 16000 for r in cf])
    res = {}
    for split in a.splits:
        root = ROOT / ("data/Vi-SRED26/public_test/data" if split == "public" else f"data/{split}")
        raw, nrm = embed(clf, files, root, ROOT / f"exp/emb/ext_{name}/{split}.npz")
        for tag, E, C in (("raw", raw, craw),):
            E, C = l2n(E), l2n(C)
            cos = (E[ia] * E[ib]).sum(1)
            qmf = QMF(use_asnorm=False).fit(C, cs, cdur).score(E, dur, ia, ib)
            res[f"{split}/{tag}"] = dict(cos_eer=compute_eer(cos, y), cos_dcf=compute_min_dcf(cos, y),
                                         qmf_eer=compute_eer(qmf, y), qmf_dcf=compute_min_dcf(qmf, y))
            r = res[f"{split}/{tag}"]
            print(f"  {split:17s} {tag:4s}  cosine EER {r['cos_eer']:6.3f}  minDCF {r['cos_dcf']:.3f}  |  "
                  f"QMF dur EER {r['qmf_eer']:6.3f}  minDCF {r['qmf_dcf']:.3f}", flush=True)
    with open(ROOT / "exp/results.jsonl", "a") as f:
        f.write(json.dumps({"model": f"external:{name}", "params_M": round(n_params / 1e6, 3), "results": res}) + "\n")


if __name__ == "__main__":
    main()
