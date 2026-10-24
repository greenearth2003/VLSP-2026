"""Post-training scoring back-end: fit a calibrator once on TRAIN data, save it, score any trial list with it.

    # 0. (optional) WiSE: blend the trained model with a partner checkpoint of the SAME architecture
    .venv/bin/python experiments/distill_noiseaug_durationaware/backend.py wise TRAINED.pt PARTNER.pt OUT.pt --alpha 0.7
    #    E4 (dense B6): partner = the approved pretrained B6 file; E5 (low-rank): partner = its compressed start
    # 1. one embedding pass for the model (train cohort clean + noisy + short crops, silver): posttrain_embed.py
    # 2. fit (CPU, seconds) -> a small JSON (+ .npy cohort when AS-norm / centering need it)
    .venv/bin/python experiments/distill_noiseaug_durationaware/backend.py fit --tag E4_step6000 --out runs/_calib/E4_step6000/qmf2.json
    # 3. score a trial list + audio folder (silver, public, private ...): "fileA fileB score" lines in trial order
    .venv/bin/python experiments/distill_noiseaug_durationaware/backend.py score MODEL --calib runs/_calib/E4_step6000/qmf2.json \\
        --trials TRIALS.txt --audio-dir DIR --out results.txt

The calibrator is our QMF2 (sred/calibrate.py): logistic regression on the cosine and per-trial quality measures
(min / max over the two clips and their products with the cosine), fitted on train-cohort trials only (posttrain_eval.py
QMF2Files: same-file pairs excluded). Options (all off = the current best candidate, duration + SNR):
  --short   add 0.5-2 s cohort crops to the fitting trials    --norm    raw embedding norm as a quality measure
  --speech  embedding = mean of whole-clip and speech-only    --center  subtract the cohort mean embedding
  --asnorm  adaptive s-norm against the cohort as an extra input
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parent
SRED = EXP.parents[1]
sys.path[:0] = [str(SRED), str(EXP)]

FLAGS = ("short", "norm", "speech", "center", "asnorm")


def wise(trained, partner, out, alpha):
    """theta = alpha * theta_trained + (1 - alpha) * theta_partner, float tensors only (keys must match exactly).
    The partner may be a pretrained-format file ({model_config, state_dict}) or a train.py / lowrank checkpoint."""
    import torch
    t = torch.load(trained, map_location="cpu", weights_only=False)
    p = torch.load(partner, map_location="cpu", weights_only=False)
    pt = p["model"] if "model" in p else p["state_dict"]
    assert set(pt) == set(t["model"]), "trained and partner checkpoints have different architectures"
    sd = {k: (alpha * v.double() + (1 - alpha) * pt[k].double()).to(v.dtype) if v.is_floating_point() else v
          for k, v in t["model"].items()}
    torch.save({**{k: v for k, v in t.items() if k != "model"}, "model": sd,
                "step": f"wise{alpha}({Path(trained).name}, {Path(partner).name})"}, out)
    print(f"WiSE alpha {alpha}: {trained} + {partner} -> {out}")


def fit(tag, out, flags):
    from posttrain_eval import QMF2Files, build, l2n
    from sred.calibrate import _topk_stats
    D = dict(np.load(EXP / "runs/_calib" / tag / "posttrain.npz"))
    ce, cs, cf, cq, _, _ = build(D, flags["short"], flags["speech"])
    center = ce.mean(0) if flags["center"] else None
    if center is not None:
        ce = l2n(ce - center)
    names = ["speech", "snr"] + (["norm"] if flags["norm"] else [])
    quality = {"speech": cq[:, 0], "snr": cq[:, 1], "norm": cq[:, 2]}
    extra_fn = None
    if flags["asnorm"]:
        cmu, csd = _topk_stats(ce, ce, 300, spk=cs)
        extra_fn = lambda I, J, s: 0.5 * ((s - cmu[I]) / csd[I] + (s - cmu[J]) / csd[J])  # noqa: E731
    m = QMF2Files(names).fit_files(ce, cs, cf, {k: quality[k] for k in names}, extra_fn)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg = {"tag": tag, "flags": flags, "names": names, "mu": m.mu.tolist(), "sd": m.sd.tolist(),
           "coef": m.lr.coef_[0].tolist(), "intercept": float(m.lr.intercept_[0]), "fit_trials": m.n_trials,
           "fit_targets": m.n_targets}
    if center is not None or flags["asnorm"]:
        np.save(out.with_suffix(".cohort.npy"), ce.astype(np.float32))
        cfg["cohort_file"] = out.with_suffix(".cohort.npy").name
    if center is not None:
        cfg["center"] = center.tolist()
    out.write_text(json.dumps(cfg, indent=1))
    print(f"calibrator -> {out} ({m.n_trials:,} cohort trials, {m.n_targets:,} same-speaker; features {names})")


class Calibrator:
    def __init__(self, path):
        from sred.calibrate import QMF2
        self.path = Path(path)
        self.c = json.loads(self.path.read_text())
        self.q = QMF2(self.c["names"])
        self.mu, self.sd = np.array(self.c["mu"]), np.array(self.c["sd"])
        self.coef, self.b = np.array(self.c["coef"]), self.c["intercept"]
        self.center = np.array(self.c["center"]) if "center" in self.c else None
        self.cohort = np.load(self.path.parent / self.c["cohort_file"]) if "cohort_file" in self.c else None

    def embed_view(self, whole, speech):
        from posttrain_eval import l2n
        e = l2n(l2n(whole) + l2n(speech)) if self.c["flags"]["speech"] else l2n(whole)
        return l2n(e - self.center) if self.center is not None else e

    def score(self, emb, q, ia, ib):
        """emb: per-file embeddings from embed_view; q: [n, 3] speech s, SNR dB, raw norm."""
        from sred.calibrate import _topk_stats
        s = np.sum(emb[ia] * emb[ib], 1)
        extra = None
        if self.c["flags"]["asnorm"]:
            mu, sd = _topk_stats(emb, self.cohort, 300)
            extra = 0.5 * ((s - mu[ia]) / sd[ia] + (s - mu[ib]) / sd[ib])
        qd = {"speech": q[:, 0], "snr": q[:, 1], "norm": q[:, 2]}
        X = self.q._feats(s, {k: qd[k][ia] for k in self.c["names"]}, {k: qd[k][ib] for k in self.c["names"]}, extra)
        return ((X - self.mu) / self.sd) @ self.coef + self.b


def score(model_spec, calib, trials, audio_dir, out):
    import soundfile as sf
    import evaluate as ev
    from calibrate_eval import CAP, quality
    from posttrain_embed import raw, speech_only
    cal = Calibrator(calib)
    model, _ = ev.load_student(model_spec)
    model = model.cuda().eval()
    lines = [ln.split() for ln in Path(trials).read_text().splitlines() if ln.strip()]
    files = sorted({x for ln in lines for x in ln[:2]})
    W, S, Q = [], [], []
    for i, f in enumerate(files):
        x, _ = sf.read(str(Path(audio_dir) / f), dtype="float32", always_2d=False)
        x = x if x.ndim == 1 else x.mean(1)
        w = raw(model, x)
        W.append(w)
        S.append(raw(model, speech_only(x[:CAP])) if cal.c["flags"]["speech"] else w)
        Q.append((*quality(x[:CAP]), float(np.linalg.norm(w))))
        if i % 5000 == 0:
            print(f"  {i}/{len(files)} files", flush=True)
    emb = cal.embed_view(np.stack(W), np.stack(S))
    pos = {f: i for i, f in enumerate(files)}
    ia, ib = np.array([pos[ln[0]] for ln in lines]), np.array([pos[ln[1]] for ln in lines])
    s = cal.score(emb, np.array(Q, np.float32), ia, ib)
    Path(out).write_text("".join(f"{ln[0]} {ln[1]} {v:.6f}\n" for ln, v in zip(lines, s)))
    print(f"wrote {out} ({len(lines):,} trials)")
    if all(len(ln) > 2 for ln in lines):
        y = np.array([ln[2] == "target" for ln in lines], int)
        print(f"EER {ev.eer(s, y):.3f} %")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--tag", required=True)
    f.add_argument("--out", required=True)
    for k in FLAGS:
        f.add_argument(f"--{k}", action="store_true")
    w = sub.add_parser("wise")
    w.add_argument("trained")
    w.add_argument("partner")
    w.add_argument("out")
    w.add_argument("--alpha", type=float, default=0.7)
    s = sub.add_parser("score")
    s.add_argument("model")
    s.add_argument("--calib", required=True)
    s.add_argument("--trials", required=True)
    s.add_argument("--audio-dir", required=True)
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "wise":
        wise(a.trained, a.partner, a.out, a.alpha)
    elif a.cmd == "fit":
        out = Path(a.out) if Path(a.out).is_absolute() else EXP / a.out
        fit(a.tag, out, {k: getattr(a, k) for k in FLAGS})
    else:
        score(a.model, a.calib, a.trials, a.audio_dir, a.out)


if __name__ == "__main__":
    main()
