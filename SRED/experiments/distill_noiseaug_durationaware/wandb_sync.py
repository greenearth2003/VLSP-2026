"""Mirror this experiment's training logs and silver evaluations to Weights & Biases (project vlsp2026-sred, group
distill-noiseaug-durationaware) without touching the training code: parses runs/<run>/log.txt and
runs/_eval/<run>__step<k>[__suffix].json. Runs trained on the other server are read from the HF repo first.

    .venv/bin/python experiments/distill_noiseaug_durationaware/wandb_sync.py E0 E1 E0_s1_l40s        # backfill
    .venv/bin/python experiments/distill_noiseaug_durationaware/wandb_sync.py --follow E4-b6-teacher-finetune

W&B run id = "dnd-<run>" with resume="allow", so re-running only appends new steps.
Logged per step: train/loss, train/<view><dim>_loss and _acc (e.g. long192_acc), train/kd_*, alpha, margin_scale,
lr, sec_per_step, gpu_mem_gb, bins; eval/<metric> at each evaluated checkpoint step.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import wandb

EXP = Path(__file__).resolve().parent
E = "experiments/distill_noiseaug_durationaware"
LINE = re.compile(r"^step (\d+)(?: \(heads only\))? loss=([0-9.]+) \[(.*?)\] alpha=([0-9.]+) margin_x=([0-9.]+) "
                  r"lr=([0-9.e+-]+) bins=([0-9.]+)/([0-9.]+)s ([0-9.]+)s/step mem=([0-9.]+)G")
HEAD = re.compile(r"([ls])(\d+):([0-9.]+)/([0-9.]+)")
KD = re.compile(r"(\w+_(?:emb|rel)):([0-9.]+)")
EVAL = re.compile(r"__step(\d+)(?:__\w+)?\.json$")


def fetch_remote(run):
    """Runs trained on server B live on HF: refresh their log and eval JSONs."""
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    paths = [f.path for f in api.list_repo_tree("pieceofcakee/vlsp2026", repo_type="dataset",
                                                path_in_repo=f"{E}/runs", recursive=True)
             if f.path.endswith(f"runs/{run}/log.txt") or (f"/_eval/{run}__step" in f.path and f.path.endswith(".json"))]
    for p in paths:
        hf_hub_download("pieceofcakee/vlsp2026", p, repo_type="dataset", local_dir=EXP.parents[1], force_download=True)


def parse(run):
    pts = {}
    f = EXP / "runs" / run / "log.txt"
    if f.exists():
        for line in f.read_text(errors="ignore").splitlines():
            m = LINE.match(line)
            if not m:
                continue
            heads, kd_part = (m.group(3).split(" | kd ") + [""])[:2]
            d = {"train/loss": float(m.group(2)), "alpha": float(m.group(4)), "margin_scale": float(m.group(5)),
                 "lr": float(m.group(6)), "bins/short_s": float(m.group(7)), "bins/long_s": float(m.group(8)),
                 "sec_per_step": float(m.group(9)), "gpu_mem_gb": float(m.group(10)),
                 "heads_only": int("(heads only)" in line)}
            for v, dim, ls, acc in HEAD.findall(heads):
                view = "long" if v == "l" else "short"
                d[f"train/{view}{dim}_loss"], d[f"train/{view}{dim}_acc"] = float(ls), float(acc)
            for k, x in KD.findall(kd_part):
                d[f"train/kd_{k}"] = float(x)
            pts[int(m.group(1))] = d  # after a resume, later lines win
    evals = {}
    for j in sorted((EXP / "runs/_eval").glob(f"{run}__step*.json")):
        m = EVAL.search(j.name)
        if m and "cleanaudio" not in j.name:
            r = json.loads(j.read_text())
            evals[int(m.group(1))] = {f"eval/{k.replace(' ', '_')}": v for k, v in r.items() if isinstance(v, (int, float))}
    done = f.exists() and "done " in f.read_text(errors="ignore")
    return pts, evals, done


def sync(run, state):
    pts, evals, done = parse(run)
    st = state.setdefault(run, {"last": 0, "evals": set()})
    new = [(s, d) for s, d in sorted(pts.items()) if s > st["last"]]
    new_ev = [(s, d) for s, d in sorted(evals.items()) if s not in st["evals"]]
    if not new and not new_ev:
        return done, len(evals)
    cfg_f = EXP / "runs" / run / "config.json"
    cfg = json.loads(cfg_f.read_text()) if cfg_f.exists() else {}
    r = wandb.init(project="vlsp2026-sred", id=f"dnd-{run}", name=run, resume="allow",
                   group="distill-noiseaug-durationaware", config=cfg, reinit="finish_previous",
                   settings=wandb.Settings(silent=True))
    r.define_metric("eval/*", step_metric="eval_step")
    for s, d in new:
        r.log(d, step=s)
        st["last"] = s
    for s, d in new_ev:  # evals arrive after training: log them on their own axis so they never go back in time
        r.log({**d, "eval_step": s})
        st["evals"].add(s)
    r.finish()
    print(f"{run}: +{len(new)} train points (last step {st['last']}), +{len(new_ev)} evals -> {r.url}", flush=True)
    return done, len(evals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--follow", action="store_true", help="keep syncing every 10 min until trained and 3 evals are in")
    ap.add_argument("--remote", action="store_true", help="fetch the runs' logs and eval JSONs from HF first")
    a = ap.parse_args()
    state = {}
    while True:
        finished = True
        for run in a.runs:
            if a.remote or a.follow:
                try:
                    fetch_remote(run)
                except Exception as e:  # HF hiccup: keep the last copy, try again next round
                    print(f"{run}: HF fetch failed ({e})", file=sys.stderr)
            done, n_eval = sync(run, state)
            finished &= done and n_eval >= 3
        if not a.follow or finished:
            break
        time.sleep(600)


if __name__ == "__main__":
    main()
