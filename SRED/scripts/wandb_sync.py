"""Mirror training logs + evaluation results to Weights & Biases (project vlsp2026-sred), without touching the
training code: parses exp/logs/<run>.log ("step N k=v ... lr=... s/it") and exp/results.jsonl.

    python scripts/wandb_sync.py RUN [RUN ...]            # backfill finished runs, then exit
    python scripts/wandb_sync.py --follow RUN [RUN ...]   # keep tailing until each run is trained and fully evaluated

W&B run id = run name, resume="allow", so re-running only appends new steps. Eval results (QMF-dur EER per
split) go to the run summary; early-review screen results (exp/logs/early_review_<run>.log) are logged at their step.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import wandb

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "exp/logs"
STEP = re.compile(r"^step (\d+) (.*?) lr=([0-9.e+-]+) ([0-9.]+)s/it")
REVIEW = re.compile(r"step(\d+)\.pt.*screen EER ([0-9.]+) \(E2b ([0-9.]+), delta ([+-][0-9.]+)")
EVAL_RUN = {  # run name -> model path whose evaluations count for it
    "M3_p80_hard": "exp/ft/W_M3_p80_hard/avg.pt", "M4_curr": "exp/ft/W_M4_curr/avg.pt",
    "M1_p50": "exp/ft/W_M1_p50/avg.pt", "M2_p80": "exp/ft/W_M2_p80/avg.pt",
}
GROUP = lambda r: "denoiser" if r.startswith("D0") else "noise-aug" if r[0] in "MN" else "other"  # noqa: E731


def parse(run):
    f = LOGS / f"{run}.log"
    pts = {}
    if f.exists():
        for line in f.read_text(errors="ignore").splitlines():
            m = STEP.match(line)
            if m:
                d = {k: float(v) for k, v in (kv.split("=") for kv in m.group(2).split())}
                d.update(lr=float(m.group(3)), sec_per_it=float(m.group(4)))
                pts[int(m.group(1))] = d  # after a resume, later lines win
    return pts, f.exists() and "done best_dev_eer" in f.read_text(errors="ignore")


def evals(run):
    model = EVAL_RUN.get(run, f"exp/ft/{run}/avg.pt")
    out = {}
    for line in open(ROOT / "exp/results.jsonl"):
        d = json.loads(line)
        if d.get("model") == model and d.get("mode", "full") == "full":
            out[f"eer/{d['split']}"] = d["results"]["qmf_dur"]["eer"]
    return out


def reviews(run):
    f = LOGS / f"early_review_{run}.log"
    out = []
    if f.exists():
        for line in f.read_text().splitlines():
            m = REVIEW.search(line)
            if m:
                out.append((int(m.group(1)), float(m.group(2)), float(m.group(4))))
    return out


def sync(run, state):
    pts, done = parse(run)
    st = state.setdefault(run, {"last": 0, "reviews": 0})
    new = [(s, d) for s, d in sorted(pts.items()) if s > st["last"]]
    revs = reviews(run)
    if not new and len(revs) == st["reviews"] and not done:
        return done
    args_f = ROOT / f"exp/ft/{run}/args.json"
    cfg = json.loads(args_f.read_text()) if args_f.exists() else {}
    r = wandb.init(project="vlsp2026-sred", id=run, name=run, resume="allow", group=GROUP(run), config=cfg,
                   reinit="finish_previous", settings=wandb.Settings(silent=True))
    for s, d in new:
        r.log({f"train/{k}": v for k, v in d.items()}, step=s)
        st["last"] = s
    for s, e, delta in revs[st["reviews"]:]:
        if s > st["last"]:
            r.log({"screen/eer": e, "screen/delta_vs_e2b": delta}, step=s)
        else:
            r.summary[f"screen/step{s}_delta_vs_e2b"] = delta
    st["reviews"] = len(revs)
    for k, v in evals(run).items():
        r.summary[k] = v
    r.finish()
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--follow", action="store_true")
    a = ap.parse_args()
    state = {}
    pending = list(a.runs)
    while pending:
        # keep following until training is done AND all three evaluations are in
        pending = [run for run in pending if not (sync(run, state) and len(evals(run)) >= 3)]
        if not a.follow:
            break
        time.sleep(120)
    if a.follow:  # one last pass for evaluations written after training ended
        for run in a.runs:
            sync(run, state)


if __name__ == "__main__":
    main()
