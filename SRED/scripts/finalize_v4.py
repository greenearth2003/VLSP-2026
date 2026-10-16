"""Pre-registered final selection after round v4 (see EXPERIMENTS.md).

Reads V comparisons from exp/logs/v4_results.jsonl and prints the training args of the final recipe:
  - recipe change  = best of S01/S02/S03 that beats S00 on V (QMF) with P(better) > 0.95
  - data change    = S04 (drop mixed-voice IDs / bad samples) if it beats S00 with P > 0.95
Writes exp/logs/final_decision.json. Prints 'NONE' if nothing qualifies (W01 stays).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RECIPES = {
    "S01_balanced": "--crop-sec 6 --crop-dist balanced",
    "S02_consistency": "--crop-sec 6 --consistency 0.1 --cons-mu 0.5",
    "S03_bal_pk_supcon": "--crop-sec 6 --crop-dist balanced --pk 4 --supcon 0.1",
}
DATA = {"S04_dropbad": "--train-list lists/train_drop_bad.txt"}

rows = [json.loads(x) for x in (ROOT / "exp/logs/v4_results.jsonl").read_text().splitlines() if x.strip()]
val = [r for r in rows if r.get("val") == "v4"]
eer = {r["sys"]: r["eer"] for r in val if "sys" in r and r["backend"] == "qmf"}
cmp = {r["cmp"]: r for r in val if "cmp" in r and r["backend"] == "qmf"}


def vs_s00(name):
    return cmp.get(f"W_{name}_avg vs W_S00_base_avg")


passing = {n: vs_s00(n) for n in list(RECIPES) + list(DATA) if vs_s00(n) and vs_s00(n)["p_better"] > 0.95}
rec = [n for n in RECIPES if n in passing]
best_rec = min(rec, key=lambda n: eer[f"W_{n}_avg"]) if rec else None
data = [n for n in DATA if n in passing]
args = " ".join(([RECIPES[best_rec]] if best_rec else []) + [DATA[n] for n in data])
dec = dict(v_eer_qmf=eer, comparisons_vs_S00={n: vs_s00(n) for n in list(RECIPES) + list(DATA)},
           passing=list(passing), chosen_recipe=best_rec, chosen_data=data, train_args=args or None)
(ROOT / "exp/logs/final_decision.json").write_text(json.dumps(dec, indent=1))
print(args if args else "NONE")
