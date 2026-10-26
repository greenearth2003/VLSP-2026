"""Compare a reproduced score file with the submitted one (same trials, same order).
  python tools/compare_results.py <reproduced results.txt> <submitted results.txt>
Prints the number of lines, pair-order mismatches, byte-identical lines and the max |score difference|.
A faithful reproduction has 0 mismatches and max |diff| <= 1e-5 (float summation order on another GPU / CPU can change
the 6th decimal of a few scores; this does not change the EER)."""
import sys

import numpy as np


def load(p):
    rows = [ln.split() for ln in open(p) if ln.strip()]
    return [(r[0], r[1]) for r in rows], np.array([float(r[2]) for r in rows]), [" ".join(r) for r in rows]


def main():
    pa, sa, la = load(sys.argv[1])
    pb, sb, lb = load(sys.argv[2])
    assert len(pa) == len(pb), f"line count {len(pa)} != {len(pb)}"
    mism = sum(x != y for x, y in zip(pa, pb))
    same = sum(x == y for x, y in zip(la, lb))
    d = float(np.max(np.abs(sa - sb)))
    print(f"lines {len(pa):,} | pair-order mismatches {mism} | identical lines {same:,} ({100 * same / len(pa):.1f} %) | "
          f"max |score diff| {d:.2e} | score correlation {np.corrcoef(sa, sb)[0, 1]:.8f}")
    ok = mism == 0 and d <= 1e-5
    print("REPRODUCED" if ok else "NOT REPRODUCED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
