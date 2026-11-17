"""Protocol table for candidate B (over-splitting).

    python validation/candidates/run_b.py          # seeds 0 1
    python validation/candidates/run_b.py 0 1 2    # final check of a winner
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), os.path.dirname(os.path.dirname(HERE))]
import report  # noqa: E402  (puts the repo root and evaluation/ on the path)
import metrica  # noqa: E402
import possession  # noqa: E402
from possession_metrics import evaluate, label_codes  # noqa: E402
from b_oversplit import postfilter  # noqa: E402

# Every value tried, fixed before scoring. Seconds.
CONFIGS = {
    "baseline (micro 1.2)": {},
    "micro 2.0": {"micro_s": 2.0},
    "micro 3.0": {"micro_s": 3.0},
    "micro 2.0 + beforeD 2.0": {"micro_s": 2.0, "before_d": 2.0},
    "micro 3.0 + beforeD 3.0": {"micro_s": 3.0, "before_d": 3.0},
    "micro 2.0 + before/afterD 2.0": {"micro_s": 2.0, "before_d": 2.0, "after_d": 2.0},
    "fillD 3.0": {"fill_d": 3.0},
    "micro 2.0 + beforeD 2.0 + fillD 3.0": {"micro_s": 2.0, "before_d": 2.0, "fill_d": 3.0},
}


def main(seeds):
    res = {name: {} for name in CONFIGS}  # name -> game -> (spell, macro, pred, true)
    for game in (1, 2, 3):
        rows = {name: [] for name in CONFIGS}
        for seed in seeds:
            tracking, is_home, halves, gt = report.prepared(game, seed)
            blocks = [(half + 1, first, last) for half, (first, last) in enumerate(halves)]
            base = possession.label_match(tracking, is_home, metrica.PITCH, blocks,
                                          possession.no_tags(len(tracking)))
            for name, kw in CONFIGS.items():
                r = evaluate(gt, label_codes(postfilter(base, blocks, **kw)))
                rows[name].append((r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"]))
        for name in CONFIGS:
            res[name][game] = np.mean(rows[name], axis=0)

    print("seeds", seeds)
    print("{:<36} {:>22} {:>22} {:>22}  {:>6} {:>6}  verdict".format(
        "config", "game 1 spell/macro/pred", "game 2", "game 3", "m12", "macro"))
    b = res["baseline (micro 1.2)"]
    b12, b3, bmac = (b[1][0] + b[2][0]) / 2, b[3][0], np.mean([b[g][1] for g in (1, 2, 3)])
    for name, r in res.items():
        m12, mac = (r[1][0] + r[2][0]) / 2, np.mean([r[g][1] for g in (1, 2, 3)])
        ok = m12 - b12 >= 0.01 and r[3][0] >= b3 and mac >= bmac - 0.01
        cells = ["{:.3f}/{:.3f}/{:.0f}:{:.0f}".format(*r[g]) for g in (1, 2, 3)]
        print("{:<36} {:>22} {:>22} {:>22}  {:.3f} {:.3f}  {}".format(
            name, *cells, m12, mac, "-" if r is b else ("ACCEPT" if ok else "reject")))


if __name__ == "__main__":
    main(tuple(int(s) for s in sys.argv[1:]) or (0, 1))
