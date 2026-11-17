"""Candidate D runner: signal diagnostic (games 1-2), then the protocol table with and without tags.

    python validation/candidates/run_d.py diag        # how often the signal names the right team
    python validation/candidates/run_d.py table       # protocol table, seeds 0 and 1
    python validation/candidates/run_d.py final W     # seed 2 check of weight W

Choose on games 1-2, confirm on game 3. Never reads BAR-ATH labels.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [ROOT, os.path.join(ROOT, "validation"), HERE, os.path.join(ROOT, "evaluation")]
import d_shape  # noqa: E402
import metrica  # noqa: E402
import possession  # noqa: E402
import report  # noqa: E402
import synthetic_tags  # noqa: E402
from possession_metrics import evaluate, label_codes  # noqa: E402

LABELS = np.array(list("HADX"))
WEIGHTS = (0.2, 0.45, 1.0)   # fixed before any table was run


def blocks_of(halves):
    return [(h + 1, a, b) for h, (a, b) in enumerate(halves)]


def diag():
    print("live frames with the ball missing: does sign(e_shape) name the team with the ball?")
    for game in (1, 2):
        tracking, is_home, halves, gt_codes = report.prepared(game, 0)
        gt = LABELS[gt_codes]
        players = tracking[:, :-1, :2].astype(float) * np.asarray(metrica.PITCH)
        e = d_shape.shape_signal(players, is_home, blocks_of(halves))
        live_missing = np.isin(gt, ["H", "A"]) & ~np.isfinite(tracking[:, -1, 0])
        truth = np.where(gt[live_missing] == "H", 1, -1)
        sig = e[live_missing]
        for lo in (0.0, 0.25, 0.5):
            m = np.abs(sig) > lo
            print("  game %d  |e| > %.2f: %5.1f%% of frames, right team %.1f%%"
                  % (game, lo, 100 * m.mean(), 100 * np.mean(np.sign(sig[m]) == truth[m])))


def score(game, seed, weight, tags):
    tracking, is_home, halves, gt = report.prepared(game, seed)
    n = len(tracking)
    info = synthetic_tags.tags_info(synthetic_tags.synthetic_events(game), n) if tags else possession.no_tags(n)
    fn = possession.label_match if weight is None else d_shape.build(weight, signal="spread")
    r = evaluate(gt, label_codes(fn(tracking, is_home, metrica.PITCH, blocks_of(halves), info)))
    return r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"]


def table(seeds=(0, 1), weights=(None,) + WEIGHTS):
    for tags in (False, True):
        print("\n%s" % ("SYNTHETIC TAGS" if tags else "NO TAGS"))
        print("config        g1 spell  g2 spell  g3 spell   m12     macro(mean)  poss g1/g2/g3")
        base = None
        for w in weights:
            res = {g: np.mean([score(g, s, w, tags) for s in seeds], axis=0) for g in (1, 2, 3)}
            m12 = np.mean([res[1][0], res[2][0]])
            macro = np.mean([res[g][1] for g in res])
            row = (m12, res[3][0], macro)
            verdict = ""
            if base is None:
                base = row
            else:
                ok = row[0] - base[0] >= 0.01 and row[1] >= base[1] and macro >= base[2] - 0.01
                verdict = "ACCEPT" if ok else "reject"
            print("%-12s  %.4f    %.4f    %.4f    %.4f  %.4f       %.0f/%.0f/%.0f  %s" % (
                "baseline" if w is None else "shape %.2f" % w, res[1][0], res[2][0], res[3][0], m12,
                macro, res[1][2], res[2][2], res[3][2], verdict), flush=True)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "table"
    if what == "diag":
        diag()
    elif what == "table":
        table()
    elif what == "final":
        table(seeds=(2,), weights=(None, float(sys.argv[2])))
