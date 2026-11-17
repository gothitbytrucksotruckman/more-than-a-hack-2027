"""Score possession.py on Metrica's open sample games with the official metric.

    python validation/report.py                 # games 1-3, two random ball-gap patterns each

Never reads the competition match. Labels come from validation/metrica.py. The sample games have
no Smart Tagging, so this measures the tracking rules only.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import metrica  # noqa: E402
import possession  # noqa: E402
from possession_metrics import evaluate, label_codes  # noqa: E402

SEEDS = (0, 1)
_cache = {}


def prepared(game, seed):
    """(tracking, is_home, halves, gt codes) for one game and one random gap pattern. Cached."""
    if (game, seed) not in _cache:
        if game not in _cache:
            _cache[game] = metrica.load(game)
        xy, is_home, period, labels = _cache[game]
        _cache[(game, seed)] = (metrica.atd_like(xy, period, seed), is_home,
                                metrica.halves_of(period), label_codes(labels))
    return _cache[(game, seed)]


def score(game, seed):
    tracking, is_home, halves, gt = prepared(game, seed)
    blocks = [(half + 1, first, last) for half, (first, last) in enumerate(halves)]
    # The sample games have no Smart Tagging, so every tag-based rule runs with empty input here.
    pred = possession.label_match(tracking, is_home, metrica.PITCH, blocks,
                                  possession.no_tags(len(tracking)))
    r = evaluate(gt, label_codes(pred))
    d = r["per_class"]["D"] if isinstance(r.get("per_class"), dict) and "D" in r["per_class"] else None
    return {"spell": r["spell_f1_mean"], "macro": r["macro_f1"], "pred_poss": r["n_pred_spells"],
            "gt_poss": r["n_gt_spells"], "dead_f1": d["f1"] if d else float("nan"),
            "alive_dead": r["alive_dead_f1"], "team": r["team_acc_both_alive"]}


def per_game(games=(1, 2, 3), seeds=SEEDS):
    """{game: mean over seeds of each number}."""
    out = {}
    for game in games:
        rows = [score(game, seed) for seed in seeds]
        out[game] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    return out


if __name__ == "__main__":
    results = per_game()
    print("game  spellF1  macroF1  alive/dead F1  team acc  possessions pred/true")
    for game, r in results.items():
        print("{:>4}  {:.4f}   {:.4f}   {:.4f}         {:.4f}    {:.0f} / {:.0f}".format(
            game, r["spell"], r["macro"], r["alive_dead"], r["team"], r["pred_poss"], r["gt_poss"]))
    print("mean  {:.4f}   {:.4f}".format(np.mean([r["spell"] for r in results.values()]),
                                         np.mean([r["macro"] for r in results.values()])))
