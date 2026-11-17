"""Candidate D: who has the ball while the tracker cannot see it, from how the two teams move.

When a team wins the ball it pushes up the pitch and the other team drops back. So while the ball
is missing (or in flight), the difference between the teams' forward speeds is evidence for who has
it:

    forward_T(t) = speed of team T's median x towards the opponent's goal, over 1 s, smoothed 2 s
    e_shape(t)   = clip((forward_H - forward_A) / SHAPE_SCALE, -1, 1)

added to the Viterbi evidence with weight SHAPE_WEIGHT, only on frames where the ball is missing or
in flight (where proximity says nothing). Which goal each team attacks is read per half from the
tracking alone: the team whose player is most often the deepest on the left defends the left.

possession.py is not edited: label_match is rebuilt from its own source with one added line, so
this module runs exactly the diff it proposes. Constants are round values set before any result.
"""

import inspect
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import possession as P  # noqa: E402

SHAPE_LAG = 1 * P.FPS          # forward speed measured over 1 s
SHAPE_SMOOTH = 2 * P.FPS + 1   # ...and smoothed over 2 s
SHAPE_SCALE = 2.0              # m/s difference that counts as full evidence
SHAPE_WEIGHT = 0.2             # weight in the Viterbi evidence (tried: 0.2, 0.45)


def home_attacks_right(players, is_home, first, last):
    """True if the home team attacks towards +x in this half: a home player (its keeper) is the
    deepest on the left more often, so home defends the left goal."""
    x = players[first:last + 1, :, 0]
    seen = np.isfinite(x).sum(axis=1) >= 10
    deepest = np.nanargmin(np.where(np.isfinite(x[seen]), x[seen], np.inf), axis=1)
    return np.mean(is_home[deepest]) > 0.5


def shape_signal(players, is_home, blocks):
    """e_shape in [-1, 1] per frame; > 0 means home is the team pushing forward."""
    n = len(players)
    out = np.zeros(n)
    with np.errstate(invalid="ignore"), __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore")
        med_h = np.nanmedian(players[:, is_home, 0], axis=1)
        med_a = np.nanmedian(players[:, ~is_home, 0], axis=1)
    for _, first, last in blocks:
        sign = 1.0 if home_attacks_right(players, is_home, first, last) else -1.0
        h = pd.Series(med_h[first:last + 1]).interpolate(limit_direction="both").values
        a = pd.Series(med_a[first:last + 1]).interpolate(limit_direction="both").values
        vh = np.zeros_like(h)
        va = np.zeros_like(a)
        vh[SHAPE_LAG:] = (h[SHAPE_LAG:] - h[:-SHAPE_LAG]) / (SHAPE_LAG / P.FPS)
        va[SHAPE_LAG:] = (a[SHAPE_LAG:] - a[:-SHAPE_LAG]) / (SHAPE_LAG / P.FPS)
        diff = sign * vh - (-sign) * va            # home forward speed minus away forward speed
        diff = pd.Series(diff).rolling(SHAPE_SMOOTH, center=True, min_periods=1).mean().values
        out[first:last + 1] = np.clip(np.nan_to_num(diff) / SHAPE_SCALE, -1.0, 1.0)
    return out


def spread_signal(players, is_home):
    """In [-1, 1]; > 0 when home is more spread out than away. The team with the ball spreads the
    pitch, the team without it compacts. Spread = std(x) * std(y) of a team's tracked players,
    compared as a ratio so no scale constant is needed, smoothed over 2 s."""
    with np.errstate(invalid="ignore"), __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore")
        def spread(team):
            return np.nanstd(players[:, team, 0], axis=1) * np.nanstd(players[:, team, 1], axis=1)
        h, a = spread(is_home), spread(~is_home)
        ratio = (h - a) / (h + a)
    ratio = pd.Series(ratio).rolling(SHAPE_SMOOTH, center=True, min_periods=1).mean().values
    return np.clip(np.nan_to_num(ratio), -1.0, 1.0)


_ANCHOR = '    base_evidence = prox_ev + tag_w * tags_info["tag_signal"]\n'
_ADDED = ('    base_evidence = base_evidence + np.where((~seen) | (nearest >= FLIGHT_DIST), '
          'SHAPE_WEIGHT * SIGNAL(players, is_home, blocks), 0.0)\n')


def build(weight=SHAPE_WEIGHT, signal="spread"):
    """label_match with the added line, running in possession's namespace."""
    src = inspect.getsource(P.label_match)
    assert src.count(_ANCHOR) == 1
    fn = shape_signal if signal == "movement" else (lambda pl, ih, bl: spread_signal(pl, ih))
    ns = dict(vars(P), SIGNAL=fn, SHAPE_WEIGHT=weight)
    exec(compile(src.replace(_ANCHOR, _ANCHOR + _ADDED), "d_shape_label_match", "exec"), ns)
    return ns["label_match"]


if __name__ == "__main__":
    # self-check: a team moving right while the other moves left gives positive evidence for it
    n, slots = 200, 22
    players = np.zeros((n, slots, 2))
    is_home = np.r_[np.ones(11, bool), np.zeros(11, bool)]
    players[:, :11, 0] = 40 + np.linspace(0, 20, n)[:, None]     # home pushes right...
    players[:, 0, 0] = 1.0                                       # ...from the left goal it defends
    players[:, 11:, 0] = 60 + np.linspace(0, 10, n)[:, None]     # away drops back to its right goal
    players[:, 11, 0] = 104.0
    assert home_attacks_right(players, is_home, 0, n - 1)
    e = shape_signal(players, is_home, [(1, 0, n - 1)])
    assert e[100] > 0, e[100]
    players[:, :11, 0] = players[::-1, :11, 0]                   # home now retreats: evidence flips
    players[:, 0, 0] = 1.0
    assert shape_signal(players, is_home, [(1, 0, n - 1)])[100] < 0
    print("ok")
