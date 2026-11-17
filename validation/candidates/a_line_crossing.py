"""Candidate A: a stoppage starts where the ball crosses a line, when tracking loses it right after.

Credited idea: Goalgorithm fork (re-implemented, no code copied).

If the ball is last seen beyond a pitch line, or its last 0.2 s of track projects across one within
1.0 s, and the tracker then loses it for >= 0.4 s, the frames from the crossing until the ball is
seen again join possession.dead_ball_per_half's out-of-bounds mask. Everything after (placed-ball
extension, restart snapping, cleanup) is the existing logic, unchanged.

possession.py is not edited: dead_ball_per_half is rebuilt from its own source with one added line,
so this module runs exactly the diff it proposes.
"""

import inspect
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import possession as P  # noqa: E402

# Set before looking at any result. Round physical values.
CROSS_GAP_MIN = int(round(0.4 * P.FPS))   # tracker loses the ball >= 0.4 s right after the crossing
CROSS_VEL_WINDOW = int(round(0.2 * P.FPS))  # last 0.2 s of track gives the ball's velocity
CROSS_PROJECT_MAX = 1.0                   # s the last velocity is carried into the gap (0 = seen only)


def line_crossing_dead(ball_raw, pitch, center, half_blocks, project_max=CROSS_PROJECT_MAX):
    """True from the frame the ball crosses a line until it is seen again, for every raw gap of
    >= 0.4 s whose last seen (or projected) ball lies beyond a line by LINE_MARGIN."""
    length, width = float(pitch[0]), float(pitch[1])
    lo = np.array([-length / 2.0 if center[0] == 0.0 else 0.0, -width / 2.0 if center[1] == 0.0 else 0.0])
    hi = lo + [length, width]

    def beyond(p):
        """Metres outside the pitch rectangle (<= 0 inside)."""
        return np.max(np.concatenate([lo - p, p - hi], axis=-1), axis=-1)

    seen = np.isfinite(ball_raw).all(axis=1)
    cross = np.zeros(len(ball_raw), dtype=bool)
    for _, first, last in half_blocks:
        for s, e in P.runs(~seen[first:last + 1]):
            if s == 0 or e == last + 1 - first or e - s < CROSS_GAP_MIN:
                continue
            s, e = first + s, first + e
            k = s - 1 - CROSS_VEL_WINDOW
            v = (ball_raw[s - 1] - ball_raw[k]) * P.FPS / CROSS_VEL_WINDOW if (
                k >= first and seen[k:s].all()) else np.zeros(2)
            steps = np.arange(min(int(round(project_max * P.FPS)), e - s) + 1)  # frame s - 1 + step
            margin = beyond(ball_raw[s - 1] + np.outer(steps / P.FPS, v))
            if not (margin > P.LINE_MARGIN).any():
                continue
            if margin[0] > 0:                 # already beyond when last seen: back to where it crossed
                c = s - 1
                while c - 1 >= first and seen[c - 1] and beyond(ball_raw[c - 1]) > 0:
                    c -= 1
            else:
                c = s - 1 + int(np.argmax(margin > 0))
            cross[c:e] = True
    return cross


def _build_dead_ball():
    src = inspect.getsource(P.dead_ball_per_half)
    anchor = "    # 1. Lost while walking"
    assert src.count(anchor) == 1, "possession.dead_ball_per_half changed: re-anchor candidate A"
    namespace = dict(vars(P))
    exec(src.replace(anchor, "    out = out | CROSSING[0]\n" + anchor), namespace)
    return namespace


_NS = _build_dead_ball()


def label_match(tracking, is_home, pitch, blocks, tags_info, project_max=CROSS_PROJECT_MAX):
    """possession.label_match with the line-crossing stoppage start."""
    pitch_m = np.array(pitch, dtype=np.float64)
    ball_raw = tracking[:, -1, :2].astype(np.float64) * pitch_m
    _NS["CROSSING"] = [line_crossing_dead(ball_raw, pitch_m, P.detect_center_spot(ball_raw, pitch_m),
                                          blocks, project_max)]
    original = P.dead_ball_per_half
    P.dead_ball_per_half = _NS["dead_ball_per_half"]
    try:
        return P.label_match(tracking, is_home, pitch, blocks, tags_info)
    finally:
        P.dead_ball_per_half = original


def _self_check():
    """Ball runs out over the right touchline at 10 m/s, then is lost for 1 s; and an in-play gap."""
    fps, n = P.FPS, 200
    ball = np.full((n, 2), np.nan)
    ball[:50] = np.c_[np.full(50, 50.0), 60.0 + np.arange(50) * 10.0 / fps - 49 * 10.0 / fps + 7.8]
    ball[75:] = [50.0, 70.0]
    blocks = [(1, 0, n - 1)]
    pitch, center = np.array([105.0, 68.0]), (52.5, 34.0)
    # last seen y = 67.8, 0.2 m inside; 10 m/s outward -> past y = 68 at frame 50, lost until 75
    cross = line_crossing_dead(ball, pitch, center, blocks, 1.0)
    assert np.flatnonzero(cross).tolist() == list(range(50, 75)), np.flatnonzero(cross)
    assert not line_crossing_dead(ball, pitch, center, blocks, 0.0).any()   # never seen beyond
    ball[:50, 1] = 30.0                                                     # still ball mid-pitch
    assert not line_crossing_dead(ball, pitch, center, blocks, 1.0).any()


_self_check()
