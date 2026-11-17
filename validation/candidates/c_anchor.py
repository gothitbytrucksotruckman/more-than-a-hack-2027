"""Candidate C: anchor each set-piece restart on the ball's still-to-moving change.

Inside every SET PIECES window (team or N/A) +/- 3 s, the first frame where a ball that has been
still (seen, < STILL_SPEED for >= STILL_MIN) starts to move (> RESTART_KICK_SPEED within 0.5 s)
with a player at it (< RESTART_KICK_DIST) is taken as the restart, and the stoppage ends there:
  - a stoppage still running at that frame is cut there (play is live from the kick);
  - a stoppage that ended earlier inside the search window is extended up to the kick.
No still-to-moving change in the window (ball unseen, say) leaves the labels as they are.

bounded=True (second and last variant, added after seeing the plain rule cut two post-goal
kick-off gates open on BAR-ATH -- a label-change count, no ground truth): skip kicks inside a
GOALS interval, never cut past the search window, and extend only when the stoppage ended at most
RESTART_REACH (5 s, possession.py's own) before the kick.

possession.py is not edited: dead_ball_per_half is wrapped for the duration of one label_match
call. The SET PIECES windows come from tags_info["set_piece_windows"], which
synthetic_tags.tags_info adds and the proposed diff adds to parse_smart_tagging.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import possession as P  # noqa: E402

# Set before looking at any result: the +/- 3 s comes from the brief, 0.5 s is a round value;
# every other threshold is possession.py's own.
ANCHOR_MARGIN = 3 * P.FPS
MOVE_CHECK = int(round(0.5 * P.FPS))


def restart_kick(lo, hi, seen, ball_speed, nearest):
    """First still-to-moving frame in [lo, hi) with a player at the ball, else None."""
    for s, e in P.runs(seen[lo:hi] & (ball_speed[lo:hi] < P.STILL_SPEED)):
        k = lo + e
        if (e - s >= P.STILL_MIN and k < hi and seen[k] and nearest[k] < P.RESTART_KICK_DIST
                and (ball_speed[k:k + MOVE_CHECK] > P.RESTART_KICK_SPEED).any()):
            return k
    return None


def anchor(dead, windows, half_blocks, seen, ball_speed, nearest, goals=(), bounded=False,
           margin=ANCHOR_MARGIN):
    dead = dead.copy()
    for ws, we in windows:
        for _, first, last in half_blocks:
            lo, hi = max(first, ws - margin), min(last + 1, we + margin)
            if hi <= lo:
                continue
            k = restart_kick(lo, hi, seen, ball_speed, nearest)
            if k is None or (bounded and any(gs <= k < ge for gs, ge in goals)):
                continue
            if dead[k]:
                end, stop = k, hi if bounded else last + 1
                while end < stop and dead[end]:
                    end += 1
                dead[k:end] = False
            else:
                before = np.flatnonzero(dead[lo:k])
                if len(before) and (not bounded or k - (lo + before[-1] + 1) <= P.RESTART_REACH):
                    dead[lo + before[-1] + 1:k] = True
    return dead


def label_match(tracking, is_home, pitch, blocks, tags_info, bounded=False):
    original = P.dead_ball_per_half

    def anchored(ball, seen, ball_speed, step_speed, nearest, *rest):
        dead = original(ball, seen, ball_speed, step_speed, nearest, *rest)
        return anchor(dead, tags_info.get("set_piece_windows", []), rest[-1], seen, ball_speed, nearest,
                      tags_info["goal_intervals"], bounded)

    P.dead_ball_per_half = anchored
    try:
        return P.label_match(tracking, is_home, pitch, blocks, tags_info)
    finally:
        P.dead_ball_per_half = original


def _self_check():
    n = 400
    seen = np.ones(n, bool)
    speed = np.full(n, 5.0)
    speed[100:150] = 0.5                       # still for 2 s, kicked at 150
    near = np.full(n, 1.0)
    blocks = [(1, 0, n - 1)]
    assert restart_kick(0, n, seen, speed, near) == 150
    dead = np.zeros(n, bool)
    dead[90:200] = True                        # stoppage runs past the kick -> cut at 150
    assert np.flatnonzero(anchor(dead, [(120, 300)], blocks, seen, speed, near))[[0, -1]].tolist() == [90, 149]
    dead = np.zeros(n, bool)
    dead[90:130] = True                        # stoppage ended before the kick -> extend to 150
    assert np.flatnonzero(anchor(dead, [(120, 300)], blocks, seen, speed, near))[[0, -1]].tolist() == [90, 149]
    assert anchor(dead, [(300, 390)], blocks, seen, speed, near).sum() == 40   # no kick: unchanged
    dead = np.zeros(n, bool)
    dead[90:110] = True                        # bounded: ended 1.6 s before the kick -> extended
    assert anchor(dead, [(120, 300)], blocks, seen, speed, near, bounded=True)[149]
    dead = np.zeros(n, bool)
    dead[0:20] = True                          # ended 5.2 s before the kick: bounded leaves it
    assert not anchor(dead, [(60, 300)], blocks, seen, speed, near, bounded=True)[149]
    assert anchor(dead, [(60, 300)], blocks, seen, speed, near)[149]
    dead = np.zeros(n, bool)
    dead[90:400] = True                        # kick inside GOALS: bounded leaves it
    assert np.array_equal(anchor(dead, [(120, 200)], blocks, seen, speed, near, [(100, 160)], True), dead)
    out = anchor(dead, [(120, 200)], blocks, seen, speed, near, bounded=True)
    assert not out[150:275].any() and out[275:].all()   # cut stops at the window end + 3 s


if __name__ == "__main__":
    _self_check()
    print("self-check ok")
