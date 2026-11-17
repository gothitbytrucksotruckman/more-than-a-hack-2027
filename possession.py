"""Team Betelgeuse -- ball possession per frame. Rule-based dead-ball detector + sticky Viterbi.

    python possession.py                     # writes evaluation/results/betelgeuse.csv
    python evaluation/validate_submission.py evaluation/results/betelgeuse.csv

No training and no labels: every constant below is a football/physics prior (walking pace, control
radius, "a controlled possession lasts about a second"), not a value fitted to the public ground
truth. Input is the tactical ATD plus the Smart Tagging team tags.

Two stages:
  1. DEAD BALL. A frame is dead when the ball is outside the lines, when the tracker has lost the
     ball while the players are walking, or when a slow ball has nobody near it. A stoppage then
     lasts until the ball has stopped being placed (stationary) and is played again.
  2. WHO HAS IT. Inside each live stretch a 2-state Viterbi path over "which team is next to the
     ball" evidence. Frames with no evidence (ball missing, ball in flight) cost nothing, so the
     last owner keeps the ball -- the challenge's own rule for loose balls -- and a change needs
     about a second of sustained control, so deflections do not split a possession.
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from baseline import HOME, team_of_each_track, write_submission  # noqa: E402
import load_smart_tagging  # noqa: E402

FPS = 25
OUT = os.path.join(ROOT, "evaluation", "results", "betelgeuse.csv")

# -- priors, in metres / seconds -----------------------------------------------------------------
INTERP_MAX = 2 * FPS        # bridge ball gaps up to 2 s with a straight line
LINE_MARGIN = 0.5           # ball this far beyond a line is out (tracking noise is ~0.3 m)
WALK_SPEED = 1.7            # mean player speed below this = players are walking, not playing
STILL_SPEED = 1.5           # ball slower than this is "placed" / stationary
STILL_MIN = 1 * FPS         # ...for at least a second
LOOSE_DIST = 3.0            # nobody within this of a stationary ball = nobody is playing it
TOUCH_DIST = 2.0            # last touch before a stoppage: a player this close
RESTART_REACH = 5 * FPS     # a placed ball within 5 s of a stoppage belongs to the same stoppage
MIN_DEAD = 2 * FPS          # stoppages shorter than this do not exist
MIN_ALIVE = 2 * FPS         # nor does live play this short between two stoppages
CONTROL_DIST = 3.0          # a team can only be "on the ball" within this distance
SWITCH_COST = 30.0          # ~1.2 s of clear control before possession changes hands
TAG_WEIGHT = 0.1            # Smart Tagging is weak evidence: 12 s of tag = one switch
# Phases whose Team tag is the team with the ball. The zone tags (BALL IN FINAL THIRD ...) are not.
POSSESSION_CODES = {"BUILD UP", "PROGRESSION", "ATTACKING TRANSITION", "CREATING CHANCES"}


def load_tracking(raw_path, n_frames):
    """(n_frames, slots + 1, 3) array of x, y, speed. Last slot is the ball. NaN = not seen."""
    rows = [None] * n_frames
    with open(raw_path) as handle:
        for line in handle:
            frame_id, players, ball = line.split(":")
            rows[int(frame_id) - 1] = np.array(
                (players + ";" + ball).replace(";", ",").split(","), dtype=np.float32)
    return np.stack(rows).reshape(n_frames, -1, 3)


def runs(mask):
    """[(start, end)] of every True run, end exclusive."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def rolling(values, window, how):
    return getattr(pd.Series(values).rolling(window, center=True, min_periods=1), how)().values


def dead_ball(ball, seen, ball_speed, nearest, activity, pitch):
    """Boolean per frame: the ball is not in play."""
    length, width = pitch
    out = seen & ((ball[:, 0] < -LINE_MARGIN) | (ball[:, 0] > length + LINE_MARGIN) |
                  (ball[:, 1] < -LINE_MARGIN) | (ball[:, 1] > width + LINE_MARGIN))
    lost_and_walking = ~seen & (activity < WALK_SPEED)
    dead = out | lost_and_walking

    still = np.zeros(len(seen), dtype=bool)   # ball placed: stationary for a second or more
    for start, end in runs(seen & (ball_speed < STILL_SPEED)):
        if end - start >= STILL_MIN:
            still[start:end] = True

    # A stationary ball nobody is near: play has stopped. It stopped at the last touch.
    for start, end in runs(still & (nearest > LOOSE_DIST)):
        touched = np.flatnonzero(nearest[max(0, start - RESTART_REACH):start] < TOUCH_DIST)
        back = start - RESTART_REACH + touched[-1] + 1 if len(touched) else start
        dead[max(0, back):end] = True

    # A stoppage runs on through every placed ball that follows it: carried to the spot, put down,
    # then played. The restart is the end of the last stationary stretch.
    still_runs = runs(still)
    changed = True
    while changed:
        changed = False
        for start, end in still_runs:
            if dead[start:end].all():
                continue
            before = dead[max(0, start - RESTART_REACH):start + 1]
            if before.any():
                last_dead = max(0, start - RESTART_REACH) + np.flatnonzero(before)[-1]
                dead[last_dead:end] = True
                changed = True

    for start, end in runs(~dead):
        if end - start < MIN_ALIVE and start > 0 and end < len(dead):
            dead[start:end] = True
    for start, end in runs(dead):
        if end - start < MIN_DEAD:
            dead[start:end] = False
    return dead


def viterbi_two_state(evidence, switch_cost):
    """Most likely H/A path. evidence > 0 favours home (state 0), < 0 away (state 1)."""
    n = len(evidence)
    score = np.array([evidence[0], -evidence[0]])
    came_from = np.zeros((n, 2), dtype=np.int8)
    for t in range(1, n):
        stay, switch = score, score[::-1] - switch_cost
        came_from[t] = np.where(switch > stay, [1, 0], [0, 1])
        score = np.maximum(stay, switch) + (evidence[t], -evidence[t])
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(score[1] > score[0])
    for t in range(n - 1, 0, -1):
        path[t - 1] = came_from[t, path[t]]
    return path


def tag_evidence(n_frames, home_name):
    """+1 where Smart Tagging says home has the ball, -1 for away, summed over overlapping tags."""
    out = np.zeros(n_frames)
    for event in load_smart_tagging.load():
        team = event["tags"].get("Team") or "N/A"
        if (event["code"] or "").strip().upper() in POSSESSION_CODES and team != "N/A":
            out[event["start_frame"]:event["end_frame"]] += 1 if team == home_name else -1
    return np.clip(out, -1, 1)


def predict(game=GAME):
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game)
    n_frames = source_frames("tactical", game)
    pitch = np.array(GAMES[game]["pitch"])
    is_home = team_of_each_track(meta_path) == HOME

    tracking = load_tracking(raw_path, n_frames)
    players = tracking[:, :-1, :2] * pitch
    ball = pd.DataFrame(tracking[:, -1, :2] * pitch)

    # Short ball gaps are a pass in flight or an occlusion: bridge them. Long ones stay missing.
    filled = ball.interpolate(limit_area="inside")
    for start, end in runs(ball[0].isna().values):
        if end - start > INTERP_MAX:
            filled.iloc[start:end] = np.nan
    ball = filled.values
    seen = np.isfinite(ball[:, 0])

    with np.errstate(invalid="ignore"), __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore")   # all-NaN frames at halftime
        activity = rolling(np.nan_to_num(np.nanmean(tracking[:, :-1, 2], axis=1)), 2 * FPS + 1, "mean")
        step = np.linalg.norm(np.diff(ball, axis=0, prepend=ball[:1]), axis=1) * FPS
        ball_speed = np.nan_to_num(rolling(step, 11, "median"), nan=np.inf)
        distance = np.linalg.norm(players - ball[:, None, :], axis=2)
        distance = np.where(np.isfinite(distance), distance, np.inf)
    home_dist = distance[:, is_home].min(axis=1)
    away_dist = distance[:, ~is_home].min(axis=1)
    nearest = np.minimum(home_dist, away_dist)

    dead = dead_ball(ball, seen, ball_speed, nearest, activity, pitch)

    # Evidence per frame: how much closer one team is than the other, only when someone is on it.
    on_ball = seen & (nearest < CONTROL_DIST)
    closer = np.minimum(away_dist, 99.0) - np.minimum(home_dist, 99.0)   # inf = team not tracked
    evidence = np.where(on_ball, np.clip(closer, -1, 1), 0.0)
    home_name = __import__("baseline").team_names(meta_path)[0]
    evidence = evidence + TAG_WEIGHT * tag_evidence(n_frames, home_name)

    labels = np.full(n_frames, "X", dtype="<U1")   # ungraded frames stay X: they are never scored
    for _, first, last in half_video_blocks(game):
        half = slice(first, last + 1)
        labels[half] = "D"
        for start, end in runs(~dead[half]):
            path = viterbi_two_state(evidence[first + start:first + end], SWITCH_COST)
            labels[first + start:first + end] = np.where(path == 0, "H", "A")
    assert len(labels) == n_frames and set(labels) <= set("HADX")
    return labels


if __name__ == "__main__":
    labels = predict()
    write_submission(OUT, labels)
    print({label: int(np.sum(labels == label)) for label in "HADX"})
    print("->", os.path.relpath(OUT, ROOT))
