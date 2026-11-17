"""Team Betelgeuse -- ball possession per frame. Physics state machine + control-gated Viterbi.

    python possession.py                     # writes evaluation/results/Betelgeuse.csv
    python evaluation/validate_submission.py evaluation/results/Betelgeuse.csv

No training and no labels: every constant below is a football/physics prior (IFAB Law 8 kick-off
geometry, walking pace, control radius, ball acceleration, minimum possession duration), not a
value fitted to the public ground truth. Input is the tactical ATD plus Smart Tagging events.

Two stages:
  1. DEAD BALL. A frame is dead during pre-kickoff and post-goal kick-off gates (IFAB Law 8),
     when the ball is out of bounds, when the tracker loses the ball while players walk (gated by
     live Smart Tagging phases), when a stationary ball has nobody closing it down, or when a
     kicker stands over a stationary set-piece ball. In-pitch stoppages are back-dated to the last
     touch before collective deceleration, restarts snap to the kick acceleration frame, and
     cleanup runs strictly within each half block.
  2. WHO HAS IT. Inside each live stretch a 2-state Viterbi decoder combines control-gated
     ball-player proximity (down-weighting fast straight-line passes in flight) with regime-weighted
     offensive/defensive Smart Tagging and set-piece restart priors. Switching cost C(t) is
     touch-aligned (lower on close control / acceleration impulses, high with persistence bias
     mid-flight or occluded), followed by a micro-possession filter absorbing < 1.2 s deflections.
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from baseline import HOME, team_names, team_of_each_track, write_submission  # noqa: E402
import load_smart_tagging  # noqa: E402

FPS = 25
OUT = os.path.join(ROOT, "evaluation", "results", "Betelgeuse.csv")

# -- priors, in metres / seconds -----------------------------------------------------------------
INTERP_MAX = 2 * FPS                      # unconditional straight-line bridge for gaps <= 2.0 s
CARRIER_INTERP_MAX = int(round(3.5 * FPS))  # bridge gaps <= 3.5 s when same team carries in & out
CARRIER_START_DIST = 2.0                  # ball-to-carrier radius before gap
CARRIER_END_DIST = 2.5                    # ball-to-carrier radius after gap
LINE_MARGIN = 0.5                         # ball this far beyond a pitch line is out of bounds

WALK_SPEED = 1.7                          # mean player speed below this = walking (no live tag)
STRICT_WALK_SPEED = 1.35                  # stricter mean speed stop when a live tag is active
STRICT_WALK_MIN = 3 * FPS                 # ...sustained for >= 3.0 s
WALK_BALL_MAX_SPEED = 5.0                 # slow ball-carry ceiling during global walking stops
SET_PIECE_GATE_MARGIN = 8 * FPS           # +/- 8 s around SET PIECES suspends live-tag gating

ABANDONED_STRICT_SPEED = 1.0              # slow ball < 1.0 m/s ...
ABANDONED_STRICT_MIN = int(round(1.5 * FPS))    # ...for >= 1.5 s
ABANDONED_MOD_SPEED = 1.5                 # or < 1.5 m/s ...
ABANDONED_MOD_MIN = int(round(2.2 * FPS))       # ...for >= 2.2 s
LOOSE_DIST = 3.0                          # no player within 3.0 m
CLOSING_WINDOW = int(round(0.5 * FPS))    # 0.5 s window for closing speed d/dt min(d_H, d_A)
CLOSING_MIN_RATE = -0.8                   # closing faster than -0.8 m/s = actively chasing ball
ABANDONED_PLAYER_MED = 2.2                # global median player speed below this = play stopped

WAIT_WINDOW = int(round(3.5 * FPS))       # rolling 3.5 s window for stationary set-piece waits
WAIT_DISP_MAX = 1.2                       # net ball displacement < 1.2 m over 3.5 s
WAIT_BALL_SPEED = 1.0                     # ...with ball speed < 1.0 m/s
WAIT_PLAYER_MED = 1.9                     # ...and global median player speed < 1.9 m/s

STILL_SPEED = 1.5                         # placed / stationary ball speed threshold
STILL_MIN = 1 * FPS                       # ...for at least 1.0 s
TOUCH_DIST = 2.0                          # controlled/contested touch radius
WHISTLE_LOOKBACK = 4 * FPS                # look back up to 4.0 s for whistle deceleration
DECEL_SPEED = 2.0                         # mean player speed drop below 2.0 m/s marks stoppage
RESTART_REACH = 5 * FPS                   # placed ball within 5.0 s extends the same stoppage
RESTART_SNAP_REACH = 2 * FPS              # +/- 2.0 s window around stoppage end to snap kick
RESTART_KICK_SPEED = 2.0                  # ball acceleration above 2.0 m/s marks restart kick
RESTART_KICK_DIST = 2.5                   # player within 2.5 m at restart kick

CENTER_CIRCLE_RADIUS = 9.15               # IFAB Law 1 / Law 8 center circle radius
KICKOFF_MIN_OWN_HALF = 8                  # >= 8 tracked players per team on own side at kick-off
KICKOFF_POSTGOAL_MIN_OWN = 7              # relaxed >= 7 threshold when post-goal kick was occluded
KICKOFF_HALF_MARGIN = 3.0                 # tolerance cx +/- 3.0 m around halfway line
KICKOFF_MOVE_DIST = 1.5                   # ball moved > 1.5 m from center spot = kicked
HALF_START_ABSORB_MAX = 5                 # absorb <= 5 initial pre-kick frames at half start

MIN_DEAD = 2 * FPS                        # stoppages shorter than 2.0 s are removed
MIN_ALIVE = 2 * FPS                       # live stretches < 2.0 s between stoppages become D

CONTROL_DIST = 3.5                        # ball-player proximity evidence radius
AMBIGUOUS_PROX_DIFF = 0.5                 # |d_A - d_H| below this = ambiguous proximity
FAST_BALL_SPEED = 8.0                     # flying pass/shot speed threshold
STRAIGHT_BALL_ACCEL = 5.0                 # straight flight acceleration ceiling
CONTROL_GATE_SCALE = 0.2                  # down-weight proximity during fast straight flight
TOUCH_ACCEL_IMPULSE = 6.0                 # clear touch/deflection acceleration impulse

SWITCH_COST_CONTROL = 22.0                # switch cost when receiver is within 2.2 m
SWITCH_COST_TOUCH = 16.0                  # switch cost when receiver < 2.2 m + touch impulse
SWITCH_COST_DEFAULT = 30.0                # intermediate switch cost
SWITCH_COST_FLIGHT = 45.0                 # high cost mid-flight (>= 2.5 m) or NaN ball
SWITCH_CTRL_DIST = 2.2                    # close control distance for touch-aligned switch cost
FLIGHT_DIST = 2.5                         # distance above which ball is considered in flight
CAUSAL_PERSIST_BIAS = 0.01                # tiny bias toward staying in s_{t-1} mid-flight/NaN

TAG_WEIGHT_TRACKED = 0.10                 # Smart Tagging weight when ball tracked < 3.5 m
TAG_WEIGHT_MISSING = 0.45                 # Smart Tagging weight when ball NaN or >= 3.5 m
RESTART_PRIOR = 0.80                      # initial set-piece team prior on ambiguous/NaN restart
RESTART_PRIOR_DUR = 2 * FPS               # duration (2.0 s) of restart prior
RESTART_TAG_REACH = 6 * FPS               # +/- 6.0 s match window for SET PIECES near restart

MICRO_POSS_MIN = int(round(1.2 * FPS))    # absorb isolated flanked spells < 1.2 s (30 frames)

OFFENSIVE_CODES_FULL = {
    "BUILD UP", "PROGRESSION", "ATTACKING TRANSITION", "CREATING CHANCES"
}
OFFENSIVE_CODES_MODERATE = {"BALL IN FINAL THIRD", "LONG BALL"}
DEFENSIVE_CODES = {
    "DEFENDING IN ATTACKING THIRD",
    "DEFENDING IN MIDDLE THIRD",
    "DEFENDING IN DEFENSIVE THIRD",
    "DEFENSIVE TRANSITION",
}
LIVE_GATE_CODES = {
    "BUILD UP", "PROGRESSION", "CREATING CHANCES", "ATTACKING TRANSITION", "BALL IN FINAL THIRD"
}
KICKOFF_TAG_CODES = {"BUILD UP", "PROGRESSION", "LONG BALL"}


def load_tracking(raw_path, n_frames):
    """(n_frames, slots + 1, 3) array of x, y, speed. Last slot is the ball. NaN = not seen."""
    rows = [None] * n_frames
    with open(raw_path) as handle:
        for line in handle:
            frame_id, players, ball = line.split(":")
            rows[int(frame_id) - 1] = np.fromstring(
                (players + ";" + ball).replace(";", ","), sep=",", dtype=np.float32
            )
    return np.stack(rows).reshape(n_frames, -1, 3)


def runs(mask):
    """[(start, end)] of every True run, end exclusive."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def rolling(values, window, how):
    return getattr(pd.Series(values).rolling(window, center=True, min_periods=1), how)().values


def detect_center_spot(ball_raw, pitch):
    """Infer center spot (cx, cy): (0, 0) for centered coordinates, else (L/2, W/2)."""
    length, width = float(pitch[0]), float(pitch[1])
    bx = ball_raw[:, 0]
    finite_x = bx[np.isfinite(bx)]
    if len(finite_x) and np.percentile(finite_x, 5) < -0.1 * length:
        return 0.0, 0.0
    return length / 2.0, width / 2.0


def bridge_ball_gaps(ball_raw, players, is_home, half_blocks):
    """Interpolate ball gaps inside each half: <= 2.0 s unconditionally, <= 3.5 s same carrier."""
    n_frames = len(ball_raw)
    ball = ball_raw.copy()
    raw_seen = np.isfinite(ball_raw[:, 0]) & np.isfinite(ball_raw[:, 1])

    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = np.linalg.norm(players - ball_raw[:, None, :], axis=2)
        d = np.where(np.isfinite(d), d, np.inf)
    h_raw = d[:, is_home].min(axis=1)
    a_raw = d[:, ~is_home].min(axis=1)

    for _, first, last in half_blocks:
        sub_raw = pd.DataFrame(ball_raw[first:last + 1])
        filled = sub_raw.interpolate(limit_area="inside").to_numpy(copy=True)
        sub_seen = raw_seen[first:last + 1]
        for s, e in runs(~sub_seen):
            if s == 0 or e == len(sub_seen):
                filled[s:e] = np.nan
                continue
            dur = e - s
            if dur <= INTERP_MAX:
                continue
            if dur <= CARRIER_INTERP_MAX:
                idx0, idx1 = first + s - 1, first + e
                t0 = 0 if h_raw[idx0] <= a_raw[idx0] else 1
                t1 = 0 if h_raw[idx1] <= a_raw[idx1] else 1
                d0 = min(h_raw[idx0], a_raw[idx0])
                d1 = min(h_raw[idx1], a_raw[idx1])
                if t0 == t1 and d0 < CARRIER_START_DIST and d1 < CARRIER_END_DIST:
                    continue
            filled[s:e] = np.nan
        ball[first:last + 1] = filled
    return ball


def parse_smart_tagging(n_frames, home_name, away_name):
    """Extract per-frame Smart Tagging evidence g(t), masks, SET PIECES, and GOALS intervals."""
    tag_signal = np.zeros(n_frames, dtype=np.float64)
    live_gate_mask = np.zeros(n_frames, dtype=bool)
    sp_near_mask = np.zeros(n_frames, dtype=bool)
    kickoff_tag_starts = np.zeros(n_frames, dtype=bool)
    named_set_pieces = []
    goal_intervals = []

    valid_teams = {home_name, away_name}
    for event in load_smart_tagging.load():
        code = (event["code"] or "").strip().upper()
        s = max(0, int(event["start_frame"]))
        e = min(n_frames, int(event["end_frame"]))
        team = event["tags"].get("Team") or "N/A"
        team_sign = (1.0 if team == home_name else -1.0) if team in valid_teams else 0.0

        if team_sign != 0.0:
            if code in OFFENSIVE_CODES_FULL:
                tag_signal[s:e] += team_sign
            elif code in OFFENSIVE_CODES_MODERATE:
                tag_signal[s:e] += 0.7 * team_sign
            elif code in DEFENSIVE_CODES:
                tag_signal[s:e] -= team_sign

        if code in LIVE_GATE_CODES:
            live_gate_mask[s:e] = True
        if code in KICKOFF_TAG_CODES and s < n_frames:
            kickoff_tag_starts[s] = True
        if code == "SET PIECES":
            sp_near_mask[max(0, s - SET_PIECE_GATE_MARGIN):min(n_frames, e + SET_PIECE_GATE_MARGIN)] = True
            if team_sign != 0.0:
                named_set_pieces.append((s, e, team_sign))
        elif code == "GOALS":
            goal_intervals.append((s, e))

    return {
        "tag_signal": np.clip(tag_signal, -1.0, 1.0),
        "live_gate_mask": live_gate_mask,
        "sp_near_mask": sp_near_mask,
        "kickoff_tag_starts": kickoff_tag_starts,
        "named_set_pieces": named_set_pieces,
        "goal_intervals": sorted(goal_intervals),
    }


def compute_half_separation(players, is_home, cx, min_own_players):
    """Per-frame check: team medians on opposite halves & >= min_own_players on own half."""
    hp_x = players[:, is_home, 0]
    ap_x = players[:, ~is_home, 0]
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        med_h = np.nanmedian(hp_x, axis=1)
        med_a = np.nanmedian(ap_x, axis=1)

    h_left = np.sum(hp_x <= cx + KICKOFF_HALF_MARGIN, axis=1)
    h_right = np.sum(hp_x >= cx - KICKOFF_HALF_MARGIN, axis=1)
    a_left = np.sum(ap_x <= cx + KICKOFF_HALF_MARGIN, axis=1)
    a_right = np.sum(ap_x >= cx - KICKOFF_HALF_MARGIN, axis=1)

    left_right = (med_h < cx) & (med_a > cx) & (h_left >= min_own_players) & (a_right >= min_own_players)
    right_left = (med_h > cx) & (med_a < cx) & (h_right >= min_own_players) & (a_left >= min_own_players)
    return left_right | right_left


def find_kickoff_frame(start_f, last_f, strict_sep, postgoal_sep, seen, dist_center,
                       nearest, ball_speed, step_speed, kickoff_tag_starts, is_postgoal):
    """First frame >= start_f in [start_f, last_f] where a valid physical Kick-Off occurs."""
    b_kicked = (ball_speed > RESTART_KICK_SPEED) | (np.nan_to_num(step_speed, nan=0.0) > RESTART_KICK_SPEED) | (
        dist_center > KICKOFF_MOVE_DIST
    )
    center_ko = (
        strict_sep[start_f:last_f + 1]
        & seen[start_f:last_f + 1]
        & (dist_center[start_f:last_f + 1] < CENTER_CIRCLE_RADIUS)
        & (nearest[start_f:last_f + 1] < RESTART_KICK_DIST)
        & b_kicked[start_f:last_f + 1]
    )
    sep_for_tag = postgoal_sep if is_postgoal else strict_sep
    if is_postgoal:
        tag_ko = sep_for_tag[start_f:last_f + 1] & kickoff_tag_starts[start_f:last_f + 1]
    else:
        tag_ko = sep_for_tag[start_f:last_f + 1] & ~seen[start_f:last_f + 1] & kickoff_tag_starts[start_f:last_f + 1]
    hits = np.flatnonzero(center_ko | tag_ko)
    if len(hits):
        return start_f + int(hits[0])
    return last_f + 1


def dead_ball_per_half(ball, seen, ball_speed, step_speed, nearest, activity, med_player_speed,
                       players, is_home, pitch, center, tags_info, half_blocks):
    """Compute upgraded per-frame dead-ball mask across each half with IFAB Law 8 kick-off gates."""
    n_frames = len(seen)
    length, width = float(pitch[0]), float(pitch[1])
    cx, cy = center
    x0 = -length / 2.0 if cx == 0.0 else 0.0
    x1 = length / 2.0 if cx == 0.0 else length
    y0 = -width / 2.0 if cy == 0.0 else 0.0
    y1 = width / 2.0 if cy == 0.0 else width

    out = seen & (
        (ball[:, 0] < x0 - LINE_MARGIN) | (ball[:, 0] > x1 + LINE_MARGIN)
        | (ball[:, 1] < y0 - LINE_MARGIN) | (ball[:, 1] > y1 + LINE_MARGIN)
    )

    # 1. Lost while walking, gated by live Smart Tagging phases unless SET PIECES is +/- 8 s
    live_gate = tags_info["live_gate_mask"] & ~tags_info["sp_near_mask"]
    lost_normal = ~seen & ~live_gate & (activity < WALK_SPEED)
    lost_strict = np.zeros(n_frames, dtype=bool)
    for s, e in runs(~seen & live_gate & (activity < STRICT_WALK_SPEED)):
        if e - s >= STRICT_WALK_MIN:
            lost_strict[s:e] = True
    # Also mark sustained global walking stops with a slow/carried ball when no live phase is active
    seen_walking = np.zeros(n_frames, dtype=bool)
    walk_cand = (
        seen
        & ~tags_info["live_gate_mask"]
        & (activity < STRICT_WALK_SPEED)
        & (med_player_speed < STRICT_WALK_SPEED)
        & (ball_speed < WALK_BALL_MAX_SPEED)
    )
    for s, e in runs(walk_cand):
        if e - s >= STRICT_WALK_MIN:
            seen_walking[s:e] = True
    lost_and_walking = lost_normal | lost_strict | seen_walking

    # 2. Abandoned ball with closing-rate and player-speed guards (no slow-pass false triggers)
    closing_rate = np.zeros(n_frames, dtype=np.float64)
    dt_close = CLOSING_WINDOW / float(FPS)
    both_fin = np.isfinite(nearest[CLOSING_WINDOW:]) & np.isfinite(nearest[:-CLOSING_WINDOW])
    closing_rate[CLOSING_WINDOW:][both_fin] = (
        nearest[CLOSING_WINDOW:][both_fin] - nearest[:-CLOSING_WINDOW][both_fin]
    ) / dt_close

    abandoned = np.zeros(n_frames, dtype=bool)
    for spd_thresh, min_dur in (
        (ABANDONED_STRICT_SPEED, ABANDONED_STRICT_MIN),
        (ABANDONED_MOD_SPEED, ABANDONED_MOD_MIN),
    ):
        for s, e in runs(seen & (nearest > LOOSE_DIST) & (ball_speed < spd_thresh)):
            if e - s >= min_dur:
                if (np.median(closing_rate[s:e]) >= CLOSING_MIN_RATE
                        and np.median(med_player_speed[s:e]) < ABANDONED_PLAYER_MED):
                    abandoned[s:e] = True

    # 3. Kicker waiting over stationary set-piece ball (3.5 s net displacement < 1.2 m)
    sp_wait = np.zeros(n_frames, dtype=bool)
    wait_cand = seen & (ball_speed < WAIT_BALL_SPEED) & (med_player_speed < WAIT_PLAYER_MED)
    for s, e in runs(wait_cand):
        if e - s >= WAIT_WINDOW:
            disp = np.linalg.norm(ball[s + WAIT_WINDOW - 1:e] - ball[s:e - WAIT_WINDOW + 1], axis=1)
            for idx in np.flatnonzero(disp < WAIT_DISP_MAX):
                sp_wait[s + int(idx):s + int(idx) + WAIT_WINDOW] = True

    still = np.zeros(n_frames, dtype=bool)
    for s, e in runs(seen & (ball_speed < STILL_SPEED)):
        if e - s >= STILL_MIN:
            still[s:e] = True

    dist_center = np.where(seen, np.hypot(ball[:, 0] - cx, ball[:, 1] - cy), np.inf)
    strict_sep = compute_half_separation(players, is_home, cx, KICKOFF_MIN_OWN_HALF)
    postgoal_sep = compute_half_separation(players, is_home, cx, KICKOFF_POSTGOAL_MIN_OWN)

    dead = np.zeros(n_frames, dtype=bool)
    for _, first, last in half_blocks:
        dead_h = (out | lost_and_walking | abandoned | sp_wait)[first:last + 1].copy()
        out_h = out[first:last + 1]
        seen_h = seen[first:last + 1]
        near_h = nearest[first:last + 1]
        act_h = activity[first:last + 1]
        still_h = still[first:last + 1]
        step_h = np.nan_to_num(step_speed[first:last + 1], nan=0.0)
        bs_h = ball_speed[first:last + 1]
        h_len = last - first + 1

        # 4. In-pitch whistle back-dating: last touch < 2.0 m followed by deceleration < 2.0 m/s
        for s, _ in runs(dead_h):
            if s == 0 or out_h[s]:
                continue
            look_a = max(0, s - WHISTLE_LOOKBACK)
            touches = np.flatnonzero(seen_h[look_a:s] & (near_h[look_a:s] < TOUCH_DIST))
            if not len(touches):
                continue
            t_touch = look_a + int(touches[-1])
            pre_a = max(0, t_touch - CLOSING_WINDOW)
            if np.max(act_h[pre_a:t_touch + 1]) >= DECEL_SPEED:
                drops = np.flatnonzero(act_h[t_touch:s + 1] < DECEL_SPEED)
                if len(drops):
                    t_drop = t_touch + int(drops[0])
                    dead_h[t_drop:s] = True

        # Extend stoppages through placed ball stretches starting within 5 s
        # (only if play has not already resumed with a fast kick and active players)
        still_runs = runs(still_h)
        changed = True
        while changed:
            changed = False
            for s, e in still_runs:
                if dead_h[s:e].all():
                    continue
                before = dead_h[max(0, s - RESTART_REACH):s + 1]
                if before.any():
                    last_d = max(0, s - RESTART_REACH) + int(np.flatnonzero(before)[-1])
                    if ((bs_h[last_d:s] > FAST_BALL_SPEED) & seen_h[last_d:s]).any() and (
                            act_h[last_d:s] >= DECEL_SPEED).any():
                        continue
                    dead_h[last_d:e] = True
                    changed = True

        # 5. Half-Start & Post-Goal Kick-Off Gates (IFAB Law 8)
        gate_locked = np.zeros(h_len, dtype=bool)
        ko_half = find_kickoff_frame(
            first, last, strict_sep, postgoal_sep, seen, dist_center,
            nearest, ball_speed, step_speed, tags_info["kickoff_tag_starts"], is_postgoal=False,
        )
        ko_rel = ko_half - first
        if ko_rel <= HALF_START_ABSORB_MAX:
            dead_h[:ko_rel] = False
        else:
            dead_h[:ko_rel] = True
            gate_locked[:ko_rel] = True

        for gs, ge in tags_info["goal_intervals"]:
            if ge <= first or gs > last:
                continue
            gs_rel = max(0, gs - first)
            ge_rel = min(h_len, ge - first)
            dead_in_goal = np.flatnonzero(dead_h[gs_rel:ge_rel])
            gate_start_rel = gs_rel + int(dead_in_goal[0]) if len(dead_in_goal) else ge_rel
            search_start = min(last, max(first + gate_start_rel, ge))
            ko_goal = find_kickoff_frame(
                search_start, last, strict_sep, postgoal_sep, seen, dist_center,
                nearest, ball_speed, step_speed, tags_info["kickoff_tag_starts"], is_postgoal=True,
            )
            ko_goal_rel = min(h_len, ko_goal - first)
            if ko_goal_rel > gate_start_rel:
                dead_h[gate_start_rel:ko_goal_rel] = True
                gate_locked[gate_start_rel:ko_goal_rel] = True

        # 6. Restart kick snapping at stoppage ends (when ball is tracked at/near restart)
        for s, e in runs(dead_h):
            if e >= h_len:
                continue
            # Only snap restarts that involve a tracked stationary/placed ball near the boundary
            lo = max(s + MIN_DEAD, e - RESTART_SNAP_REACH)
            hi = min(h_len, e + RESTART_SNAP_REACH)
            if hi <= lo:
                continue
            has_slow_pre = (seen_h[max(s, e - RESTART_SNAP_REACH):e] &
                            (step_h[max(s, e - RESTART_SNAP_REACH):e] <= RESTART_KICK_SPEED)).any()
            if not has_slow_pre:
                continue
            # Look for frame k where ball accelerates > 2.0 m/s from <= 2.0 m/s with player < 2.5 m
            prev_step = np.concatenate(([step_h[0]], step_h[:-1]))
            kicks = np.flatnonzero(
                seen_h[lo:hi]
                & (near_h[lo:hi] < RESTART_KICK_DIST)
                & (step_h[lo:hi] > RESTART_KICK_SPEED)
                & (prev_step[lo:hi] <= RESTART_KICK_SPEED)
                & (bs_h[lo:hi] > RESTART_KICK_SPEED)
            )
            valid_kicks = [
                lo + int(idx)
                for idx in kicks
                if not still_h[lo + int(idx):min(h_len, lo + int(idx) + STILL_MIN)].any()
            ]
            if valid_kicks:
                k = min(valid_kicks, key=lambda idx: abs(idx - e))
                if k < e and not gate_locked[k:e].any():
                    dead_h[k:e] = False
                elif k > e and not dead_h[e:k].any():
                    dead_h[e:k] = True

        # 7. Boundary-safe cleanup strictly inside the half
        for s, e in runs(~dead_h):
            if e - s < MIN_ALIVE and s > 0 and e < h_len:
                dead_h[s:e] = True
        for s, e in runs(dead_h):
            if e - s < MIN_DEAD:
                dead_h[s:e] = False

        dead[first:last + 1] = dead_h

    return dead


def viterbi_two_state(evidence, cost_to_home, cost_to_away, in_flight_or_nan):
    """Time-varying 2-state Viterbi decoder (0 = Home, 1 = Away) with causal flight persistence."""
    n = len(evidence)
    score = np.array([evidence[0], -evidence[0]], dtype=np.float64)
    came_from = np.zeros((n, 2), dtype=np.int8)
    for t in range(1, n):
        stay_bonus = CAUSAL_PERSIST_BIAS if in_flight_or_nan[t] else 0.0
        stay = score + stay_bonus
        switch = np.array([score[1] - cost_to_home[t], score[0] - cost_to_away[t]], dtype=np.float64)
        came_from[t] = np.where(switch > stay, [1, 0], [0, 1])
        score = np.maximum(stay, switch) + np.array([evidence[t], -evidence[t]], dtype=np.float64)
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(score[1] > score[0])
    for t in range(n - 1, 0, -1):
        path[t - 1] = came_from[t, path[t]]
    return path


def absorb_micro_possessions(path, min_len=MICRO_POSS_MIN):
    """Absorb isolated < 1.2 s interior spells flanked by the opposing team into that team."""
    out = path.copy()
    while True:
        segs = []
        breaks = np.flatnonzero(out[1:] != out[:-1]) + 1
        bounds = np.concatenate(([0], breaks, [len(out)]))
        for i in range(len(bounds) - 1):
            segs.append((int(bounds[i]), int(bounds[i + 1]), int(out[bounds[i]])))
        if len(segs) < 3:
            break
        interior_short = [
            (e - s, s, e, segs[i - 1][2])
            for i, (s, e, team) in enumerate(segs)
            if 0 < i < len(segs) - 1 and (e - s) < min_len and segs[i - 1][2] == segs[i + 1][2]
        ]
        if not interior_short:
            break
        _, s, e, surround_team = min(interior_short)
        out[s:e] = surround_team
    return out


def nearest_set_piece_team(restart_frame, named_set_pieces):
    """Return +1/-1 for named SET PIECES within +/- 6 s of restart_frame, else 0.0."""
    best_dist = RESTART_TAG_REACH + 1
    best_sign = 0.0
    for s, e, team_sign in named_set_pieces:
        if s <= restart_frame <= e:
            dist = 0
        elif restart_frame < s:
            dist = s - restart_frame
        else:
            dist = restart_frame - e
        if dist <= RESTART_TAG_REACH and dist < best_dist:
            best_dist = dist
            best_sign = team_sign
    return best_sign


def no_tags(n_frames):
    """Smart Tagging evidence for a match that has none (validation/ runs on other matches)."""
    return {"tag_signal": np.zeros(n_frames), "live_gate_mask": np.zeros(n_frames, dtype=bool),
            "sp_near_mask": np.zeros(n_frames, dtype=bool),
            "kickoff_tag_starts": np.zeros(n_frames, dtype=bool),
            "named_set_pieces": [], "goal_intervals": []}


def label_match(tracking, is_home, pitch, blocks, tags_info):
    """Labels for any match. tracking: (frames, slots + 1, 3) x/y normalised 0-1 and speed m/s,
    last slot the ball. blocks: [(half, first, last)] inclusive frame ranges; the rest is X."""
    n_frames = len(tracking)
    pitch = np.array(pitch, dtype=np.float64)
    players = tracking[:, :-1, :2].astype(np.float64) * pitch
    pspeed = tracking[:, :-1, 2].astype(np.float64)
    ball_raw = tracking[:, -1, :2].astype(np.float64) * pitch

    center = detect_center_spot(ball_raw, pitch)
    ball = bridge_ball_gaps(ball_raw, players, is_home, blocks)
    seen = np.isfinite(ball[:, 0]) & np.isfinite(ball[:, 1])

    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        activity = rolling(np.nan_to_num(np.nanmean(pspeed, axis=1)), 2 * FPS + 1, "mean")
        med_player_speed = rolling(np.nan_to_num(np.nanmedian(pspeed, axis=1)), 2 * FPS + 1, "median")

        vel = np.diff(ball, axis=0, prepend=ball[:1]) * FPS
        both_seen = seen & np.concatenate(([False], seen[:-1]))
        vel[~both_seen] = np.nan
        step_speed = np.linalg.norm(vel, axis=1)
        ball_speed = np.nan_to_num(rolling(step_speed, 11, "median"), nan=np.inf)

        vs = rolling(np.where(seen, ball_speed, np.nan), 5, "mean")
        a_ball = np.zeros(n_frames, dtype=np.float64)
        if n_frames > 4:
            v_ok = seen[4:] & seen[:-4] & np.isfinite(vs[4:]) & np.isfinite(vs[:-4])
            a_ball[2:n_frames - 2][v_ok] = np.abs(vs[4:][v_ok] - vs[:-4][v_ok]) / (4.0 / FPS)

        distance = np.linalg.norm(players - ball[:, None, :], axis=2)
        distance = np.where(np.isfinite(distance), distance, np.inf)

    home_dist = distance[:, is_home].min(axis=1)
    away_dist = distance[:, ~is_home].min(axis=1)
    nearest = np.minimum(home_dist, away_dist)

    dead = dead_ball_per_half(
        ball, seen, ball_speed, step_speed, nearest, activity, med_player_speed,
        players, is_home, pitch, center, tags_info, blocks,
    )

    # Control-gated proximity evidence + regime-dependent Smart Tagging
    tracked_ctrl = seen & (nearest < CONTROL_DIST)
    closer = np.minimum(away_dist, 99.0) - np.minimum(home_dist, 99.0)
    prox = np.where(tracked_ctrl, np.clip(closer, -1.0, 1.0), 0.0)
    w_ctrl = np.where(
        seen & (ball_speed > FAST_BALL_SPEED) & (a_ball < STRAIGHT_BALL_ACCEL),
        CONTROL_GATE_SCALE,
        1.0,
    )
    prox_ev = w_ctrl * prox

    tag_w = np.where(tracked_ctrl, TAG_WEIGHT_TRACKED, TAG_WEIGHT_MISSING)
    base_evidence = prox_ev + tag_w * tags_info["tag_signal"]

    # Touch-aligned state-dependent switching costs C(t)
    in_flight_or_nan = (~seen) | (nearest >= FLIGHT_DIST)
    touch_cost = np.where(a_ball > TOUCH_ACCEL_IMPULSE, SWITCH_COST_TOUCH, SWITCH_COST_CONTROL)
    cost_to_home = np.where(
        seen & (home_dist < SWITCH_CTRL_DIST),
        touch_cost,
        np.where(in_flight_or_nan, SWITCH_COST_FLIGHT, SWITCH_COST_DEFAULT),
    )
    cost_to_away = np.where(
        seen & (away_dist < SWITCH_CTRL_DIST),
        touch_cost,
        np.where(in_flight_or_nan, SWITCH_COST_FLIGHT, SWITCH_COST_DEFAULT),
    )

    labels = np.full(n_frames, "X", dtype="<U1")
    for _, first, last in blocks:
        half = slice(first, last + 1)
        labels[half] = "D"
        for start, end in runs(~dead[half]):
            s_abs, e_abs = first + start, first + end
            ev_slice = base_evidence[s_abs:e_abs].copy()
            if start > 0:
                sp_sign = nearest_set_piece_team(s_abs, tags_info["named_set_pieces"])
                if sp_sign != 0.0:
                    w_end = min(e_abs - s_abs, RESTART_PRIOR_DUR)
                    ambig = (~seen[s_abs:s_abs + w_end]) | (nearest[s_abs:s_abs + w_end] >= CONTROL_DIST) | (
                        np.abs(closer[s_abs:s_abs + w_end]) < AMBIGUOUS_PROX_DIFF
                    )
                    ev_slice[:w_end] = np.where(
                        ambig, prox_ev[s_abs:s_abs + w_end] + RESTART_PRIOR * sp_sign, ev_slice[:w_end]
                    )
            path = viterbi_two_state(
                ev_slice,
                cost_to_home[s_abs:e_abs],
                cost_to_away[s_abs:e_abs],
                in_flight_or_nan[s_abs:e_abs],
            )
            path = absorb_micro_possessions(path)
            labels[s_abs:e_abs] = np.where(path == 0, "H", "A")

    assert len(labels) == n_frames and set(labels) <= set("HADX")
    return labels


def predict(game=GAME):
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game)
    n_frames = source_frames("tactical", game)
    is_home = team_of_each_track(meta_path) == HOME
    home_name, away_name = team_names(meta_path)
    tracking = load_tracking(raw_path, n_frames)
    return label_match(tracking, is_home, GAMES[game]["pitch"], half_video_blocks(game),
                       parse_smart_tagging(n_frames, home_name, away_name))


if __name__ == "__main__":
    labels = predict()
    write_submission(OUT, labels)
    print({label: int(np.sum(labels == label)) for label in "HADX"})
    print("->", os.path.relpath(OUT, ROOT))
