"""External validation data: Metrica Sports' open sample games 1-3, with labels from their events.

The competition match has no labels we may use, so the pipeline's constants are checked and tuned
on these three other matches instead (RULES.md: labelled data from other games is fine).

    git clone https://github.com/metrica-sports/sample-data   # or download data/Sample_Game_1..3
    # files go in validation/sample-data/ (git-ignored)

Two things make the sample games usable as a stand-in for BAR-ATH:

1. LABELS FROM EVENTS, built by code, never by hand. Each control event (pass, carry, set piece,
   recovery, shot, ball lost) gives its team the ball from its start frame. A ball out, a foul or
   a shot that goes out or in starts a stoppage that lasts until the next restart (set piece or goal
   kick). Then the competition's "controlled possession" rule: a spell shorter than 3 s that
   interrupts the other team's possession is given back to that team (CHALLENGE.md: touches and
   deflections do not change possession). Idea credited to the MRH2818 fork.

2. ATD-LIKE BALL. In the sample games the ball goes missing almost only during stoppages, which
   would make "ball missing" a perfect dead-ball signal. In BAR-ATH it is not. So the native gaps
   are filled and new ones are cut at random -- log-normal lengths with BAR-ATH's median (268 frames)
   -- until 41% of frames lack the ball, as in BAR-ATH, plus 0.3 m of smooth ball noise. Idea
   credited to the FrancescoRomeo02 (InkSight) fork.

The tracking has no speed column, so player speed is computed from positions.
"""

import json
import os
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "sample-data")
FPS = 25
PITCH = (105.0, 68.0)

CONTROL = {"PASS", "CARRY", "SET PIECE", "RECOVERY", "SHOT", "BALL LOST"}
CONTROL_MIN = 3 * FPS            # a possession shorter than this is not a controlled possession
GAP_TARGET = 0.41                # BAR-ATH: ball missing on 41% of graded frames
GAP_MEDIAN = 268                 # BAR-ATH: median length of a ball gap, frames
GAP_SIGMA = 0.9                  # spread of the log-normal; BAR-ATH p90/median is about 3.2
BALL_NOISE = 0.3                 # metres, smooth


# -- loading ---------------------------------------------------------------------------------------

def _csv_game(n):
    def read(side):
        path = os.path.join(DATA, "Sample_Game_{}_RawTrackingData_{}_Team.csv".format(n, side))
        return pd.read_csv(path, skiprows=3, header=None).values
    home, away = read("Home"), read("Away")
    period = home[:, 0].astype(int)
    xy = np.concatenate([home[:, 3:-2], away[:, 3:-2], home[:, -2:]], axis=1).astype(float)
    xy = xy.reshape(len(xy), -1, 2)
    is_home = np.r_[np.ones((home.shape[1] - 5) // 2, bool), np.zeros((away.shape[1] - 5) // 2, bool)]

    events = pd.read_csv(os.path.join(DATA, "Sample_Game_{}_RawEventsData.csv".format(n)))
    ev = [{"team": "H" if t == "Home" else "A", "type": ty, "sub": str(sub) if sub == sub else "",
           "period": int(pe), "start": int(sf) - 1, "end": int(ef) - 1 if ef > 0 else int(sf) - 1}
          for t, ty, sub, pe, sf, ef in zip(events["Team"], events["Type"], events["Subtype"],
                                            events["Period"], events["Start Frame"], events["End Frame"])]
    return xy, is_home, period, ev


def _epts_game():
    root = ET.parse(os.path.join(DATA, "Sample_Game_3_metadata.xml")).getroot()
    team_of = {p.get("id"): p.get("teamId") for p in root.iter("Player")}
    player_of = {c.get("id"): c.get("playerId") for c in root.iter("PlayerChannel")}
    home_id = [t.get("id") for t in root.iter("Team")][0]
    specs = []
    for spec in root.iter("DataFormatSpecification"):
        refs = [r.get("playerChannelId") for r in spec.iter("PlayerChannelRef")
                if r.get("playerChannelId").endswith("_x")]
        specs.append((int(spec.get("startFrame")), int(spec.get("endFrame")),
                      np.array([team_of[player_of[r]] == home_id for r in refs])))

    rows = []
    with open(os.path.join(DATA, "Sample_Game_3_tracking.txt")) as handle:
        for line in handle:
            _, players, ball = line.strip().split(":")
            rows.append(np.array((players + ";" + ball).replace(";", ",").split(","), dtype=float))
    xy = np.stack(rows).reshape(len(rows), -1, 2)

    # Substitutions reshuffle which slot is which team: put home slots first in every segment.
    n_home = int(specs[0][2].sum())
    for start, end, home in specs:
        order = np.r_[np.flatnonzero(home), np.flatnonzero(~home), len(home)]
        xy[start - 1:end] = xy[start - 1:end][:, order]
        assert home.sum() == n_home, "substitution changed team sizes"
    is_home = np.r_[np.ones(n_home, bool), np.zeros(xy.shape[1] - 1 - n_home, bool)]

    params = {p.findtext("Name"): p.findtext("Value") for p in root.iter("ProviderParameter")}
    period = np.zeros(len(xy), int)
    period[int(params["first_half_start"]) - 1:int(params["first_half_end"])] = 1
    period[int(params["second_half_start"]) - 1:int(params["second_half_end"])] = 2

    ev = []
    for e in json.load(open(os.path.join(DATA, "Sample_Game_3_events.json")))["data"]:
        subs = e["subtypes"]
        sub = "-".join(s["name"] for s in subs) if isinstance(subs, list) else (subs or {}).get("name", "")
        ev.append({"team": "H" if e["team"]["id"] == home_id else "A", "type": e["type"]["name"],
                   "sub": sub or "", "period": e["period"], "start": e["start"]["frame"] - 1,
                   "end": (e["end"]["frame"] or e["start"]["frame"]) - 1})
    return xy, is_home, period, ev


# -- labels from events ----------------------------------------------------------------------------

def _stops_play(e):
    sub = e["sub"].upper()
    if e["type"] in ("BALL OUT", "FAULT RECEIVED"):
        return True
    if e["type"] == "SHOT" and ("OUT" in sub or "GOAL" in sub):
        return True
    return e["type"] == "BALL LOST" and any(k in sub for k in ("OFFSIDE", "END HALF", "HAND BALL"))


def _restarts_play(e):
    """A set piece, a goal kick (logged as PASS or BALL LOST), or failing those the first pass:
    a keeper picking the ball up (RECOVERY) or carrying it to the spot (CARRY) is still dead."""
    return e["type"] in ("SET PIECE", "PASS") or "GOAL KICK" in e["sub"].upper()


def labels_from_events(events, n_frames):
    """Per-frame H/A/D/X. X before the first and after the last event of each period."""
    labels = np.full(n_frames, "X", dtype="<U1")
    for period in sorted({e["period"] for e in events}):
        evs = sorted((e for e in events if e["period"] == period), key=lambda e: e["start"])
        changes, dead = [], True              # before the kick-off the ball is dead
        for e in evs:
            if _restarts_play(e):
                dead = False
            if not dead and e["type"] in CONTROL:
                changes.append((e["start"], e["team"]))
            if _stops_play(e):
                at = e["end"] if e["type"] in ("BALL OUT", "SHOT") else e["start"]
                changes.append((at, "D"))
                dead = True
        first, last = evs[0]["start"], max(e["end"] for e in evs)
        changes.sort(key=lambda c: c[0])
        for (start, label), (end, _) in zip(changes, changes[1:] + [(last + 1, None)]):
            labels[max(start, first):min(end, last + 1)] = label
        labels[first:changes[0][0]] = "D"
    return give_back_short(labels)


def give_back_short(labels):
    """A possession under CONTROL_MIN between two spells of the other team is not a change."""
    out = labels.copy()
    edges = np.flatnonzero(out[1:] != out[:-1]) + 1
    starts, ends = np.r_[0, edges], np.r_[edges, len(out)]
    for i in range(1, len(starts) - 1):
        a, b = starts[i], ends[i]
        if out[a] in "HA" and b - a < CONTROL_MIN:
            before, after = out[starts[i - 1]], out[starts[i + 1]]
            if before in "HA" and before != out[a] and after != out[a]:
                out[a:b] = before
    return out


# -- making it look like BAR-ATH --------------------------------------------------------------------

def _smooth_noise(rng, n, sd):
    noise = pd.Series(rng.normal(size=n)).rolling(15, center=True, min_periods=1).mean().values
    return noise / noise.std() * sd


def atd_like(xy, period, seed, pitch=PITCH):
    """(frames, slots + 1, 3) like possession.load_tracking: x, y normalised, speed in m/s."""
    rng = np.random.default_rng(seed)
    xy = xy.copy()
    ball = pd.DataFrame(xy[:, -1])
    for p in (1, 2):
        rows = period == p
        ball[rows] = ball[rows].interpolate(limit_direction="both")
    xy[:, -1] = ball.values
    for axis, size in enumerate(pitch):
        xy[:, -1, axis] += _smooth_noise(rng, len(xy), BALL_NOISE) / size

    in_play = np.flatnonzero(period > 0)
    hidden = np.zeros(len(xy), bool)
    while hidden[in_play].mean() < GAP_TARGET:
        length = int(rng.lognormal(np.log(GAP_MEDIAN), GAP_SIGMA))
        start = rng.choice(in_play)
        hidden[start:start + length] = True
    xy[hidden, -1] = np.nan

    metres = xy[:, :-1] * np.asarray(pitch)
    step = np.full(metres.shape[:2], np.nan)
    step[2:-2] = np.linalg.norm(metres[4:] - metres[:-4], axis=2) * FPS / 4
    step[step > 12] = np.nan                                # tracking jumps, not running
    speed = np.concatenate([step, np.full((len(xy), 1), np.nan)], axis=1)
    return np.concatenate([xy, speed[:, :, None]], axis=2).astype(np.float32)


def halves_of(period):
    return [(int(np.flatnonzero(period == p)[0]), int(np.flatnonzero(period == p)[-1])) for p in (1, 2)]


def load(game):
    """(xy, is_home, period, labels) for sample game 1, 2 or 3."""
    xy, is_home, period, events = _epts_game() if game == 3 else _csv_game(game)
    return xy, is_home, period, labels_from_events(events, len(xy))


if __name__ == "__main__":
    for game in (1, 2, 3):
        xy, is_home, period, labels = load(game)
        graded = labels != "X"
        gaps = np.isnan(xy[graded, -1, 0]).mean()
        seg = labels[graded]
        edges = np.flatnonzero(seg[1:] != seg[:-1]) + 1
        kinds = seg[np.r_[0, edges]]
        print("game {}: {} frames, {} home / {} away slots, ball missing {:.0%} natively".format(
            game, len(xy), is_home.sum(), (~is_home).sum(), gaps))
        print("   labels H {:.0%} A {:.0%} D {:.0%};  {} possessions, {} stoppages, {:.1f} poss/min".format(
            *[np.mean(seg == c) for c in "HAD"], np.sum(kinds != "D"), np.sum(kinds == "D"),
            np.sum(kinds != "D") / (graded.sum() / FPS / 60)))
