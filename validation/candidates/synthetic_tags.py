"""Smart-Tagging-like tags for Metrica sample games 1-3, generated from their events.

Credited idea: InkSight (FrancescoRomeo02) fork -- synthesise tags for games that have none.
Re-implemented here, no code copied.

    python validation/candidates/synthetic_tags.py      # calibration check against BAR-ATH

The output goes through possession.parse_smart_tagging itself (its loader is swapped for the
synthetic list), so the tag rules under test are exactly the ones the submission runs.

ASSUMPTIONS. Every number below is a BAR-ATH Smart Tagging statistic (DATA.md counts, or window
lengths / timing measured from the XML and BAR-ATH tracking -- never its ground truth), or a round
value set before any scoring. This is an approximation of Metrica's tagger, not a copy of it.
 1. Phases come from possession spells (runs of H/A in metrica.labels_from_events). A spell of
    >= 5 s is tagged with probability 0.7, so phase tags arrive at ~1.3 / min as in BAR-ATH.
 2. Code: ATTACKING TRANSITION when the spell starts by winning the ball from the other team in
    open play, else BUILD UP / PROGRESSION / CREATING CHANCES by the ball's third at the spell
    start. (BAR-ATH has fewer transitions, 17%; the code only matters for kick-off detection.)
 3. Window: starts 0-2 s after the spell starts (tagger latency), length uniform between BAR-ATH's
    10th and 90th percentile for that code (CREATING CHANCES is always 30 s). Not cut at the end of
    the spell: on BAR-ATH windows are fixed-ish lengths, so they overrun into the next spell.
 4. Every phase tag gets its defensive mirror for the other team over the same window (BAR-ATH:
    142 of 142 offensive phases have one, except 2 CREATING CHANCES tagged N/A).
 5. Team noise: none added (p_flip 0). Check: share of phase-tagged frames whose nearest player
    (ball seen, < 1.5 m) is on the tag's team -- BAR-ATH 0.78; synthetic 0.72-0.78 from the window
    overrun alone; the event labels themselves score 0.89-0.91 (the ceiling of this proxy).
    Sensitivity runs: p_flip 0.15 (noisier, 0.66-0.68) and clip=True (cleaner, 0.91).
 6. BALL IN FINAL THIRD: from the first frame of a spell where the ball enters the attacking third,
    same 0.7 detection, length uniform 22-40 s (BAR-ATH p10-p90). LONG BALL and the box codes
    are not generated (possession.py ignores the box codes; LONG BALL is 29 tags at weight 0.7).
 7. SET PIECES: one window per restart event. 35 s long, the restart 0-6 s after the window start
    (BAR-ATH, tracking only: corner kicker leaves the flag ~1.5 s in, thrower leaves the touchline
    ~5 s in; the ball's still-to-moving change alone is too sparse to place it). Kick-offs: 5 s,
    restart 3.4 s in (the published kick-off frames). Detection per type = BAR-ATH count / Metrica
    mean count, capped at 1: throw-in 0.7, corner 1, goal kick 0.6, free kick 0.2, kick-off 1.
    Throw-ins and kick-offs are Team N/A (as in BAR-ATH); the rest carry the taking team.
    Plus 2 false-positive windows per half at random frames, team H/A/N/A at random.
 8. GOALS: from 5-20 s before the goal event to 2.4 s before the next kick-off (BAR-ATH: GOALS
    ends 1 s after the kick-off tag starts; its high-activity attack sits in the first ~20 s).
"""

import contextlib
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "validation"))
import possession as P  # noqa: E402
import metrica  # noqa: E402

FPS = P.FPS
TAG_SEED = 2027
NAMES = {"H": "Home", "A": "Away"}

# Every generator setting, all set before scoring (see the module docstring for where each comes from).
SETTINGS = {
    "min_spell": 5.0,        # s, shortest spell the tagger recognises as a phase
    "p_phase": 0.7,          # detection probability of a phase / BALL IN FINAL THIRD
    "latency": 2.0,          # s, tag starts 0..latency after the spell
    "p_flip": 0.0,           # wrong team (calibrated: the overrun already matches BAR-ATH)
    "clip": False,           # True = cut phase windows at the spell end (sensitivity only)
    "sp_lead": 6.0,          # s, restart sits 0..sp_lead after the SET PIECES window start
    "fp_per_half": 2,        # false-positive SET PIECES windows per half
}
PHASE_LEN = {"BUILD UP": (14, 25), "PROGRESSION": (11, 28), "CREATING CHANCES": (30, 30),
             "ATTACKING TRANSITION": (11, 21), "BALL IN FINAL THIRD": (22, 40)}
MIRROR = {"BUILD UP": "DEFENDING IN ATTACKING THIRD", "PROGRESSION": "DEFENDING IN MIDDLE THIRD",
          "CREATING CHANCES": "DEFENDING IN DEFENSIVE THIRD", "ATTACKING TRANSITION": "DEFENSIVE TRANSITION"}
SP_DETECT = {"Throw-in": 0.7, "Corner Kick": 1.0, "Goal Kick": 0.6, "Free Kick": 0.2, "Kick Off": 1.0}
SP_LEN, KO_LEN, KO_LEAD = 35 * FPS, 5 * FPS, int(round(3.4 * FPS))
GOAL_END_BEFORE_KO = int(round(2.4 * FPS))


def raw_game(game):
    """(xy, is_home, period, events) straight from metrica's loaders."""
    return metrica._epts_game() if game == 3 else metrica._csv_game(game)


def set_piece_type(e):
    sub = e["sub"].upper()
    if "GOAL KICK" in sub:
        return "Goal Kick"
    if e["type"] != "SET PIECE":
        return None
    for key, kind in (("THROW", "Throw-in"), ("CORNER", "Corner Kick"), ("KICK OFF", "Kick Off")):
        if key in sub:
            return kind
    return "Free Kick"                                   # free kicks and penalties


def attack_pos(xy, is_home, period):
    """Per-frame ball x in the direction each team attacks: {'H': array, 'A': array}, 0 = own goal."""
    bx = np.full(len(xy), np.nan)
    home_attacks_right = np.zeros(len(xy), bool)
    for p in (1, 2):
        rows = period == p
        bx[rows] = pd.Series(xy[rows, -1, 0]).interpolate(limit_direction="both").values
        home_attacks_right[rows] = np.nanmean(xy[rows][:, :-1][:, is_home, 0]) < 0.5
    return {"H": np.where(home_attacks_right, bx, 1 - bx), "A": np.where(home_attacks_right, 1 - bx, bx)}


def synthetic_events(game, settings=None, seed=TAG_SEED, data=None):
    """List of {code, start_frame, end_frame, tags} like load_smart_tagging.load(), plus 'type'."""
    s = dict(SETTINGS, **(settings or {}))
    rng = np.random.default_rng([seed, game])
    xy, is_home, period, events = data or raw_game(game)
    n = len(xy)
    labels = metrica.labels_from_events(events, n)
    pos = attack_pos(xy, is_home, period)
    out = []

    def add(code, start, end, team, kind=None):
        start, end = max(0, int(start)), min(n, int(end))
        if end > start:
            out.append({"code": code, "start_frame": start, "end_frame": end, "type": kind,
                        "tags": {"Team": NAMES.get(team, "N/A")}})

    other = {"H": "A", "A": "H"}
    edges = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts, ends = np.r_[0, edges], np.r_[edges, n]
    for i, (a, b) in enumerate(zip(starts, ends)):
        team = labels[a]
        if team not in "HA" or b - a < s["min_spell"] * FPS:
            continue
        # phase tag
        if rng.random() < s["p_phase"]:
            if i > 0 and labels[starts[i - 1]] == other[team]:
                code = "ATTACKING TRANSITION"
            else:
                code = ("BUILD UP", "PROGRESSION", "CREATING CHANCES")[min(2, int(pos[team][a] * 3))]
            t0 = a + rng.uniform(0, s["latency"]) * FPS
            t1 = t0 + rng.uniform(*PHASE_LEN[code]) * FPS
            t1 = min(t1, b) if s["clip"] else t1
            tag_team = other[team] if rng.random() < s["p_flip"] else team
            add(code, t0, t1, tag_team)
            add(MIRROR[code], t0, t1, other[tag_team])
        # ball in final third
        entry = np.flatnonzero(pos[team][a:b] > 2 / 3)
        if len(entry) and rng.random() < s["p_phase"]:
            t0 = a + entry[0]
            t1 = t0 + rng.uniform(*PHASE_LEN["BALL IN FINAL THIRD"]) * FPS
            t1 = min(t1, b) if s["clip"] else t1
            add("BALL IN FINAL THIRD", t0, t1, other[team] if rng.random() < s["p_flip"] else team)

    # set pieces (restart events), goals
    evs = sorted(events, key=lambda e: e["start"])
    for e in evs:
        kind = set_piece_type(e)
        if kind is None or rng.random() >= SP_DETECT[kind]:
            continue
        r = e["start"]
        if kind == "Kick Off":
            add("SET PIECES", r - KO_LEAD, r - KO_LEAD + KO_LEN, None, kind)
        else:
            w0 = r - rng.uniform(0, s["sp_lead"]) * FPS
            add("SET PIECES", w0, w0 + SP_LEN, None if kind == "Throw-in" else e["team"], kind)
    for p in (1, 2):
        frames = np.flatnonzero(period == p)
        for _ in range(s["fp_per_half"]):
            w0 = rng.choice(frames)
            add("SET PIECES", w0, w0 + SP_LEN, rng.choice(["H", "A", None]), "False positive")
    for e in evs:
        sub = e["sub"].upper()
        if e["type"] in ("SHOT", "BALL OUT") and "GOAL" in sub and "GOAL KICK" not in sub:
            ko = next((k["start"] for k in evs if k["start"] > e["start"] and set_piece_type(k) == "Kick Off"), None)
            if ko is not None:
                add("GOALS", e["start"] - rng.uniform(5, 20) * FPS, ko - GOAL_END_BEFORE_KO, e["team"])
    return out


@contextlib.contextmanager
def _loader(events):
    saved = P.load_smart_tagging.load
    P.load_smart_tagging.load = lambda *a, **k: events
    try:
        yield
    finally:
        P.load_smart_tagging.load = saved


def tags_info(events, n_frames):
    """possession.parse_smart_tagging on the synthetic list, plus every SET PIECES window
    (team or N/A) under 'set_piece_windows' for the anchoring candidate."""
    with _loader(events):
        info = P.parse_smart_tagging(n_frames, NAMES["H"], NAMES["A"])
    info["set_piece_windows"] = sorted((e["start_frame"], e["end_frame"]) for e in events
                                       if e["code"] == "SET PIECES")
    return info


def calibration(game, settings=None, gap_seed=0):
    """Numbers to hold against BAR-ATH: phase tags / min (1.34), share of graded frames under a
    phase tag (0.41), tag-team = nearest-player team on seen ball < 1.5 m (0.78)."""
    data = raw_game(game)
    xy, is_home, period, events = data
    ev = synthetic_events(game, settings, data=data)
    labels = metrica.labels_from_events(events, len(xy))
    graded = labels != "X"
    tr = metrica.atd_like(xy, period, gap_seed)
    pitch = np.asarray(metrica.PITCH)
    ball = tr[:, -1, :2].astype(float) * pitch
    d = np.linalg.norm(tr[:, :-1, :2].astype(float) * pitch - ball[:, None], axis=2)
    d = np.where(np.isfinite(d), d, np.inf)
    hd, ad = d[:, is_home].min(1), d[:, ~is_home].min(1)
    ok = np.isfinite(ball[:, 0]) & (np.minimum(hd, ad) < 1.5)
    cover = np.zeros(len(xy), bool)
    agree = tot = 0
    phases = [e for e in ev if e["code"] in P.OFFENSIVE_CODES_FULL]
    for e in phases:
        sl = slice(e["start_frame"], e["end_frame"])
        cover[sl] = True
        if e["tags"]["Team"] in ("Home", "Away"):
            agree += np.sum((hd[sl] < ad[sl])[ok[sl]] == (e["tags"]["Team"] == "Home"))
            tot += ok[sl].sum()
    kinds = {}
    for e in ev:
        if e["code"] == "SET PIECES":
            kinds[e["type"]] = kinds.get(e["type"], 0) + 1
    return {"phase/min": len(phases) / (graded.sum() / FPS / 60), "cover": cover[graded].mean(),
            "agree": agree / max(tot, 1), "set pieces": kinds,
            "goals": sum(e["code"] == "GOALS" for e in ev)}


def _self_check():
    """Structure checks on a tiny fake game: mirror tags, N/A throw-ins, goal window before kick-off."""
    n = 6000
    xy = np.full((n, 3, 2), 0.5)
    xy[:, 0, 0], xy[:, 1, 0] = 0.3, 0.7                     # home left, attacks right
    xy[:, -1, 0] = np.linspace(0.1, 0.9, n)
    period = np.r_[np.ones(n // 2, int), 2 * np.ones(n // 2, int)]
    ev = [{"team": "H", "type": "SET PIECE", "sub": "KICK OFF", "period": 1, "start": 100, "end": 100},
          {"team": "H", "type": "PASS", "sub": "", "period": 1, "start": 120, "end": 130},
          {"team": "H", "type": "SHOT", "sub": "ON TARGET-GOAL", "period": 1, "start": 1500, "end": 1510},
          {"team": "A", "type": "SET PIECE", "sub": "KICK OFF", "period": 1, "start": 2500, "end": 2500},
          {"team": "A", "type": "PASS", "sub": "", "period": 1, "start": 2510, "end": 2520},
          {"team": "A", "type": "BALL OUT", "sub": "", "period": 1, "start": 2900, "end": 2900},
          {"team": "H", "type": "SET PIECE", "sub": "THROW IN", "period": 2, "start": 3100, "end": 3100},
          {"team": "H", "type": "PASS", "sub": "", "period": 2, "start": 3110, "end": 3120}]
    out = synthetic_events(0, {"p_phase": 1.0, "p_flip": 0.0, "fp_per_half": 0},
                           data=(xy, np.array([True, False]), period, ev))
    phases = [e for e in out if e["code"] in MIRROR]
    mirrors = [e for e in out if e["code"] in MIRROR.values()]
    assert phases and len(phases) == len(mirrors)
    assert all(m["tags"]["Team"] != p["tags"]["Team"] and m["start_frame"] == p["start_frame"]
               for p, m in zip(phases, mirrors))
    sp = {e["type"]: e for e in out if e["code"] == "SET PIECES"}
    assert sp["Throw-in"]["tags"]["Team"] == "N/A" and sp["Kick Off"]["tags"]["Team"] == "N/A"
    assert 3100 - 6 * FPS <= sp["Throw-in"]["start_frame"] <= 3100 < sp["Throw-in"]["end_frame"]
    goal = next(e for e in out if e["code"] == "GOALS")
    assert goal["start_frame"] < 1500 and goal["end_frame"] == 2500 - GOAL_END_BEFORE_KO
    info = tags_info(out, n)
    assert info["goal_intervals"] and len(info["set_piece_windows"]) == len(sp) + 1  # two kick-offs


if __name__ == "__main__":
    _self_check()
    print("self-check ok")
    print("BAR-ATH targets: phase/min 1.34, cover 0.41, agree 0.78, 59 SET PIECES / match")
    for g in (1, 2, 3):
        c = calibration(g)
        print("game {}: phase/min {:.2f}  cover {:.2f}  agree {:.2f}  goals {}  set pieces {} = {}".format(
            g, c["phase/min"], c["cover"], c["agree"], c["goals"], sum(c["set pieces"].values()), c["set pieces"]))
