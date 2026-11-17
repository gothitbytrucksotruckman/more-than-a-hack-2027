"""Candidate A (stoppage start at the line crossing): diagnosis and protocol table.

    python validation/candidates/run_a.py

1. External diagnosis (Metrica games 1-3, seeds 0, 1): how late the current pipeline starts
   stoppages, and how often a ball gap follows a line crossing there.
2. BAR-ATH frequency of the same pattern, from tracking only (no labels of any kind).
3. Protocol table, baseline vs both variants; seed 2 is added if a variant passes.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import a_line_crossing as A  # noqa: E402  (puts the repo root on sys.path)
import possession as P  # noqa: E402
import report  # noqa: E402
import metrica  # noqa: E402
from possession_metrics import evaluate, label_codes  # noqa: E402

D = 2
FPS = P.FPS
VARIANTS = {"baseline": None, "A proj 1.0 s": 1.0, "A seen-only": 0.0}   # every value tried


def predict(variant, tracking, is_home, pitch, blocks, tags):
    proj = VARIANTS[variant]
    if proj is None:
        return P.label_match(tracking, is_home, pitch, blocks, tags)
    return A.label_match(tracking, is_home, pitch, blocks, tags, project_max=proj)


def external(game, seed, variant):
    tracking, is_home, halves, gt = report.prepared(game, seed)
    blocks = [(h + 1, a, b) for h, (a, b) in enumerate(halves)]
    pred = label_codes(predict(variant, tracking, is_home, metrica.PITCH, blocks, P.no_tags(len(tracking))))
    return tracking, blocks, gt, pred


def beyond(p, lo, hi):
    return np.max(np.concatenate([lo - p, p - hi], axis=-1), axis=-1)


def ball_and_box(tracking, pitch):
    pitch = np.asarray(pitch, dtype=np.float64)
    ball = tracking[:, -1, :2].astype(np.float64) * pitch
    cx, cy = P.detect_center_spot(ball, pitch)
    lo = np.array([-pitch[0] / 2 if cx == 0 else 0.0, -pitch[1] / 2 if cy == 0 else 0.0])
    return ball, lo, lo + pitch, (cx, cy)


def pattern(tracking, pitch, blocks):
    """Tracking-only counts: ball gaps >= 0.4 s; out episodes (seen beyond a line by LINE_MARGIN
    for >= 0.2 s) and how many are followed by a >= 0.4 s gap within 1 s; rule triggers."""
    ball, lo, hi, center = ball_and_box(tracking, pitch)
    seen = np.isfinite(ball).all(axis=1)
    margin = np.where(seen, beyond(np.nan_to_num(ball), lo, hi), -np.inf)
    out_eps = out_lost = 0
    minutes = sum(b - a + 1 for _, a, b in blocks) / FPS / 60
    gap_starts = []
    for _, first, last in blocks:
        gap_starts += [first + s for s, e in P.runs(~seen[first:last + 1])
                       if s > 0 and e < last + 1 - first and e - s >= A.CROSS_GAP_MIN]
    gap_starts = np.array(gap_starts)
    for _, first, last in blocks:
        for s, e in P.runs(margin[first:last + 1] > P.LINE_MARGIN):
            if e - s < 5:
                continue
            out_eps += 1
            out_lost += bool(((gap_starts >= first + s) & (gap_starts <= first + s + FPS)).any())
    cross = {proj: A.line_crossing_dead(ball, np.asarray(pitch, float), center, blocks, proj)
             for proj in (1.0, 0.0)}
    return {"min": minutes, "gaps": len(gap_starts), "gap_starts": gap_starts, "out": out_eps,
            "out_lost": out_lost, "cross": cross,
            "trig": {p: (len(P.runs(c)), c.sum() / FPS / 60) for p, c in cross.items()}}


def d_runs(codes, true_starts_only=False):
    rs = P.runs(codes == D)
    if true_starts_only:   # in-play stoppages: preceded by a possession, not the pre-kick-off X
        rs = [(s, e) for s, e in rs if s > 0 and codes[s - 1] in (0, 1)]
    return rs


def diagnose_external():
    print("== External diagnosis: games 1-3 x seeds 0, 1 (baseline pipeline) ==")
    lags = {"out": [], "other": []}
    missed = {"out": 0, "other": 0}
    gap_at_start = {"out": 0, "other": 0}
    trig_hit = trig_all = trig_true_hit = 0
    tot = {"min": 0, "gaps": 0, "out": 0, "out_lost": 0, "t1": 0, "t0": 0}
    for game in (1, 2, 3):
        for seed in (0, 1):
            tracking, blocks, gt, pred = external(game, seed, "baseline")
            native = report._cache[game][0][:, -1] * np.asarray(metrica.PITCH)
            nat_out = np.where(np.isfinite(native).all(axis=1),
                               beyond(np.nan_to_num(native), np.zeros(2), np.asarray(metrica.PITCH)), -np.inf) > 0
            pred_runs = d_runs(pred)
            st = pattern(tracking, metrica.PITCH, blocks)
            for k in ("min", "gaps", "out", "out_lost"):
                tot[k] += st[k]
            tot["t1"] += st["trig"][1.0][0]
            tot["t0"] += st["trig"][0.0][0]
            cross_starts = np.array([cs for cs, _ in P.runs(st["cross"][1.0])])
            true_runs = d_runs(gt, true_starts_only=True)
            for cs in cross_starts:
                trig_all += 1
                trig_hit += bool(any(ts - 2 * FPS <= cs < te for ts, te in true_runs))
            for ts, te in true_runs:
                kind = "out" if nat_out[max(0, ts - FPS):ts + FPS].any() else "other"
                gap_at_start[kind] += bool((np.abs(st["gap_starts"] - ts) <= FPS).any())
                trig_true_hit += bool((np.abs(cross_starts - ts) <= 2 * FPS).any())
                over = [ps for ps, pe in pred_runs if ps < te and pe > ts]
                if over:
                    lags[kind].append((min(over, key=lambda ps: abs(ps - ts)) - ts) / FPS)
                else:
                    missed[kind] += 1
    for kind in ("out", "other"):
        n = len(lags[kind]) + missed[kind]
        lg = np.array(lags[kind])
        print("  true stoppages, {:5s} type: {:3d}; predicted start lag median {:+.1f} s "
              "(p25 {:+.1f}, p75 {:+.1f}), late > 1 s {:.0%}, missed {:.0%}; a >= 0.4 s ball gap starts within +/-1 s: {:.0%}".format(
                  kind, n, np.median(lg), np.percentile(lg, 25), np.percentile(lg, 75),
                  np.mean(lg > 1.0), missed[kind] / n, gap_at_start[kind] / n))
    n_true = sum(len(lags[k]) + missed[k] for k in lags)
    print("  all {} true stoppages: predicted start lag median {:+.1f} s".format(
        n_true, np.median(lags["out"] + lags["other"])))
    print("  per 90 min: {:.0f} ball gaps >= 0.4 s, {:.1f} out episodes ({:.0%} lost within 1 s), "
          "triggers proj 1.0 s {:.1f}, seen-only {:.1f}".format(
              *(tot[k] / tot["min"] * 90 for k in ("gaps", "out")), tot["out_lost"] / max(tot["out"], 1),
              tot["t1"] / tot["min"] * 90, tot["t0"] / tot["min"] * 90))
    print("  proj 1.0 s triggers: {} total, {:.0%} start within a true stoppage (or <= 2 s before); "
          "{:.0%} of {} true stoppages get one within +/-2 s".format(
              trig_all, trig_hit / max(trig_all, 1), trig_true_hit / n_true, n_true))


def diagnose_bar_ath():
    print("== BAR-ATH, tracking only (no labels) ==")
    meta_path, raw_path = P.require(["atd_meta", "atd_raw"], P.GAME)
    n = P.source_frames("tactical", P.GAME)
    tracking = P.load_tracking(raw_path, n)
    blocks = P.half_video_blocks(P.GAME)
    pitch = P.GAMES[P.GAME]["pitch"]
    st = pattern(tracking, pitch, blocks)
    per90 = 90 / st["min"]
    print("  {:.0f} min; per 90 min: {:.0f} ball gaps >= 0.4 s, {:.1f} out episodes ({:.0%} lost within 1 s), "
          "triggers proj 1.0 s {:.1f} ({:.1f} min), seen-only {:.1f} ({:.1f} min)".format(
              st["min"], st["gaps"] * per90, st["out"] * per90, st["out_lost"] / max(st["out"], 1),
              st["trig"][1.0][0] * per90, st["trig"][1.0][1], st["trig"][0.0][0] * per90, st["trig"][0.0][1]))
    is_home = P.team_of_each_track(meta_path) == P.HOME
    tags = P.parse_smart_tagging(n, *P.team_names(meta_path))
    base = P.label_match(tracking, is_home, pitch, blocks, tags)
    for variant in ("A proj 1.0 s", "A seen-only"):
        cand = predict(variant, tracking, is_home, pitch, blocks, tags)
        changed = base != cand
        print("  {}: {} frames change label ({:.2f}%), {} H/A->D, {} D->H/A".format(
            variant, changed.sum(), 100 * changed.mean(), (changed & (cand == "D")).sum(),
            (changed & (base == "D")).sum()))


def table(seeds):
    print("== Protocol table, seeds {} (mean over seeds) ==".format(seeds))
    print("{:14s} game  spellF1  macroF1  poss pred/true".format("variant"))
    res = {}
    for variant in VARIANTS:
        for game in (1, 2, 3):
            rows = []
            for seed in seeds:
                _, _, gt, pred = external(game, seed, variant)
                r = evaluate(gt, pred)
                rows.append((r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"]))
            res[variant, game] = np.mean(rows, axis=0)
            print("{:14s} {:>4}  {:.4f}   {:.4f}   {:.0f} / {:.0f}".format(variant, game, *res[variant, game]))
    passed = []
    for variant in list(VARIANTS)[1:]:
        d12 = np.mean([res[variant, g][0] - res["baseline", g][0] for g in (1, 2)])
        d3 = res[variant, 3][0] - res["baseline", 3][0]
        dmacro = np.mean([res[variant, g][1] - res["baseline", g][1] for g in (1, 2, 3)])
        ok = d12 >= 0.01 and d3 >= 0 and dmacro >= -0.01
        print("  {}: d spellF1 games 1-2 {:+.4f}, game 3 {:+.4f}, d macroF1 {:+.4f} -> {}".format(
            variant, d12, d3, dmacro, "ACCEPT" if ok else "REJECT"))
        if ok:
            passed.append(variant)
    return passed


if __name__ == "__main__":
    diagnose_external()
    diagnose_bar_ath()
    if table((0, 1)):
        table((0, 1, 2))
