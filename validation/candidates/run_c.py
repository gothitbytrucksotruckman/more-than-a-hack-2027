"""Candidate C (Smart Tagging rules): do the tag rules help, and does set-piece anchoring?

    python validation/candidates/run_c.py

1. Tags vs no tags: the CURRENT pipeline on Metrica games 1-3 (gap seeds 0, 1) with no_tags (what
   validation/report.py runs) and with synthetic Smart Tagging (synthetic_tags.py), plus two
   generator sensitivity settings (noisier team, cleaner windows) and a drop-one ablation of
   each tag rule.
2. Set-piece anchoring (c_anchor.py, plain and bounded) vs the current pipeline, both on the same
   synthetic tags. Chosen on games 1-2, confirmed on game 3; seed 2 is added if one passes.
   How often each variant fires is shown for Metrica and for BAR-ATH (label changes only).
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import synthetic_tags as S  # noqa: E402  (puts the repo root and validation/ on sys.path)
import c_anchor  # noqa: E402
import possession as P  # noqa: E402
import report  # noqa: E402
import metrica  # noqa: E402
from possession_metrics import evaluate, label_codes  # noqa: E402

# Every configuration tried: (generator settings or None for no_tags, anchoring, tag key dropped)
CONFIGS = {
    "no tags": (None, None, None),
    "synthetic": ({}, None, None),
    "synth flip 0.15": ({"p_flip": 0.15}, None, None),
    "synth clip": ({"clip": True}, None, None),
    "drop tag_signal": ({}, None, "tag_signal"),
    "drop live gate": ({}, None, "live_gate_mask"),
    "drop restart prior": ({}, None, "named_set_pieces"),
    "drop goals gate": ({}, None, "goal_intervals"),
    "drop kickoff tags": ({}, None, "kickoff_tag_starts"),
    "anchor": ({}, "plain", None),
    "anchor bounded": ({}, "bounded", None),
}
_raw, _tags, _pred = {}, {}, {}


def tags_for(game, settings, n, drop):
    if settings is None:
        return P.no_tags(n)
    key = (game, tuple(sorted(settings.items())))
    if key not in _tags:
        _tags[key] = S.tags_info(S.synthetic_events(game, settings, data=_raw[game]), n)
    tags = dict(_tags[key])
    if drop:
        tags[drop] = [] if isinstance(tags[drop], list) else np.zeros_like(tags[drop])
    return tags


def run(anchoring, tracking, is_home, pitch, blocks, tags):
    if anchoring is None:
        return P.label_match(tracking, is_home, pitch, blocks, tags)
    return c_anchor.label_match(tracking, is_home, pitch, blocks, tags, bounded=anchoring == "bounded")


def predict(game, seed, config):
    if (game, seed, config) not in _pred:
        if game not in _raw:
            _raw[game] = S.raw_game(game)
            xy, is_home, period, events = _raw[game]
            report._cache[game] = (xy, is_home, period, metrica.labels_from_events(events, len(xy)))
        tracking, is_home, halves, gt = report.prepared(game, seed)
        blocks = [(h + 1, a, b) for h, (a, b) in enumerate(halves)]
        settings, anchoring, drop = CONFIGS[config]
        tags = tags_for(game, settings, len(tracking), drop)
        _pred[game, seed, config] = (gt, label_codes(run(anchoring, tracking, is_home, metrica.PITCH, blocks, tags)))
    return _pred[game, seed, config]


def score(game, seed, config):
    gt, pred = predict(game, seed, config)
    r = evaluate(gt, pred)
    changed = np.mean(pred != predict(game, seed, "synthetic")[1])
    return (r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"],
            r["alive_dead_f1"], r["team_acc_both_alive"], 100 * changed)


def table(configs, seeds):
    print("\n== seeds {} (mean over seeds) ==".format(seeds))
    print("{:18s} game  spellF1  macroF1  poss pred/true  alive/dead F1  team acc  % frames != synthetic".format("config"))
    res = {}
    for config in configs:
        for game in (1, 2, 3):
            res[config, game] = np.mean([score(game, seed, config) for seed in seeds], axis=0)
            print("{:18s} {:>4}  {:.4f}   {:.4f}   {:3.0f} / {:3.0f}      {:.4f}         {:.4f}    {:.2f}".format(
                config, game, *res[config, game]))
    return res


def verdict(res, base, cand):
    d12 = np.mean([res[cand, g][0] - res[base, g][0] for g in (1, 2)])
    d3 = res[cand, 3][0] - res[base, 3][0]
    dmacro = np.mean([res[cand, g][1] - res[base, g][1] for g in (1, 2, 3)])
    ok = d12 >= 0.01 and d3 >= 0 and dmacro >= -0.01
    print("  {:18s} vs {:10s}: d spellF1 games 1-2 {:+.4f}, game 3 {:+.4f}, d macroF1 {:+.4f} -> {}".format(
        cand, base, d12, d3, dmacro, "ACCEPT" if ok else "REJECT"))
    return ok


def bar_ath_firing():
    """Share of BAR-ATH frames each anchoring variant relabels, with the real tags. No labels."""
    meta_path, raw_path = P.require(["atd_meta", "atd_raw"], P.GAME)
    n = P.source_frames("tactical", P.GAME)
    tracking = P.load_tracking(raw_path, n)
    is_home = P.team_of_each_track(meta_path) == P.HOME
    tags = P.parse_smart_tagging(n, *P.team_names(meta_path))
    tags["set_piece_windows"] = sorted((e["start_frame"], e["end_frame"]) for e in P.load_smart_tagging.load()
                                       if (e["code"] or "").strip().upper() == "SET PIECES")
    args = (tracking, is_home, P.GAMES[P.GAME]["pitch"], P.half_video_blocks(P.GAME), tags)
    base = run(None, *args)
    for anchoring in ("plain", "bounded"):
        changed = run(anchoring, *args) != base
        print("  BAR-ATH anchor {}: {:.2f}% of frames relabelled in {} runs".format(
            anchoring, 100 * changed.mean(), len(P.runs(changed))))


if __name__ == "__main__":
    res = table(CONFIGS, (0, 1))
    print("\nTag rules on vs off (protocol applied to 'tags' vs 'no tags'):")
    tags_ok = [verdict(res, "no tags", cand) for cand in ("synthetic", "synth flip 0.15", "synth clip")][0]
    print("Drop-one ablation (ACCEPT here = that rule hurts):")
    for cand in [c for c in CONFIGS if c.startswith("drop")]:
        verdict(res, "synthetic", cand)
    print("Set-piece anchoring, on synthetic tags:")
    passed = [cand for cand in ("anchor", "anchor bounded") if verdict(res, "synthetic", cand)]
    bar_ath_firing()
    if tags_ok or passed:
        res2 = table(["no tags", "synthetic"] + passed, (0, 1, 2))
        verdict(res2, "no tags", "synthetic")
        for cand in passed:
            verdict(res2, "synthetic", cand)
