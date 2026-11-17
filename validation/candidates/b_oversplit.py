"""Candidate B: over-splitting. Post-filters on possession.label_match's labels; possession.py untouched.

    micro_s   interior spell shorter than this, flanked by the other team, goes back to it
              (possession.absorb_micro_possessions with a longer window; 1.2 s is the baseline)
    before_d  last spell of a live stretch shorter than this, i.e. a touch before the ball goes dead,
              goes to the team before it
    after_d   first spell of a live stretch shorter than this goes to the team after it
    fill_d    stoppage shorter than this between two spells of the same team becomes that team

Re-running the absorption on the final labels equals running it inside label_match: H/A runs in
the labels are exactly the Viterbi stretches, and absorption is shortest-first, so 1.2 s then 2 s
is the same as 2 s directly.
"""

import numpy as np

import possession

FPS = possession.FPS


def _spans(path):
    b = np.concatenate(([0], np.flatnonzero(path[1:] != path[:-1]) + 1, [len(path)]))
    return list(zip(b[:-1].tolist(), b[1:].tolist()))


def absorb_edges(path, before_len, after_len):
    """First / last spell of a live stretch shorter than the limit joins its only neighbour."""
    out = path.copy()
    s = _spans(out)
    if len(s) > 1 and s[0][1] - s[0][0] < after_len:
        out[:s[0][1]] = out[s[1][0]]
    s = _spans(out)
    if len(s) > 1 and s[-1][1] - s[-1][0] < before_len:
        out[s[-1][0]:] = out[s[-2][0]]
    return out


def postfilter(labels, blocks, micro_s=1.2, before_d=0.0, after_d=0.0, fill_d=0.0):
    out = labels.copy()
    for _, first, last in blocks:
        half = out[first:last + 1]  # view: writes go to out
        for s, e in possession.runs(half != "D"):
            path = (half[s:e] == "A").astype(np.int8)
            path = possession.absorb_micro_possessions(path, int(round(micro_s * FPS)))
            path = absorb_edges(path, int(round(before_d * FPS)), int(round(after_d * FPS)))
            half[s:e] = np.where(path == 1, "A", "H")
        for s, e in possession.runs(half == "D"):
            if 0 < s and e < len(half) and e - s < fill_d * FPS and half[s - 1] == half[e]:
                half[s:e] = half[e]
    return out


def label_match(tracking, is_home, pitch, blocks, tags_info, **kw):
    return postfilter(possession.label_match(tracking, is_home, pitch, blocks, tags_info), blocks, **kw)


if __name__ == "__main__":
    lab = np.array(list("DDHHHAAHHHHHHDDDHHDDDDAHHHHHHHHD"))
    blk = [(1, 0, len(lab) - 1)]
    assert "".join(postfilter(lab, blk, micro_s=0)) == "".join(lab)                      # no-op
    assert "".join(postfilter(lab, blk, micro_s=3 / FPS)) == "DDHHHHHHHHHHHDDDHHDDDDAHHHHHHHHD"
    assert "".join(postfilter(lab, blk, micro_s=0, after_d=2 / FPS)) == "DDHHHAAHHHHHHDDDHHDDDDHHHHHHHHHD"
    assert "".join(postfilter(lab, blk, micro_s=0, fill_d=4 / FPS)) == "DDHHHAAHHHHHHHHHHHDDDDAHHHHHHHHD"
    print("ok")
