# Team Betelgeuse — More than a Hack 2027, Phase 1: Ball Possession

Our submission to **More than a Hack 2027**, organised by Metrica Sports, FC Barcelona and the
Mobile World Capital Barcelona. The task: for every frame of FC Barcelona vs Athletic Club
(`BAR-ATH`, LaLiga 2024-25), say who has the ball — `H` (home), `A` (away), `D` (dead ball) or `X`
(ungraded). One CSV, 199,319 rows, ranked on `spellF1`.

This repository is a fork of the
[official challenge repository](https://github.com/metrica-sports/more-than-a-hack-2027). The
organisers' documents are unchanged: [CHALLENGE.md](CHALLENGE.md), [DATA.md](DATA.md),
[EVALUATION.md](EVALUATION.md), [SUBMISSION.md](SUBMISSION.md), [RULES.md](RULES.md),
[FAQ.md](FAQ.md). The full technical document is [Betelgeuse.tex](Betelgeuse.tex) (submitted as
PDF).

---

## Results at a glance

| test | spellF1 | macroF1 | note |
|---|---|---|---|
| Public 5-minute window | **0.4533** | **0.8570** | 17 possessions only; bootstrap interval 0.27–0.64 |
| External matches (mean of 3) | **0.466** | **0.729** | Metrica open games, labels from events — see below |
| Official baseline, full match | 0.2433 | 0.6724 | the published floor to beat |
| Official baseline, external matches | 0.256 | — | close to its real 0.2433, so the external test tracks the real one |

The full-match score is private and unknown to us. We trust the external test more than the public
window: it covers about 670 possessions on three matches we never designed against, against 17
possessions on a window we looked at while building the rules.

---

## Reproduce the submission

Python 3.8+ with `numpy` and `pandas` only. Developed on Python 3.14, numpy 2.5.3, pandas 3.0.6.

1. Unpack the official data bundle into `data/` (or set `MTAH_DATA_DIR`). Only these are read:
   `data/tactical/BAR-ATH_tactical_atd.txt`, `BAR-ATH_tactical_atd.xml`,
   `BAR-ATH_tactical_smart_tagging.xml`.
2. Run:

```bash
pip install -r requirements.txt
```

```bash
python possession.py
```

```bash
python evaluation/validate_submission.py evaluation/results/Betelgeuse.csv
```

About 8 seconds. No randomness: the same data gives the same file, byte for byte. The match data is
git-ignored and never committed.

---

## The method

One file, [possession.py](possession.py), no trained model. Every constant is a football or physics
prior stated in metres and seconds at the top of the file.

### 1. Dead ball (`D`) — a rule-based state machine, run inside each half

- **Kick-off gates (IFAB Law 8).** At each half start and after each goal, play stays dead until a
  real kick-off: both teams in their own halves (at least 8 tracked players each) and the ball
  kicked from the centre circle — or, if the ball is not tracked, a live Smart Tagging phase begins.
- **Out of bounds.** The ball more than 0.5 m beyond a line.
- **Lost while walking.** Ball not tracked and players walking (2 s mean speed < 1.7 m/s), gated by
  Smart Tagging: during a tagged live phase a stricter stop is required (< 1.35 m/s for 3 s).
- **Abandoned ball.** A slow ball (< 1.0 m/s for 1.5 s, or < 1.5 m/s for 2.2 s) with nobody within
  3 m, nobody closing in on it, and slow players.
- **Kicker waiting.** A ball that barely moves for 3.5 s while players are slow (a set piece being
  prepared).
- **Boundaries.** Stoppage starts are back-dated to the last touch before players slow down (up to
  4 s back); restarts snap to the frame the ball is kicked (within ±2 s). Live stretches under 2 s
  between stoppages become dead; stoppages under 2 s are removed.

### 2. Who has it (`H` / `A`) — Viterbi decoding inside each live stretch

- **Evidence per frame:** which team is closer to the ball, when someone is within 3.5 m, clipped
  to ±1; down-weighted to 0.2 while the ball flies fast and straight (a pass in flight belongs to
  nobody).
- **Smart Tagging** adds team evidence: offensive phases for the tagged team, defensive phases for
  the opponent. Weight 0.10 when the ball is tracked near a player, 0.45 when it is missing. Near a
  restart, a `SET PIECES` tag naming a team gives that team a ±0.8 prior for 2 s.
- **Switching cost** depends on the situation: cheap (16–22) when a player of the new team is on
  the ball, expensive (45) while the ball is in flight or missing. So the last owner keeps the ball
  through passes and occlusions — the challenge's own rule for loose balls.
- **Touch rule.** A spell under 2.0 s is a touch, not a controlled possession (the challenge
  defines possession as keeping the ball "a few seconds"): a short spell inside the other team's
  possession goes back to that team, and so does a short last spell before a stoppage.

Frames outside the two graded halves are `X`. Every graded frame gets `H`, `A` or `D`.

---

## How we validated it without breaking the rules

The rules forbid hand labels on this match and forbid tuning on the public 5 minutes. With only 17
public possessions, any comparison there is noise anyway. So we built an **external test** on
other matches ([validation/](validation/)):

- **Data:** Metrica Sports' three
  [open sample games](https://github.com/metrica-sports/sample-data), about 430,000 frames.
  Download them into `validation/sample-data/` (git-ignored).
- **Labels from events, by code** ([validation/metrica.py](validation/metrica.py)): passes, carries,
  set pieces, recoveries and shots give their team the ball; a ball out, a foul or a shot out/in
  starts a stoppage until the next restart. A spell under 3 s that interrupts the other team's
  possession is given back, following the challenge's "controlled possession" definition.
- **Made to look like our tracking.** In the sample games the ball goes missing almost only during
  stoppages — a shortcut our match does not have. So we fill those gaps and cut new random ones
  (median 268 frames, as in our match) until 41% of frames lack the ball, plus 0.3 m of ball noise.
- **Run it:**

```bash
python validation/report.py
```

| method | game 1 | game 2 | game 3 | mean spellF1 |
|---|---|---|---|---|
| **Betelgeuse (final)** | **0.456** | **0.467** | **0.475** | **0.466** |
| before the 2 s touch rule | 0.437 | 0.454 | 0.458 | 0.450 |
| our first, simpler version | 0.385 | 0.395 | 0.435 | 0.405 |
| official baseline (`smooth50`) | 0.257 | 0.265 | 0.245 | 0.256 |

### Protocol for every change

Written down before any experiment ([validation/IMPROVEMENT_PROMPT.md](validation/IMPROVEMENT_PROMPT.md)):
choose on games 1–2, confirm on held-out game 3; accept only with ≥ +0.01 spellF1 on games 1–2, no
drop on game 3, and macroF1 down by no more than 0.01. Constants are round values fixed before
results are seen; every value tried is reported. The public window is never used to choose.

| change | verdict | evidence |
|---|---|---|
| Our first version vs the current dead-ball state machine | **current kept** | wins on all 3 external games (0.450 vs 0.405) |
| Random-search tuning of the first version's constants | rejected | best setting did not improve held-out game 3 (0.426 vs 0.435) |
| 2.0 s touch rule | **accepted** | better in all 9 runs (3 games × 3 gap patterns), +0.012 to +0.017 per game, macroF1 unchanged |
| 3.0 s touch rule | rejected | scored higher, but the external labels use a 3 s rule themselves, so the gain is partly artificial |
| Stoppage starts at the line crossing | rejected | +0.002 / −0.004; ball-outs already start on time — fouls and offsides are the late ones |
| Set-piece anchoring | rejected | no measurable effect; on our match it reopened two post-goal kick-off waits |
| Smart Tagging rules (tested with synthetic tags) | **kept** | +0.036 on games 1–2, +0.056 on game 3; removing any one rule lowers the score |

**The public window disagreed with the touch rule.** It moved from 0.4947 (21 possessions) to
0.4533 (13), against 17 true. We kept the rule: 17 possessions is within noise, and over the whole
match the rule brings us from 5.8 to 5.0 possessions per minute of live play — still above the
public truth's 4.2, so it reduces over-splitting without overshooting.

The candidate experiments are in [validation/candidates/](validation/candidates/), each with its
own runner (`run_a.py`, `run_b.py`, `run_c.py`).

---

## Rules we followed

- **No hand labels** on the provided match, for any purpose. No human-made data about it.
- **The public 5 minutes** were inspected to understand what the labels mean and scored as a
  correctness check — never used to fit or choose anything.
- **Labelled data from other matches** (Metrica open games) is used for validation only; its labels
  are derived by code from published events.
- **No match data** was uploaded to any hosted LLM or service, and none is in this repository.
- **Credit.** Validating on open matches, degrading the ball to resemble the automatic tracking,
  returning short spells to the surrounding team, and synthetic Smart Tagging are ideas we adapted
  from other participants' public forks — FrancescoRomeo02 "InkSight", MRH2818 and Mahmoudmoh-cse
  "Goalgorithm". No code was copied; the implementation is ours.

---

## Limitations

- **Late stoppage starts for fouls and offsides** (median 1.4 s late on the external games), when
  players keep jogging after the whistle.
- **Long live occlusions** (ball missing > 3.5 s in open play) lean on Smart Tagging, which is
  automatic and noisy.
- **Throw-ins** are tagged without a team, so an occluded throw-in's taker is guessed from
  proximity alone.
- **Genuine possessions under 2 s** are absorbed by the touch rule.
- **The external test is an approximation:** event-based labels, about 15% coarser than the
  organisers' (3.6–3.9 vs 4.2 possessions per live minute), and no real Smart Tagging.

---

## Repository map (our additions)

```
possession.py                 the method; writes evaluation/results/Betelgeuse.csv
Betelgeuse.tex                technical document (LaTeX; submitted as PDF)
validation/
  metrica.py                  Metrica open games -> labels from events + tracking degraded like ours
  report.py                   scores possession.py on the three games with the official metric
  IMPROVEMENT_PROMPT.md       the protocol every change had to pass
  candidates/                 the tested changes (A line crossing, B touch rule, C tags/anchoring)
  sample-data/                Metrica open data, downloaded separately (git-ignored)
```

Everything else — `evaluation/`, `starter/`, `games.py` and the challenge documents — is the
organisers' and unchanged. The scorer in `evaluation/` is the one that produces the official result.

## Licence

Code and documentation: MIT, as the upstream repository ([LICENSE](LICENSE)). The match data is not
covered and is not redistributed here.
