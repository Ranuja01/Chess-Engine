# Time manager

How AlphaUX decides how long to think on a move. This is the reference for the system as shipped in **gen 6.2**
(2026-10-27). The design history, the reference-engine comparison and the step-by-step test record live in
[TIME-MANAGER-RESEARCH-2026-10-25.md](TIME-MANAGER-RESEARCH-2026-10-25.md); the ship entries are in
[OPTIMIZATION_LOG.md](OPTIMIZATION_LOG.md) (2026-10-25, 10-26, 10-27).

Code: the budget is computed in `cmd_go` (`uci_main.cpp`); the per-iteration decisions are in the iterative-deepening loop
of `get_engine_move` (`search_engine.cpp`, `may_start_iteration` and the `TM_SCALE` block). Knobs: `Config::TM_*` in
`search_engine.h`.

---

## 1. Which rule applies

The UCI `go` command decides which rule runs. Only a real clock (`wtime`/`btime`) uses the time manager.

| how the engine is asked to move | what decides the time | time manager? |
|---|---|---|
| `go wtime … btime … [winc binc] [movestogo]` (fastchess, cutechess, a GUI clock, the UI's `clock=`) | sections 2-4 below | **yes** |
| `go movetime N` | exactly N ms less 30 ms overhead | no |
| `go depth N` / `go nodes N` / `go infinite` | the limit given, no time limit | no |
| bare `go` (the UI presets LIGHTNING / BLITZ / STANDARD / LONG_FORMAT) | the preset's per-depth `MOVE_TIMES` and `TIME_LIMIT` | no |
| the Cython path (`main.py`, the old in-process UI class) | the preset | no |

So a time-manager change can only move Elo in clock games. Preset strength (the UI's STANDARD) is untouched by it.

## 2. The budget for one move: optimum and hard limit

Each `go` with a clock computes two numbers once, before the search starts.

- **Optimum (soft limit)**: the time the move is expected to take. Checked only *between* iterations (section 3).
- **Hard limit**: an absolute stop, checked *inside* the search. Never exceeded.

Notation: `rem` = our remaining time, `inc` = our increment, `mo` = move overhead (UCI option `Move Overhead`, default
30 ms), `mtg` = `movestogo` if the GUI sends it.

```
h    = min(mtg, 50) if movestogo is given, else 50        # the horizon: how many moves the budget must last
left = rem + inc * (h - 1) - mo * (2 + h)                 # time available over the horizon, after the overhead reserve

with movestogo:  optimum = min(0.95 / h, 0.88 * rem / left) * left
sudden death:    optimum = min(0.012 + (ply + 3)^0.45 * 0.0039, 0.2 * rem / left) * left

hard = min(5 * optimum, 0.8 * rem - mo)                   # never more than 80% of the clock
```

- **With movestogo** (classical controls such as 40/15) the remaining time is simply split over the moves left before
  the control, 0.95 / h each. As the control approaches, h falls, so each move's share rises; on the last move before the
  control (h = 1) the move may use up to 80% of what is left. Time arriving *after* the control is never counted: it
  cannot help before the control, and overspending before it loses on time.
- **Sudden death / increment** (60+0.6, 3+2, 5+0) has no move count, so the horizon is a fixed guess of 50 moves. The
  share of `left` grows with the game ply (`TM_OPT_PLY`), because a game that has lasted longer has fewer moves left:

| ply | 0 | 10 | 20 | 40 | 60 | 80 | 120 | 200 |
|---|---|---|---|---|---|---|---|---|
| optimum as % of `left` | 1.84 | 2.44 | 2.80 | 3.32 | 3.72 | 4.05 | 4.60 | 5.46 |

### Examples with a full clock (optimum / hard, seconds)

| control | ply 0 | ply 20 | ply 40 | ply 80 | ply 120 |
|---|---|---|---|---|---|
| 10 s + 0.1 (internal A/B) | 0.25 / 1.2 | 0.37 / 1.9 | 0.44 / 2.2 | 0.54 / 2.7 | 0.61 / 3.1 |
| 60 s + 0.6 (internal A/B) | 1.6 / 8.1 | 2.5 / 12.3 | 2.9 / 14.6 | 3.6 / 17.8 | 4.0 / 20.2 |
| 1 min + 0 | 1.1 / 5.4 | 1.6 / 8.2 | 1.9 / 9.7 | 2.4 / 11.8 | 2.7 / 13.4 |
| 3 min + 2 | 5.1 / 25 | 7.7 / 39 | 9.2 / 46 | 11.2 / 56 | 12.7 / 64 |
| 5 min + 0 | 5.5 / 27 | 8.4 / 42 | 9.9 / 50 | 12.1 / 60 | 13.7 / 69 |
| 15 min + 10 | 26 / 128 | 39 / 194 | 46 / 230 | 56 / 281 | 64 / 319 |
| 90 min + 30 | 126 / 632 | 192 / 961 | 228 / 1140 | 278 / 1390 | 316 / 1580 |

(Each column assumes the full starting clock at that ply, to show the ply effect alone. In a real game the clock has
been spent, so the numbers shrink accordingly; the next table shows that.)

### 40/15 move by move (900 s per 40 moves, no increment)

If every move spent exactly its optimum:

| move | moves to the control | time left (s) | optimum (s) | hard (s) |
|---|---|---|---|---|
| 1 | 40 | 900.0 | 21.4 | 106.7 |
| 10 | 31 | 706.8 | 21.6 | 108.2 |
| 20 | 21 | 488.7 | 22.1 | 110.4 |
| 30 | 11 | 264.9 | 22.8 | 114.2 |
| 35 | 6 | 149.4 | 23.6 | 118.1 |
| 38 | 3 | 77.7 | 24.6 | 62.2 |
| 39 | 2 | 53.2 | 25.2 | 42.5 |
| 40 | 1 | 28.0 | 22.4 | 22.4 |

About 5.6 s would be left at the control and carried into the next 15 minutes. Real moves rarely spend exactly the optimum
(section 3 scales it). Unspent time raises every later budget automatically: if every move spent only half its optimum,
move 20 would get 30 s, move 39 109 s and move 40 140 s, and about 105 s would still carry over.

### Low clock

Sudden death, around ply 60 (optimum / hard, seconds):

| time left | 30 s | 10 s | 5 s | 2 s | 1 s | 0.5 s |
|---|---|---|---|---|---|---|
| no increment | 1.06 / 5.3 | 0.31 / 1.6 | 0.13 / 0.64 | 0.016 / 0.08 | ~0 / 0.01 | ~0 / 0.01 |
| + 0.1 s | 1.24 / 6.2 | 0.50 / 2.5 | 0.31 / 1.6 | 0.20 / 0.99 | 0.16 / 0.77 | 0.10 / 0.37 |
| + 1 s | 2.88 / 14.4 | 2.00 / 8.0 | 1.00 / 4.0 | 0.40 / 1.6 | 0.20 / 0.77 | 0.10 / 0.37 |

With any increment the engine keeps thinking roughly the increment per move. With none, see weakness 1 (section 6).

## 3. Inside the budget: spend more or less

The search deepens one iteration at a time (depth 1, 2, 3, …). After each **completed** iteration it decides whether to
start another. Nothing in this section can extend a move past the hard limit.

**The start rule (`TM_START=175`).** Another iteration starts only if

```
time spent so far + 1.75 * (duration of the last iteration)  <=  optimum * scale
```

The next iteration typically costs about 1.8x the last one (measured), so this refuses to start an iteration that would
run far past the budget and then be cut by the hard limit with its work thrown away (`TM_PARTIAL` is off: the move of
the last *completed* iteration is played).

**The scale (`TM_SCALE=7`, from depth 5 on; 1.0 before).** Three factors, recomputed after every completed iteration
from depth 5, multiply the optimum:

| factor | input | formula | easy end | hard end |
|---|---|---|---|---|
| node share | share of this search's root nodes spent under the current best move | `max(0.563, 0.4499 + 2.2669 * (1 - share))` | 0.563 (best move took ≥ 95%) | 2.72 (share near 0) |
| stability | completed iterations in a row with the same best move, 0-10 | `1.311 - 0.0533 * n` | 0.78 (10 in a row) | 1.31 (just changed) |
| score drop | fall in the score over the last 3 iterations, millipawns (rises count as 0) | `clamp(0.1127 + 0.00262 * drop, 0.5028, 1.6561)` | 0.50 (drop < 0.15 pawn) | 1.66 (drop ≥ 0.59 pawn) |

`scale = node share × stability × score drop × 0.95` (`TM_NORM=95`). The range is about 0.21 to 5.6, and the hard
limit (5x the optimum) caps the top. The score-drop factor sits at 0.50 for a steady score and reaches 1.0 at a drop of
about 0.34 pawn, so a quiet position spends well under the optimum and a collapsing one well over it.

Worked example, 40/15 move 1 (optimum 21.4 s, hard 107 s):

| position | factors | scaled limit | likely time |
|---|---|---|---|
| one legal move (`TM_SINGLE=1`) | stops after the first iteration | — | ~0 s |
| obvious recapture: ~95% of nodes on the best move, stable, steady score | 0.56 × 0.78 × 0.50 × 0.95 | ~4.5 s | 2-4 s |
| ordinary quiet move: ~70% share, best move stable for 5 iterations | 1.13 × 1.05 × 0.50 × 0.95 | ~12 s | 7-11 s |
| best move keeps changing, ~40% share | 1.81 × 1.31 × 0.50 × 0.95 | ~24 s | 15-24 s |
| the same, and the score has dropped half a pawn | 1.81 × 1.31 × 1.42 × 0.95 | ~68 s | 45-68 s |
| everything maxed | | over 107 s | capped at 107 s |

"Likely time" is lower than the scaled limit because the start rule stops at an iteration boundary. With iterations
growing ~1.8x, the arithmetic of the rule puts that stop between about 56% and 100% of the limit. These rows are worked from the formulas, not measured move by move.

## 4. One move, start to finish

1. Book move? Played at once (`OwnBook`, from the start position, up to `BookMaxPly`).
2. Compute optimum and hard limit (section 2). One legal move: optimum = 0.
3. Iterate depth 1, 2, 3, …; the hard limit is checked during the search and aborts it.
4. After each completed iteration: record the best move and score; from depth 5 recompute the scale; apply the start rule.
5. Play the best move of the last completed iteration. A cut iteration's partial result is discarded.

## 5. The knobs (gen 6.2 defaults)

| knob | default | meaning |
|---|---|---|
| `TM_MODE` | 1 | 1 = this time manager; 0 = the pre-6.1 placeholder rule (preset share of a fixed budget) |
| `TM_OPT_PLY` | 1 | sudden-death optimum grows with the ply; 0 = flat `min(0.025, 0.214 * rem / left)` |
| `TM_START` | 175 | start rule, percent of the last iteration; 0 = start whenever the optimum has not passed |
| `TM_SCALE` | 7 | bits: 1 node share, 2 stability, 4 score drop; 0 = no scaling |
| `TM_NORM` | 95 | percent applied to the scale product |
| `TM_SINGLE` | 1 | stop after the first iteration with one legal move |
| `TM_PARTIAL` | 0 | 1 = play a cut iteration's move when its own search finished and beat alpha |
| UCI `Move Overhead` | 30 ms | per-move allowance for GUI / transport latency |

Gen 6.1 behaviour from a gen 6.2 binary: `Knobs TM_START=0,TM_SINGLE=0,TM_SCALE=0,TM_NORM=100,TM_OPT_PLY=0`.

## 6. Weaknesses and open items

1. **No increment, under ~2 s left.** Sudden death reserves the overhead for the whole 50-move horizon (30 ms × 52 =
   1.56 s). With no increment and a clock near that size, `left` collapses and the engine plays almost instantly at
   depth ~3 (2 s left → 16 ms; 1 s → ~0). It never flags this way, but it plays the last seconds far below its strength.
   The cause is the fixed 50-move guess rather than the formula: sudden death does not say how many moves remain, and
   reserving overhead for 50 of them is too pessimistic once the clock is short. Any increment removes the problem.
   A fix (shrink the reserve or the horizon as the clock runs down) is parked for the release-readiness sweep.
2. **The horizon ignores the position.** Sudden death always assumes 50 more moves, whether it is move 10 of a
   middlegame or a simple pawn ending at move 70. The ply curve compensates on average only.
3. **The soft limit is checked only between iterations.** Within an iteration only the hard limit (5x the optimum) can
   stop the search. If the best move fails low halfway through a long iteration, the engine cannot react until the
   iteration completes; the score-drop factor sees it only afterwards.
4. **The scale factors are borrowed.** Their constants come from Stockfish's search shape. Only `TM_START` and
   `TM_NORM` were set for us. The factors alone measured flat (#104); they paid only together with the start rule (#106).
   The set was tuned at 60+0.6, mostly in self-play.
5. **No pondering**, and the opponent's clock is not used.
6. **Revisit as the engine changes.** A stronger or faster search changes how iteration cost grows and how often the
   best move changes, which is what these rules are fitted to. Re-check at least after search v2.

## 7. Evidence

All at 60+0.6 unless noted; Elo ± 95% interval.

| step | vs | result |
|---|---|---|
| gen 6.1: optimum / hard rule (`TM_MODE=1`) | the placeholder rule, self-play | +85 |
| gen 6.1 | Mediocre 0.5 | +96.2 |
| gen 6.1, movestogo path, 40/60 | Mediocre 0.5, 100 g | +56, 0 forfeits |
| step 4 factors alone (#104) | gen 6.1 | A +7.0, B +24.4, C +12.2 (± ~41 each) |
| start rule + single move (D, #106) | gen 6.1, 200 g | +3.5 ± 44.1 |
| **everything together (E, #106) = gen 6.2** | gen 6.1, 200 g | **+59.6 ± 46** |
| gen 6.2 (#107) | Mediocre 0.5, 200 g | **+119.1 ± 48.2**, 0 forfeits by us |
| gen 6.2 (#108), us 60+0.6 vs it 20+0.2 | Stockfish 1.1, 200 g | 21.0% (gen 6.1: 22.5% / 100 g) |
