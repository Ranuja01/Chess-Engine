# Time manager: what the reference engines do, what we do, and the design (2026-10-25)

Scope: time management only (the transition plan's item #1 after gen 6). This was asked for because gen 6 is +77 against
Mediocre 0.5 at a fixed 1 s per move but only +7 at 60+0.6. Sources: Stockfish 11 / 15.1 / 16.1 / 18 read from the local
trees, and Ethereal, Weiss, Berserk, Koivisto, Stormphrax and Obsidian fetched from GitHub master on 2026-10-25. File:line
citations are to those copies. Constants are given in each engine's own units.

## 1. What we do now

**Time allocation:** `uci_main.cpp:188-197`, placeholder rule.
- `budget = rem/30 + 0.8*inc` (`rem/movestogo` when movestogo is given).
- `hard = min(4*budget, rem/4) - 0.03 s` goes into `TIME_LIMIT`. The search checks it every 2048 nodes against the time since the search started.
- `MOVE_TIMES[d] = budget * preset.MOVE_TIMES[d] / preset.TIME_LIMIT`, capped at `hard`.

**Start rule:** `search_engine.cpp:3696`. Another iteration starts only when `elapsed <= MOVE_TIMES[depth]`, and `elapsed` is **the last iteration's own duration** (`t0` is reset at :3728). Two consequences:
- The ratio comes from the preset, so **under a clock the preset changes how much time we spend.**
  - The UCI default is STANDARD, ratio 6/45 = 0.133. All the Mediocre and self-play clock matches ran with it.
  - LIGHTNING has ratio 0.75, about five times more willing.
- Combined with the measured growth below, the rule stops once the total time reaches about 0.28 × budget. The final iteration then finishes around 0.53 × budget.
  - So we use roughly 0.3–0.55 of the budget. At the start of a 60+0.6 game the budget is 2.48 s.
  - That fits the measured 0.94 s per move at 60+0.6, against Mediocre's 1.38 s.
  - The hard limit (4 × budget) almost never binds.

**Starting depth:** iterative deepening starts at **depth 3** (:3663), and the first pass cannot be interrupted (:3604).

**Cut iterations:** a cut iteration's result is **always thrown away** (`completed_move`, :3680-3685 and :3798-3807).
- At the root, a move becomes `best_move` only after its own search finishes uncut (the `time_up` check at :4540 comes before the update at :4584). So a cut iteration's `best_move` is the best of its fully searched moves, but it is never played.
- In the fixed-time probe, 36.1% of all move time went into cut iterations (10,913 moves). The cut iteration's best move differed from the completed one in 8% of cuts.
- Under a fixed time per move (st=1 and the UI presets), every move ends in a cut iteration. Under a clock, cuts are rare, because the start rule stops long before the hard limit.

**Move overhead:** a fixed 30 ms, not a UCI option.

### Measured: per-iteration time growth (2026-10-25)

Probe: production gen 6, 30 UHO positions, `go movetime 8000`, from the `info ... time` lines. The script is in the scratchpad, `iter_growth.py`.

| depth | 9 | 10 | 11 | 12 | 13 | 14 | 15 | all (219) |
|---|---|---|---|---|---|---|---|---|
| median T(d)/T(d−1), cumulative | 1.76 | 2.02 | 1.75 | 2.01 | 1.62 | 2.03 | 1.73 | **1.93** (mean 2.08) |

The cost of the next iteration as a fraction of the time already spent: median **0.86**, p75 1.35, p90 1.87. The median maximum depth at 8 s was 14.5.

## 2. The reference engines

### 2.1 Time allocation

| engine | optimum (soft), sudden death | maximum (hard) | move overhead |
|---|---|---|---|
| SF18 `timeman.cpp:90-134` | `timeLeft = T + (inc·(50.5−1) − MO·(2+50.5))`; `opt = min(0.0121 + (ply+2.95)^0.461 · c(logT), 0.213·T/timeLeft) · adj(TC)` × timeLeft. At 60+0.6, ply 20: **≈ 2.80 s** | `min(0.825·T − MO, maxScale·opt) − 10 ms`, maxScale 3.4 + ply/12 rising to 6.7 (≈ 6.7× opt) | 10 ms |
| SF11 `timeman.cpp:84-133` | minimum over hypothetical moves-to-go 1..50 of a ply "move importance" share | the same with MaxRatio 7.3 and StealRatio 0.34 | 30 ms |
| Ethereal `timeman.c:49-80` | `(T − MO + 25·inc)/20` | 4 × ideal, ≤ T − MO | 300 ms |
| Weiss `time.c:25-56` | `min(0.022·(T + 50·(inc − 6)), 0.2·T)` | `min(5·opt, 0.8·T)` | 6 ms (fixed) |
| Berserk `uci.c ParseGo` | `min(0.419·T, 0.0575·(T + 50·inc − 50·MO))` | `min(0.922·T − MO, 5.93·alloc) − 10` | 50 ms |
| Koivisto `timemanager.cpp` | `T/10 + 2·inc` (movestogo default 20) | `T/2` | 0 (+50 if inc = 0) |
| Stormphrax `limit.cpp` | `0.68·((T − MO)/19 + 0.94·inc)` | `0.65·(T − MO)` | n/r |
| Obsidian `timeman.cpp` | `min(0.025, 0.214·T/timeLeft)·timeLeft`, with timeLeft as in SF | `0.8·T − MO` | n/r |

Common ground:
- **The base budget is about 1/40 to 1/20 of a "total"** that counts roughly 50 future increments (T + ~50·inc).
- **The hard limit is 4–6× the base**, capped at 0.8–0.92 of the remaining time.
- **Movestogo is clamped to 50** wherever it is used.
- **Under `go movetime`, Stockfish, Ethereal and Berserk use no soft logic at all:** they search until the time is up. Weiss can stop at 0.52 × movetime.

### 2.2 The soft stop: decided only between iterations

Every engine: `stop when elapsed > base × (product of factors)`. Within an iteration only the hard limit can stop the search. No engine predicts the next iteration's cost explicitly. SF has a weaker form: past 0.5 × totalTime (0.6 in SF11), the next iteration repeats the same depth instead of going deeper (`increaseDepth`).

| factor | SF18 `search.cpp:485-528` | Ethereal | Berserk | Obsidian | Stormphrax | Weiss | Koivisto |
|---|---|---|---|---|---|---|---|
| **best-move node share f** (cumulative over all iterations of this move) | f ≥ 93.3% → ×0.76 (since 2024, from Koivisto) | max(0.5, 2(1−f)+0.4) | max(0.563, 0.45+2.27(1−f)) | 0.63+2.0(1−f) | max(0.188, 2.59−1.6f) | 0.52+3.73(1−f) (the only factor) | 1.1−min(f, 0.9) |
| **stability** (consecutive iterations with the same best move) | instability 1.02+2.14·(best-move changes, halved each iteration); timeReduction sigmoid centred 12 iterations after the last change | 1.20−0.04s (s ≤ 10), only from 4 completed iterations | 1.311−0.053s (s ≤ 10) | 1.71−0.08s (s ≤ 8) | power law 2.36 → 0.80 | – (the "uncertain" flag is probably inert) | – |
| **score drop** (one-sided: a stable or rising score sits at the floor) | clamp((11.85 + 2.24Δavg + 0.93Δ4iter)/100, 0.57, 1.70) | clamp(0.05·(s[d−3]−s[d]), 0.75, 1.25) | clamp(0.113 + 0.026Δ3iter + 0.026Δprevmove, 0.50, 1.66) | clamp(0.86 + 0.01Δiter + 0.025Δmove, 0.81, 1.5) | moving-average form, 0.66–1.40 | – | 0.77–1.23 |
| scaling starts from | depth 1 | 4 completed | depth 5 | depth 4 | 6 (stability term) | depth 1 | depth 1 |
| single legal move | totalTime ≤ 502 ms | – | – | stop after 200 ms | max 0.5 s | opt ≤ 500 ms | stop immediately |

Points to carry over:
- **The best-move node share is now universal.** Seven of the eight engines have it, including Stockfish since 2024.
  - It came from Koivisto. Ethereal's rewrite with it gained **+9.7 at 60+0.6, +9.9 at 10+0.1 and +20.9 at 40/10** (OpenBench 26080/1/93).
  - Weiss's node-share stop gained **+11.4 at 8+0.08 and +11.8 at 40+0.4**.
  - Berserk reports that changing only how nodes are counted moved Elo through this term by up to −4 ⇒ **the node accounting has to be deliberate.** For us that means deciding whether the root pre-search's nodes count.
- **The score-drop factor only adds time,** and when the score is stable it sits at about 0.5–0.8. So in quiet positions the effective budget is well below the base.
- **Stability counters span about ×1.3 when unstable to ×0.8 when stable.** Stockfish's instability multiplier is larger (×3.2 after one change in the last iteration).
- The measured gain of "don't start an iteration you can't use" in Weiss (#254) was **+57 at 10+0.1**. We already have a crude version of it, through the preset ratio.

### 2.3 Hard stop

- The clock is checked every 512–4096 nodes; ours is 2048.
- SF16.1+, Ethereal, Weiss and Stormphrax **always finish depth 1**, even past the maximum. Koivisto always records the root best at depth ≤ 2.

### 2.4 A cut iteration: which move is played

**No engine ever takes a move whose own subtree was cut.** They split on what they do with the moves that did finish:

| rule | engines |
|---|---|
| **Play the cut iteration's best so far**: a move whose search completed in the cut iteration and beat alpha (or the first move, scored with its fail-low bound) replaces the previous best. A stable sort leaves moves not yet searched in their old order | **Stockfish (11 and 18)**, Stormphrax, Obsidian, Koivisto |
| **Play the last completed aspiration pass** (a completed fail-high pass of the cut depth counts; a new best found partway through a pass does not) | Ethereal, Weiss, Berserk |
| Play only the last fully completed iteration | **us** (more conservative than all of them) |

Stockfish's own comment (`search.cpp:383-389`): the sort after an abort "is safe because RootMoves is still valid, although it refers to the previous iteration". In a fail-low pass no other move beat alpha, so the previous best stays. This supports the owner's point that a deeper iteration's best is rarely worse than the previous best.

### 2.5 Starting depth

**All eight engines start iterative deepening at depth 1.** Ours starts at 3.

### 2.6 Do the constants depend on engine strength?

**No source says so.**
- **Tuning:** the constants are SPSA- or hand-tuned on each engine's own fishtest/OpenBench TCs (10+0.1, 60+0.6; one SF tune at 180+1.8).
- **The one explicit dependence:** SF doubled the weight of best-move changes in 2020 (5f426d8) because "NNUE gives a more stable eval". So the constants depend on **eval stability and search shape**, not on Elo as such.
- **Our search shape is in the same range:** cumulative growth per iteration is 1.93, close to what they assume. So their shapes should transfer, but their constants have to be retuned on our engine.
- **Two signals have to be measured before choosing thresholds:**
  - **our node-share distribution.** Ordering quality and the root pre-search both affect it; SF's 93% threshold assumes SF's ordering.
  - **our best-move change rate.** At our depth 13–15, SF18's 12-iteration sigmoid sits near its midpoint even for a move that never changed, so a deep-engine centre doesn't fit us.
- **Units:** score slopes have to be converted to millipawns. Ethereal 0.05/cp and Berserk 0.026/cp work out to about 0.003–0.005 per millipawn.

## 3. Design: four steps, each behind a knob, built and tested one at a time

The internal A/B tests use plain builds, and the production PGO build comes only after the matches. One knob, `TM_MODE` (0 = today's rule), is extended step by step; each step is its own commit and match.

**Step 1: allocation and soft stop (affects clock games only).**
- `opt = min(K_OPT·(T + 50·inc − 50·MO), 0.2·T)`, `max = min(5·opt, 0.8·T − MO)`, with movestogo clamped to 50 when given.
- Under a clock, the start rule becomes "stop after an iteration when total elapsed > opt" and no longer depends on the preset.
- Hard stop at max. `Move Overhead` becomes a UCI option (default 30 ms).
- `K_OPT` starts at **0.025** (Obsidian / Weiss level, which is about SF's at ply 20 without the factors): at the start of 60+0.6 that is opt ≈ 2.2 s and max ≈ 11 s.
- With growth of 1.93, the average use should be about 1.2–1.5 × opt ⇒ **about 2.5–3 s per move early in a 60+0.6 game, against our 0.94 s today.**
- **Identity:** st=1, fixed depth and the bench stay node-identical, because movetime and depth are untouched. This is what makes st=1 a control for this step.

**Step 2: depth 1 start and a guaranteed depth 1.**
- The first pass then becomes trivially short, so the "first pass can't be interrupted" problem goes away. This is what makes a low-time policy possible.
- This changes the tree at every time control, including the bench, so it **gets its own fingerprint row and its own match.** Expected effect at normal time controls: small (owner: "no benefit seen").

**Step 3: use the cut iteration (Stockfish's rule).**
- When the clock cuts an iteration, play its `best_move` if a move fully searched in that iteration beat the pass's alpha in an in-window or full-window pass. Otherwise keep the completed move.
- A fail-low pass's move-0 bound never replaces the completed move. Our move 0 is the pre-search's top, not necessarily the previous best, so we're stricter than Stockfish here.
- **Where it pays:** fixed-time play (st=1, the UI presets), where every move ends in a cut. The existing `g_tm_cut_differs` counter (8%) is the number of moves it touches.
- **Test:** st=1 self-play against step 2.

**Step 4: the scaling factors, measured first (instruments before terms).**
- Counters first:
  - the distribution of the best move's node share, with and without pre-search nodes;
  - the best-move change rate per depth;
  - the score change over 3 iterations, in millipawns.
- Then the consensus form, which multiplies the soft limit:
  - node factor `max(0.5, 0.45 + 2.2(1−f))`;
  - stability `1.3 − 0.05s` (s ≤ 10);
  - score drop `clamp(a + b·Δ, 0.5, 1.6)` with b converted to millipawns;
  - starting from depth 5.
- Each factor is judged on its own first (search changes are antagonistic, not additive), then together.
- Also here: the single-legal-move cap (500 ms) and the low-time policy (opt floor, emergency when T < ~1 s).

**How each step is judged:**
1. Actual time used against time available, from the PGNs (`fastchess_report.py`).
2. Self-play at 60+0.6 against gen 6 (200–400 games).
3. The **Mediocre 60+0.6 rematch** (200 games), against the pooled **+7.4** baseline.
4. st=1 as the control wherever the step leaves it identical.
- Every match: `-recover`, a disconnect count, a check for new dumps, at most 3 games at a time.

**Predictions registered for step 1 (before any build):**
- self-play 60+0.6 against gen 6: **+25 … +70**;
- Mediocre 60+0.6: from +7 to **+35 … +75** (toward the st=1 +77);
- time per move at 60+0.6 in moves 1–40: **2.0–3.0 s**;
- 0 time forfeits;
- st=1: node-identical.

## 4. Owner decisions (2026-10-25)

- **K_OPT stays at the standard 0.025, with no bracket and no fine-tuning now.** The optimum moves as our branching factor falls, so constants are retuned once the engine is stronger.
- **Step 3: Stockfish's rule.** Plus our extra guard: a fail-low pass's move 0 never replaces the completed move.
- **Under `go movetime`: search to the end,** as Stockfish does.

Step 1 is built as `build_uci/chess_engine_uci_tm1` (plain build, commit 28b0e97). All four fingerprints are identical. The opening move at 60+0.6 went from 0.78 s to 3.61 s; movetime 1000 is unchanged (0.97 s).

**Queue #102 result (step 1):**

| match | result | registered |
|---|---|---|
| self-play 60+0.6, 200 g | **+85.0** (95% +40.7 … +132.3) | +25 … +70 |
| Mediocre 60+0.6, 200 g | **+96.2** (95% +51.9 … +143.8) | +35 … +75 |

- The Mediocre baseline was +7.4.
- 0 disconnects. The only 3 time forfeits were Mediocre's.
- Time per move in moves 1–20 / 21–40 / 41+: 2.83 / 1.36 / about 0.6 s. That is front-loaded; see OPTIMIZATION_LOG 2026-10-25.
- Step 2 is dropped for now (the depth-3 first pass takes at most 10 ms). Step 3 is built as `TM_PARTIAL`.

**Step 3 test, registered before running (2026-10-25 evening):**
1. **Where it fires** (`diagnostics/_partial_iter_judge.py`): 6,000 middlegame positions from the #102 PGNs (moves 10–60, |score| ≤ 3), at movetime 1000 with TM_PARTIAL=1. SF18 scores both moves at 2M nodes each.
   - The smoke test changed 2.5% of moves (3 of 120), so expect about 150 changed moves.
   - Predicted: the adopted move is better in ≥ 55% of the non-tied cases (|Δ| > 0.5 pp); mean Δwin% (adopted − completed) **+0.3 … +1.5 pp**.
2. **st=1 self-play, 400 g, TM_PARTIAL=1 vs 0** (plain `chess_engine_uci_g61`): predicted **−10 … +25**. This is a no-harm check only: with about 2.5% of moves touched, 400 g cannot resolve the expected effect.
- **Ship rule:** ship if (1) is positive and (2) is above −1σ.

**Known gap, left for step 4:** with no increment and 1–2 s left, the 30 ms × 52 overhead reserve leaves almost nothing, so the move is played at once at depth 3. That is no worse than today. With any increment the gap is gone: 1 s + 0.1 → optimum 0.11 s.
