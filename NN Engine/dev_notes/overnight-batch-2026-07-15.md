# Overnight autonomous batch — post-falsification (2026-07-15)

Context: the falsification (`falsification-bent-ruler-2026-07-15.md`) proved the fixed-NODE gauntlet was a
bent ruler — mobility is +20 Elo at fixed TIME, a real gain the venue threw away. The eval lanes are REOPENED.
This batch exploits that: re-gate previously-"closed" eval levers at the CORRECT venue (fixed-TIME lightning
SPRT), find the bundle bricks, then combine.

## Venue discipline (carry into every job here)
- The REAL arbiter = fixed-TIME lightning SPRT (`gate` sub). Charges the eval its true cost.
- Fixed-DEPTH (`fast_tourney`) OVER-credits eval (cost free) — use only for cheap RELATIVE ranking among
  variants of the SAME lever (e.g. MOBILITY_SCALE magnitudes), never as a ship signal.
- Fixed-NODE = biased against eval accuracy — do NOT use to judge eval levers.
- Run one job at a time; read the result; decide the next (CLAUDE.md unattended discipline). Each `gate` is
  ~1.5-2.5 hr (caps at 600 games; a real +15-20 effect won't trip the +2.94 bound — read the point estimate).
- All dispatcher calls SINGLE-LINE, `bash '<runner>' <sub> ...` prefix. Read task .output with the Read tool.
- byte-id 247 after any build (none needed — all knobs are env-read into the existing .so).

Runner = `/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine/selfplay/overnight_runner.sh`

## ⚠️ MID-SESSION FINDING — Mediocre exposes real strength (2026-07-15 eve)
Preliminary 6-game FAIR equal-time (our LIGHTNING ~1s vs Mediocre 1s): **8.3% for us (0W/1D/5L)**. NOT a harness
bug (verified: we search depth 12-14 @ ~600k nodes vs Mediocre depth 9-12 — full strength). Mediocre 0.5 = ~2319
CCRL (2012, stronger than the 2007 ~2200). Implied real strength ~1900-1950 = coherent with "can't beat SF1";
the ~2700 self-estimate is inflated by the over-optimistic eval. Collapse decomposition on ONE collapse (game 5,
peak +2.35): our MOVE matched SF 100% at the peak+drop, but the peak is a CALM position — the real blunder is
elsewhere. Two reassuring facts are THINNER than they look: (1) "deeper search" is NOMINAL — EBF 3.2 vs SF ~2 =
bushy/soft, effective depth on the critical line may be worse; (2) "SF-like eval" only shown at calm peaks, not
the losing moves. Candidates for what's wrong: search EFFICIENCY (bushy tree cuts critical lines), eval
calibration at SHARP positions (hides from calm probes AND self-play), known bugs (root-razoring `break`,
root-child TT depth over-trust). STRATEGIC: Mediocre is a real external yardstick (self-play CANNOT surface
collapse holes) → gate collapse-relevant eval items vs MEDIOCRE, not just self-play.

## QUEUE (priority order)

### 0. Mediocre BASELINE + collapse decomposition (do FIRST — establishes the yardstick)
Get a trustworthy baseline score (6g + UHO = huge CI). Then decompose the LOSING moves (not peaks):
```
pyrun selfplay/vs_sf.py --our-label ours --our-config 'PRESET=LIGHTNING MAX_DEPTH=64 USE_OPENING_BOOK=0' --sf-path /home/ranuja/mediocre_uci.sh --sf-arb-path '/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/stockfish_18_linux/stockfish-ubuntu-x86-64-avx2' --opponent-raw --sf-elo 0 --sf-movetime 1.0 --games 40 --concurrency 2 --seed 0 --openings selfplay/openings_uho.txt --adjudicate-draw --quiet --tag mediocre_base
```
Then run the COLLAPSE AUTOPSY (design: collapse-autopsy-design-2026-07-15.md) — per losing move, a toggle-recovery
matrix that LABELS each blunder by mechanism: shallow prune (RFP/razoring/futility) vs deep prune (LMR) vs LMP/null
vs HORIZON (only depth recovers) vs EVAL-MIS-RANKS (nothing recovers) vs BUG (root-razoring break) vs E-drift
(no single blunder = calibration drift). Build diagnostics/collapse_autopsy.py; prototype on the game-5 collapse,
run on the 40-game set. The output DISTRIBUTION picks the lever. Target the DROP move, not the peak.

### 0b. KEY cross-venue test — does mobility's +20 SELF-PLAY transfer vs Mediocre?
Self-play can't see collapse holes. Gate mobility (the confirmed +20 self-play brick) AS the candidate config vs
Mediocre — same baseline openings/seed — and compare score to the base baseline (0). If mobility LIFTS the
Mediocre score, the self-play gain is real vs a real opponent (ship default). If flat, self-play gains may be
orthogonal to the collapse weakness (important negative result).
```
pyrun selfplay/vs_sf.py --our-label ours --our-config 'PRESET=LIGHTNING MAX_DEPTH=64 USE_OPENING_BOOK=0 ENABLE_MOBILITY=1' --sf-path /home/ranuja/mediocre_uci.sh --sf-arb-path '/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/stockfish_18_linux/stockfish-ubuntu-x86-64-avx2' --opponent-raw --sf-elo 0 --sf-movetime 1.0 --games 40 --concurrency 2 --seed 0 --openings selfplay/openings_uho.txt --adjudicate-draw --quiet --tag mediocre_mob
```
(Same idea for endgame-scale/damping if they gate positive in self-play — validate vs Mediocre.)

### 1. corrhist fixed-TIME gate (the pending real-arbiter read)
Fixed-depth was NEUTRAL (+2.6 ±39.9). Does it hold/help at fixed time?
```
gate 'ENABLE_CORR_HIST=1 CORR_W=32' corrhist sprt_corrhist_time 600 5 4
```
Decision: ≥ ~neutral → a bundle brick; clearly negative → corrhist genuinely doesn't convert (unlike mobility).

### 2. MOBILITY_SCALE magnitude — cheap fixed-DEPTH rank, then gate the winner
Mobility@40 = +20 at time. Peak magnitude unknown. First a cheap relative rank (fixed-depth is OK for ranking
variants of ONE lever — same cost structure), base vs each scale:
```
fast_tourney 40 10 'ENABLE_MOBILITY=1 MOBILITY_SCALE=25' 4 mob_s25
fast_tourney 40 10 'ENABLE_MOBILITY=1 MOBILITY_SCALE=60' 4 mob_s60
fast_tourney 40 10 'ENABLE_MOBILITY=1 MOBILITY_SCALE=80' 4 mob_s80
```
Then fixed-TIME gate the best 1-2 scales vs base:
```
gate 'ENABLE_MOBILITY=1 MOBILITY_SCALE=<best>' mob_sbest sprt_mob_sbest 600 5 4
```

### 3. Re-gate the OTHER "closed" eval-feature levers at fixed-TIME (the big payoff)
These were closed on the bent ruler; re-test on the real one, one at a time. Candidates (knobs verified in
search_engine.h / runner subs):
```
gate 'IMBALANCE_SCALE=4' imbalance sprt_imbalance_time 600 5 4
gate 'ENABLE_KS_REPLACE_LT=1 KING_SAFETY_MAG=4000 MOD_KS_CONTROL=256' ksctl sprt_ksctl_time 600 5 4
```
(Pull more from mobility-term-2026-07-12.md / the eval-feature closed list; gate each. KEEP the fixed-time
survivors as bricks.)

### 4. Mobility overlap at fixed-TIME (confirm clean brick)
2x2 {cheap-rook × mobility}. cheap-rook is default-ON. Gate mobility-with-cheap-rook-OFF and the pair vs base
to see if mobility's +20 is additive or overlaps the shipped cheap-rook term:
```
gate 'ENABLE_MOBILITY=1 ENABLE_CHEAP_ROOK_MOBILITY=0' mob_norook sprt_mob_norook 600 5 4
```
(Compare to mobility@default and to the fixed-depth STS 2x2 seeded pre-game, tag ov_*.)

### 5. Mediocre for-fun exhibition (harness now robust — respawn-on-hang)
FAIR EQUAL-TIME: our LIGHTNING (TIME_LIMIT=1.0 = hard 1s/move cap, deepens to ~0.75s) vs Mediocre 1.0s/move.
This replaces the old node-handicap recipe (our 250k nodes vs their 1s) which was deterministic + Mediocre-
favoring but NOT the fair TC. No NODE_LIMIT. Driver = raw_uci.py + vs_sf.py --opponent-raw (validated: no
desync, self-heals Mediocre's nondeterministic deadlock by kill+respawn). conc2-3 (keras OOM cap):
```
pyrun selfplay/vs_sf.py --our-label ours --our-config 'PRESET=LIGHTNING MAX_DEPTH=64 USE_OPENING_BOOK=0' --sf-path /home/ranuja/mediocre_uci.sh --sf-arb-path '/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/stockfish_18_linux/stockfish-ubuntu-x86-64-avx2' --opponent-raw --sf-elo 0 --sf-movetime 1.0 --games 30 --concurrency 2 --seed 0 --openings selfplay/openings_uho.txt --adjudicate-draw --quiet --tag mediocre
```
NOTE: a low-budget 6-game smoke (120k nodes vs Mediocre 0.3s) scored only 16.7% (0W/2D/4L, 4 by resign, one
+4.2 collapse). At that budget Mediocre 0.3s (~depth 9 / 200k nodes) out-searched our 120k — so it is NOT a
strength read; equal-time is the fair one. WATCH: if we still lose/collapse at 1s vs 1s, that's a real finding
(weaker vs Mediocre's style than the ~2700 estimate, or collapse-prone vs a different-style opponent) → Mediocre
becomes a useful collapse-mining opponent, not just a fun anchor. If we WIN clearly, the anchor is confirmed.
Optional future: add a MOVE_TIME_OVERRIDE knob for exact N-second parity (memory-noted; needs a build).

### 3b. Re-gate the NODE-SAVING ordering/pruning levers at fixed-TIME (SAME bent-ruler insight)
These were closed as "regression-to-mean" on the fixed-NODE gauntlet — which STRUCTURALLY cannot see a
node-saving lever (same node cap both sides ⇒ savings never convert to depth). They are effectively SPEED
levers (fewer nodes/pos → more depth in TIME). Re-gate at fixed-time, where savings→depth→Elo IS visible:
```
gate 'ENABLE_CHECK_ORDER=1' chkorder sprt_chkorder_time 600 5 4
gate 'ENABLE_HISTORY_MALUS=1' malus sprt_malus_time 600 5 4
gate 'ENABLE_CHECK_ORDER=1 ENABLE_HISTORY_MALUS=1' chkmalus sprt_chkmalus_time 600 5 4
```
(chk+malus was −13% nodes offline with tactics UP — the strongest node-saver; at fixed time that −13% is real
depth. piece×to −4.9% nodes is another candidate but had a fixed-depth STS strategic cost — gate cautiously.)
This roughly DOUBLES the reopened territory vs the eval-only re-gate.

### 3c. Other eval candidates to re-gate at fixed-time
- ENABLE_PIECE_MOBILITY (the FULLER per-piece mobility; NOTE it disables the cheap surrogates, so it's a
  DIFFERENT config than ENABLE_MOBILITY — but the +20 net-square result suggests the fuller term may gate well):
  `gate 'ENABLE_PIECE_MOBILITY=1' piecemob sprt_piecemob_time 600 5 4`
- corrhist higher weight if #1 is neutral-positive: gate CORR_W=64/96 (fixed-depth was neutral at 32).
- endgame-scale / rule-50 damping (the lever load-bearing-optimism PREDICTED should help; now that the doctrine
  is suspect, worth a clean fixed-time gate). Pull the exact knob from the HCE-eval brief.

### 6. First BUNDLE (after 1-4 resolve)
Combine the fixed-time survivors (mobility@best_scale + corrhist-if-positive + imbalance/KS-if-positive), gate
the bundle vs base. Watch for non-additivity/overlap. If several survive → consider a joint fixed-TIME SPSA
(NOTE: the `spsa` sub routes eval knobs to fixed-DEPTH = over-credits → needs a fixed-time variant first).

## STATE
byte-id 247 default; nothing committed; all levers env-gated default-off. Mediocre harness files: raw_uci.py
(new), vs_sf.py (+--opponent-raw flag, gated), _raw_uci_smoke.py / _raw_uci_probe.py (temp diagnostics — delete).
