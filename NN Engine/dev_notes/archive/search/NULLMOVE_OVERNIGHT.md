# Pending experiment — depth-adaptive null-move (`NULLMOVE_PROGRESSIVE`) overnight A/B

**Status (2026-06-04):** code IN (uncommitted, default-off = byte-identical), launcher saved at
`selfplay/nmp_overnight.sh`. Waiting to run the overnight A/B, then analyze. Flip the knob on by default
only if the data earns it; otherwise leave env-gated default-off (or clean up — see "If abandoned").

## What we're testing
Null-move reduction had a latent bug: `if(depth_limit>=10) -=1; else if(>=12) -=2; else if(>=14) -=3;` — the
`else if`s are unreachable, so it only ever reduced `-1`. Making it progressive (`-2`@d≥12, `-3`@d≥14) is a
**new, more-aggressive tuning**, not an obvious bugfix (the engine was validated WITH the `-1`-only behavior).

First A/B (machine-independent node counts) showed it's **depth-dependent**: d12 main-search **−18% nodes**
(good) but d10 **+4.7% nodes** (bad). Cause: `depth_limit` is inflated by check extensions, so at shallow
iterations the `>=12` test fires only on check-extended forcing lines where a shallower null-move
verification fails to cut → more full search.

**Fix = depth-adaptive gate** (in BOTH null-move blocks, `search_engine.cpp` minimizer ~1980 + maximizer
~2347): gate the extra reductions on the **genuine iteration depth** `base_depth = depth_limit -
g_check_extensions` (strip check-extension inflation), so they fire only when the search is *really* at
depth ≥12. Auto-adapts to time control: LIGHTNING/BLITZ (≈d9–11) → never fires → no harm; STANDARD
(≈d12–14) → fires → the −18%. Invariant: the only mid-search `depth_limit++` are the two guarded check-ext
sites (1250/1480); the ID loop sets the base (740) → `depth_limit = base + g_check_extensions`.

## The knob (with / without, both anchored to baseline)
`NULLMOVE_PROGRESSIVE` — `=0` reproduces the **pre-all-changes** null-move behavior (byte-identical baseline);
`=1` is the new depth-adaptive system. Both A/B arms also carry the permanent crash fix (`d839c4e`) +
`-fno-semantic-interposition` (`5acfb7f`) — held constant, so the A/B isolates only the null-move system.

## Run sequence
```bash
# 1. rebuild with the gate
python setupAI.py build_ext --inplace --force

# 2. anchor BOTH arms to the original baseline at d10 (criteria a) -- BOTH must print 254,973,405:
for v in 0 1; do
  NULLMOVE_PROGRESSIVE=$v PRESET=LONG_FORMAT MAX_DEPTH=10 python diagnostics/tactical_test.py wac.epd nmp_gate_$v
  awk -F, -v v=$v 'NR>1{n+=$8} END{printf "NULLMOVE_PROGRESSIVE=%s nodes=%d (want 254,973,405)\n", v, n}' diagnostics/results/tactical_results_nmp_gate_$v.csv
done
#    off==baseline (by construction), on==baseline-at-shallow (by the gate). If either differs, fix first.

# 3. launch the overnight A/B (off vs on; BLITZ 50 / LIGHTNING 30 / STANDARD 6; tunable):
bash selfplay/nmp_overnight.sh
#    e.g. more games:  BLITZ_GAMES=80 LIGHTNING_GAMES=40 STANDARD_GAMES=8 bash selfplay/nmp_overnight.sh
```
Per-game cost (old timings ×1.2): LIGHTNING ~2.1min, BLITZ ~6.8min, STANDARD ~27min. Defaults ≈ 9h.

## Morning analysis — TELEMETRY FIRST
Elo is a blunt tool here (a ~½-ply gain ≈ +20 Elo needs ~700 games to resolve). So read the **per-move
telemetry** the JSONL records (`depth`, `nodes`, `nps`, `engine_time`, `eval`), averaged per arm:
- Does `=1` reach **deeper average depth** / better eval at BLITZ & STANDARD in the same clock? (the win)
- Is LIGHTNING **neutral** (knob inert → arms ≈ identical)? (the no-regression check)
- Optional `annotate.py` SF pass for eval-quality.
Then W/L/D + Elo as the coarse/directional secondary read (results in `selfplay/games/nmp_*/`).

## Decision
Keep `NULLMOVE_PROGRESSIVE` on by default only if telemetry shows **deeper/equal-eval at BLITZ/STANDARD AND
LIGHTNING neutral**; else leave env-gated default-off.

## If abandoned
Don't restore the broken dead branches — clean up to the byte-identical `if (depth_limit >= 10)
reduced_depth -= 1;` (delete the dead `else if`s), strictly better than the original.
