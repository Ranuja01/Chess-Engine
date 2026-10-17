#!/bin/bash
# 2026-10-17, queue #77 — gate the CELLS-ONLY candidate C64 (POT_V2_WIN_PAWNEND=64 POT_V2_WIN_ONEMINOR=16; no race) vs gen 3
# under the NARROW-ITEM rule (memory narrow-items-are-judged-where-they-fire). Why C64: at d12 it beats set A on held-out pawn
# rows (61.1 vs 63.9, bias −0.71) — the race's edge vanishes with depth while its cost remains — and it costs no cycles.
#  1. balanced endgame starts (eg_balanced_fens.txt), book OFF, node-limited @50k, 1,200 g, seed 270. PRED: +5…+25.
#  2. same starts, TIMED (LIGHTNING), 600 g, seed 271 (daytime). PRED: positive, ≈ step 1 (no speed cost).
#  3. standard UHO @50k, 2,000 g, seed 272 — "not harmful" (> −1σ). PRED: −5…+5.
#  4. SF18 @1100, 2 seeds × 500 (+ gen-3 baseline per seed) — "not harmful". PRED: within ±10 of baseline.
# ≤ 3 processes. Steps 1, 3, 4 node-limited (safe any hour); step 2 timed, scheduled to end before 9pm.
C="POT_V2_WIN_PAWNEND=64 POT_V2_WIN_ONEMINOR=16"
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
cd "$ND" || exit 1
echo "[q77] $(date) start"
FP=$(env V2_PRESET=shipped bash "$R" wac fp_v2 2>&1 | grep -E "SOLVED|NODES" | tr '\n' ' ')
echo "$FP" | grep -q "259/300.*47869841" || { echo "[q77] ☠️ FINGERPRINT MISMATCH (expected gen 3) — aborting"; exit 1; }
SYM=$(env V2_PRESET=shipped $C bash "$R" pyrun diagnostics/_eval_symmetry.py IN=/mnt/e/chess_data/retune/EGX_fens.csv N=4000 2>&1 | grep -E "violations")
echo "[q77] symmetry (C64, EGX endgames): $(echo "$SYM" | tr '\n' ' ')"
[ "$(echo "$SYM" | grep -c 'violations 0 ')" = "2" ] || { echo "[q77] ☠️ symmetry FAILED — aborting"; exit 1; }
match() {   # $1 tag, $2 openings, $3 games, $4 seed, $5 preset, $6 extra config (node limit / book)
  bash "$R" pyrun selfplay/tournament.py --p1-label $1 --p1-config "V2_PRESET=shipped $C MAX_DEPTH=64 $6" \
    --p2-label shipped --p2-config "V2_PRESET=shipped MAX_DEPTH=64 $6" --games $3 --concurrency 3 \
    --preset $5 --max-plies 400 --openings $2 --seed $4 --tag $1 --quiet
  echo "[q77] $(date) $1: $(grep -oE '"elo(_margin)?": [-0-9.]+' selfplay/games/$1/tournament.json | tr '\n' ' ')"
}
match c64_egbal270 selfplay/eg_balanced_fens.txt 1200 270 LONG_FORMAT "USE_OPENING_BOOK=0 NODE_LIMIT=50000"
match c64_egbal_timed271 selfplay/eg_balanced_fens.txt 600 271 LIGHTNING "USE_OPENING_BOOK=0"
match c64_sp272 selfplay/openings_uho.txt 2000 272 LONG_FORMAT "NODE_LIMIT=50000"
for s in 273 274; do
  bash "$R" gauntlet 500 1100 3 g1100_g3ship_s$s $s V2_PRESET=shipped
  bash "$R" gauntlet 500 1100 3 g1100_c64_s$s $s V2_PRESET=shipped $C
  echo "[q77] $(date) SF18 @1100 seed $s done"
done
echo "[q77] $(date) QUEUE 77 DONE"
