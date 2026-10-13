#!/bin/bash
# 2026-10-13, queue #63 — WHY did R3A split (self-play +20.9 ± 17.9 vs gen 3, SF18 @1000 pooled −14 ± 14)? Owner: "seems sus,
# more work before considering". R3A's δ split into two arms on top of gen 3, each exported from the SAME fit (fit_hybrid_r3a):
#   r3a_val   = piece-value deltas only (gen-3 values + δ: R eg 127→148, Q eg −68→−104, Q mg −3→−24, minors eg …)
#   r3a_cells = every table block (C1 · KSB · KFL · KPROT · PX · CONN · Kaufman · PST), values and caps at gen 3
# SF18 @1000, fresh seeds 224/225, 500 g per arm per seed, gen-3 baseline per seed; then self-play @50k 2,000 g each vs gen 3
# (seed 226). PREDICTIONS (registered): the SF18 loss sits mostly in r3a_val (queen eg −104 overshoots); r3a_cells ≈ 0 on SF18.
# Diagnostic only — nothing ships. Node-limited (safe any hour). ≤ 4 engines.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
D=/mnt/e/chess_data/retune
cd "$ND" || exit 1
echo "[q63] $(date) start"
FP=$(env V2_PRESET=shipped bash "$R" wac fp_v2 2>&1 | grep -E "SOLVED|NODES" | tr '\n' ' ')
echo "$FP" | grep -q "259/300.*47869841" || { echo "[q63] ☠️ FINGERPRINT MISMATCH (expected gen 3) — aborting"; exit 1; }
env V2_PRESET=shipped bash "$R" pyrun diagnostics/_retune_export.py MODE=tables NPZ=fit_hybrid_r3a.npz ARM_I=0 ONLY=val OUT=$D/arm_r3a_val 2>&1 | grep -E "values:|caps:|EXPORT"
env V2_PRESET=shipped bash "$R" pyrun diagnostics/_retune_export.py MODE=tables NPZ=fit_hybrid_r3a.npz ARM_I=0 ONLY=c1,ksb,kfl,kprot,px,conn,kauf,pst OUT=$D/arm_r3a_cells 2>&1 | grep -E "values:|caps:|EXPORT"
for arm in r3a_val r3a_cells; do
  SYM=$(env V2_PRESET=shipped $(cat $D/arm_$arm/env.txt) bash "$R" pyrun diagnostics/_eval_symmetry.py N=4000 2>&1 | grep -E "violations")
  echo "[q63] $arm symmetry: $(echo "$SYM" | tr '\n' ' ')"
  [ "$(echo "$SYM" | grep -c 'violations 0 ')" = "2" ] || { echo "[q63] ☠️ $arm symmetry FAILED — aborting"; exit 1; }
done
for s in 224 225; do
  bash "$R" gauntlet 500 1000 4 g1000_g3ship_s$s $s V2_PRESET=shipped
  for arm in r3a_val r3a_cells; do
    bash "$R" gauntlet 500 1000 4 g1000_${arm}_s$s $s V2_PRESET=shipped $(cat $D/arm_$arm/env.txt)
  done
  echo "[q63] $(date) SF18 seed $s done"
done
for arm in r3a_val r3a_cells; do
  tag=split_${arm}_sp226
  bash "$R" pyrun selfplay/tournament.py --p1-label $tag --p1-config "V2_PRESET=shipped $(cat $D/arm_$arm/env.txt) MAX_DEPTH=64 NODE_LIMIT=50000" \
    --p2-label shipped --p2-config "V2_PRESET=shipped MAX_DEPTH=64 NODE_LIMIT=50000" --games 2000 --concurrency 4 \
    --preset LONG_FORMAT --max-plies 400 --openings selfplay/openings_uho.txt --seed 226 --tag $tag --quiet
  echo "[q63] $(date) $tag: $(grep -o '"elo": [-0-9.]*' selfplay/games/$tag/tournament.json)"
done
echo "[q63] $(date) QUEUE 63 DONE"
