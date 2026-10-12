#!/bin/bash
# 2026-10-12, queue #57 — retune round 2, the SF18 leg of the owner's ship rule, both arms that passed self-play (#56):
#   g2l003 (no values, λ 0.03) +31.7 ± 17.9 · g2v003 (V2_VAL_* + PST pin, λ 0.03) +22.8 ± 17.9 (2,000 g each vs ship gen 2).
# SF18 @1000 nodes vs ours @250k, fresh seeds 216/217, 500 g per arm per seed, ship gen 2 baseline per seed (pooled reads).
# PREDICTIONS (registered): g2l003 +5…+20, g2v003 +0…+20 vs ship. Node-limited (safe any hour). ≤ 4 engines. No build.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
D=/mnt/e/chess_data/retune
cd "$ND" || exit 1
echo "[q57] $(date) start"
FP=$(env V2_PRESET=shipped bash "$R" wac fp_v2 2>&1 | grep -E "SOLVED|NODES" | tr '\n' ' ')
echo "[q57] v2 shipped: $FP"
echo "$FP" | grep -q "256/300.*47473313" || { echo "[q57] ☠️ FINGERPRINT MISMATCH — aborting"; exit 1; }
L=$(cat $D/arm_g2l003/env.txt); V=$(cat $D/arm_g2v003/env.txt)
for s in 216 217; do
  bash "$R" gauntlet 500 1000 4 g1000_g2ship_s$s $s V2_PRESET=shipped
  bash "$R" gauntlet 500 1000 4 g1000_g2l003_s$s $s V2_PRESET=shipped $L
  bash "$R" gauntlet 500 1000 4 g1000_g2v003_s$s $s V2_PRESET=shipped $V
  echo "[q57] $(date) SF18 seed $s done"
done
echo "[q57] $(date) QUEUE 57 DONE"
