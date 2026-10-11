#!/bin/bash
# 2026-10-10 night, queue #51 — STRUCTURAL RETUNE, the external (defensive-half) instrument after the #50 self-play screen.
# #50: hybrid λ0.1 (arm_h01t) +56.6 ± 17.9 self-play @50k (2,000 g, seed 211); λ1 (arm_h1t) pending when this was written.
# Ship rule (owner 10-09): self-play ≥ +2σ AND SF18 better than −1σ AND Mediocre (≥ 200 g/arm) not negative.
# Here: SF18 @1000 nodes vs ours @250k (the calibrated judge, ship ≈ 48.6%), fresh seeds 212/213, 500 games per arm per seed,
# ship baseline per seed (pooled reads only). Node-limited ⇒ safe in the owner's evening. ≤ 4 engines. No build.
# PREDICTION (registered before the run): h01t vs ship pooled +10…+30 (self-play has overstated external gains ~3×: Fit A
# +111 → +38); h1t smaller.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
D=/mnt/e/chess_data/retune
until grep -q "QUEUE 50 DONE" /mnt/e/chess_data/q50_retune_screen.log 2>/dev/null; do sleep 60; done
cd "$ND" || exit 1
echo "[q51] $(date) start"
FP=$(env V2_PRESET=shipped bash "$R" wac fp_v2 2>&1 | grep -E "SOLVED|NODES" | tr '\n' ' ')
echo "[q51] v2 shipped: $FP"
echo "$FP" | grep -q "254/300.*50622239" || { echo "[q51] ☠️ FINGERPRINT MISMATCH — aborting"; exit 1; }
H01=$(cat $D/arm_h01t/env.txt); H1=$(cat $D/arm_h1t/env.txt)
for s in 212 213; do
  bash "$R" gauntlet 500 1000 4 g1000_ship_s$s $s V2_PRESET=shipped
  bash "$R" gauntlet 500 1000 4 g1000_rt_h01t_s$s $s V2_PRESET=shipped $H01
  bash "$R" gauntlet 500 1000 4 g1000_rt_h1t_s$s $s V2_PRESET=shipped $H1
  echo "[q51] $(date) SF18 seed $s done"
done
echo "[q51] $(date) QUEUE 51 DONE"
