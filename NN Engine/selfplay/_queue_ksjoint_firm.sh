#!/bin/bash
# 2026-10-03, queue #12 (owner chose option A: firm up each candidate separately): joint KS + KS-B (depth fit, λ 1e-2) on the
# CALIBRATED judge — +11.2 [−12.9, +35.8] on seeds 60/61, self-play +17.7 ± 17.9 (C3 doc §18o). 1,000 more paired games on
# FRESH seeds 64/65 with fresh baselines. Waits for queue #11 (Kaufman firm-up).
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
Q11LOG="/mnt/c/Users/Kumodth/AppData/Local/Temp/claude/c--Users-Kumodth-OneDrive-Desktop-Programming-Chess-Engine-Chess-Engine/97a258a9-0c6e-4a2c-8ae5-fa3a9a6ce239/scratchpad/q11.log"
TX=/mnt/e/chess_data/texel
cd "$ND" || exit 1
echo "[q12] $(date) waiting for queue #11"
until grep -q "QUEUE 11 DONE" "$Q11LOG" 2>/dev/null; do sleep 300; done
J="$(cat $TX/ks_depth_L1e-2_ks.txt) KSB_V2=1 KSB_V2_FILE=$TX/ks_depth_L1e-2_ksb.txt"
for s in 64 65; do
  bash "$R" gauntlet 500 800 4 g800_ship_s$s $s V2_PRESET=shipped
  bash "$R" gauntlet 500 800 4 g800_ksjoint_s$s $s V2_PRESET=shipped $J
done
echo "[q12] $(date) QUEUE 12 DONE"
