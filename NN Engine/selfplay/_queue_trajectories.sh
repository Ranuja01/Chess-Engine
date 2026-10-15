#!/bin/bash
# 2026-10-15, queue #67 — POT TRAJECTORY SET (owner's idea): _build_trajectories.py, ~40k starts (70% unresolved systems +
# 30% random control), SF18 plays both sides @50k nodes/move, checkpoints +8/16/24 with our static + SF18 d14. Waits for #66.
# 4 shards, resumable, fixed-node (safe any hour). ≤ 4 engines. Output E:\chess_data\traj\traj_s*of4.csv.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
until grep -qE "QUEUE 66 DONE|☠️" /mnt/e/chess_data/q66_eg_enrich.log 2>/dev/null; do sleep 120; done
cd "$ND" || exit 1
echo "[q67] $(date) start"
FP=$(env V2_PRESET=shipped bash "$R" wac fp_v2 2>&1 | grep -E "SOLVED|NODES" | tr '\n' ' ')
echo "$FP" | grep -q "259/300.*47869841" || { echo "[q67] ☠️ FINGERPRINT MISMATCH (expected gen 3) — aborting"; exit 1; }
for s in 0 1 2 3; do
  env V2_PRESET=shipped bash "$R" pyrun diagnostics/_build_trajectories.py N=40000 SHARD=$s/4 > /mnt/e/chess_data/traj/q67_s$s.log 2>&1 &
done
wait
echo "[q67] $(date) trajectories: $(cat /mnt/e/chess_data/traj/traj_s*of4.csv 2>/dev/null | grep -vc '^start_fen')"
echo "[q67] $(date) QUEUE 67 DONE"
