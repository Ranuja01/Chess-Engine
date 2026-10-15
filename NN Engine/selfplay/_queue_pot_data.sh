#!/bin/bash
# 2026-10-15, queue #70 — POT DATA, the long runs (OVERNIGHT; owner: long runs overnight, interactive ones while awake). Needs
# #69's build + prep (pe/light_src_s*.csv). ≤ 3 processes (the box hosts a live server). Fixed-depth / node-limited (safe any hour).
#  1. SF18 d14 multi-PV (K=8) on 6,000 LIGHT endgames (pawns, no rooks / queens) → critical points for the race prototype
#     stage 2 (does the race margin rank the only move above the losing alternatives?).
#  2. ENDGAME trajectory slice: _build_trajectories.py MODE=eg, 12,000 starts (endgame markers + 30% control, colour-mirrored
#     balance), SF18 @50k nodes/move, checkpoints +8/16/24 → the POT fit (potential for / against the mover, realised or not).
# Both resumable: rerun this queue after an interruption and it continues.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
P=/mnt/e/chess_data/pe
cd "$ND" || exit 1
echo "[q70] $(date) start"
[ -f $P/light_src_s2.csv ] || { echo "[q70] ☠️ no light_src files — run queue #69 (prep) first"; exit 1; }
FP=$(env V2_PRESET=shipped bash "$R" wac fp_v2 2>&1 | grep -E "SOLVED|NODES" | tr '\n' ' ')
echo "$FP" | grep -q "259/300.*47869841" || { echo "[q70] ☠️ FINGERPRINT MISMATCH (expected gen 3) — aborting"; exit 1; }
for s in 0 1 2; do
  bash "$R" pyrun diagnostics/_label_positions.py SRC=$P/light_src_s$s.csv OUT=$P/light_mpv_s$s.csv N=6000 K=8 SF_DEPTH=14 \
    > $P/label_s$s.log 2>&1 &
done
wait
echo "[q70] $(date) multi-PV labels: $(cat $P/light_mpv_s*.csv | grep -vc '^fen') rows"
for s in 0 1 2; do
  env V2_PRESET=shipped bash "$R" pyrun diagnostics/_build_trajectories.py MODE=eg N=12000 SHARD=$s/3 > $P/traj_eg_s$s.log 2>&1 &
done
wait
echo "[q70] $(date) eg trajectories: $(tail -qn1 $P/traj_eg_s*.log | tr '\n' ' ')"
echo "[q70] $(date) QUEUE 70 DONE"
