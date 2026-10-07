#!/bin/bash
# 2026-10-07 evening, queue #33 — CONTROL for queue #32. The stored ship d10 (`ours1004`, 10-04) reproduced on only 146/300
# rows today (re-run +4.6% MSE on those rows), so the arm must be compared with a ship pass run under IDENTICAL conditions:
# same rows, same SHARD=4 / CHUNK=800 layout, same session. No build. ≤ 4 engines.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
cd "$ND" || exit 1
echo "[q33] $(date) start"
for s in 0 1 2 3; do
  ( until env V2_PRESET=shipped PRESET=LONG_FORMAT MAX_DEPTH=10 USE_OPENING_BOOK=0 bash "$R" pyrun \
        diagnostics/_depth_residual_pass.py IN=ks_sets/dual_val_sf18.csv OUT=ks_sets/dual_val_shipctl_d10.csv \
        SHARD=$s/4 CHUNK=800 > /tmp/q33_ship_$s.txt 2>&1; [ $? -ne 3 ]; do :; done ) &
done
wait
N=$(cat diagnostics/ks_sets/dual_val_shipctl_d10_s*of4.csv 2>/dev/null | grep -vc "^fen,")
echo "[q33] $(date) ship control rows: $N"
echo "[q33] $(date) QUEUE 33 DONE"
