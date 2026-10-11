#!/bin/bash
# 2026-10-11, queue #53 — our d10 DEPTH PASS of the NEW ship (gen 2, 256 / 47,473,313) on slice A (140k SF18-labelled rows):
# the depth-target base for every fit after the 10-11 ship (each fit must start from the CURRENT ship's d10 search).
# Fixed-depth (safe in the owner's evening), 4 shards, CHUNK loop (bounded memory), resumable: re-running skips done rows.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
D=/mnt/e/chess_data/retune
cd "$ND" || exit 1
echo "[q53] $(date) start"
FP=$(env V2_PRESET=shipped bash "$R" wac fp_v2 2>&1 | grep -E "SOLVED|NODES" | tr '\n' ' ')
echo "[q53] v2 shipped: $FP"
echo "$FP" | grep -q "256/300.*47473313" || { echo "[q53] ☠️ FINGERPRINT MISMATCH (expected gen 2) — aborting"; exit 1; }
for s in 0 1 2 3; do
  ( until env V2_PRESET=shipped PRESET=LONG_FORMAT MAX_DEPTH=10 USE_OPENING_BOOK=0 bash "$R" pyrun \
        diagnostics/_depth_residual_pass.py IN=$D/A_sf18.csv OUT=$D/A_ours1011_d10.csv SHARD=$s/4 CHUNK=800 \
        > $D/q53_d10_$s.log 2>&1; [ $? -ne 3 ]; do :; done ) &
done
wait
echo "[q53] $(date) A depth rows: $(cat $D/A_ours1011_d10_s*of4.csv 2>/dev/null | grep -vc '^fen,')"
echo "[q53] $(date) QUEUE 53 DONE"
