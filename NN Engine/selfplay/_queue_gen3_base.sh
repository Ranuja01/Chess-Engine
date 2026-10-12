#!/bin/bash
# 2026-10-12 night, queue #59 — the base for the next retune pieces after the gen-3 ship (259 / 47,869,841): our d10 DEPTH PASS
# of gen 3 on slice A (A_ours1012_d10) + the gen-3 feature/static export (feat_gen3). Fixed-depth / static (safe any hour),
# 4 shards, CHUNK loop, resumable.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
D=/mnt/e/chess_data/retune
cd "$ND" || exit 1
echo "[q59] $(date) start"
FP=$(env V2_PRESET=shipped bash "$R" wac fp_v2 2>&1 | grep -E "SOLVED|NODES" | tr '\n' ' ')
echo "[q59] v2 shipped: $FP"
echo "$FP" | grep -q "259/300.*47869841" || { echo "[q59] ☠️ FINGERPRINT MISMATCH (expected gen 3) — aborting"; exit 1; }
for s in 0 1 2 3; do
  ( until env V2_PRESET=shipped PRESET=LONG_FORMAT MAX_DEPTH=10 USE_OPENING_BOOK=0 bash "$R" pyrun \
        diagnostics/_depth_residual_pass.py IN=$D/A_sf18.csv OUT=$D/A_ours1012_d10.csv SHARD=$s/4 CHUNK=800 \
        > $D/q59_d10_$s.log 2>&1; [ $? -ne 3 ]; do :; done ) &
done
wait
echo "[q59] $(date) A depth rows: $(cat $D/A_ours1012_d10_s*of4.csv 2>/dev/null | grep -vc '^fen,')"
env V2_PRESET=shipped bash "$R" pyrun diagnostics/_px_export.py IN=$D/corpus_v1.csv OUT=$D/feat_gen3.npz 2>&1 | grep -E "CORPUS"
echo "[q59] $(date) QUEUE 59 DONE"
