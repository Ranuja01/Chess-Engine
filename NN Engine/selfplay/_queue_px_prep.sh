#!/bin/bash
# 2026-10-03 night, queue #15: PASSER SYSTEM (PX) prep — after queue #14 (the pawn gates run from the working tree, so NO
# build before). Build → guards → eg depth pass → labelled export → per-block fit → closure + symmetry. STOPS before any
# gate: which arms to gate is decided with the owner once queue #14's passer/structure results are known.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
Q14LOG="/mnt/c/Users/Kumodth/AppData/Local/Temp/claude/c--Users-Kumodth-OneDrive-Desktop-Programming-Chess-Engine-Chess-Engine/97a258a9-0c6e-4a2c-8ae5-fa3a9a6ce239/scratchpad/q14.log"
TX=/mnt/e/chess_data/texel
cd "$ND" || exit 1
echo "[q15] $(date) waiting for queue #14"
until grep -qE "QUEUE 14 DONE|gates skipped" "$Q14LOG" 2>/dev/null; do sleep 300; done
# wait for any straggling engine job of queue #14 (its own script may still be exiting)
while pgrep -f "vs_sf.py|tournament.py" >/dev/null; do sleep 60; done
bash "$R" build > /tmp/q15_build.txt 2>&1
FP1=$(bash "$R" wac fp_v1 2>&1 | grep -E "SOLVED|NODES" | tr '\n' ' ')
FP2=$(bash "$R" wac fp_v2 V2_PRESET=shipped 2>&1 | grep -E "SOLVED|NODES" | tr '\n' ' ')
echo "[q15] v1: $FP1"; echo "[q15] v2 shipped: $FP2"
if ! echo "$FP1" | grep -q "250/300.*35310778" || ! echo "$FP2" | grep -q "255/300.*47218480"; then
  echo "[q15] ☠️ FINGERPRINT MISMATCH — aborting"; exit 1
fi
for s in 0 1 2 3; do
  ( until env V2_PRESET=shipped PRESET=LONG_FORMAT MAX_DEPTH=10 USE_OPENING_BOOK=0 bash "$R" pyrun \
        diagnostics/_depth_residual_pass.py IN=ks_sets/fitC_eg_sf18.csv OUT=ks_sets/fitC_eg_ours1003_d10.csv \
        SHARD=$s/4 CHUNK=800 > /tmp/q15_depth_$s.txt 2>&1; [ $? -ne 3 ]; do :; done ) &
done
wait
echo "[q15] $(date) eg depth rows: $(cat ks_sets/fitC_eg_ours1003_d10_s*of4.csv | wc -l)"
bash "$R" pyrun diagnostics/_px_export.py V2_PRESET=shipped 2>&1 | grep "PX EXPORT" | sed 's/^/[q15] /'
bash "$R" pyrun diagnostics/_px_depth_fit.py 2>&1 | sed 's/^/[q15] /'
# closure: the engine's v2_pxpass must equal Σ count × fitted θ (feature pass, block v2_pxpass) + symmetry
P="PX_V2=1 PX_V2_FILE=$TX/px_depth_px.txt"
env V2_PRESET=shipped $P bash "$R" pyrun diagnostics/_texel_feature_pass.py IN=$TX/fitC_stage1.csv.gz LIMIT=20000 \
  OUT=/tmp/q15_px.npz 2>&1 | grep -E "v2_pxpass|v2_passers" | sed 's/^/[q15] closure /'
SYM=$(env V2_PRESET=shipped $P bash "$R" pyrun diagnostics/_eval_symmetry.py N=4000 2>&1 | grep -E "violations")
echo "[q15] symmetry: $(echo "$SYM" | tr '\n' ' ')"
echo "[q15] $(date) QUEUE 15 DONE (prep only — gates await the owner)"
