#!/bin/bash
# 2026-10-04 night, queue #19 (owner's order: finish the eval first). PREP for the mobility / placement depth fits:
# re-run the depth pass on the 10-04 ship (connected pawns) so every later depth-target fit nests on the CURRENT engine,
# then re-anchor the calibrated judge (SF18 @800, fresh seeds 76/77) after the ship. No build (built + fingerprinted
# 10-04 evening); the fingerprint guard still runs. ≤ 4 engines; CHUNK loops bound memory (run_one leaks ~1.2 MB/FEN).
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
cd "$ND" || exit 1
echo "[q19] $(date) start"
FP1=$(bash "$R" wac fp_v1 2>&1 | grep -E "SOLVED|NODES" | tr '\n' ' ')
FP2=$(env V2_PRESET=shipped bash "$R" wac fp_v2 2>&1 | grep -E "SOLVED|NODES" | tr '\n' ' ')
echo "[q19] v1: $FP1"; echo "[q19] v2 shipped: $FP2"
if ! echo "$FP1" | grep -q "250/300.*35310778" || ! echo "$FP2" | grep -q "254/300.*50622239"; then
  echo "[q19] ☠️ FINGERPRINT MISMATCH — aborting"; exit 1
fi
for set in mg eg; do
  for s in 0 1 2 3; do
    ( until env V2_PRESET=shipped PRESET=LONG_FORMAT MAX_DEPTH=10 USE_OPENING_BOOK=0 bash "$R" pyrun \
          diagnostics/_depth_residual_pass.py IN=ks_sets/fitC_${set}_sf18.csv OUT=ks_sets/fitC_${set}_ours1004_d10.csv \
          SHARD=$s/4 CHUNK=800 > /tmp/q19_depth_${set}_$s.txt 2>&1; [ $? -ne 3 ]; do :; done ) &
  done
  wait
  N=$(cat diagnostics/ks_sets/fitC_${set}_ours1004_d10_s*of4.csv 2>/dev/null | grep -vc "^fen,")
  echo "[q19] $(date) $set depth rows: $N"
  [ "$N" -gt 1000 ] || { echo "[q19] ☠️ $set depth pass wrote too few rows — aborting"; exit 1; }
done
for s in 76 77; do
  bash "$R" gauntlet 500 800 4 g800_ship1004_s$s $s V2_PRESET=shipped
  echo "[q19] $(date) anchor seed $s done"
done
echo "[q19] $(date) QUEUE 19 DONE"
