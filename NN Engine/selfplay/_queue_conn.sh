#!/bin/bash
# 2026-10-04, queue #17: the CONNECTED-PAWN gate. v2's connected term ships at PS_V2_CONN_MAG=0 (closed 09-12 on §I
# corpus MSE); the first DEPTH-target fit (`diagnostics/_conn_depth_fit.py`, KNOBS arm) wants CONN_MAG=21 SUPPORT=99
# EG_RATIO=101 (val −0.59%; FREE −0.64%; the unfitted CONN_MAG=100 form +2.21% = the old harm, reproduced).
# Guards ALREADY PASSED before launch (no rebuild since; env knobs only): closure EXACT (1,922/3,000 rows live, worst
# 1 mp) · symmetry colour 0/4000, file 0/3170. Fresh seeds + fresh baseline (memory gate-new-candidates-on-fresh-seeds).
# SF18 @800 seeds 73/74 (500 each, baseline + conn) → self-play 2,000 @50k seed 75.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
C="PS_V2_CONN_MAG=21 PS_V2_SUPPORT=99 PS_V2_EG_RATIO=101"
cd "$ND" || exit 1
echo "[q17] $(date) start"
for s in 73 74; do
  bash "$R" gauntlet 500 800 4 g800_ship1003_s$s $s V2_PRESET=shipped
  bash "$R" gauntlet 500 800 4 g800_conn_s$s $s V2_PRESET=shipped $C
  echo "[q17] $(date) seed $s done"
done
bash "$R" pyrun selfplay/tournament.py --p1-label conn --p1-config "V2_PRESET=shipped $C MAX_DEPTH=64 NODE_LIMIT=50000" \
  --p2-label shipped --p2-config "V2_PRESET=shipped MAX_DEPTH=64 NODE_LIMIT=50000" --games 2000 --concurrency 4 \
  --preset LONG_FORMAT --max-plies 400 --openings selfplay/openings_uho.txt --seed 75 --tag conn_selfplay --quiet
echo "[q17] $(date) QUEUE 17 DONE"
