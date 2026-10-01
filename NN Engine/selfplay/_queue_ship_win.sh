#!/bin/bash
# 2026-10-01, queue #5: build the POT winnability SHIP (V2_PRESET=shipped now carries POT_V2_WIN) after queue #4, then
#   1 fingerprints: v1 unchanged (ABORT otherwise) · shipped with POT_V2_WIN=0 must reproduce the OLD v2 fingerprint
#     252 / 49,094,807 (proves the ship changed nothing else; ABORT otherwise) · record the NEW shipped fingerprint
#   2 re-score the stage-1 fit data with the new shipped eval (future fits nest on top of winnability)
#   3 self-play at 250k nodes, winnability ON vs OFF, 1,000 games (seed 53): DEPTH vs OPPONENT-STRENGTH (C3 doc §16c)
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
Q4LOG="/mnt/c/Users/Kumodth/AppData/Local/Temp/claude/c--Users-Kumodth-OneDrive-Desktop-Programming-Chess-Engine-Chess-Engine/97a258a9-0c6e-4a2c-8ae5-fa3a9a6ce239/scratchpad/q4.log"
cd "$ND" || exit 1
echo "[q5] $(date) waiting for queue #4"
until grep -q "QUEUE 4 DONE" "$Q4LOG" 2>/dev/null; do sleep 300; done
bash "$R" build > /tmp/q5_build.txt 2>&1
FP1=$(bash "$R" wac fp_v1 2>&1 | grep -E "SOLVED|NODES|EBF" | tr '\n' ' ')
FPOLD=$(bash "$R" wac fp_v2off V2_PRESET=shipped POT_V2_WIN=0 2>&1 | grep -E "SOLVED|NODES|EBF" | tr '\n' ' ')
FPNEW=$(bash "$R" wac fp_v2 V2_PRESET=shipped 2>&1 | grep -E "SOLVED|NODES|EBF" | tr '\n' ' ')
echo "[q5] v1: $FP1"; echo "[q5] shipped, POT_V2_WIN=0: $FPOLD"; echo "[q5] NEW shipped: $FPNEW"
if ! echo "$FP1" | grep -q "250/300.*35310778" || ! echo "$FPOLD" | grep -q "252/300.*49094807"; then
  echo "[q5] ☠️ FINGERPRINT MISMATCH — aborting"; exit 1
fi
env V2_PRESET=shipped bash "$R" pyrun diagnostics/_texel_win_pass.py OUT=/mnt/e/chess_data/texel/fitC_win_shipwin.npz \
  > /tmp/q5_rescore.txt 2>&1
echo "[q5] $(date) re-score: $(grep -E 'WIN PASS|wrote' /tmp/q5_rescore.txt | tr '\n' ' ')"
bash "$R" pyrun selfplay/tournament.py --p1-label win_on --p1-config "V2_PRESET=shipped MAX_DEPTH=64 NODE_LIMIT=250000" \
  --p2-label win_off --p2-config "V2_PRESET=shipped POT_V2_WIN=0 MAX_DEPTH=64 NODE_LIMIT=250000" --games 1000 \
  --concurrency 4 --preset LONG_FORMAT --max-plies 400 --openings selfplay/openings_uho.txt --seed 53 \
  --tag win_selfplay_250k --quiet
echo "[q5] $(date) QUEUE 5 DONE"
