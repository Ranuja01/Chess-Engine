#!/bin/bash
# 2026-10-18, queue #78 — first run of the CRITICAL-POINT venue (selfplay/eg_critical_fens.txt: 464 starts 2-4 moves before an
# SF18 only-move position, still close by SF18 d14). C64 (winnability cells, narrow-item pool) vs gen 3, book OFF, @50k, both
# colours (928 g), seed 280. Questions: does the venue DISCRIMINATE (pairs not decided by colour), and is the per-game effect
# larger than on generic balanced endgames (C64 there: +11.9 ± 23.1)?
# PREDICTION (registered): C64 +15…+40; discriminating pairs ≥ 85% of 464. ≤ 3 processes, node-limited (safe any hour).
C="POT_V2_WIN_PAWNEND=64 POT_V2_WIN_ONEMINOR=16"
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
cd "$ND" || exit 1
echo "[q78] $(date) start"
FP=$(env V2_PRESET=shipped bash "$R" wac fp_v2 2>&1 | grep -E "SOLVED|NODES" | tr '\n' ' ')
echo "$FP" | grep -q "259/300.*47869841" || { echo "[q78] ☠️ FINGERPRINT MISMATCH (expected gen 3) — aborting"; exit 1; }
tag=c64_crit280
bash "$R" pyrun selfplay/tournament.py --p1-label $tag --p1-config "V2_PRESET=shipped $C USE_OPENING_BOOK=0 MAX_DEPTH=64 NODE_LIMIT=50000" \
  --p2-label shipped --p2-config "V2_PRESET=shipped USE_OPENING_BOOK=0 MAX_DEPTH=64 NODE_LIMIT=50000" --games 928 --concurrency 3 \
  --preset LONG_FORMAT --max-plies 400 --openings selfplay/eg_critical_fens.txt --seed 280 --tag $tag --quiet
echo "[q78] $(date) $tag: $(grep -oE '"elo(_margin)?": [-0-9.]+' selfplay/games/$tag/tournament.json | tr '\n' ' ')"
echo "[q78] $(date) QUEUE 78 DONE"
