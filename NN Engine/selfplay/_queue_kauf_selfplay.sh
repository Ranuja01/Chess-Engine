#!/bin/bash
# 2026-10-01, queue #4: the OPPONENT-STRENGTH hypothesis (C3 doc §18f). Kaufman Texel cells lost vs SF18 @400 nodes
# (full −29.6, queen −20.0) although the queen misjudgement persists at depth. Self-play (equal strength) was never run:
# positive here + negative there ⇒ the weak-opponent gauntlet biases material; negative both ⇒ the fits are wrong.
# Waits for queue #3 (WSF re-gate). Per-game engine processes (bounded memory).
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
Q3LOG="/mnt/c/Users/Kumodth/AppData/Local/Temp/claude/c--Users-Kumodth-OneDrive-Desktop-Programming-Chess-Engine-Chess-Engine/97a258a9-0c6e-4a2c-8ae5-fa3a9a6ce239/scratchpad/q3.log"
TX=/mnt/e/chess_data/texel
cd "$ND" || exit 1
echo "[q4] $(date) waiting for queue #3"
until grep -q "QUEUE 3 DONE" "$Q3LOG" 2>/dev/null; do sleep 300; done
BASE="V2_PRESET=shipped MAX_DEPTH=64 NODE_LIMIT=50000"
for arm in queen full; do
  bash "$R" pyrun selfplay/tournament.py --p1-label kauf$arm \
    --p1-config "$BASE KAUF_V2_MAG=1000 KAUF_V2_FORM=3 KAUF_V2_FILE=$TX/kauf_$arm.txt" \
    --p2-label shipped --p2-config "$BASE" --games 2000 --concurrency 4 --preset LONG_FORMAT --max-plies 400 \
    --openings selfplay/openings_uho.txt --seed 52 --tag kauf${arm}_selfplay --quiet
done
echo "[q4] $(date) QUEUE 4 DONE"
