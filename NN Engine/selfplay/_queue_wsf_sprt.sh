#!/bin/bash
# 2026-10-01: POT winnability reference form (WSF_V2, 2-feature fit) passed the fresh-seed SF18 gate (+20.2 [+3.7, +37.4]).
# Protocol next: SPRT @50k vs shipped (fresh seed 45), then a 2,000-game fixed replication (seed 46). Sequential.
R="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine/selfplay/overnight_runner.sh"
CAND="V2_PRESET=shipped WSF_V2=1 WSF_V2_BASE=-37 WSF_V2_SP=34 WSF_V2_OCB=-80 MAX_DEPTH=64 NODE_LIMIT=50000"
BASE="V2_PRESET=shipped MAX_DEPTH=64 NODE_LIMIT=50000"
bash "$R" pyrun selfplay/sprt.py --p1-label wsf --p1-config "$CAND" --p2-label shipped --p2-config "$BASE" \
  --openings selfplay/openings_uho.txt --seed 45 --preset LONG_FORMAT --concurrency 4 --elo0 0 --elo1 10 \
  --max-games 6000 --max-minutes 600 --tag sprt_wsf --quiet
bash "$R" pyrun selfplay/tournament.py --p1-label wsf --p1-config "$CAND" --p2-label shipped --p2-config "$BASE" \
  --games 2000 --concurrency 4 --preset LONG_FORMAT --max-plies 400 --openings selfplay/openings_uho.txt --seed 46 \
  --tag wsf_rep --quiet
echo "WSF SPRT QUEUE DONE"
