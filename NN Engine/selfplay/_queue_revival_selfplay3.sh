#!/bin/bash
# 2026-10-06 night, queue #26: SELF-PLAY for KFL (king flank C3-b, built at 0, DEPTH fit): SF18 @800 s78 +2.9 / s79 +2.3 ⇒
# +2.6pp ≈ +18 Elo (~1.3σ). 2,000 @50k, fresh seed 83. Waits for queue #25.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
O=/mnt/e/chess_data/texel/revival
Q25LOG="$ND/selfplay/games/q25_kprot_selfplay.log"
cd "$ND" || exit 1
echo "[q26] $(date) waiting for queue #25"
until grep -qE "QUEUE 25 DONE" "$Q25LOG" 2>/dev/null; do sleep 300; done
while pgrep -f "vs_sf.py|tournament.py" >/dev/null; do sleep 60; done
bash "$R" pyrun selfplay/tournament.py --p1-label revkfl --p1-config "V2_PRESET=shipped KFL_V2=1 KFL_V2_FILE=$O/kfl_depth.txt MAX_DEPTH=64 NODE_LIMIT=50000" \
  --p2-label shipped --p2-config "V2_PRESET=shipped MAX_DEPTH=64 NODE_LIMIT=50000" --games 2000 --concurrency 4 \
  --preset LONG_FORMAT --max-plies 400 --openings selfplay/openings_uho.txt --seed 83 --tag revkfl_selfplay --quiet
echo "[q26] $(date) revkfl self-play done: $(grep -o '"elo": [-0-9.]*' selfplay/games/revkfl_selfplay/tournament.json)"
echo "[q26] $(date) QUEUE 26 DONE"
