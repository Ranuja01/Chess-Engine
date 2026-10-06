#!/bin/bash
# 2026-10-06 night, queue #25: SELF-PLAY for KPROT (KingProtector C3-c, built at 0, DEPTH fit): SF18 @800 s78 +3.7 / s79 +4.1 ⇒
# +3.9pp ≈ +27 Elo (~2σ on SF alone; collapses 50/48 vs 76/80). 2,000 @50k, fresh seed 82. Waits for queue #24.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
O=/mnt/e/chess_data/texel/revival
Q24LOG="$ND/selfplay/games/q24_revival_selfplay.log"
cd "$ND" || exit 1
echo "[q25] $(date) waiting for queue #24"
until grep -qE "QUEUE 24 DONE" "$Q24LOG" 2>/dev/null; do sleep 300; done
while pgrep -f "vs_sf.py|tournament.py" >/dev/null; do sleep 60; done
bash "$R" pyrun selfplay/tournament.py --p1-label revkprot --p1-config "V2_PRESET=shipped KPROT_V2=1 KPROT_V2_FILE=$O/kprot_depth.txt MAX_DEPTH=64 NODE_LIMIT=50000" \
  --p2-label shipped --p2-config "V2_PRESET=shipped MAX_DEPTH=64 NODE_LIMIT=50000" --games 2000 --concurrency 4 \
  --preset LONG_FORMAT --max-plies 400 --openings selfplay/openings_uho.txt --seed 82 --tag revkprot_selfplay --quiet
echo "[q25] $(date) revkprot self-play done: $(grep -o '"elo": [-0-9.]*' selfplay/games/revkprot_selfplay/tournament.json)"
echo "[q25] $(date) QUEUE 25 DONE"
