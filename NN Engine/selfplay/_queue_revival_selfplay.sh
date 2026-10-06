#!/bin/bash
# 2026-10-05, queue #24: SELF-PLAY (the second instrument) for the revival arms that read positive on SF18 @800 (queue #22):
# KAUF depth re-fit +2.25pp (s78 +4.8 / s79 −0.3) · PST depth re-fit +1.35pp (+2.5 / +0.2). 2,000 games @50k each, fresh
# seeds 80/81, UHO openings. Waits for queue #23 (≤ 4 engines). The q23 arms get their self-play after the owner reads them.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
O=/mnt/e/chess_data/texel/revival
Q23LOG="$ND/selfplay/games/q23_revival_gates2.log"
cd "$ND" || exit 1
echo "[q24] $(date) waiting for queue #23"
until grep -qE "QUEUE 23 DONE" "$Q23LOG" 2>/dev/null; do sleep 300; done
while pgrep -f "vs_sf.py|tournament.py" >/dev/null; do sleep 60; done
sp() {  # $1 tag  $2 seed  $3.. knobs
  bash "$R" pyrun selfplay/tournament.py --p1-label $1 --p1-config "V2_PRESET=shipped ${*:3} MAX_DEPTH=64 NODE_LIMIT=50000" \
    --p2-label shipped --p2-config "V2_PRESET=shipped MAX_DEPTH=64 NODE_LIMIT=50000" --games 2000 --concurrency 4 \
    --preset LONG_FORMAT --max-plies 400 --openings selfplay/openings_uho.txt --seed $2 --tag ${1}_selfplay --quiet
  echo "[q24] $(date) $1 self-play done: $(grep -o '"elo": [-0-9.]*' selfplay/games/${1}_selfplay/tournament.json)"
}
sp revkauf 80 KAUF_V2_FILE=$O/kauf_depth.txt
sp revpst 81 PST_V2_FILE=$O/pst_depth.txt
echo "[q24] $(date) QUEUE 24 DONE"
