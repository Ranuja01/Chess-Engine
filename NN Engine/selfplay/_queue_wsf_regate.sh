#!/bin/bash
# 2026-09-30 night, queue #3: the WSF instruments DISAGREE — SF18 gauntlet +20.2 [+3.7, +37.4] (seeds 42/43) vs self-play
# SPRT ≈ −2 Elo at 3,480 games (heading to H0). Tie-breaker = a SECOND external gate on new seeds 50 + 51 (fresh
# baselines). Waits for queue #2 to finish (one engine job at a time).
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
Q2LOG="/mnt/c/Users/Kumodth/AppData/Local/Temp/claude/c--Users-Kumodth-OneDrive-Desktop-Programming-Chess-Engine-Chess-Engine/97a258a9-0c6e-4a2c-8ae5-fa3a9a6ce239/scratchpad/overnight_q2.log"
cd "$ND" || exit 1
echo "[q3] $(date) waiting for queue #2"
until grep -qE "OVERNIGHT QUEUE 2 DONE|aborting the whole queue" "$Q2LOG" 2>/dev/null; do sleep 300; done
echo "[q3] $(date) start"
for s in 50 51; do
  bash "$R" gauntlet 500 400 4 gauntlet_ship6_s$s $s V2_PRESET=shipped
  bash "$R" gauntlet 500 400 4 gauntlet_wsf2_s$s $s V2_PRESET=shipped WSF_V2=1 WSF_V2_BASE=-37 WSF_V2_SP=34 WSF_V2_OCB=-80
done
echo "[q3] $(date) QUEUE 3 DONE"
