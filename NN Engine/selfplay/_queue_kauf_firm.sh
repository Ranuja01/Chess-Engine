#!/bin/bash
# 2026-10-02 night, queue #11: firm up Kaufman full (FORM 3) on the CALIBRATED judge — +18.9 [−6.5, +44.9] on seeds 58/59,
# self-play +11.8 ± 17.9 (C3 doc §18m). Seeds 60/61 reuse queue #9's fresh baselines (g800_ship_s60/61) ⇒ 1,000 more
# paired games at no baseline cost. Waits for queue #10.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
Q10LOG="/mnt/c/Users/Kumodth/AppData/Local/Temp/claude/c--Users-Kumodth-OneDrive-Desktop-Programming-Chess-Engine-Chess-Engine/97a258a9-0c6e-4a2c-8ae5-fa3a9a6ce239/scratchpad/q10.log"
cd "$ND" || exit 1
echo "[q11] $(date) waiting for queue #10"
until grep -qE "QUEUE 10 DONE|aborting" "$Q10LOG" 2>/dev/null; do sleep 300; done
for s in 60 61; do
  [ -f "$ND/selfplay/games/g800_ship_s$s/results.csv" ] || bash "$R" gauntlet 500 800 4 g800_ship_s$s $s V2_PRESET=shipped
  bash "$R" gauntlet 500 800 4 g800_kauffull_s$s $s V2_PRESET=shipped KAUF_V2_MAG=1000 KAUF_V2_FORM=3 \
    KAUF_V2_FILE=/mnt/e/chess_data/texel/kauf_full.txt
done
echo "[q11] $(date) QUEUE 11 DONE"
