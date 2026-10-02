#!/bin/bash
# 2026-10-02, queue #7: RECALIBRATE the external gauntlet. The SF18 @400-node anchor was set to score ~45-55%; v2 now
# scores ~75% there, and THREE material terms read negative vs SF@400 but positive/neutral in self-play (Kaufman full
# −30/+12, Kaufman queen −20/+3, MCL all −24/+9) ⇒ the weak opponent may bias material. Sweep SF18's node budget with
# the SHIPPED config (ours at 250k) to find the ~50% anchor. 200 games each (feasibility, not a verdict).
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
cd "$ND" || exit 1
for n in 800 1600 3200; do
  bash "$R" gauntlet 200 $n 4 calib_sf${n}_s57 57 V2_PRESET=shipped
done
echo "[q7] $(date) QUEUE 7 DONE"
