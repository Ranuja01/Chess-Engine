#!/bin/bash
# 2026-10-05, queue #23: the three column arms queue #22 dropped. Their guards failed on TOOLING, not on the tables:
# the feature-pass closure refuses any live knob outside its model (flag 4) and connected pawns (PS_V2_CONN_MAG=21,
# shipped 10-04) is one. Re-run with PS_V2_CONN_MAG=0 (the checked blocks do not read it): mobility max|res| 1.9 mp
# (19,775 live) · v2_kprot 1.0 (17,833) · v2_kflank 1.0 (15,695); symmetry 0/4000 + 0/3170 each (from queue #22).
# Same seeds 78/79, SHARED baseline = queue #22's g800_ship1004b_s78/79 (identical config + seed). Waits for #22.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
O=/mnt/e/chess_data/texel/revival
Q22LOG="$ND/selfplay/games/q22_revival_gates.log"
cd "$ND" || exit 1
echo "[q23] $(date) waiting for queue #22"
until grep -qE "QUEUE 22 DONE|aborting|nothing to gate" "$Q22LOG" 2>/dev/null; do sleep 300; done
while pgrep -f "vs_sf.py|tournament.py" >/dev/null; do sleep 60; done
declare -A ARM
ARM[mob]="C1_V2_FIT=1 C1_V2_FILE=$O/mob_depth_c1.txt"
ARM[kprot]="KPROT_V2=1 KPROT_V2_FILE=$O/kprot_depth.txt"
ARM[kfl]="KFL_V2=1 KFL_V2_FILE=$O/kfl_depth.txt"
for s in 78 79; do
  for a in mob kprot kfl; do
    bash "$R" gauntlet 500 800 4 g800_rev_${a}_s$s $s V2_PRESET=shipped ${ARM[$a]}
    echo "[q23] $(date) $a seed $s done"
  done
done
echo "[q23] $(date) QUEUE 23 DONE"
