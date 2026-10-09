#!/bin/bash
# 2026-10-09, queue #46 — verify the path-ladder fix (passer_path_kw shared by the constant and C1 paths). Static dumps of the
# depth rows: C1 passer table alone vs C1 + PASSER_V2_PATH_PCT=100 (must now DIFFER — before the fix they were identical),
# plus the constant path with PATH 100 vs the ship (must differ, as before). Symmetry with C1 + PATH on. ≤ 4 engines. No build.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
O=/mnt/e/chess_data/texel/revival
TX=/mnt/e/chess_data/texel
C1="C1_V2_FIT=1 C1_V2_FILE=$TX/px_depth_c1.txt"
cd "$ND" || exit 1
echo "[q46] $(date) start"
d() { env V2_PRESET=shipped ${@:2} bash "$R" pyrun diagnostics/_revival_screen.py MODE=dump OUT=$O/$1.csv 2>&1 | grep -E "REVIVAL DUMP|rror"; }
d pc_c1 $C1 & d pc_c1path $C1 PASSER_V2_PATH_PCT=100 & d pc_path PASSER_V2_PATH_PCT=100 & wait
SYM=$(env V2_PRESET=shipped $C1 PASSER_V2_PATH_PCT=100 bash "$R" pyrun diagnostics/_eval_symmetry.py N=4000 2>&1 | grep -E "violations")
echo "[q46] C1+path symmetry: $(echo "$SYM" | tr '\n' ' ')"
echo "[q46] $(date) QUEUE 46 DONE"
