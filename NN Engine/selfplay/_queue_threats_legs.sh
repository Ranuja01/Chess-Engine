#!/bin/bash
# 2026-10-05, queue #21: THREATS per-leg retry (owner: "threats in SF11's breakdown edged us out in several situations —
# perhaps it has a place now if tuned correctly"; the Kaufman lesson: fit the SHAPE, not one magnitude). The 10-05 screen
# tested threats as ONE multiplier on the core form (α 0.12, val −0.05%). Here: core (THREAT_V2_PCT=100, GATE 0) and each
# optional leg as an INCREMENT over core, one multiplier per leg, fitted jointly. Static dumps; ≤ 3 engines at a time.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
O=/mnt/e/chess_data/texel/revival   # persistent — WSL /tmp is wiped when the distro idles down (lost the q20 dumps 10-05)
mkdir -p "$O"
cd "$ND" || exit 1
d() { env V2_PRESET=shipped THREAT_V2_PCT=100 "${@:2}" bash "$R" pyrun diagnostics/_revival_screen.py MODE=dump OUT=$O/$1.csv 2>&1 | grep -E "REVIVAL DUMP|rror"; }
echo "[q21] $(date) start"
env V2_PRESET=shipped bash "$R" pyrun diagnostics/_revival_screen.py MODE=dump OUT=$O/off.csv 2>&1 | grep -E "REVIVAL DUMP|rror" &
d threats100 &
wait
[ -s $O/off.csv ] && [ -s $O/threats100.csv ] || { echo "[q21] ☠️ base dumps missing — aborting"; exit 1; }
d th_hanging THREAT_V2_HANGING=1 &
d th_restrict THREAT_V2_RESTRICT=1 &
d th_king THREAT_V2_KING=1 &
wait
d th_pawntgt THREAT_V2_PAWN_TARGETS=1 &
d th_push THREAT_V2_PUSH=1 &
wait
bash "$R" pyrun diagnostics/_revival_screen.py MODE=knobs JOINT=1 KNOBS=core:$O/threats100.csv:$O/off.csv,hanging:$O/th_hanging.csv:$O/threats100.csv,restrict:$O/th_restrict.csv:$O/threats100.csv,king:$O/th_king.csv:$O/threats100.csv,pawntgt:$O/th_pawntgt.csv:$O/threats100.csv,push:$O/th_push.csv:$O/threats100.csv
echo "[q21] $(date) QUEUE 21 DONE"
