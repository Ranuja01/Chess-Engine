#!/bin/bash
# 2026-10-05, queue #20: REVIVAL SCREEN part 2 dumps (static ev_breakdown totals on the 34k SF18-labelled rows, one process
# per knob setting — knobs latch at init). Reference magnitudes per the 10-05 knob audit (each term ~linear in its knob,
# so the fitted multiplier α carries the magnitude). KPROT/KFL are screened as cells (part 1); heat map is v1-only
# (unreachable at EVAL_ARM=1); tempo is a side-to-move constant (= the STM nuisance) — not dumped. ≤ 3 engines at a time.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
O=/tmp/revival
mkdir -p "$O"
cd "$ND" || exit 1
d() { env V2_PRESET=shipped "${@:2}" bash "$R" pyrun diagnostics/_revival_screen.py MODE=dump OUT=$O/$1.csv 2>&1 | grep -E "REVIVAL DUMP|rror"; }
echo "[q20] $(date) start"
d off &
d taper650 EVAL_V2_PAWN_MG=650 &
d threats100 THREAT_V2_PCT=100 &
wait
d space1320 SPACE_V2_MAG=1320 &
d longdiag100 LONGDIAG_V2_PCT=100 &
d reach100 REACH_V2_PCT=100 &
wait
d latent100 LATENT_V2_PCT=100 &
d rookfile ROOKFILE_V2_OPEN=200 ROOKFILE_V2_SEMI=90 &
wait
bash "$R" pyrun diagnostics/_revival_screen.py MODE=knobs KNOBS=taper650:$O/taper650.csv:$O/off.csv,threats100:$O/threats100.csv:$O/off.csv,space1320:$O/space1320.csv:$O/off.csv,longdiag100:$O/longdiag100.csv:$O/off.csv,reach100:$O/reach100.csv:$O/off.csv,latent100:$O/latent100.csv:$O/off.csv,rookfile200_90:$O/rookfile.csv:$O/off.csv
echo "[q20] $(date) QUEUE 20 DONE"
