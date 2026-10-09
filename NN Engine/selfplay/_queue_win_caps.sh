#!/bin/bash
# 2026-10-09, queue #45 — WINNABILITY MATERIAL-CLASS CAPS (built + fingerprint byte-identical at 0, 10-09). Each rule at SF15.1's
# value: symmetry gate (all four on), then a static dump of the depth rows per rule (≤ 4 engines), then MODE=dual: fire rate,
# static + depth-target effect (structural ⇒ the depth proxy is fair as a HARM screen) at α = 1 (SF values) and fitted.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
O=/mnt/e/chess_data/texel/revival
OCBX="POT_V2_WIN_OCBX=-42 POT_V2_WIN_OCBX_PC=3"
ROOKE="POT_V2_WIN_ROOKE=-28"
QNOQ="POT_V2_WIN_QNOQ=-27 POT_V2_WIN_QNOQ_MINOR=3"
LONE="POT_V2_WIN_LONEMINOR=-64"
cd "$ND" || exit 1
echo "[q45] $(date) start"
SYM=$(env V2_PRESET=shipped $OCBX $ROOKE $QNOQ $LONE bash "$R" pyrun diagnostics/_eval_symmetry.py N=4000 2>&1 | grep -E "violations")
echo "[q45] all-caps symmetry: $(echo "$SYM" | tr '\n' ' ')"
[ "$(echo "$SYM" | grep -c 'violations 0 ')" = "2" ] || { echo "[q45] ☠️ symmetry FAILED — aborting"; exit 1; }
d() { env V2_PRESET=shipped ${@:2} bash "$R" pyrun diagnostics/_revival_screen.py MODE=dump OUT=$O/$1.csv 2>&1 | grep -E "REVIVAL DUMP|rror"; }
d wc_ocbx $OCBX & d wc_rooke $ROOKE & d wc_qnoq $QNOQ & d wc_lone $LONE & wait
echo "[q45] $(date) dumps done"
bash "$R" pyrun diagnostics/_revival_screen.py MODE=dual BASE=$O/off.csv FIXED=1,1,1,1 \
  KNOBS=ocbx:$O/wc_ocbx.csv:$O/off.csv,rooke:$O/wc_rooke.csv:$O/off.csv,qnoq:$O/wc_qnoq.csv:$O/off.csv,lone:$O/wc_lone.csv:$O/off.csv \
  2>&1 | tail -8
echo "[q45] $(date) QUEUE 45 DONE"
