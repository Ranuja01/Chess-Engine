#!/bin/bash
# 2026-10-05, queue #22: REVIVAL GATES (C3 doc §20/§20a). Five arms, each ONE change on top of V2_PRESET=shipped, files from
# `_revival_screen.py MODE=gateexport` (E:/chess_data/texel/revival/). Per arm: closure + symmetry guards — an arm that fails
# a guard is DROPPED from the games (logged), the queue continues. Then calibrated SF18 @800, fresh seeds 78/79, ONE shared
# baseline per seed + every surviving arm (500 games each). Self-play is decided with the owner after reading these.
# Batch ≠ bundle: every arm is judged alone vs the shared baseline; survivors get a combined confirmation later.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
O=/mnt/e/chess_data/texel/revival
TX=/mnt/e/chess_data/texel
cd "$ND" || exit 1
declare -A ARM
ARM[kauf]="KAUF_V2_FILE=$O/kauf_depth.txt"
ARM[mob]="C1_V2_FIT=1 C1_V2_FILE=$O/mob_depth_c1.txt"
ARM[pst]="PST_V2_FILE=$O/pst_depth.txt"
ARM[kprot]="KPROT_V2=1 KPROT_V2_FILE=$O/kprot_depth.txt"
ARM[kfl]="KFL_V2=1 KFL_V2_FILE=$O/kfl_depth.txt"
echo "[q22] $(date) start"
FP=$(env V2_PRESET=shipped bash "$R" wac fp_v2 2>&1 | grep -E "SOLVED|NODES" | tr '\n' ' ')
echo "[q22] v2 shipped: $FP"
echo "$FP" | grep -q "254/300.*50622239" || { echo "[q22] ☠️ FINGERPRINT MISMATCH — aborting"; exit 1; }
PASS=""
for a in kauf mob pst kprot kfl; do
  K="${ARM[$a]}"; ok=1
  case $a in
    kauf)  env V2_PRESET=shipped KAUF_V2_MAG=1000 $K bash "$R" pyrun diagnostics/_texel_kauf_fit.py MODE=closure > /tmp/q22_clo_$a.txt 2>&1 || ok=0
           grep "KAUF CLOSURE" /tmp/q22_clo_$a.txt | sed "s/^/[q22] $a /" ;;
    pst)   env V2_PRESET=shipped $K PST_V2_DUMP=/tmp/q22_pst_rt.txt PRESET=LONG_FORMAT USE_OPENING_BOOK=0 /home/ranuja/anaconda3/bin/python \
             -c "import sys; sys.path.insert(0,'.'); import chess, ChessAI; ChessAI.ChessAI(None, None, chess.Board(), True)" > /tmp/q22_clo_$a.txt 2>&1
           if diff -q <(grep -v '^#' $O/pst_depth.txt | tr -s ' \n' '\n') <(grep -v '^#' /tmp/q22_pst_rt.txt | tr -s ' \n' '\n') >/dev/null; then
             echo "[q22] pst round-trip EXACT (768 values)"; else echo "[q22] pst round-trip ☠️ DIFFERS"; ok=0; fi ;;
    *)     blk=$( [ $a = mob ] && echo mobility || ([ $a = kprot ] && echo v2_kprot || echo v2_kflank) )
           env V2_PRESET=shipped $K bash "$R" pyrun diagnostics/_texel_feature_pass.py IN=$TX/fitC_stage1.csv.gz LIMIT=20000 \
             OUT=/tmp/q22_$a.npz > /tmp/q22_clo_$a.txt 2>&1
           L=$(grep -E "^ *$blk " /tmp/q22_clo_$a.txt)
           echo "[q22] $a closure: $L"
           [ -n "$L" ] || ok=0
           # max|res| (last field) within ±10 mp AND rows_live (2nd field) > 0 — a block that never fires passes vacuously
           echo "$L" | awk '{ if ($NF+0 > 10 || $2+0 <= 0) exit 1 }' || ok=0 ;;
  esac
  SYM=$(env V2_PRESET=shipped $K bash "$R" pyrun diagnostics/_eval_symmetry.py N=4000 2>&1 | grep -E "violations")
  echo "[q22] $a symmetry: $(echo "$SYM" | tr '\n' ' ')"
  [ "$(echo "$SYM" | grep -c 'violations 0 ')" = "2" ] || ok=0
  if [ $ok = 1 ]; then PASS="$PASS $a"; echo "[q22] $a GUARDS OK"; else echo "[q22] ☠️ $a guards FAILED — dropped from the gates"; fi
done
echo "[q22] arms passing guards:$PASS"
[ -n "$PASS" ] || { echo "[q22] nothing to gate"; exit 1; }
for s in 78 79; do
  bash "$R" gauntlet 500 800 4 g800_ship1004b_s$s $s V2_PRESET=shipped
  for a in $PASS; do
    bash "$R" gauntlet 500 800 4 g800_rev_${a}_s$s $s V2_PRESET=shipped ${ARM[$a]}
    echo "[q22] $(date) $a seed $s done"
  done
done
echo "[q22] $(date) QUEUE 22 DONE"
