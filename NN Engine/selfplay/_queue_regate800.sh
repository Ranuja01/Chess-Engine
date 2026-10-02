#!/bin/bash
# 2026-10-02, queue #8: RE-GATE on the RECALIBRATED external judge. Sweep (queue #7, shipped v2 @250k): SF18 @400 ≈ 75%,
# @800 54.2%, @1600 29.8%, @3200 13.0% ⇒ the new anchor is SF18 @800 nodes. Fresh seeds 58 + 59, fresh baselines.
# Arms: shipped (baseline, winnability ON) · winnability OFF (⇒ WSF's value on the new judge) · MCL all · MCL queen ·
# Kaufman full (FORM 3). 500 games each. Per-game engine processes (bounded memory).
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
TX=/mnt/e/chess_data/texel
cd "$ND" || exit 1
QV="MCL_V2_Q0=-430 MCL_V2_QR=80 MCL_V2_QM=-190 MCL_V2_QP=20"
ALL="MCL_V2=1 $QV MCL_V2_R2M=-170 MCL_V2_MP0=320 MCL_V2_MPP=20 MCL_V2_PAIR=200"
for s in 58 59; do
  bash "$R" gauntlet 500 800 4 g800_ship_s$s $s V2_PRESET=shipped
  bash "$R" gauntlet 500 800 4 g800_winoff_s$s $s V2_PRESET=shipped POT_V2_WIN=0
  bash "$R" gauntlet 500 800 4 g800_mclall_s$s $s V2_PRESET=shipped $ALL
  bash "$R" gauntlet 500 800 4 g800_mclq_s$s $s V2_PRESET=shipped MCL_V2=1 $QV
  bash "$R" gauntlet 500 800 4 g800_kauffull_s$s $s V2_PRESET=shipped KAUF_V2_MAG=1000 KAUF_V2_FORM=3 KAUF_V2_FILE=$TX/kauf_full.txt
done
echo "[q8] $(date) QUEUE 8 DONE"
