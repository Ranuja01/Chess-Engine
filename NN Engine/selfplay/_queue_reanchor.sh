#!/bin/bash
# 2026-10-06 night, queue #28 (owner: "better to be sure"): RE-ANCHOR the calibrated judge + re-gate the KFL+PST pair on it.
# SF18 @800 now reads 57-59% vs the ship (q27 s84 57.4 / s85 59.2) — drifting toward the ~60%+ zone where the @400 judge
# flipped material verdicts (10-02, memory the-sf18-gauntlet-anchor-drifted-too-weak). Sweep @1000 and @1200 (fresh seeds
# 87/88): the shipped baseline AND the KFL+PST pair on every block, so the recalibration and the pair's second fair read
# come from the same games. Pair so far: SF @800 −0.6pp · self-play +15.1 ± 9 (C3 §20d).
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
O=/mnt/e/chess_data/texel/revival
cd "$ND" || exit 1
COMBO="KFL_V2=1 KFL_V2_FILE=$O/kfl_depth.txt PST_V2_FILE=$O/pst_depth.txt"
echo "[q28] $(date) start"
FP=$(env V2_PRESET=shipped bash "$R" wac fp_v2 2>&1 | grep -E "SOLVED|NODES" | tr '\n' ' ')
echo "[q28] v2 shipped: $FP"
echo "$FP" | grep -q "254/300.*50622239" || { echo "[q28] ☠️ FINGERPRINT MISMATCH — aborting"; exit 1; }
[ -s $O/kfl_depth.txt ] && [ -s $O/pst_depth.txt ] || { echo "[q28] ☠️ arm files missing — aborting"; exit 1; }
for n in 1000 1200; do
  for s in 87 88; do
    bash "$R" gauntlet 500 $n 4 g${n}_ship1004_s$s $s V2_PRESET=shipped
    bash "$R" gauntlet 500 $n 4 g${n}_combo_kflpst_s$s $s V2_PRESET=shipped $COMBO
    echo "[q28] $(date) @$n seed $s done"
  done
done
echo "[q28] $(date) QUEUE 28 DONE"
