#!/bin/bash
# 2026-10-20, queue #87 — the numbers for the "v2 is the official engine" record (owner: d10 benches, win% MSE comparisons, the
# v1-vs-v2 games). ONE process at a time (timing benches need a quiet box). Waits for queue #86 unless NO_WAIT=1.
#   benches, v2 (V2_PRESET=shipped) and v1 (no env): WAC d10 fingerprint · STS300 d10 · STS full 15 themes d10 ·
#   WAC timed at LIGHTNING (depth reached per move) · WAC speed (pinned core, 3 reps, max NPS)
#   accuracy (_v1_v2_accuracy.py): static v1 / v2 / SF11 and d10 search v1 / v2 on 5,988 held-out quiet rows, cp + WDL labels
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
D=/mnt/e/chess_data/wdlcal
cd "$ND" || exit 1
echo "[q87] $(date) armed"
[ "${NO_WAIT:-0}" = "1" ] || until grep -q "QUEUE 86 DONE" /mnt/e/chess_data/q86_fc_day.log 2>/dev/null; do sleep 60; done
echo "[q87] $(date) start"
V2="V2_PRESET=shipped"
for arm in v2 v1; do
  K=""; [ $arm = v2 ] && K=$V2
  echo "[q87] == $arm WAC d10";        bash "$R" wac rec_$arm $K | head -3
  echo "[q87] == $arm STS300 d10";     bash "$R" sts rec_$arm $K | tail -2
  echo "[q87] == $arm STS full d10";   bash "$R" sts_full recf_$arm $K | tail -1
  echo "[q87] == $arm WAC LIGHTNING";  bash "$R" wac_timed_depth rect_$arm $K | head -6
  echo "[q87] == $arm WAC speed";      bash "$R" wac_speed recs_$arm 3 $K | grep -E "NODES:|MAX_NPS|MEDIAN|SPREAD"
done
echo "[q87] $(date) accuracy dumps"
env OMP_NUM_THREADS=1 bash "$R" pyrun diagnostics/_v1_v2_accuracy.py MODE=prep > /dev/null 2>&1
env OMP_NUM_THREADS=1 $V2 bash "$R" pyrun diagnostics/_v1_v2_accuracy.py MODE=static TAG=v2 > /dev/null 2>&1
env OMP_NUM_THREADS=1 bash "$R" pyrun diagnostics/_v1_v2_accuracy.py MODE=static TAG=v1 > /dev/null 2>&1
env OMP_NUM_THREADS=1 bash "$R" pyrun diagnostics/_v1_v2_accuracy.py MODE=sf11 > /dev/null 2>&1
for arm in v2 v1; do
  K=""; [ $arm = v2 ] && K=$V2
  until timeout 3600 env OMP_NUM_THREADS=1 PRESET=LONG_FORMAT MAX_DEPTH=10 USE_OPENING_BOOK=0 RESIGN_THRESHOLD=-99999999 $K \
        bash "$R" pyrun diagnostics/_depth_residual_pass.py IN=$D/cal_d10_in.csv OUT=$D/d10_$arm.csv CHUNK=1000 > /dev/null 2>&1; [ $? -ne 3 ]; do :; done
  echo "[q87] $(date) d10 $arm: $(grep -vc '^fen' $D/d10_$arm.csv) rows"
done
bash "$R" pyrun diagnostics/_v1_v2_accuracy.py MODE=read 2>/dev/null | sed 's/^/[q87] /'
echo "[q87] $(date) QUEUE 87 DONE"
