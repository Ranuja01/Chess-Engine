#!/bin/bash
# 2026-10-19, queue #82 — FULL-SIZE WDL check of the boosted-tree ceiling test (POT brief §5n). SF18 d14 WDL on the 22,313
# trajectory TRAIN starts (3 shards), then _pot_gbm_ceiling.py MODE=wdlfull (train → unseen val, old cp label vs WDL).
# Waits until START_AT (default 23:30, after the owner's evening) unless already later. ≤ 3 processes, Threads 1 each.
# ~1 s / row ⇒ ~2.1 h on 3 shards.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
START_AT=${START_AT:-23:30}
cd "$ND" || exit 1
echo "[q82] $(date) armed, start at $START_AT"
until [ "$(date +%H:%M)" \> "$START_AT" ] || [ "$(date +%H:%M)" \< "06:00" ]; do sleep 60; done
echo "[q82] $(date) start"
for s in 0 1 2; do
  env OMP_NUM_THREADS=1 bash "$R" pyrun diagnostics/_sf18_wdl_pass.py SPLIT=train OUT=/mnt/e/chess_data/deep/mg_wdl_train.csv \
    SHARD=$s/3 > /mnt/e/chess_data/deep/mg_wdl_train_s$s.log 2>&1 &
done
wait
echo "[q82] $(date) labels: $(cat /mnt/e/chess_data/deep/mg_wdl_train_s*of3.csv | grep -vc '^fen') train rows"
env OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 bash "$R" pyrun diagnostics/_pot_gbm_ceiling.py MODE=wdlfull 2>&1 | tail -4
echo "[q82] $(date) QUEUE 82 DONE"
