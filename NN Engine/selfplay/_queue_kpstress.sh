#!/bin/bash
# 2026-10-04, queue #18: the K+PAWNS STRESS CHECK (check-only, held-out; diagnostics/_kp_stress_check.py).
# Fresh set (SEED 41, never used) → SF18 d14 labels → our d10 search, connected pawns OFF (CONN_MAG=0) and ON (the ship),
# one arm at a time, 2 shards each (≤ 2 engines: the owner is still up), CHUNK loops (run_one leaks ~1.2 MB/FEN).
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
cd "$ND" || exit 1
echo "[q18] $(date) start"
bash "$R" pyrun diagnostics/gen_kp_fens.py N=600 SEED=41 MIX=dense,dense,pure OUT=diagnostics/ks_sets/kp_stress_fens.txt
bash "$R" pyrun diagnostics/_kp_stress_check.py MODE=label IN=diagnostics/ks_sets/kp_stress_fens.txt OUT=ks_sets/kp_stress_sf18.csv
for arm in off on; do
  if [ "$arm" = off ]; then K="PS_V2_CONN_MAG=0"; else K=""; fi
  for s in 0 1; do
    ( until env PRESET=LONG_FORMAT MAX_DEPTH=10 USE_OPENING_BOOK=0 V2_PRESET=shipped $K bash "$R" pyrun \
          diagnostics/_depth_residual_pass.py IN=ks_sets/kp_stress_sf18.csv OUT=ks_sets/kp_stress_${arm}_d10.csv \
          SHARD=$s/2 CHUNK=300; [ $? -ne 3 ]; do :; done ) &
  done
  wait
  echo "[q18] $(date) arm $arm done"
done
bash "$R" pyrun diagnostics/_kp_stress_check.py MODE=read ARMS=off:kp_stress_off_d10,on:kp_stress_on_d10
echo "[q18] $(date) QUEUE 18 DONE"
