#!/bin/bash
# 2026-10-19, queue #81 — MIDDLEGAME DEEP-RESIDUAL set (owner: middlegame POT before cleanup; also the search-transition
# instrument). Our candidate (gen 3 + C64) d12 SEARCH score + best move on the 26,362 mg / mgl trajectory starts
# (E:\chess_data\deep\mg_starts.csv, seeded random order ⇒ any finished prefix per shard is an unbiased sample). The trajectory set
# already holds SF18 d14 at t = 0 / 8 / 16 / 24 plies ⇒ targets: deep residual sf(0) − ours_d12(0) and realised potential sf(t+k).
# Pilot: 30 rows = 178 s (~5.9 s/row, 1.2M nodes mean) ⇒ 3 procs × ~8.5 h ≈ 13-14k rows. Stops at EXIT_AT (default 08:30).
# ≤ 3 processes, fixed depth, single-threaded math, resign disabled; CHUNK bounds run_one's per-FEN leak.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
D=/mnt/e/chess_data/deep
EXIT_AT=${EXIT_AT:-08:30}
cd "$ND" || exit 1
echo "[q81] $(date) start (exit at $EXIT_AT)"
FP=$(env V2_PRESET=candidate bash "$R" wac fp_q81 2>&1 | grep -E "SOLVED|NODES" | tr '\n' ' ')
echo "[q81] fingerprint: $FP"
echo "$FP" | grep -q "258/300.*47689413" || { echo "[q81] ☠️ FINGERPRINT MISMATCH (expected candidate 258 / 47,689,413) — aborting"; exit 1; }
past_exit() { [ "$(date +%H:%M)" \> "$EXIT_AT" ] && [ "$(date +%H:%M)" \< "18:00" ]; }
for s in 0 1 2; do
  ( until past_exit; do
      timeout 3600 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PRESET=LONG_FORMAT MAX_DEPTH=12 \
        USE_OPENING_BOOK=0 V2_PRESET=candidate RESIGN_THRESHOLD=-99999999 bash "$R" pyrun diagnostics/_depth_residual_pass.py \
        IN=$D/mg_starts.csv OUT=$D/mg_cand_d12.csv SHARD=$s/3 CHUNK=150 MOVE=1 > /dev/null 2>&1
      rc=$?
      [ $rc -eq 3 ] || { echo "[q81] $(date) shard $s ended rc=$rc"; break; }
    done ) &
done
while sleep 1800; do
  n=$(cat $D/mg_cand_d12_s*of3.csv 2>/dev/null | grep -vc '^fen')
  echo "[q81] $(date +%H:%M) rows $n · free RAM $(free -g | awk '/Mem:/{print $7}') GB"
  jobs -r | grep -q . || break
done
wait
echo "[q81] $(date) QUEUE 81 DONE: $(cat $D/mg_cand_d12_s*of3.csv | grep -vc '^fen') rows"
