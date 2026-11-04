#!/bin/bash
# 2026-11-04 morning, queue #145: build-identity game check for speed batch 8 (the #142 speed match read -13.2 (-31.7 ... +5.2)
# although real play was +4.0% nps; the knobs are result-identical by construction and by their verify modes, and this closes it).
# Waits for queue #144, then plays the batch-7 and batch-8 production binaries against themselves from the same starts at a fixed
# node budget (deterministic; one engine process per side, as in real play) and compares every move. Every set must report
# "diverged: 0" and 0 crashes; any divergence means a real behaviour change and reopens the batch.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
STOP=/mnt/e/chess_data/q145.stop
Q144=/mnt/e/chess_data/q144.log
OLD="$ND/build_uci/chess_engine_uci_g65fpgo"
NEW="$ND/build_uci/chess_engine_uci_g65b8pgo"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"
echo "[q145] $(date) waiting for queue #144 to finish (stop: $STOP)"
until grep -qE 'QUEUE 144 DONE|QUEUE 144 ABORTED' "$Q144" 2>/dev/null; do
  [ -e "$STOP" ] && { echo "[q145] $(date) stop file present — QUEUE 145 DONE (nothing run)"; exit 0; }
  sleep 60
done
sleep 30
pgrep -f "fastchess" >/dev/null && { echo "[q145] matches are still running after #144 — QUEUE 145 ABORTED"; exit 1; }
echo "[q145] $(date) start: batch 7 ($OLD) vs batch 8 ($NEW), fixed 30k nodes, 40 starts per set, one process per side"
bash "$R" pyrun diagnostics/_fix_firecount.py BIN="$OLD" BIN_ON="$NEW" ON= N=40 NODES=30000 PLIES=200 SIDES=2 JOBS=2 2>&1 \
  | grep -vE '^\s*$' | sed 's/^/[q145]   /'
echo "[q145] $(date) QUEUE 145 DONE"
