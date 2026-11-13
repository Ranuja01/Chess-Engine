#!/bin/bash
# 2026-11-13 ~12:05, queue #224: the result-identical build's remaining ship checks (BASELINE_PERF per-build tracking 4 and 5),
# after #223 installed chess_engine_uci_g73fpgo (MOVEGEN_FAST=1 in the shipped preset).
#  1 Build identity games: gen 7.3's first build (g73pgo) vs the installed one, fixed 30k nodes, 40 starts per set, one process
#    per side. REGISTERED: every set "diverged: 0" and no crash (the same moves in every game).
#  2 Cython module (the UI's engine) nps: depth_nps_bench, 60 midgame FENs, PRESET=LIGHTNING, V2_PRESET=shipped, pinned, x3.
#    A record for the ledger, no bar.
# Stop: touch /mnt/e/chess_data/q224.stop (checked between steps). Never edit this file, the binaries or the sources while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
OLD="$ND/build_uci/chess_engine_uci_g73pgo"
PROD="$ND/build_uci/chess_engine_uci"
STOP=/mnt/e/chess_data/q224.stop
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"
echo "[q224] $(date '+%Y-%m-%d %H:%M:%S') start"
cmp -s "$PROD" build_uci/chess_engine_uci_g73fpgo && echo "[q224] production == chess_engine_uci_g73fpgo" \
	|| { echo "[q224] ☠️ production is not g73fpgo — QUEUE 224 ABORTED"; exit 1; }
echo "[q224] $(date '+%H:%M') build identity: g73pgo vs production, fixed 30k nodes, 40 starts per set, one process per side"
bash "$R" pyrun diagnostics/_fix_firecount.py BIN="$OLD" BIN_ON="$PROD" ON= N=40 NODES=30000 PLIES=200 SIDES=2 JOBS=2 2>&1 \
	| grep -E '===|diverged|CRASH' | sed 's/^/[q224]   /'
[ -e "$STOP" ] && { echo "[q224] $(date) stop file present before the Cython nps — QUEUE 224 DONE"; exit 0; }
echo "[q224] $(date '+%H:%M') Cython depth_nps_bench, 60 midgame FENs, PRESET=LIGHTNING, shipped, pinned, x3"
for r in 1 2 3; do
	echo "[q224]   $(env PRESET=LIGHTNING USE_OPENING_BOOK=0 V2_PRESET=shipped taskset -c 5 bash "$R" pyrun diagnostics/depth_nps_bench.py --n 60 2>&1 | grep -E 'DEPTH reached|NPS: median' | tr '\n' ' ')"
done
echo "[q224] $(date) QUEUE 224 DONE"
