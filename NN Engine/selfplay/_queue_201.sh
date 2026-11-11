#!/bin/bash
# 2026-11-11 ~14:00, queue #201: POT DUE DILIGENCE, test C (search v2 plan §7d; registered in diagnostics/_line_trace.py's
# docstring before any trace ran). How does our search treat the moves of Stockfish's line in the owner's game positions?
# The C++ line tracer (search_engine.cpp TraceRec, TRACE_TARGETS) is off without its file, so production is unchanged:
# step 1 checks that on a test UCI build. Fixed depth only. Test copies only: production binary and live UI module untouched.
#  1 test UCI build (plain) chess_engine_uci_trace: bench must equal gen 7.2 (247 / 39,861,964), the tracer off.
#  2 test Cython module /mnt/e/chess_data/so_trace from the same sources.
#  3 SF18 d18 lines (8 plies) for the 17 cases; then our d10 and d12 searches traced; then the registered report.
# Stop: touch /mnt/e/chess_data/q201.stop (checked between steps). Never edit this file, the tools or the sources while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
TB="$ND/build_uci/chess_engine_uci_trace"
SO=/mnt/e/chess_data/so_trace
L=/mnt/e/chess_data/linetrace
STOP=/mnt/e/chess_data/q201.stop
LATEST=$(date -d '2026-11-11 21:00' +%s)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"; mkdir -p "$L"
show() { sed "s/^/[q201]   /"; }
stopped() { [ -e "$STOP" ] && { echo "[q201] $(date) stop file present before $1 — QUEUE 201 DONE"; exit 0; }; }
[ $(date +%s) -lt $LATEST ] || { echo "[q201] $(date) past 21:00 — skipped"; exit 0; }
START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[q201] $START start"

# ── 1 the tracer is inert at the default ────────────────────────────────────────────────────────────────────────────────
rm -f "$TB"
OUT=chess_engine_uci_trace bash selfplay/build_uci.sh > $L/build_uci.out 2>&1
[ -x "$TB" ] || { echo "[q201] ☠️ UCI test build failed — QUEUE 201 ABORTED"; tail -5 build_uci/build.log | show; exit 1; }
fp=$(printf 'bench\nquit\n' | "$TB" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+')
echo "[q201] test UCI build, tracer off (want 247 / 39861964): $fp"
[ "$fp" = "solved 247/300 nodes 39861964" ] || { echo "[q201] ☠️ the tracer changes the default — QUEUE 201 ABORTED"; exit 1; }

# ── 2 the test Cython module ────────────────────────────────────────────────────────────────────────────────────────────
stopped "the Cython test build"
rm -rf "$SO"; mkdir -p "$SO"
bash "$R" pyrun setupAI.py build_ext --build-lib "$SO" --build-temp "$SO/tmp" > "$SO/build.log" 2>&1
ls "$SO"/ChessAI*.so >/dev/null 2>&1 || { echo "[q201] ☠️ Cython test build failed — QUEUE 201 ABORTED"; tail -5 "$SO/build.log" | show; exit 1; }
echo "[q201] $(date) test module built in $SO"

# ── 3 lines, traces, report ─────────────────────────────────────────────────────────────────────────────────────────────
stopped "the SF lines"
bash "$R" pyrun diagnostics/_line_trace.py MODE=sf > $L/sf.log 2>&1
echo "[q201] $(date) SF18 lines: $(grep -c "'" $L/sf.log) cases"
for dep in 10 12; do
	stopped "the d$dep trace"
	env CHESSAI_DIR=$SO V2_PRESET=shipped MAX_DEPTH=$dep bash "$R" pyrun diagnostics/_line_trace.py MODE=trace TAG=d$dep > $L/trace_d$dep.log 2>&1
	echo "[q201] $(date) d$dep traced:"
	grep -E 'trace records' $L/trace_d$dep.log | show
done
for dep in 10 12; do
	echo "[q201] == REPORT d$dep (final iteration; the registered read)"
	bash "$R" pyrun diagnostics/_line_trace.py MODE=report TAG=d$dep > $L/report_d$dep.txt 2>&1
	sed -n '/quiet SF moves at k/,$p' $L/report_d$dep.txt | show
	echo "[q201]   all iterations pooled (secondary):"
	bash "$R" pyrun diagnostics/_line_trace.py MODE=report TAG=d$dep ALL_IT=1 2>&1 | sed -n '/quiet SF moves at k/,$p' | show
done
echo "[q201] full tables: $L/report_d10.txt, $L/report_d12.txt"
echo "[q201] $(date) QUEUE 201 DONE"
