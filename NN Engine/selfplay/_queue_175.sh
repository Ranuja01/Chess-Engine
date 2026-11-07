#!/bin/bash
# 2026-11-07 ~10:20, queue #175 (owner 11-07: ship S, "we can monitor the results in that case and continue on into the evening"):
# gen 7 = gen 6.7 + the big retune's stage 1 tables (arm S) compiled in ship_tables_v5.h, V2_SHIP_GEN 7 by default.
# Evidence: timed self-play +16.0 (+6.2 … +25.8) over 4,000 g (#167 + #171, z 3.20); SF18 @ 1,100 −0.90 pp ± 4.88 and Mediocre
# not negative (#167); 50k nodes +15.4 (+5.6 … +25.2) over 4,000 g (#168).
#  1 PGO build chess_engine_uci_g7pgo from the repo sources. Gate: default bench must equal S on gen 6.7 (246 / 36,509,830), S's
#    file knobs on the new binary must give the same, and V2_SHIP_GEN=6 must restore gen 6.7 (251 / 36,859,850). Install as
#    production (gen 6.7 stays as chess_engine_uci_g67pgo). The binaries the running queues use are not touched.
#  2 after the held-out moves redo (holdfix.log) is over: Cython module rebuild (its .so backed up first) and its fingerprints.
#  3 after #173 is over, on a quiet box and before 18:30: UCI pinned bench, gen 6.7 and gen 7 alternating x6.
# Stop: touch /mnt/e/chess_data/q175.stop (checked between steps). Never edit this file or the sources while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
P=/mnt/e/chess_data/pgo_g7
OLD="$ND/build_uci/chess_engine_uci_g67pgo"
NEW="$ND/build_uci/chess_engine_uci_g7pgo"
SKN=$(cat /mnt/e/chess_data/q166/knobs_v3s.txt)
STOP=/mnt/e/chess_data/q175.stop
BENCH_LATEST=$(date -d '2026-11-07 18:30' +%s)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"
show() { sed "s/^/[q175]   /"; }
stopped() { [ -e "$STOP" ] && { echo "[q175] $(date) stop file present before $1 — QUEUE 175 DONE"; exit 0; }; }
over() { grep -qE "QUEUE $1 (DONE|ABORTED)" /mnt/e/chess_data/q$1.log 2>/dev/null; }
busy() { pgrep -f "fastchess-linux-x86-64/fastchess|vs_sf.py|_depth_residual_pass|_build_regret_set|_px_export|_retune_fit|_reference_ceiling|sts_test|tactical_test" >/dev/null; }
k() { printf 'setoption name Knobs value %s\nbench\nquit\n' "$2" | "$1" 2>/dev/null | grep '^bench'; }
echo "[q175] $(date) start"
cmp -s build_uci/chess_engine_uci "$OLD" && echo "[q175] backup ok: production == chess_engine_uci_g67pgo" \
  || { echo "[q175] ☠️ production differs from chess_engine_uci_g67pgo — QUEUE 175 ABORTED"; exit 1; }
grep -q "KAUF_SHIP7_OURS" ship_tables_v5.h || { echo "[q175] ☠️ ship_tables_v5.h missing — QUEUE 175 ABORTED"; exit 1; }

# ── 1 build, gate, install ────────────────────────────────────────────────────────────────────────────────────────────
rm -rf "$P"; mkdir -p "$P"
OUT=chess_engine_uci_g7gen PGO=gen PGO_DIR=$P bash selfplay/build_uci.sh || { echo "[q175] ☠️ gen build failed — QUEUE 175 ABORTED"; exit 1; }
nice -n 5 "$FC" -engine cmd="$ND/build_uci/chess_engine_uci_g7gen" name=a option.OwnBook=false st=0.25 \
  -engine cmd="$ND/build_uci/chess_engine_uci_g7gen" name=b option.OwnBook=false st=0.25 \
  -openings file=$O/openings_uho.epd format=epd order=random -srand 7 -rounds 4 -games 2 -concurrency 1 > /tmp/pgo_g7_train.log 2>&1
echo "[q175] training: $(grep -E '^Games:' /tmp/pgo_g7_train.log | tail -1) · profile files $(find "$P" -name '*.gcda' | wc -l)"
OUT=chess_engine_uci_g7pgo PGO=use PGO_DIR=$P bash selfplay/build_uci.sh || { echo "[q175] ☠️ use build failed — QUEUE 175 ABORTED"; exit 1; }
d=$(printf 'bench\nquit\n' | "$NEW" 2>/dev/null | grep '^bench')
s=$(k "$NEW" "$SKN")
g6=$(k "$NEW" V2_SHIP_GEN=6)
echo "[q175] gen 7 default (want 246 / 36509830):              $d"
echo "[q175] S file knobs on gen 7 (want 246 / 36509830):      $s"
echo "[q175] V2_SHIP_GEN=6 = gen 6.7 (want 251 / 36859850):    $g6"
echo "[q175] V2_SHIP_GEN=5 (want 258 / 47689586):              $(k "$NEW" V2_SHIP_GEN=5)"
echo "[q175] v1 EvalArm 0 (want 250 / 35310778 or 35310775):   $(printf 'setoption name EvalArm value 0\nbench\nquit\n' | "$NEW" 2>/dev/null | grep '^bench')"
echo "[q175] gen 6.7 binary, S knobs (on record 246 / 36509830): $(k "$OLD" "$SKN")"
echo "[q175] id: $(printf 'uci\nquit\n' | "$NEW" 2>/dev/null | grep '^id name')"
echo "$d" | grep -q "solved 246/300 nodes 36509830 " && echo "$s" | grep -q "solved 246/300 nodes 36509830 " \
  && echo "$g6" | grep -q "solved 251/300 nodes 36859850 " \
  || { echo "[q175] ☠️ fingerprints differ — production NOT replaced, QUEUE 175 ABORTED"; exit 1; }
cp "$NEW" build_uci/.chess_engine_uci.tmp && mv -f build_uci/.chess_engine_uci.tmp build_uci/chess_engine_uci \
  && cmp -s build_uci/chess_engine_uci "$NEW" && echo "[q175] $(date) INSTALLED: build_uci/chess_engine_uci == chess_engine_uci_g7pgo" \
  || { echo "[q175] ☠️ install failed — QUEUE 175 ABORTED"; exit 1; }

# ── 2 Cython module ───────────────────────────────────────────────────────────────────────────────────────────────────
echo "[q175] $(date) waiting for the held-out moves redo before the Cython rebuild"
until grep -q "HOLDFIX DONE" /mnt/e/chess_data/holdfix.log 2>/dev/null && ! pgrep -f "_depth_residual_pass|tactical_test|sts_test" >/dev/null; do
  stopped "the Cython rebuild"; sleep 60
done
stopped "the Cython rebuild"
SO=$(ls ChessAI.*.so 2>/dev/null | head -1)
[ -n "$SO" ] && cp "$SO" "$P/g67_$SO" && echo "[q175] backed up $SO to $P/g67_$SO"
bash "$R" build 2>&1 | grep -iE 'error|copying|FAILED' | tail -n 3 | show
ls ChessAI.*.so >/dev/null 2>&1 || { echo "[q175] ☠️ Cython build left no .so — restoring the gen 6.7 one"; cp "$P/g67_$SO" "$SO"; exit 1; }
echo "[q175] Cython gen 7 shipped (want 246 / 36509830):   $(bash "$R" wac fp_g7 V2_PRESET=shipped 2>&1 | grep -E '^SOLVED|^NODES|^EBF' | tr '\n' ' ')"
echo "[q175] Cython V2_SHIP_GEN=6 (want 251 / 36859850):   $(bash "$R" wac fp_g7as67 V2_PRESET=shipped V2_SHIP_GEN=6 2>&1 | grep -E '^SOLVED|^NODES|^EBF' | tr '\n' ' ')"
echo "[q175] Cython v1 (want 250 / 35310778):              $(bash "$R" wac fp_g7v1 2>&1 | grep -E '^SOLVED|^NODES|^EBF' | tr '\n' ' ')"

# ── 3 pinned bench ────────────────────────────────────────────────────────────────────────────────────────────────────
echo "[q175] $(date) waiting for #173 and a quiet box before the pinned bench"
until { over 173 && ! busy; } || [ $(date +%s) -gt $BENCH_LATEST ]; do stopped "the pinned bench"; sleep 60; done
if [ $(date +%s) -gt $BENCH_LATEST ]; then
  echo "[q175] $(date) too late for the pinned bench today — skipped"
else
  echo "[q175] $(date) == TIMED: UCI bench, gen 6.7 and gen 7 production binaries, pinned, alternating x6"
  for i in 1 2 3 4 5 6; do
    echo "[q175]   g67 $(printf 'bench\nquit\n' | taskset -c 5 "$OLD" 2>/dev/null | grep '^bench')"
    echo "[q175]   g7  $(printf 'bench\nquit\n' | taskset -c 5 "$NEW" 2>/dev/null | grep '^bench')"
  done
fi
echo "[q175] $(date) QUEUE 175 DONE"
