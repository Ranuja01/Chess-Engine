#!/bin/bash
# 2026-11-16 ~17:40, queue #258: phase H-S step 1 on the new base, a fresh profile of gen 7.5 (the gen 7.4 one is #247's). Fixed
# work, one process, nothing timed: safe beside #257's node-limited SF18 legs and while the owner games.
# Copy chess_engine_uci_g75prof = production flags + the gen 7.5 PGO profile + -g -fno-omit-frame-pointer (gate: bench
# 252 / 35,235,460). cachegrind with cache and branch simulation on `bench game12 10` (12 game positions, depth 10), cg_annotate
# per function → /mnt/e/chess_data/q258/cg_annotate.txt, beside #247's gen 7.4 report. Descriptive: it ranks the speed items
# after the eval speed batches and C2c, no rule.
# Stop: touch /mnt/e/chess_data/q258.stop. Never edit the sources before "build done".
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
W=/mnt/e/chess_data/q258
P=/mnt/e/chess_data/pgo_g75
G=/mnt/e/chess_data/sv2_speed/game12.txt
PB="$ND/build_uci/chess_engine_uci_g75prof"
STOP=/mnt/e/chess_data/q258.stop
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"; mkdir -p "$W"
show() { sed "s/^/[q258]   /"; }
echo "[q258] $(date '+%Y-%m-%d %H:%M:%S') start"
OUT=chess_engine_uci_g75prof PGO=use PGO_DIR=$P EXTRA_FLAGS="-g -fno-omit-frame-pointer" nice -n 10 bash selfplay/build_uci.sh > "$W/prof_build.out" 2>&1 \
	|| { echo "[q258] ☠️ profiling build failed — QUEUE 258 DONE"; tail -5 "$W/prof_build.out" | show; exit 1; }
echo "[q258] $(date '+%H:%M') build done — the sources are free to edit"
[ -e "$STOP" ] && { echo "[q258] stop file — QUEUE 258 DONE"; exit 0; }
f=$(printf 'bench\nquit\n' | "$PB" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+')
echo "[q258] profiling copy bench (want 252 / 35235460): [$f]"
[ "$f" = "solved 252/300 nodes 35235460" ] || { echo "[q258] ☠️ profiling copy moved a node — profile skipped — QUEUE 258 DONE"; exit 1; }
printf "bench $G 10\nquit\n" | nice -n 10 valgrind --tool=cachegrind --cache-sim=yes --branch-sim=yes \
	--cachegrind-out-file="$W/cg.out" "$PB" > "$W/cg_bench.txt" 2> "$W/cg.log"
echo "[q258] $(date '+%H:%M') cachegrind: $(grep -oE 'solved [0-9]+/[0-9]+ nodes [0-9]+' "$W/cg_bench.txt")"
grep -E 'I *refs|D1 *misses|LLd *misses|Mispred' "$W/cg.log" | sed 's/^==[0-9]*== //' | show
cg_annotate "$W/cg.out" > "$W/cg_annotate.txt" 2>&1
echo "[q258] per-function report: $W/cg_annotate.txt ($(wc -l < "$W/cg_annotate.txt") lines); gen 7.4's: /mnt/e/chess_data/q247/cg_annotate.txt"
echo "[q258] $(date) QUEUE 258 DONE"
