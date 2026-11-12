#!/bin/bash
# 2026-11-12 ~13:10, queue #215: why early pruning saved instructions but not time, and MOVEGEN_FAST's cost profile, from
# cachegrind with its cache and branch simulation (WSL2 exposes no hardware counters, so perf cannot count cycles or misses
# here). Deterministic and load-insensitive, so it runs beside #211's node-limited games. Waits for #214 (one build at a time).
# The 12 game positions of #204 / #208 at depth 10 (game12.txt), identical trees within each pair:
#  A early pruning: build_uci/chess_engine_uci_sv2pe (plain, #208), SEARCH_ARM=1, SV2_PRUNE_EARLY 0 vs 1
#  B MOVEGEN_FAST: build_uci/mgf (plain, #212), SEARCH_ARM=0 (production's search), MOVEGEN_FAST 0 vs 1
# Reported per run: instructions, D1 and last-level data misses, branch mispredictions (cachegrind's simple model: read the
# differences within a pair, not the absolute rates). Not registered as a gate: a diagnostic for the early-pruning miss.
# Prediction (A): fewer instructions (−3.16%, #208) but no fewer last-level misses, the cost that time follows.
# Stop: touch /mnt/e/chess_data/q215.stop.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
O=/mnt/e/chess_data/cg_215
G=/mnt/e/chess_data/sv2_speed/game12.txt
STOP=/mnt/e/chess_data/q215.stop
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"; mkdir -p "$O"
echo "[q215] $(date '+%Y-%m-%d %H:%M:%S') queued; waiting for #214"
until grep -qE "QUEUE 214 (DONE|ABORTED)|stopped" /mnt/e/chess_data/q214.log 2>/dev/null; do [ -e "$STOP" ] && { echo "[q215] stopped while waiting"; exit 0; }; sleep 30; done
echo "[q215] $(date '+%H:%M') start"
cg() {  # tag, binary, env knobs
	[ -e "$STOP" ] && { echo "[q215] stopped"; exit 0; }
	printf "bench $G 10\nquit\n" | env $3 valgrind --tool=cachegrind --cache-sim=yes --branch-sim=yes \
		--cachegrind-out-file="$O/cg_$1.out" "$2" > "$O/bench_$1.txt" 2> "$O/cg_$1.log"
	echo "[q215] $1 [$3]: $(grep -oE 'solved [0-9]+/[0-9]+ nodes [0-9]+' "$O/bench_$1.txt")"
	grep -E 'I *refs|D1  misses|LLd misses|LL misses|Mispred' "$O/cg_$1.log" | sed -E 's/==[0-9]*== *//; s/^/[q215]     /'
}
cg pe0 "$ND/build_uci/chess_engine_uci_sv2pe" "SEARCH_ARM=1 SV2_PRUNE_EARLY=0"
cg pe1 "$ND/build_uci/chess_engine_uci_sv2pe" "SEARCH_ARM=1 SV2_PRUNE_EARLY=1"
cg mgf0 "$ND/build_uci/mgf" "SEARCH_ARM=0 MOVEGEN_FAST=0"
cg mgf1 "$ND/build_uci/mgf" "SEARCH_ARM=0 MOVEGEN_FAST=1"
num() { grep -E "$2" "$O/cg_$1.log" | head -1 | sed -E 's/==[0-9]*== *//' | awk -F: '{print $2}' | awk '{gsub(/,/, "", $1); print $1}'; }
for pair in "pe0 pe1" "mgf0 mgf1"; do
	set -- $pair
	for m in 'I *refs' 'D1  misses' 'LLd misses' 'Mispred'; do
		a=$(num $1 "$m"); b=$(num $2 "$m")
		[ -n "$a" ] && [ -n "$b" ] && echo "[q215] $2 - $1, $m: $(awk -v a="$a" -v b="$b" 'BEGIN { printf "%+.2f%% (%d -> %d)", 100 * (b - a) / a, a, b }')"
	done
done
echo "[q215] $(date) QUEUE 215 DONE"
