#!/bin/bash
# 2026-11-11 ~20:40, queue #208: search v2 S3 item "late-move pruning before make_move" (SV2_PRUNE_EARLY; plan §7e, S3 design
# §4). The verify build (-DSV2_PRUNE_VERIFY) already agreed on all 19,756,269 early prunes of the WAC bench (0 mismatches,
# bench 247 / 39,861,964). Here, on a plain build:
#  1 SW1 identity: diagnostics/_sv2_identity.sh with SV2_PRUNE_EARLY=1 (arm 0 ignores it, so arm 1 + early is held to today's
#    search node for node on every bench set).
#  2 SW2 identity: diagnostics/_sv2_game_identity.py with SV2_PRUNE_EARLY=1 (12 self-play games, tables carried, node limits).
#  3 Speed, instructions: cachegrind, the 12 game positions of #204 at depth 10, arm 1 with SV2_PRUNE_EARLY 0 vs 1.
#    REGISTERED: identity SAME everywhere; instructions lower with early pruning (prediction −3 to −8%: 56-63% of made moves
#    were pruned straight away, S3 step 0).
# Fixed depth throughout (safe beside the owner's evening). Stop: touch /mnt/e/chess_data/q208.stop. Never edit this file,
# the tools or the binary while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
B="$ND/build_uci/chess_engine_uci_sv2pe"
O=/mnt/e/chess_data/sv2_prune_early
STOP=/mnt/e/chess_data/q208.stop
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"; mkdir -p "$O"
stopped() { [ -e "$STOP" ] && { echo "[q208] $(date) stop file present before $1 — QUEUE 208 DONE"; exit 0; }; }
echo "[q208] $(date '+%Y-%m-%d %H:%M:%S') start"
rm -f "$B"
OUT=chess_engine_uci_sv2pe bash selfplay/build_uci.sh > "$O/build.out" 2>&1
[ -x "$B" ] || { echo "[q208] ☠️ build failed — QUEUE 208 ABORTED"; tail -5 build_uci/build.log; exit 1; }
for e in "SEARCH_ARM=0" "SEARCH_ARM=1" "SEARCH_ARM=1 SV2_PRUNE_EARLY=1"; do
	echo "[q208] bench [$e]: $(printf 'bench\nquit\n' | env $e "$B" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+')"
done

stopped "SW1"
echo "[q208] == SW1 identity with SV2_PRUNE_EARLY=1"
SV2_PRUNE_EARLY=1 bash diagnostics/_sv2_identity.sh "$B" /mnt/e/chess_data/sv2_id 10 2>&1 | sed 's/^/[q208]   /'

stopped "SW2"
echo "[q208] == SW2 game identity with SV2_PRUNE_EARLY=1"
SV2_PRUNE_EARLY=1 bash "$R" pyrun diagnostics/_sv2_game_identity.py BIN=build_uci/chess_engine_uci_sv2pe STARTS=6 PLIES=60 2>&1 | grep -E 'SAME|DIFFERENT|arm|game identity' | sed 's/^/[q208]   /'

stopped "the instruction count"
echo "[q208] == instructions (cachegrind), arm 1, SV2_PRUNE_EARLY 0 vs 1"
for pe in 0 1; do
	printf "bench /mnt/e/chess_data/sv2_speed/game12.txt 10\nquit\n" | env SEARCH_ARM=1 SV2_PRUNE_EARLY=$pe valgrind --tool=cachegrind --cache-sim=no \
		--cachegrind-out-file="$O/cg_pe$pe.out" "$B" > "$O/bench_pe$pe.txt" 2> "$O/cg_pe$pe.log"
	echo "[q208] early $pe: $(grep -oE 'solved [0-9]+/[0-9]+ nodes [0-9]+' "$O/bench_pe$pe.txt") · $(grep -E 'I *refs' "$O/cg_pe$pe.log" | sed 's/==[0-9]*== *//')"
done
i0=$(grep -E 'I *refs' "$O/cg_pe0.log" | grep -oE '[0-9,]+$' | tr -d ,)
i1=$(grep -E 'I *refs' "$O/cg_pe1.log" | grep -oE '[0-9,]+$' | tr -d ,)
echo "[q208] early 1 - early 0 instructions: $(awk -v a="$i0" -v b="$i1" 'BEGIN { printf "%+.4f%%", 100 * (b - a) / a }')"
echo "[q208] $(date) QUEUE 208 DONE"
