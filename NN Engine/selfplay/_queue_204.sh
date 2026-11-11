#!/bin/bash
# 2026-11-11 ~15:15, queue #204: search v2 S1 speed read, part 1 (plan §7e; the proposed replacement of S1's games read).
# At node-for-node identity the arms differ only in speed. Instructions executed (cachegrind `I refs`, no cache simulation)
# are insensitive to machine load, so this runs beside Memcheck (#203). Same plain binary, SEARCH_ARM 0 vs 1, 12 game
# positions (game_regret_set rows 1, 1251, 2501, ... : every phase), depth 10, tables cleared per position (bench).
# Registered before the run: LEVEL = |difference| <= 0.5% of instructions; v2 SLOWER if > +0.5% (then S2 looks at why first);
# v2 FASTER if < -0.5%. Prediction: within ±1% (same work, template code laid out differently), sign unknown.
# Stop: touch /mnt/e/chess_data/q204.stop (checked between steps). Never edit this file or the binary while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
B="$ND/build_uci/chess_engine_uci_sv2"
O=/mnt/e/chess_data/sv2_speed
STOP=/mnt/e/chess_data/q204.stop
cd "$ND" || exit 1
rm -f "$STOP"; mkdir -p "$O"
echo "[q204] $(date '+%Y-%m-%d %H:%M:%S') start"
awk -F, 'NR > 1 && (NR - 2) % 1250 == 0 { print $1 " | " $3 }' diagnostics/ks_sets/game_regret_set.csv > "$O/game12.txt"
echo "[q204] positions: $(wc -l < "$O/game12.txt")"
for arm in 0 1; do
	[ -e "$STOP" ] && { echo "[q204] stop file — QUEUE 204 DONE"; exit 0; }
	printf "bench $O/game12.txt 10\nquit\n" | env SEARCH_ARM=$arm valgrind --tool=cachegrind --cache-sim=no \
		--cachegrind-out-file="$O/cg_arm$arm.out" "$B" > "$O/bench_arm$arm.txt" 2> "$O/cg_arm$arm.log"
	echo "[q204] arm $arm: $(grep -oE 'solved [0-9]+/[0-9]+ nodes [0-9]+' "$O/bench_arm$arm.txt") · $(grep -E 'I *refs' "$O/cg_arm$arm.log" | sed 's/==[0-9]*== *//')"
done
i0=$(grep -E 'I *refs' "$O/cg_arm0.log" | grep -oE '[0-9,]+$' | tr -d ,)
i1=$(grep -E 'I *refs' "$O/cg_arm1.log" | grep -oE '[0-9,]+$' | tr -d ,)
echo "[q204] arm 1 - arm 0 instructions: $(awk -v a="$i0" -v b="$i1" 'BEGIN { printf "%+.4f%%", 100 * (b - a) / a }')"
echo "[q204] $(date) QUEUE 204 DONE"
