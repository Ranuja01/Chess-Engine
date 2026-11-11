#!/bin/bash
# 2026-11-11 ~16:25, queue #206: search v2 S2, the profile that chooses its first item (plan §3 S2, §6.4's method). A profiling
# copy of today's sources (production flags + -g -fno-omit-frame-pointer, no PGO) runs 24 game positions (game_regret_set,
# every 625th row: all phases) to depth 12 under perf, SEARCH_ARM 1 and 0; self time per symbol, top 45 each.
# perf writes its data to /tmp (it cannot write to the DrvFs mount: "Bad address"). Fixed depth (the shares are what is read). Box otherwise idle. Never edit this file or the binary while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
TB="$ND/build_uci/chess_engine_uci_sv2prof"
O=/mnt/e/chess_data/sv2_profile
PERF=/usr/lib/linux-tools/5.15.0-203-generic/perf
cd "$ND" || exit 1
mkdir -p "$O"
echo "[q206] $(date '+%Y-%m-%d %H:%M:%S') start"
[ -x "$PERF" ] || { echo "[q206] ☠️ no perf at $PERF — QUEUE 206 ABORTED"; exit 1; }
# (the profiling copy is reused when it exists; its bench is checked below)
[ -x "$TB" ] || EXTRA_FLAGS="-g -fno-omit-frame-pointer" OUT=chess_engine_uci_sv2prof bash selfplay/build_uci.sh > "$O/build.out" 2>&1
[ -x "$TB" ] || { echo "[q206] ☠️ profiling build failed — QUEUE 206 ABORTED"; exit 1; }
echo "[q206] profiling build: $(printf 'bench\nquit\n' | SEARCH_ARM=1 "$TB" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+') (want 247 / 39861964)"
awk -F, 'NR > 1 && (NR - 2) % 625 == 0 { print $1 " | " $3 }' diagnostics/ks_sets/game_regret_set.csv > "$O/game24.txt"
for arm in 1 0; do
	printf "bench $O/game24.txt 12\nquit\n" | SEARCH_ARM=$arm "$PERF" record -e cpu-clock:u -o "/tmp/sv2_perf_arm$arm.data" "$TB" > "$O/bench_arm$arm.txt" 2> "$O/perf_arm$arm.log"
	"$PERF" report -i "/tmp/sv2_perf_arm$arm.data" --stdio --no-children --sort symbol 2>/dev/null | grep -E '^ +[0-9]' | head -45 > "$O/self_arm$arm.txt"
	echo "[q206] arm $arm: $(grep -oE 'solved [0-9]+/[0-9]+ nodes [0-9]+ time [0-9]+' "$O/bench_arm$arm.txt")"
done
echo "[q206] $(date) QUEUE 206 DONE"
