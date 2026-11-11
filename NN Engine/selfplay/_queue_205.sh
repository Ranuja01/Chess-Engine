#!/bin/bash
# 2026-11-11 ~16:10, queue #205: owner-cpp-ideas §15.2, step 2. #203: Memcheck CLEAN (0 errors) and the v1 bench reads
# 35,310,775 on every run with ASLR off but 777 / 778 with it on ⇒ the wobble is address-dependent. Where is it? A test UCI
# build with BENCH_PER_POS (uci_main.cpp: each position's node count) runs the v1 bench 3x with ASLR on and 2x with it off;
# the first position whose count moves between runs locates the dependence (shared search code, or v1-only eval code).
# Registered: v2 (default) per-position counts must not move at all (its bench never wobbles); the v1 runs differ at one or a
# few positions. Fixed depth, one process at a time, nothing timed. Never edit this file or the binary while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
TB="$ND/build_uci/chess_engine_uci_pp"
O=/mnt/e/chess_data/wobble
cd "$ND" || exit 1
mkdir -p "$O"
echo "[q205] $(date '+%Y-%m-%d %H:%M:%S') start"
rm -f "$TB"
OUT=chess_engine_uci_pp bash selfplay/build_uci.sh > "$O/build.out" 2>&1
[ -x "$TB" ] || { echo "[q205] ☠️ test build failed — QUEUE 205 ABORTED"; tail -5 build_uci/build.log; exit 1; }
fp=$(printf 'bench\nquit\n' | "$TB" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+')
echo "[q205] test build, default (want 247 / 39861964): $fp"
pp() { printf 'bench\nquit\n' | env BENCH_PER_POS=1 "$@" 2>/dev/null | grep -oE 'pos [0-9]+ nodes [0-9]+' | awk '{print $2, $4}'; }
for r in 1 2 3; do pp EVAL_ARM=0 "$TB" > "$O/v1_aslr_on_$r.txt"; done
for r in 1 2; do pp EVAL_ARM=0 setarch x86_64 -R "$TB" > "$O/v1_aslr_off_$r.txt"; done
for r in 1 2; do pp "$TB" > "$O/v2_aslr_on_$r.txt"; done
for f in "$O"/v1_*.txt "$O"/v2_*.txt; do echo "[q205] $(basename "$f"): $(awk '{s += $2} END {print NR " positions, " s " nodes"}' "$f")"; done
echo "[q205] positions whose count differs from v1_aslr_off_1:"
for f in "$O"/v1_aslr_on_*.txt "$O"/v1_aslr_off_2.txt; do
	echo "[q205]   $(basename "$f"): $(paste "$O/v1_aslr_off_1.txt" "$f" | awk '$2 != $4 {printf "pos %d (%d vs %d) ", $1, $2, $4}')"
done
echo "[q205]   v2_aslr_on_2 vs _1: $(paste "$O/v2_aslr_on_1.txt" "$O/v2_aslr_on_2.txt" | awk '$2 != $4 {printf "pos %d ", $1}')"
echo "[q205] $(date) QUEUE 205 DONE"
