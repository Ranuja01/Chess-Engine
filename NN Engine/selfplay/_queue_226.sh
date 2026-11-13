#!/bin/bash
# 2026-11-13 ~13:00, queue #226: S3 root R1, phase A (owner 11-13: "ok lets go"; skeptical of the pre-search, wants to "cleanly
# remove or reduce reliance" by measurement). Design: dev_notes/active/S3-ROOT-AUDIT-2026-11-13.md §6.2-6.4. Existing knobs only,
# no code change: what each root part costs and does on TODAY's engine (gen 7.3, v2 eval; the September numbers were v1-era).
# Fixed depth 10, LONG_FORMAT, book off, V2_PRESET=shipped, the production Cython module, one process per setting, serial.
# Venues (memory presearch-economics: WAC = solve sanity only; STS = accuracy; the quiet corpus = the node judge):
#   WAC 300 (solves, nodes) · STS300 (score, nodes) · quiet corpus cploss 100 (median nodes) · game set v2 100 (median nodes);
#   counters from each run's stderr, last line = suite totals: [node_split] [tail_mode] [razor_audit] [aspiration].
# Arms:
#   A0 defaults
#   A1 PRESEARCH_TAIL_MODE=3: the same pre-search, its root SCORES replaced by the fill value (= the score channel alone)
#   A2 ENABLE_ROOT_RAZOR=0: the pre-search on, nothing skipped (razoring's node saving and accuracy cost)
#   A3 ENABLE_PRESEARCH_SUBSET=1: the pre-search skips the prefix the previous main search reached (its warming there removed)
#   A4 the September best "cascading tables" arm: pre-search off, razor off, off-path sorted by provenance tiers + root table
#   A5 ROOT_PRESEARCH_REDUCTION=2: the pre-search one ply shallower
# REGISTERED predictions (directions, not verdicts; no verdict rides on fixed-depth nodes): A0 pre-search 30-45% of nodes;
# A1 nodes within ±5% of A0; A2 nodes +15-40%; A3 nodes higher than A0; A4 nodes +0-15%; A5 nodes −5 to +10%.
# Stop: touch /mnt/e/chess_data/q226.stop (between runs). Never edit this file or rebuild the Cython module while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
W=/mnt/e/chess_data/r1a
STOP=/mnt/e/chess_data/q226.stop
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"; mkdir -p "$W"
BASE="MAX_DEPTH=10 USE_OPENING_BOOK=0 PRESET=LONG_FORMAT V2_PRESET=shipped"
declare -A ARM
ARM[A0]=""
ARM[A1]="PRESEARCH_TAIL_MODE=3"
ARM[A2]="ENABLE_ROOT_RAZOR=0"
ARM[A3]="ENABLE_PRESEARCH_SUBSET=1"
ARM[A4]="ENABLE_ROOT_PRESEARCH=0 ENABLE_ROOT_RAZOR=0 ENABLE_PRESEARCH_OFF_SORT=1 PRESEARCH_OFF_SORT_MODE=2 ENABLE_ROOT_TABLE=1"
ARM[A5]="ROOT_PRESEARCH_REDUCTION=2"
last() { grep -E "^\[$1\]" "$2" | tail -1; }
counters() { echo "$(last node_split "$1") | $(last tail_mode "$1" | grep -oE 'razor_fires=[0-9]+') | $(last razor_audit "$1") | $(last aspiration "$1")"; }
echo "[q226] $(date '+%Y-%m-%d %H:%M:%S') start"
for a in A0 A1 A2 A3 A4 A5; do
	k="${ARM[$a]}"
	[ -e "$STOP" ] && { echo "[q226] $(date) stop file — QUEUE 226 DONE"; exit 0; }
	echo "[q226] $(date '+%H:%M') == $a ${k:-(defaults)}"
	# WAC (the runner's wac sub keeps stderr in /tmp/wac_<tag>.err; tags are unique per arm)
	echo "[q226]   WAC: $(bash "$R" wac r1a_$a V2_PRESET=shipped $k 2>&1 | grep -E '^SOLVED|^NODES|^EBF' | tr '\n' ' ')"
	cp /tmp/wac_r1a_$a.err "$W/wac_$a.err" 2>/dev/null
	echo "[q226]     $(counters "$W/wac_$a.err")"
	# STS300
	env $BASE $k bash "$R" pyrun diagnostics/sts_test.py sts300.epd r1a_$a > "$W/sts_$a.out" 2> "$W/sts_$a.err"
	echo "[q226]   STS: $(grep -E 'STS score' "$W/sts_$a.out" | tail -1)"
	echo "[q226]     $(counters "$W/sts_$a.err")"
	# quiet corpus and game set, median nodes at depth 10
	bash "$R" pyrun diagnostics/depth_nps_bench.py $BASE $k --n 100 > "$W/quiet_$a.out" 2> "$W/quiet_$a.err"
	echo "[q226]   quiet: $(grep -E 'NPS: median' "$W/quiet_$a.out" | grep -oE 'median nodes [0-9,]+')"
	echo "[q226]     $(counters "$W/quiet_$a.err")"
	bash "$R" pyrun diagnostics/depth_nps_bench.py $BASE $k --n 100 --corpus diagnostics/ks_sets/game_regret_set_v2.csv > "$W/game_$a.out" 2> "$W/game_$a.err"
	echo "[q226]   game: $(grep -E 'NPS: median' "$W/game_$a.out" | grep -oE 'median nodes [0-9,]+')"
	echo "[q226]     $(counters "$W/game_$a.err")"
done
echo "[q226] $(date) QUEUE 226 DONE"
