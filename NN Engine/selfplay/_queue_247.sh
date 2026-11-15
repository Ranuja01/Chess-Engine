#!/bin/bash
# 2026-11-15 ~11:45, queue #247: phase H (search v2 plan §0b), the light half, safe while the owner games (one process at a
# time, fixed work, nothing timed).
#  1 H-S1: a fresh profile of gen 7.4. Copy chess_engine_uci_g74prof = production flags + the gen 7.4 PGO profile + -g
#    -fno-omit-frame-pointer (gate: bench 250 / 37,320,382). cachegrind with cache and branch simulation on `bench game12 10`
#    (12 game positions, depth 10), cg_annotate per function → /mnt/e/chess_data/q247/cg_annotate.txt. Descriptive: it ranks the
#    speed items, no rule.
#  2 H-C screens on STS1500 (STS1-STS15_LAN_v3.epd, d10, LONG_FORMAT) and WAC, base = gen 7.4 (STS1500 9,945 on #245's D7 arm,
#    same settings; neutral arm 9,927):
#    C3 SV2_STATIC_CUT_STORE=1 · C4 SV2_HIST_ALLNODE_GATE=1 · C5 SV2_MALUS_SEARCHED_ONLY=1 ·
#    C2 as a 2x2: C2a FIX_RESEARCH=1 · C2b FIX_MIN_SCOUT=1 FIX_MIN_RD1=1 · C2c all three.
#    REGISTERED: an arm goes to games if STS1500 >= 9,875 (base − 70, the 1,500-position noise per arm pair) and WAC >= 247;
#    below 9,875 = a possible real STS cost (per-position sign test next, as #245); WAC < 247 = tactical veto. Nodes are
#    reported, never judged here (time decides; memory the-nps-gap-is-mostly-not-eval).
# Stop: touch /mnt/e/chess_data/q247.stop (between runs). Never edit the sources before "build done".
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
W=/mnt/e/chess_data/q247
P=/mnt/e/chess_data/pgo_g74
G=/mnt/e/chess_data/sv2_speed/game12.txt
PB="$ND/build_uci/chess_engine_uci_g74prof"
STOP=/mnt/e/chess_data/q247.stop
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"; mkdir -p "$W"
show() { sed "s/^/[q247]   /"; }
stopped() { [ -e "$STOP" ] && { echo "[q247] $(date) stop file before $1 — QUEUE 247 DONE"; exit 0; }; }
echo "[q247] $(date '+%Y-%m-%d %H:%M:%S') start"

# ── 1 profile ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
OUT=chess_engine_uci_g74prof PGO=use PGO_DIR=$P EXTRA_FLAGS="-g -fno-omit-frame-pointer" nice -n 10 bash selfplay/build_uci.sh > "$W/prof_build.out" 2>&1 \
	|| { echo "[q247] ☠️ profiling build failed"; tail -5 "$W/prof_build.out" | show; }
echo "[q247] $(date '+%H:%M') build done — the sources are free to edit"
if [ -x "$PB" ]; then
	f=$(printf 'bench\nquit\n' | "$PB" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+')
	echo "[q247] profiling copy bench (want 250 / 37320382): [$f]"
	if [ "$f" = "solved 250/300 nodes 37320382" ]; then
		printf "bench $G 10\nquit\n" | nice -n 10 valgrind --tool=cachegrind --cache-sim=yes --branch-sim=yes \
			--cachegrind-out-file="$W/cg.out" "$PB" > "$W/cg_bench.txt" 2> "$W/cg.log"
		echo "[q247] $(date '+%H:%M') cachegrind: $(grep -oE 'solved [0-9]+/[0-9]+ nodes [0-9]+' "$W/cg_bench.txt")"
		grep -E 'I *refs|D1 *misses|LLd *misses|Mispred' "$W/cg.log" | sed 's/^==[0-9]*== //' | show
		cg_annotate "$W/cg.out" > "$W/cg_annotate.txt" 2>&1
		echo "[q247] per-function report: $W/cg_annotate.txt ($(wc -l < "$W/cg_annotate.txt") lines)"
	else
		echo "[q247] ☠️ profiling copy moved a node — profile skipped"
	fi
fi

# ── 2 STS1500 + WAC screens ───────────────────────────────────────────────────────────────────────────────────────────────
BASE="MAX_DEPTH=10 USE_OPENING_BOOK=0 PRESET=LONG_FORMAT V2_PRESET=shipped"
declare -A ARM
ARM[C3]="SV2_STATIC_CUT_STORE=1"; ARM[C4]="SV2_HIST_ALLNODE_GATE=1"; ARM[C5]="SV2_MALUS_SEARCHED_ONLY=1"
ARM[C2a]="FIX_RESEARCH=1"; ARM[C2b]="FIX_MIN_SCOUT=1 FIX_MIN_RD1=1"; ARM[C2c]="FIX_RESEARCH=1 FIX_MIN_SCOUT=1 FIX_MIN_RD1=1"
for a in C3 C4 C5 C2a C2b C2c; do
	stopped "$a"
	w=$(bash "$R" wac q247_wac_$a V2_PRESET=shipped ${ARM[$a]} 2>&1 | grep -E '^SOLVED|^NODES' | tr '\n' ' ')
	env $BASE ${ARM[$a]} nice -n 10 bash "$R" pyrun diagnostics/sts_test.py STS1-STS15_LAN_v3.epd q247_full_$a > "$W/full_$a.out" 2> "$W/full_$a.err"
	echo "[q247] $(date '+%H:%M') $a (${ARM[$a]}): WAC $w· $(grep -E 'STS score' "$W/full_$a.out" | tail -1)"
done
echo "[q247] base for the reads: gen 7.4 STS1500 9,945 (#245 r12a_full_D7), WAC 250 / 37,320,382; bar 9,875 and WAC 247"
echo "[q247] $(date) QUEUE 247 DONE"
