#!/bin/bash
# 2026-11-17 ~21:30, queue #265: SORT_KEYED (H-S speed item 1, result-identical): the move-ordering sort on unique 64-bit keys
# with a compare-exchange network per run of 8 and branch-free merges. Gen 7.6's profile (#258 on gen 7.5): the insertion sort
# and merge lines of stableSortScoredDesc carry ~12.2 M of ~100 M branch mispredictions on `bench game12 10`.
# Fixed work only (the owner's gaming window): a plain test build, the gates, two cachegrind runs side by side.
#  1 plain build chess_engine_uci_sktest from the current sources (gen 7.6 + SORT_KEYED, default 0). Production untouched.
#  2 GATES: default 246 / 35,138,434 == SORT_KEYED=1 == SORT_KEYED=1,SCRATCH_REUSE=2 (every list checked against
#    std::stable_sort, abort on a difference); SORT_KEYED=1,SV2_VERIFY_PROMOTE=0 = gen 7.5 252 / 35,235,460; v1 with SORT_KEYED=1
#    250 / 35,310,775-778; game12 d10 default == SORT_KEYED=1 == SORT_KEYED=1,SCRATCH_REUSE=2.
#  3 cachegrind (cache + branch simulation) on `bench game12 10`, SORT_KEYED=0 and =1 in parallel.
# REGISTERED before the run: mispredictions −8 … −12% (the sort's 12.2 M mostly gone); instructions −1 … +2% (the network and the
# key build/decode against the insertion sort's moves). Time to be read later on a quiet box (pinned bench), prediction −1.5 … −4%.
# Stop: touch /mnt/e/chess_data/q265.stop.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
W=/mnt/e/chess_data/q265
G=/mnt/e/chess_data/sv2_speed/game12.txt
B="$ND/build_uci/chess_engine_uci_sktest"
STOP=/mnt/e/chess_data/q265.stop
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"; mkdir -p "$W"
show() { sed "s/^/[q265]   /"; }
fp() { local k=""; [ -n "$2" ] && k="setoption name Knobs value $2"; printf '%s\nbench\nquit\n' "$k" | "$1" 2>&1 | grep -oE 'solved [0-9]+/300 nodes [0-9]+|mismatch.*|abort.*' | head -1; }
g12() { local k=""; [ -n "$1" ] && k="setoption name Knobs value $1"; printf "%s\nbench $G 10\nquit\n" "$k" | "$B" 2>&1 | grep -oE 'solved [0-9]+/[0-9]+ nodes [0-9]+|mismatch.*|abort.*' | head -1; }
echo "[q265] $(date '+%Y-%m-%d %H:%M:%S') start"
pgrep -f "[f]astchess-linux|[b]uild_uci.sh|[v]algrind" >/dev/null && { echo "[q265] ☠️ games, a build or valgrind running — QUEUE 265 ABORTED"; exit 1; }
OUT=chess_engine_uci_sktest nice -n 10 bash selfplay/build_uci.sh > "$W/build.out" 2>&1 \
	|| { echo "[q265] ☠️ build failed — QUEUE 265 ABORTED"; grep -iE 'error' "$W/build.out" | head -8 | show; exit 1; }
echo "[q265] $(date '+%H:%M') build done — the sources are free to edit"
a=$(fp "$B" ""); k=$(fp "$B" "SORT_KEYED=1"); v=$(fp "$B" "SORT_KEYED=1,SCRATCH_REUSE=2"); p=$(fp "$B" "SORT_KEYED=1,SV2_VERIFY_PROMOTE=0")
v1=$(printf 'setoption name EvalArm value 0\nsetoption name Knobs value SORT_KEYED=1\nbench\nquit\n' | "$B" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+')
echo "[q265] default [$a] (246 / 35138434) · SORT_KEYED=1 [$k] · +SCRATCH_REUSE=2 verify [$v] · +SV2_VERIFY_PROMOTE=0 [$p] (252 / 35235460) · v1 [$v1]"
ga=$(g12 ""); gk=$(g12 "SORT_KEYED=1"); gv=$(g12 "SORT_KEYED=1,SCRATCH_REUSE=2")
echo "[q265] game12 d10: default [$ga] · SORT_KEYED=1 [$gk] · verify [$gv]"
if [ "$a" = "solved 246/300 nodes 35138434" ] && [ "$k" = "$a" ] && [ "$v" = "$a" ] && [ "$p" = "solved 252/300 nodes 35235460" ] \
	&& echo "$v1" | grep -qE 'solved 250/300 nodes 3531077[5-8]' && [ -n "$ga" ] && [ "$gk" = "$ga" ] && [ "$gv" = "$ga" ]; then
	echo "[q265] ✅ every gate equal — SORT_KEYED is result-identical on every set"
else
	echo "[q265] ☠️ a gate differs — SORT_KEYED is NOT result-identical; cachegrind skipped — QUEUE 265 DONE"; exit 1
fi
[ -e "$STOP" ] && { echo "[q265] stop file — QUEUE 265 DONE"; exit 0; }
for s in 0 1; do
	( printf "setoption name Knobs value SORT_KEYED=$s\nbench $G 10\nquit\n" | nice -n 10 valgrind --tool=cachegrind --cache-sim=yes --branch-sim=yes \
		--cachegrind-out-file="$W/cg_sk$s.out" "$B" > "$W/cg_sk$s.txt" 2> "$W/cg_sk$s.log" ) &
done
wait
for s in 0 1; do
	echo "[q265] $(date '+%H:%M') SORT_KEYED=$s: $(grep -oE 'solved [0-9]+/[0-9]+ nodes [0-9]+' "$W/cg_sk$s.txt")"
	grep -E 'I *refs|Mispred' "$W/cg_sk$s.log" | sed 's/^==[0-9]*== //' | show
done
ir() { grep -E 'I *refs' "$W/cg_sk$1.log" | grep -oE '[0-9,]+$' | tr -d ','; }
bm() { grep -E 'Mispredicts:' "$W/cg_sk$1.log" | head -1 | awk '{print $3}' | tr -d ','; }
echo "[q265] SORT_KEYED=1 vs 0: instructions $(awk -v a=$(ir 1) -v b=$(ir 0) 'BEGIN { printf "%+.3f%%", 100 * (a / b - 1) }') · mispredictions $(awk -v a=$(bm 1) -v b=$(bm 0) 'BEGIN { printf "%+.2f%% (%d → %d)", 100 * (a / b - 1), b, a }') (registered: mispredictions −8 … −12%, instructions −1 … +2%)"
cg_annotate "$W/cg_sk1.out" > "$W/cg_sk1_annotate.txt" 2>&1
echo "[q265] $(date) QUEUE 265 DONE"
