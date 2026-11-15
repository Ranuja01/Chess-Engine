#!/bin/bash
# 2026-11-15 ~14:30, queue #250: phase H-S, eval speed batch 2 (the audit's items 2 and 5a), nothing timed; starts when #249 ends.
#   EVAL_FAST bit 4  build_side_attacks as one loop per piece type, type fixed at compile time (attacks_mask's switch folds)
#   EVAL_FAST bit 8  PX skips a side with no passed pawn (its 51 counts are all zero, so its score is exactly 0)
#  1 Test build chess_engine_uci_e2test (plain). GATES: bench 250 / 37,320,382 for EVAL_FAST 4, 8, 12, and with batch 1
#    (EVAL_MAT_CACHE=1,EVAL_FAST=15, and the verify mode EVAL_MAT_CACHE=2,EVAL_FAST=15); `bench game12 10` equal to default
#    with everything on; gen 7.3 knob 249 / 37,279,382 and v1 250 / 35,310,775-778 with everything on.
#  2 cachegrind instructions on `bench game12 10`, base = batch 1 on (EVAL_MAT_CACHE=1,EVAL_FAST=3): +bit 4, +bit 8, +both.
#    REGISTERED: an item counts when it removes more than 0.2% with the same nodes. Audit estimates (INFERRED): bit 4 about 2%
#    of instructions plus most indirect mispredictions (instructions cannot see those; the PGO pinned bench will), bit 8 ~0.5%.
# Stop: touch /mnt/e/chess_data/q250.stop (between runs). Never edit the sources before "build done".
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
W=/mnt/e/chess_data/q250
G=/mnt/e/chess_data/sv2_speed/game12.txt
T="$ND/build_uci/chess_engine_uci_e2test"
STOP=/mnt/e/chess_data/q250.stop
cd "$ND" || exit 1
rm -f "$STOP"; mkdir -p "$W"
stopped() { [ -e "$STOP" ] && { echo "[q250] $(date) stop file before $1 — QUEUE 250 DONE"; exit 0; }; }
fp() { local k=""; [ -n "$1" ] && k="setoption name Knobs value $1"; printf '%s\nbench\nquit\n' "$k" | "$T" 2>&1 | grep -oE 'solved [0-9]+/300 nodes [0-9]+|mismatch.*' | head -1; }
irefs() { grep -E 'I *refs' "$1" | grep -oE '[0-9,]+$' | tr -d ,; }
echo "[q250] $(date '+%Y-%m-%d %H:%M:%S') queued; waiting for #249 to end"
until grep -qE 'QUEUE 249 (DONE|ABORTED)' /mnt/e/chess_data/q249.log; do stopped "the start"; sleep 30; done
sleep 20
echo "[q250] $(date '+%H:%M') start"
OUT=chess_engine_uci_e2test nice -n 10 bash selfplay/build_uci.sh > "$W/build.out" 2>&1 || { echo "[q250] ☠️ build failed — QUEUE 250 ABORTED"; grep -iE 'error' "$W/build.out" | head -8 | sed 's/^/[q250]   /'; exit 1; }
echo "[q250] $(date '+%H:%M') build done — the sources are free to edit"
ok=1
for k in "" "EVAL_FAST=4" "EVAL_FAST=8" "EVAL_FAST=12" "EVAL_MAT_CACHE=1,EVAL_FAST=15" "EVAL_MAT_CACHE=2,EVAL_FAST=15"; do
	f=$(fp "$k"); echo "[q250] bench [${k:-default}]: $f"
	[ "$f" = "solved 250/300 nodes 37320382" ] || ok=0
done
d0=$(fp "EVAL_MAT_CACHE=1,EVAL_FAST=15,SV2_VERIFY_DEEPER=0"); echo "[q250] gen 7.3 knob, all on (249 / 37279382): $d0"
[ "$d0" = "solved 249/300 nodes 37279382" ] || ok=0
v1=$(printf 'setoption name EvalArm value 0\nsetoption name Knobs value EVAL_MAT_CACHE=1,EVAL_FAST=15\nbench\nquit\n' | "$T" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+')
echo "[q250] v1, all on (250 / 35310775-778): $v1"
echo "$v1" | grep -qE 'solved 250/300 nodes 3531077[5-8]' || ok=0
gv=$(printf "setoption name Knobs value EVAL_MAT_CACHE=2,EVAL_FAST=15\nbench $G 10\nquit\n" | "$T" 2>&1 | grep -oE 'solved [0-9]+/[0-9]+ nodes [0-9]+|mismatch.*' | head -1)
gb=$(printf "bench $G 10\nquit\n" | "$T" 2>&1 | grep -oE 'solved [0-9]+/[0-9]+ nodes [0-9]+' | head -1)
echo "[q250] game12 d10: verify + all on [$gv] · default [$gb]"
[ -n "$gb" ] && [ "$gv" = "$gb" ] || ok=0
[ $ok = 1 ] || { echo "[q250] ☠️ a gate failed — no measurement — QUEUE 250 DONE"; exit 1; }
for arm in b1:"EVAL_MAT_CACHE=1,EVAL_FAST=3" b4:"EVAL_MAT_CACHE=1,EVAL_FAST=7" b8:"EVAL_MAT_CACHE=1,EVAL_FAST=11" b12:"EVAL_MAT_CACHE=1,EVAL_FAST=15"; do
	stopped "cachegrind ${arm%%:*}"
	n=${arm%%:*}; k=${arm#*:}
	printf "setoption name Knobs value %s\nbench $G 10\nquit\n" "$k" | nice -n 10 valgrind --tool=cachegrind --cache-sim=no \
		--cachegrind-out-file="$W/cg_$n.out" "$T" > "$W/bench_$n.txt" 2> "$W/cg_$n.log"
	echo "[q250] cachegrind $n [$k]: $(grep -oE 'solved [0-9]+/[0-9]+ nodes [0-9]+' "$W/bench_$n.txt") · Ir $(irefs "$W/cg_$n.log")"
done
b=$(irefs "$W/cg_b1.log")
for n in b4 b8 b12; do echo "[q250]   $n vs batch 1: $(awk -v a="$b" -v x="$(irefs "$W/cg_$n.log")" 'BEGIN { printf "%+.3f%%", 100 * (x - a) / a }')"; done
echo "[q250] $(date) QUEUE 250 DONE"
