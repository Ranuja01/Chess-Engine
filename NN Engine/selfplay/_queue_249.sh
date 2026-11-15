#!/bin/bash
# 2026-11-15 ~13:00, queue #249: phase H-S, the first eval speed batch (from the 11-15 read-only audit of the gen 7.4 profile),
# plus the rest of #247's correctness screens. Nothing timed: it starts when queue #248 ends (its games finish or are paused
# for the owner's gaming) and is safe while the owner games (one process at a time, fixed work).
#  1 Test build chess_engine_uci_e1test (plain, no PGO: both arms of every comparison run on this one binary) of the sources
#    with three result-identical items, each behind a switch defaulting to off:
#      EVAL_MAT_CACHE 1  the piece-count part of the eval (non-pawn material, phase, census, Kaufman) cached per exact count key
#                        (2 = recompute and abort on any difference)
#      EVAL_FAST bit 1   kfl_cells: the king's nearest-pawn distance from cumulative distance rings, not a loop over pawns
#      EVAL_FAST bit 2   the C3 blocks call their side functions directly, not through a function pointer
#    GATES (all must hold, else no measurement): bench 250 / 37,320,382 with every combination below; EVAL_MAT_CACHE=2 (verify)
#    with everything on completes the bench and `bench game12 10` without an abort; v1 bench 250 / 35,310,775-778; gen 7.3 knob
#    SV2_VERIFY_DEEPER=0 249 / 37,279,382 with everything on.
#  2 cachegrind instructions (no cache simulation) on `bench game12 10` (the same 2,372,218 nodes): base, each item alone, all
#    three. REGISTERED: an item is counted when its instructions fall by more than 0.2% with the same nodes; the audit's
#    estimates (INFERRED from the per-line profile): material cache −4.5%, rings −2.6%, direct calls −0.3%. Instructions are not
#    time: a PGO build and a pinned bench on a quiet box come next for whatever counts.
#  3 #247's remaining correctness screens on the production module (STS1500 d10 + WAC; base gen 7.4 STS1500 9,945, WAC 250;
#    bar STS1500 >= 9,875 and WAC >= 247): C4 SV2_HIST_ALLNODE_GATE=1 · C5 SV2_MALUS_SEARCHED_ONLY=1 · C2a FIX_RESEARCH=1 ·
#    C2b FIX_MIN_SCOUT=1 FIX_MIN_RD1=1 · C2c all three.
# Stop: touch /mnt/e/chess_data/q249.stop (between runs). Never edit the sources before "build done".
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
W=/mnt/e/chess_data/q249
G=/mnt/e/chess_data/sv2_speed/game12.txt
T="$ND/build_uci/chess_engine_uci_e1test"
STOP=/mnt/e/chess_data/q249.stop
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"; mkdir -p "$W"
show() { sed "s/^/[q249]   /"; }
stopped() { [ -e "$STOP" ] && { echo "[q249] $(date) stop file before $1 — QUEUE 249 DONE"; exit 0; }; }
fp() { local k=""; [ -n "$1" ] && k="setoption name Knobs value $1"; printf '%s\nbench\nquit\n' "$k" | "$T" 2>&1 | grep -oE 'solved [0-9]+/300 nodes [0-9]+|mismatch.*' | head -1; }
irefs() { grep -E 'I *refs' "$1" | grep -oE '[0-9,]+$' | tr -d ,; }
echo "[q249] $(date '+%Y-%m-%d %H:%M:%S') queued; waiting for #248 to end"
until grep -qE 'QUEUE 248 (DONE|ABORTED)|☠' /mnt/e/chess_data/q248.log; do stopped "the start"; sleep 30; done
sleep 30
if pgrep -f "[f]astchess-linux-x86-64/fastchess" >/dev/null; then echo "[q249] ☠️ games still running — QUEUE 249 ABORTED"; exit 1; fi
echo "[q249] $(date '+%H:%M') start"

# ── 1 build and gates ─────────────────────────────────────────────────────────────────────────────────────────────────────
OUT=chess_engine_uci_e1test nice -n 10 bash selfplay/build_uci.sh > "$W/build.out" 2>&1 || { echo "[q249] ☠️ build failed — QUEUE 249 ABORTED"; grep -iE 'error' "$W/build.out" | head -8 | show; exit 1; }
echo "[q249] $(date '+%H:%M') build done — the sources are free to edit"
ok=1
for k in "" "EVAL_MAT_CACHE=1" "EVAL_MAT_CACHE=2" "EVAL_FAST=1" "EVAL_FAST=2" "EVAL_FAST=3" "EVAL_MAT_CACHE=1,EVAL_FAST=3" "EVAL_MAT_CACHE=2,EVAL_FAST=3"; do
	f=$(fp "$k"); echo "[q249] bench [${k:-default}]: $f"
	[ "$f" = "solved 250/300 nodes 37320382" ] || ok=0
done
d0=$(fp "EVAL_MAT_CACHE=1,EVAL_FAST=3,SV2_VERIFY_DEEPER=0"); echo "[q249] gen 7.3 knob with all on (want 249 / 37279382): $d0"
[ "$d0" = "solved 249/300 nodes 37279382" ] || ok=0
v1=$(printf 'setoption name EvalArm value 0\nsetoption name Knobs value EVAL_MAT_CACHE=1,EVAL_FAST=3\nbench\nquit\n' | "$T" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+')
echo "[q249] v1 with all on (250 / 35310775-778): $v1"
echo "$v1" | grep -qE 'solved 250/300 nodes 3531077[5-8]' || ok=0
gv=$(printf "setoption name Knobs value EVAL_MAT_CACHE=2,EVAL_FAST=3\nbench $G 10\nquit\n" | "$T" 2>&1 | grep -oE 'solved [0-9]+/[0-9]+ nodes [0-9]+|mismatch.*' | head -1)
gb=$(printf "bench $G 10\nquit\n" | "$T" 2>&1 | grep -oE 'solved [0-9]+/[0-9]+ nodes [0-9]+' | head -1)
echo "[q249] game12 d10: verify + all on [$gv] · default [$gb]"
[ -n "$gb" ] && [ "$gv" = "$gb" ] || ok=0
[ $ok = 1 ] || { echo "[q249] ☠️ a gate failed — no measurement (the screens below still run)"; }

# ── 2 instructions ────────────────────────────────────────────────────────────────────────────────────────────────────────
if [ $ok = 1 ]; then
	for arm in base:"" mat:"EVAL_MAT_CACHE=1" rings:"EVAL_FAST=1" direct:"EVAL_FAST=2" all:"EVAL_MAT_CACHE=1,EVAL_FAST=3"; do
		stopped "cachegrind ${arm%%:*}"
		n=${arm%%:*}; k=${arm#*:}; kl=""; [ -n "$k" ] && kl="setoption name Knobs value $k"
		printf "%s\nbench $G 10\nquit\n" "$kl" | nice -n 10 valgrind --tool=cachegrind --cache-sim=no \
			--cachegrind-out-file="$W/cg_$n.out" "$T" > "$W/bench_$n.txt" 2> "$W/cg_$n.log"
		echo "[q249] cachegrind $n [${k:-default}]: $(grep -oE 'solved [0-9]+/[0-9]+ nodes [0-9]+' "$W/bench_$n.txt") · Ir $(irefs "$W/cg_$n.log")"
	done
	b=$(irefs "$W/cg_base.log")
	for n in mat rings direct all; do
		echo "[q249]   $n vs base: $(awk -v a="$b" -v x="$(irefs "$W/cg_$n.log")" 'BEGIN { printf "%+.3f%%", 100 * (x - a) / a }')"
	done
fi

# ── 3 the rest of #247's screens ──────────────────────────────────────────────────────────────────────────────────────────
BASE="MAX_DEPTH=10 USE_OPENING_BOOK=0 PRESET=LONG_FORMAT V2_PRESET=shipped"
declare -A ARM
ARM[C4]="SV2_HIST_ALLNODE_GATE=1"; ARM[C5]="SV2_MALUS_SEARCHED_ONLY=1"
ARM[C2a]="FIX_RESEARCH=1"; ARM[C2b]="FIX_MIN_SCOUT=1 FIX_MIN_RD1=1"; ARM[C2c]="FIX_RESEARCH=1 FIX_MIN_SCOUT=1 FIX_MIN_RD1=1"
for a in C4 C5 C2a C2b C2c; do
	stopped "$a"
	w=$(bash "$R" wac q249_wac_$a V2_PRESET=shipped ${ARM[$a]} 2>&1 | grep -E '^SOLVED|^NODES' | tr '\n' ' ')
	env $BASE ${ARM[$a]} nice -n 10 bash "$R" pyrun diagnostics/sts_test.py STS1-STS15_LAN_v3.epd q249_full_$a > "$W/full_$a.out" 2> "$W/full_$a.err"
	echo "[q249] $(date '+%H:%M') $a (${ARM[$a]}): WAC $w· $(grep -E 'STS score' "$W/full_$a.out" | tail -1)"
done
echo "[q249] base for the screens: gen 7.4 STS1500 9,945, WAC 250 / 37,320,382; bar 9,875 and WAC 247"
echo "[q249] $(date) QUEUE 249 DONE"
