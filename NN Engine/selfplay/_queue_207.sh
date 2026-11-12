#!/bin/bash
# 2026-11-11 ~20:55, queue #207 (builds now from the S1 sources; the timed steps wait until 00:30, after the owner's evening): search v2 watch
# SW3, part 2 (plan §7e), and the S1 games smoke agreed with the owner in place of S1's 2,000-game read.
#  1 Two PGO builds of today's sources, each trained on its own arm (the arm switch changes which code the profile sees):
#    sv2_pgo0 trained and run with SEARCH_ARM=0, sv2_pgo1 trained and run with SEARCH_ARM=1. Gate on both: bench 247 / 39,861,964
#    under both arms (layout must not change a result).
#  2 Real-play speed: _ebf_per_iter.py, 40 positions per phase from game_regret_set (seed 1), depth 13, the two binaries
#    alternated three times (arm0, arm1, arm0, ...). Nodes per iteration must be equal between the arms (same tree).
#    REGISTERED: arm 1's mean time to depth 13 within +1% of arm 0's = "not slower" (the S1 speed gate); prediction −1 to −4%
#    (#204: −1.87% instructions; #206 one run −4.8%).
#  3 Games smoke, 10+0.1, 300 games (UHO, random order, seed 207), arm 1 (sv2_pgo1, Knobs SEARCH_ARM=1) vs arm 0 (sv2_pgo0).
#    REGISTERED: 0 "have option" warnings, no disconnects or stalls, time forfeits no more than arm 0's, Elo inside ±2σ of 0
#    or positive. It also exercises the clock-abort paths that SW2's node-limit games could not.
# Stop: touch /mnt/e/chess_data/q207.stop (checked between steps). ≤ 3 engine processes at a time. Never edit this file, the
# tools or the sources while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
W=/mnt/e/chess_data/sv2_sw3
STOP=/mnt/e/chess_data/q207.stop
START_AT=$(date -d '2026-11-12 00:30' +%s)
LATEST=$(date -d '2026-11-12 09:30' +%s)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"; mkdir -p "$W"
show() { sed "s/^/[q207]   /"; }
stopped() { [ -e "$STOP" ] && { echo "[q207] $(date) stop file present before $1 — QUEUE 207 DONE"; exit 0; }; }
fp() { local k=""; [ -n "$2" ] && k="setoption name Knobs value $2"; printf '%s\nbench\nquit\n' "$k" | "$1" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+'; }
echo "[q207] $(date '+%Y-%m-%d %H:%M:%S') start: PGO builds now, timed steps at 00:30"

# ── 1 PGO builds, one per arm ─────────────────────────────────────────────────────────────────────────────────────────
for arm in 0 1; do
	stopped "the arm $arm PGO build"
	P=/mnt/e/chess_data/pgo_sv2_arm$arm
	rm -rf "$P"; mkdir -p "$P"
	OUT=sv2_gen$arm PGO=gen PGO_DIR=$P bash selfplay/build_uci.sh > "$W/build_gen$arm.out" 2>&1 || { echo "[q207] ☠️ arm $arm gen build failed — QUEUE 207 ABORTED"; exit 1; }
	nice -n 5 "$FC" -engine cmd="$ND/build_uci/sv2_gen$arm" name=a option.OwnBook=false option.Knobs=SEARCH_ARM=$arm st=0.25 \
		-engine cmd="$ND/build_uci/sv2_gen$arm" name=b option.OwnBook=false option.Knobs=SEARCH_ARM=$arm st=0.25 \
		-openings file=$O/openings_uho.epd format=epd order=random -srand 207 -rounds 4 -games 2 -concurrency 1 > "$W/train$arm.log" 2>&1
	echo "[q207] arm $arm training: $(grep -E '^Games:' "$W/train$arm.log" | tail -1) · warnings $(grep -c 'have option' "$W/train$arm.log") · profile files $(find "$P" -name '*.gcda' | wc -l)"
	OUT=sv2_pgo$arm PGO=use PGO_DIR=$P bash selfplay/build_uci.sh > "$W/build_use$arm.out" 2>&1 || { echo "[q207] ☠️ arm $arm use build failed — QUEUE 207 ABORTED"; exit 1; }
	a0=$(fp "$ND/build_uci/sv2_pgo$arm" "SEARCH_ARM=0"); a1=$(fp "$ND/build_uci/sv2_pgo$arm" "SEARCH_ARM=1")
	echo "[q207] sv2_pgo$arm bench: arm 0 [$a0] arm 1 [$a1] (want 247 / 39861964 both)"
	[ "$a0" = "solved 247/300 nodes 39861964" ] && [ "$a1" = "solved 247/300 nodes 39861964" ] || { echo "[q207] ☠️ fingerprint moved — QUEUE 207 ABORTED"; exit 1; }
done

echo "[q207] $(date '+%H:%M') builds done; waiting for 00:30"
while [ $(date +%s) -lt $START_AT ]; do [ -e "$STOP" ] && { echo "[q207] stopped while waiting"; exit 0; }; sleep 60; done
echo "[q207] $(date '+%Y-%m-%d %H:%M:%S') timed steps start"

# ── 2 real-play speed, alternated ────────────────────────────────────────────────────────────────────────────────────
for rep in 1 2 3; do
	for arm in 0 1; do
		stopped "speed rep $rep arm $arm"
		[ $(date +%s) -lt $LATEST ] || { echo "[q207] $(date) past 09:30 — QUEUE 207 DONE"; exit 0; }
		timeout 3000 bash "$R" pyrun diagnostics/_ebf_per_iter.py E1=arm$arm="$ND/build_uci/sv2_pgo$arm" \
			O1="OwnBook:false;Knobs:SEARCH_ARM=$arm" D1=13 N=40 SEED=1 OUT="$W/speed_r${rep}_arm$arm" > "$W/speed_r${rep}_arm$arm.log" 2>&1
		echo "[q207] speed rep $rep arm $arm: $(grep -iE 'time to depth 13|EBF over' "$W/speed_r${rep}_arm$arm.log" | head -3 | tr '\n' ' ')"
	done
done
bash "$R" pyrun diagnostics/_sv2_speed_read.py DIR="$W" > "$W/speed_read.txt" 2>&1
show < "$W/speed_read.txt"

# ── 3 games smoke ────────────────────────────────────────────────────────────────────────────────────────────────────
stopped "the games smoke"
[ $(date +%s) -lt $LATEST ] || { echo "[q207] $(date) past 09:30 — QUEUE 207 DONE"; exit 0; }
echo "[q207] $(date '+%H:%M') games smoke: arm 1 vs arm 0 at 10+0.1, 300 games"
COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
nice -n 5 "$FC" -engine "cmd=$ND/build_uci/sv2_pgo1" name=arm1 option.OwnBook=false option.Knobs=SEARCH_ARM=1 tc=10+0.1 timemargin=250 \
	-engine "cmd=$ND/build_uci/sv2_pgo0" name=arm0 option.OwnBook=false option.Knobs=SEARCH_ARM=0 tc=10+0.1 timemargin=250 \
	$COMMON -srand 207 -rounds 150 -games 2 -pgnout file=$W/smoke.pgn notation=san nodes=true append=false > $W/smoke.log 2>&1
echo "[q207] smoke: $(grep -E '^Games:' $W/smoke.log | tail -1) · warnings $(grep -c 'have option' $W/smoke.log) · disconnects/stalls $(grep -ciE 'disconnect|stall|not responsive' $W/smoke.log) · time forfeits $(grep -c 'loses on time' $W/smoke.pgn)"
grep -E 'Elo:|Ptnml' $W/smoke.log | tail -2 | show
timeout 900 bash "$R" pyrun selfplay/fastchess_report.py $W/smoke.pgn --p1 arm1 2>&1 | sed -n '/per engine/,$p' | show
echo "[q207] $(date) QUEUE 207 DONE"
