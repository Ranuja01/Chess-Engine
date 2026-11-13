#!/bin/bash
# Queue #221, chained after #220 (starts when q220.log says "QUEUE 220 DONE"; #220's go file is the owner's "box is free").
#  A MOVEGEN_FAST production candidate, a gen 7.3 speed build: the shipped preset sets MOVEGEN_FAST=1 (search_engine.cpp,
#    the gen-7 block, before the start-up read: MOVEGEN_FAST=0 restores today's path; results are identical either way).
#    #219: −3.52% time to depth 13 (sd 1.79, 5 of 5 reps faster), same tree, nps +3.7%. PGO build chess_engine_uci_g73fpgo.
#    Gates: default 249 / 37,279,382 and MOVEGEN_FAST=0 the same; SEE_PER_MOVE=0 = gen 7.2 247 / 39,861,964; + MOPUP_V2=0 =
#    gen 7.1 247 / 39,852,210; V2_SHIP_GEN=6 = gen 6.7 251 / 36,859,850; v1 printed. NOT INSTALLED: the owner's yes.
#    Pinned bench, production (g73pgo) and the candidate alternating x6, on a quiet box. REGISTERED: the candidate faster in
#    mean bench time; prediction −2 to −4%.
#  B Lazy generation on search v2 (SV2_LAZY_GEN, S3 option B). PGO build build_uci/sv2lazy_pgo profiled with
#    SEARCH_ARM=1,SV2_PRUNE_EARLY=1 and SV2_LAZY_GEN 0 / 1. Gates: v2 with SV2_LAZY_GEN 0 and 2 = 249 / 37,279,382 (2 = the same
#    code with phase 2 at once, proven node-identical before this queue); SV2_LAZY_GEN=1 = 249 / 37,086,375 (the test build's).
#    Speed: _ebf_per_iter, 40 positions per phase, depth 13, lazy 0 / 1 alternated, 5 reps (the trees differ: time AND nodes
#    to depth 13 are read). Games: SV2_LAZY_GEN 1 vs 0 on SEARCH_ARM=1, SV2_PRUNE_EARLY=1, 2,000 at 10+0.1, UHO, seed 221.
#    REGISTERED before any game: A − B ≥ +2σ (pentanomial) = KEEP as search v2's default; within ±2σ = NEUTRAL (a knob);
#    ≤ −2σ = REJECT. Prediction 0 to +5 (#214: 40-43% of the generated quiet moves are never needed; list build ~20% of time).
# Games start only before 07:00 11-13. Stop: touch /mnt/e/chess_data/q221.stop (between steps).
# Never edit this file, the tools, the binaries or the sources while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
W=/mnt/e/chess_data/s3_c
PA=/mnt/e/chess_data/pgo_g73f
PB=/mnt/e/chess_data/pgo_lazy
PROD="$ND/build_uci/chess_engine_uci"
OLD="$ND/build_uci/chess_engine_uci_g73pgo"
NEW="$ND/build_uci/chess_engine_uci_g73fpgo"
LZ="$ND/build_uci/sv2lazy_pgo"
LAZY1_FP="solved 249/300 nodes 37086375"
STOP=/mnt/e/chess_data/q221.stop
GAMES_LATEST=$(date -d '2026-11-13 07:00' +%s)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"; mkdir -p "$W"
show() { sed "s/^/[q221]   /"; }
stopped() { [ -e "$STOP" ] && { echo "[q221] $(date) stop file present before $1 — QUEUE 221 DONE"; exit 0; }; }
busy() { pgrep -f "fastchess-linux-x86-64/fastchess|vs_sf.py|sts_test|tactical_test|_ebf_per_iter|_grade_game|_depth_ladder|chess_ui" >/dev/null; }
# Waits up to an hour for a quiet box (no UI, no other job); returns 1 when it is still busy, so the timed step is skipped.
wait_quiet() { local t=$(( $(date +%s) + 3600 )); until ! busy; do stopped "$1"; [ $(date +%s) -gt $t ] && { echo "[q221] $(date) box still busy after an hour — $1 skipped"; return 1; }; sleep 20; done; }
fp() { local k=""; [ -n "$2" ] && k="setoption name Knobs value $2"; printf '%s\nbench\nquit\n' "$k" | "$1" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+'; }
echo "[q221] $(date '+%Y-%m-%d %H:%M:%S') queued; waiting for #220"
until grep -qE "QUEUE 220 DONE" /mnt/e/chess_data/q220.log 2>/dev/null; do stopped "the start"; sleep 30; done
echo "[q221] $(date '+%Y-%m-%d %H:%M:%S') start"
cmp -s "$PROD" "$OLD" && echo "[q221] production == chess_engine_uci_g73pgo" || echo "[q221] ⚠️ production differs from chess_engine_uci_g73pgo"

# ── A MOVEGEN_FAST production candidate ─────────────────────────────────────────────────────────────────────────────────
rm -rf "$PA"; mkdir -p "$PA"
if OUT=chess_engine_uci_g73fgen PGO=gen PGO_DIR=$PA bash selfplay/build_uci.sh > "$W/a_gen.out" 2>&1; then
	nice -n 5 "$FC" -engine cmd="$ND/build_uci/chess_engine_uci_g73fgen" name=a option.OwnBook=false st=0.25 \
		-engine cmd="$ND/build_uci/chess_engine_uci_g73fgen" name=b option.OwnBook=false st=0.25 \
		-openings file=$O/openings_uho.epd format=epd order=random -srand 731 -rounds 4 -games 2 -concurrency 1 > "$W/a_train.log" 2>&1
	echo "[q221] A training: $(grep -E '^Games:' "$W/a_train.log" | tail -1) · profile files $(find "$PA" -name '*.gcda' | wc -l)"
	if OUT=chess_engine_uci_g73fpgo PGO=use PGO_DIR=$PA bash selfplay/build_uci.sh > "$W/a_use.out" 2>&1; then
		d=$(fp "$NEW" ""); d0=$(fp "$NEW" MOVEGEN_FAST=0); g72=$(fp "$NEW" SEE_PER_MOVE=0); g71=$(fp "$NEW" SEE_PER_MOVE=0,MOPUP_V2=0); g67=$(fp "$NEW" V2_SHIP_GEN=6)
		echo "[q221] A default (want 249 / 37279382):              [$d]"
		echo "[q221] A MOVEGEN_FAST=0 (want 249 / 37279382):       [$d0]"
		echo "[q221] A SEE_PER_MOVE=0 = gen 7.2 (247 / 39861964):  [$g72]"
		echo "[q221] A + MOPUP_V2=0 = gen 7.1 (247 / 39852210):    [$g71]"
		echo "[q221] A V2_SHIP_GEN=6 = gen 6.7 (251 / 36859850):   [$g67]"
		echo "[q221] A v1 EvalArm 0 (250 / 35310775-778):          $(printf 'setoption name EvalArm value 0\nbench\nquit\n' | "$NEW" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+')"
		if [ "$d" = "solved 249/300 nodes 37279382" ] && [ "$d0" = "$d" ] && [ "$g72" = "solved 247/300 nodes 39861964" ] \
			&& [ "$g71" = "solved 247/300 nodes 39852210" ] && [ "$g67" = "solved 251/300 nodes 36859850" ]; then
			echo "[q221] A fingerprints OK — candidate ready, NOT installed (owner's yes)"
			if wait_quiet "the pinned bench"; then
				echo "[q221] $(date '+%H:%M') == TIMED: UCI bench, production g73pgo and candidate g73fpgo, pinned, alternating x6"
				for i in 1 2 3 4 5 6; do
					echo "[q221]   g73  $(printf 'bench\nquit\n' | taskset -c 5 "$OLD" 2>/dev/null | grep '^bench')"
					echo "[q221]   g73f $(printf 'bench\nquit\n' | taskset -c 5 "$NEW" 2>/dev/null | grep '^bench')"
				done
			fi
		else
			echo "[q221] ☠️ A fingerprints differ — candidate rejected"
		fi
	else
		echo "[q221] ☠️ A use build failed"; tail -5 "$W/a_use.out" | show
	fi
else
	echo "[q221] ☠️ A gen build failed"; tail -5 "$W/a_gen.out" | show
fi

# ── B lazy generation on search v2 ──────────────────────────────────────────────────────────────────────────────────────
stopped "the lazy generation build"
rm -rf "$PB"; mkdir -p "$PB"
OUT=sv2lazy_gen PGO=gen PGO_DIR=$PB bash selfplay/build_uci.sh > "$W/b_gen.out" 2>&1 || { echo "[q221] ☠️ B gen build failed — QUEUE 221 DONE"; exit 1; }
nice -n 5 "$FC" -engine cmd="$ND/build_uci/sv2lazy_gen" name=a option.OwnBook=false option.Knobs=SEARCH_ARM=1,SV2_PRUNE_EARLY=1,SV2_LAZY_GEN=1 st=0.25 \
	-engine cmd="$ND/build_uci/sv2lazy_gen" name=b option.OwnBook=false option.Knobs=SEARCH_ARM=1,SV2_PRUNE_EARLY=1,SV2_LAZY_GEN=0 st=0.25 \
	-openings file=$O/openings_uho.epd format=epd order=random -srand 221 -rounds 4 -games 2 -concurrency 1 > "$W/b_train.log" 2>&1
echo "[q221] B training: $(grep -E '^Games:' "$W/b_train.log" | tail -1) · warnings $(grep -c 'have option' "$W/b_train.log") · profile files $(find "$PB" -name '*.gcda' | wc -l)"
OUT=sv2lazy_pgo PGO=use PGO_DIR=$PB bash selfplay/build_uci.sh > "$W/b_use.out" 2>&1 || { echo "[q221] ☠️ B use build failed — QUEUE 221 DONE"; exit 1; }
l0=$(fp "$LZ" SEARCH_ARM=1,SV2_PRUNE_EARLY=1,SV2_LAZY_GEN=0); l2=$(fp "$LZ" SEARCH_ARM=1,SV2_PRUNE_EARLY=1,SV2_LAZY_GEN=2); l1=$(fp "$LZ" SEARCH_ARM=1,SV2_PRUNE_EARLY=1,SV2_LAZY_GEN=1)
echo "[q221] B v2 lazy 0 (want 249 / 37279382): [$l0] · lazy 2 (want the same): [$l2] · lazy 1 (want $LAZY1_FP): [$l1]"
[ "$l0" = "solved 249/300 nodes 37279382" ] && [ "$l2" = "$l0" ] && [ "$l1" = "$LAZY1_FP" ] || { echo "[q221] ☠️ B fingerprints — QUEUE 221 DONE"; exit 1; }

speed() {  # dir, reps, binary, knobs arm 0, knobs arm 1
	local d=$1 n=$2 bin=$3 rep a k
	mkdir -p "$d"
	for rep in $(seq 1 $n); do
		for a in 0 1; do
			stopped "speed $d rep $rep arm $a"
			[ $a = 0 ] && k=$4 || k=$5
			timeout 3000 bash "$R" pyrun diagnostics/_ebf_per_iter.py E1=arm$a="$bin" O1="OwnBook:false;Knobs:$k" D1=13 N=40 SEED=1 \
				OUT="$d/speed_r${rep}_arm$a" > "$d/speed_r${rep}_arm$a.log" 2>&1
		done
	done
	bash "$R" pyrun diagnostics/_sv2_speed_read.py DIR="$d" 2>&1 | grep -v '^REGISTERED (#207)' | show
}
if wait_quiet "the lazy speed read"; then
	echo "[q221] == lazy generation speed, SEARCH_ARM=1, arm 0 = SV2_LAZY_GEN 0, arm 1 = 1 (5 reps; trees differ)"
	speed "$W/lazy_speed" 5 "$LZ" "SEARCH_ARM=1,SV2_PRUNE_EARLY=1,SV2_LAZY_GEN=0" "SEARCH_ARM=1,SV2_PRUNE_EARLY=1,SV2_LAZY_GEN=1"
fi

stopped "the lazy generation games"
[ $(date +%s) -lt $GAMES_LATEST ] || { echo "[q221] $(date) past 07:00 — games skipped — QUEUE 221 DONE"; exit 0; }
echo "[q221] $(date '+%H:%M') games: SV2_LAZY_GEN 1 vs 0 (search v2) at 10+0.1, 2,000 games, seed 221"
COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
nice -n 5 "$FC" -engine "cmd=$LZ" name=lazy1 option.OwnBook=false option.Knobs=SEARCH_ARM=1,SV2_PRUNE_EARLY=1,SV2_LAZY_GEN=1 tc=10+0.1 timemargin=250 \
	-engine "cmd=$LZ" name=lazy0 option.OwnBook=false option.Knobs=SEARCH_ARM=1,SV2_PRUNE_EARLY=1,SV2_LAZY_GEN=0 tc=10+0.1 timemargin=250 \
	$COMMON -srand 221 -rounds 1000 -games 2 -pgnout file=$W/lazy.pgn notation=san nodes=true append=false > $W/lazy.log 2>&1
echo "[q221] games: $(grep -E '^Games:' $W/lazy.log | tail -1) · warnings $(grep -c 'have option' $W/lazy.log) · disconnects/stalls $(grep -ciE 'disconnect|stall|not responsive' $W/lazy.log) · time forfeits $(grep -c 'loses on time' $W/lazy.pgn)"
grep -E 'Elo:|Ptnml' $W/lazy.log | tail -2 | show
timeout 900 bash "$R" pyrun selfplay/fastchess_report.py $W/lazy.pgn --p1 lazy1 2>&1 | sed -n '/vs the field/,$p' | show
echo "[q221] $(date) QUEUE 221 DONE"
