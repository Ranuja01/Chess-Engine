#!/bin/bash
# 2026-11-12 ~13:30, queue #216 (tonight, after the owner's gaming; starts on the go file /mnt/e/chess_data/q216.go, written
# when the owner says the box is free). Waits for #211; writes #213's go file and waits for it too (both need a quiet box).
#  1 PGO build of today's sources, profiled with MOVEGEN_FAST both 0 and 1 (one engine each in the training games), so both
#    paths are optimised: build_uci/mgf_pgo. Gates: default 247 / 39,861,964 and the same with MOVEGEN_FAST=1.
#  2 MOVEGEN_FAST speed read, production's search (SEARCH_ARM=0): _ebf_per_iter.py, 40 positions per phase (seed 1), depth 13,
#    the same binary with the knob 0 / 1 alternated, 5 repetitions (#209's 3 gave sd 1.77%). Same tree required.
#    REGISTERED: MOVEGEN_FAST=1 faster in mean time to depth 13 = a speed item for the ledger; prediction −2 to −5% (#215:
#    −6.47% instructions, −5.75% mispredictions, memory traffic level).
#  3 Early pruning re-read, search v2: build_uci/sv2_pgo1pe (#209) with SEARCH_ARM=1, SV2_PRUNE_EARLY 0 / 1 alternated,
#    8 repetitions (to resolve #215's estimate of −1 to −1.5%). REGISTERED: faster by more than 2 standard errors of the
#    mean difference = counted in the ledger; otherwise stays uncounted. Prediction −1 to −1.5%.
#  4 GAMES, SEE_PER_MOVE bit 4 (search v2's losing-capture prune by the move's own exchange value): A = SEE_PER_MOVE=7,
#    B = SEE_PER_MOVE=3, both build_uci/sv2_pgo1pe with SEARCH_ARM=1, SV2_PRUNE_EARLY=1; 2,000 games at 10+0.1, UHO, seed 216.
#    Fingerprints (WAC d10, arm 1): 7 = 247 / 39,685,447; 3 = 249 / 37,279,382.
#    REGISTERED before any game: A − B ≥ +2σ (pentanomial) = KEEP as search v2's default; within ±2σ = NEUTRAL (a knob);
#    ≤ −2σ = REJECT. Prediction 0 to +5 (the prune is 4% of the moves pruned after being made, S3 step 0).
# Speed steps end by 09:30 11-13, games start only before 04:00. Stop: touch /mnt/e/chess_data/q216.stop (between steps).
# Never edit this file, the tools or the sources while it builds.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
W=/mnt/e/chess_data/s3_b
P=/mnt/e/chess_data/pgo_mgf
STOP=/mnt/e/chess_data/q216.stop
GO=/mnt/e/chess_data/q216.go
GAMES_LATEST=$(date -d '2026-11-13 04:00' +%s)
LATEST=$(date -d '2026-11-13 09:30' +%s)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP" "$GO"; mkdir -p "$W"
show() { sed "s/^/[q216]   /"; }
stopped() { [ -e "$STOP" ] && { echo "[q216] $(date) stop file present before $1 — QUEUE 216 DONE"; exit 0; }; }
fp() { local k=""; [ -n "$2" ] && k="setoption name Knobs value $2"; printf '%s\nbench\nquit\n' "$k" | "$1" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+'; }
echo "[q216] $(date '+%Y-%m-%d %H:%M:%S') queued; waiting for the go file and #211"
until [ -e "$GO" ] && grep -qE "QUEUE 211 (DONE|ABORTED)" /mnt/e/chess_data/q211.log 2>/dev/null; do stopped "the start"; sleep 30; done
touch /mnt/e/chess_data/q213.go
until grep -qE "QUEUE 213 DONE|stopped" /mnt/e/chess_data/q213.log 2>/dev/null; do stopped "#213"; sleep 30; done
echo "[q216] $(date '+%Y-%m-%d %H:%M:%S') start"

# ── 1 PGO build, both paths profiled ────────────────────────────────────────────────────────────────────────────────────
rm -rf "$P"; mkdir -p "$P"
OUT=mgf_gen PGO=gen PGO_DIR=$P bash selfplay/build_uci.sh > "$W/build_gen.out" 2>&1 || { echo "[q216] ☠️ gen build failed — QUEUE 216 ABORTED"; exit 1; }
nice -n 5 "$FC" -engine cmd="$ND/build_uci/mgf_gen" name=a option.OwnBook=false option.Knobs=MOVEGEN_FAST=1 st=0.25 \
	-engine cmd="$ND/build_uci/mgf_gen" name=b option.OwnBook=false option.Knobs=MOVEGEN_FAST=0 st=0.25 \
	-openings file=$O/openings_uho.epd format=epd order=random -srand 216 -rounds 4 -games 2 -concurrency 1 > "$W/train.log" 2>&1
echo "[q216] training: $(grep -E '^Games:' "$W/train.log" | tail -1) · warnings $(grep -c 'have option' "$W/train.log") · profile files $(find "$P" -name '*.gcda' | wc -l)"
OUT=mgf_pgo PGO=use PGO_DIR=$P bash selfplay/build_uci.sh > "$W/build_use.out" 2>&1 || { echo "[q216] ☠️ use build failed — QUEUE 216 ABORTED"; exit 1; }
B="$ND/build_uci/mgf_pgo"
d0=$(fp "$B" ""); d1=$(fp "$B" "MOVEGEN_FAST=1")
echo "[q216] mgf_pgo bench: default [$d0] · MOVEGEN_FAST=1 [$d1] (want 247 / 39861964 both)"
[ "$d0" = "solved 247/300 nodes 39861964" ] && [ "$d1" = "solved 247/300 nodes 39861964" ] || { echo "[q216] ☠️ fingerprint moved — QUEUE 216 ABORTED"; exit 1; }

# ── 2 MOVEGEN_FAST speed ────────────────────────────────────────────────────────────────────────────────────────────────
speed() {  # dir, reps, binary, knobs arm 0, knobs arm 1
	local d=$1 n=$2 bin=$3 rep a k
	mkdir -p "$d"
	for rep in $(seq 1 $n); do
		for a in 0 1; do
			stopped "speed $d rep $rep arm $a"
			[ $(date +%s) -lt $LATEST ] || { echo "[q216] $(date) past 09:30 — QUEUE 216 DONE"; exit 0; }
			[ $a = 0 ] && k=$4 || k=$5
			timeout 3000 bash "$R" pyrun diagnostics/_ebf_per_iter.py E1=arm$a="$bin" O1="OwnBook:false;Knobs:$k" D1=13 N=40 SEED=1 \
				OUT="$d/speed_r${rep}_arm$a" > "$d/speed_r${rep}_arm$a.log" 2>&1
		done
	done
	bash "$R" pyrun diagnostics/_sv2_speed_read.py DIR="$d" 2>&1 | grep -v '^REGISTERED (#207)' | show
}
echo "[q216] == MOVEGEN_FAST speed, SEARCH_ARM=0, arm 0 = knob 0, arm 1 = knob 1 (5 reps)"
speed "$W/mgf_speed" 5 "$B" "MOVEGEN_FAST=0" "MOVEGEN_FAST=1"

# ── 3 early pruning re-read ─────────────────────────────────────────────────────────────────────────────────────────────
echo "[q216] == early pruning speed, SEARCH_ARM=1, arm 0 = SV2_PRUNE_EARLY 0, arm 1 = 1 (8 reps)"
speed "$W/pe_speed" 8 "$ND/build_uci/sv2_pgo1pe" "SEARCH_ARM=1,SV2_PRUNE_EARLY=0" "SEARCH_ARM=1,SV2_PRUNE_EARLY=1"

# ── 4 games: SEE_PER_MOVE bit 4 ─────────────────────────────────────────────────────────────────────────────────────────
stopped "the SEE bit 4 games"
[ $(date +%s) -lt $GAMES_LATEST ] || { echo "[q216] $(date) past 04:00 — games skipped — QUEUE 216 DONE"; exit 0; }
G="$ND/build_uci/sv2_pgo1pe"
s7=$(fp "$G" "SEARCH_ARM=1,SV2_PRUNE_EARLY=1,SEE_PER_MOVE=7"); s3=$(fp "$G" "SEARCH_ARM=1,SV2_PRUNE_EARLY=1,SEE_PER_MOVE=3")
echo "[q216] fingerprints: SEE 7 [$s7] (want 247 / 39685447) · SEE 3 [$s3] (want 249 / 37279382)"
[ "$s7" = "solved 247/300 nodes 39685447" ] && [ "$s3" = "solved 249/300 nodes 37279382" ] || { echo "[q216] ☠️ fingerprints — games skipped — QUEUE 216 DONE"; exit 1; }
echo "[q216] $(date '+%H:%M') games: SEE_PER_MOVE=7 vs 3 (search v2) at 10+0.1, 2,000 games"
COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
nice -n 5 "$FC" -engine "cmd=$G" name=see7 option.OwnBook=false option.Knobs=SEARCH_ARM=1,SV2_PRUNE_EARLY=1,SEE_PER_MOVE=7 tc=10+0.1 timemargin=250 \
	-engine "cmd=$G" name=see3 option.OwnBook=false option.Knobs=SEARCH_ARM=1,SV2_PRUNE_EARLY=1,SEE_PER_MOVE=3 tc=10+0.1 timemargin=250 \
	$COMMON -srand 216 -rounds 1000 -games 2 -pgnout file=$W/see7.pgn notation=san nodes=true append=false > $W/see7.log 2>&1
echo "[q216] games: $(grep -E '^Games:' $W/see7.log | tail -1) · warnings $(grep -c 'have option' $W/see7.log) · disconnects/stalls $(grep -ciE 'disconnect|stall|not responsive' $W/see7.log) · time forfeits $(grep -c 'loses on time' $W/see7.pgn)"
grep -E 'Elo:|Ptnml' $W/see7.log | tail -2 | show
timeout 900 bash "$R" pyrun selfplay/fastchess_report.py $W/see7.pgn --p1 see7 2>&1 | sed -n '/vs the field/,$p' | show
echo "[q216] $(date) QUEUE 216 DONE"
