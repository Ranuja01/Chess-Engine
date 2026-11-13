#!/bin/bash
# Queue #220 = the rest of #219, after the owner's gaming (#219's games were interrupted cleanly at 540 games = 270 complete
# pairs so the owner could play; its early-pruning speed read was disturbed by other load in reps 3-5, unreadable).
# Starts on the go file /mnt/e/chess_data/q220.go (written when the owner says the box is free) once #219 has finished.
#  1 GAMES, SEE_PER_MOVE bit 4, continued: the same two engines and settings as #219 step 4, 1,460 more games (730 rounds),
#    fresh openings (seed 2191), see7b.pgn. Verdict on the POOLED 2,000 (see7.pgn + see7b.pgn passed as separate files):
#    the rule registered in #219 before any game: A − B ≥ +2σ = KEEP as search v2's default; within ±2σ = NEUTRAL (a knob);
#    ≤ −2σ = REJECT. Prediction 0 to +5.
#  2 Early pruning speed read again, on a quiet box: build_uci/sv2_pgo1pe with SEARCH_ARM=1, SV2_PRUNE_EARLY 0 / 1 alternated,
#    12 repetitions, a new directory (#219's disturbed read is not pooled). REGISTERED: faster by more than 2 standard errors
#    of the mean difference = counted in the ledger; otherwise stays uncounted. Prediction −1 to −1.5% (#215 cachegrind).
# Games start only before 03:00, speed reads end by 09:30 11-13. Stop: touch /mnt/e/chess_data/q220.stop (between steps).
# Never edit this file, the tools or the binaries it uses while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
W=/mnt/e/chess_data/s3_b
STOP=/mnt/e/chess_data/q220.stop
GO=/mnt/e/chess_data/q220.go
GAMES_LATEST=$(date -d '2026-11-13 03:00' +%s)
LATEST=$(date -d '2026-11-13 09:30' +%s)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP" "$GO"
show() { sed "s/^/[q220]   /"; }
stopped() { [ -e "$STOP" ] && { echo "[q220] $(date) stop file present before $1 — QUEUE 220 DONE"; exit 0; }; }
fp() { local k=""; [ -n "$2" ] && k="setoption name Knobs value $2"; printf '%s\nbench\nquit\n' "$k" | "$1" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+'; }
echo "[q220] $(date '+%Y-%m-%d %H:%M:%S') queued; waiting for the go file and #219"
until [ -e "$GO" ] && grep -qE "QUEUE 219 (DONE|ABORTED)" /mnt/e/chess_data/q219.log 2>/dev/null; do stopped "the start"; sleep 30; done
echo "[q220] $(date '+%Y-%m-%d %H:%M:%S') start"

# ── 1 games: SEE_PER_MOVE bit 4, the remaining 1,460 ────────────────────────────────────────────────────────────────────
G="$ND/build_uci/sv2_pgo1pe"
if [ $(date +%s) -lt $GAMES_LATEST ]; then
	s7=$(fp "$G" "SEARCH_ARM=1,SV2_PRUNE_EARLY=1,SEE_PER_MOVE=7"); s3=$(fp "$G" "SEARCH_ARM=1,SV2_PRUNE_EARLY=1,SEE_PER_MOVE=3")
	echo "[q220] fingerprints: SEE 7 [$s7] (want 247 / 39685447) · SEE 3 [$s3] (want 249 / 37279382)"
	if [ "$s7" = "solved 247/300 nodes 39685447" ] && [ "$s3" = "solved 249/300 nodes 37279382" ]; then
		echo "[q220] $(date '+%H:%M') games: SEE_PER_MOVE=7 vs 3 (search v2) at 10+0.1, 1,460 games, seed 2191"
		COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
		nice -n 5 "$FC" -engine "cmd=$G" name=see7 option.OwnBook=false option.Knobs=SEARCH_ARM=1,SV2_PRUNE_EARLY=1,SEE_PER_MOVE=7 tc=10+0.1 timemargin=250 \
			-engine "cmd=$G" name=see3 option.OwnBook=false option.Knobs=SEARCH_ARM=1,SV2_PRUNE_EARLY=1,SEE_PER_MOVE=3 tc=10+0.1 timemargin=250 \
			$COMMON -srand 2191 -rounds 730 -games 2 -pgnout file=$W/see7b.pgn notation=san nodes=true append=false > $W/see7b.log 2>&1
		echo "[q220] games: $(grep -E '^Games:' $W/see7b.log | tail -1) · warnings $(grep -c 'have option' $W/see7b.log) · disconnects/stalls $(grep -ciE 'disconnect|stall|not responsive' $W/see7b.log) · time forfeits $(grep -c 'loses on time' $W/see7b.pgn)"
		grep -E 'Elo:|Ptnml' $W/see7b.log | tail -2 | show
		echo "[q220] POOLED see7.pgn + see7b.pgn (separate files; pairs must = games / 2):"
		timeout 900 bash "$R" pyrun selfplay/fastchess_report.py $W/see7.pgn $W/see7b.pgn --p1 see7 2>&1 | show
	else
		echo "[q220] ☠️ fingerprints moved — games skipped"
	fi
else
	echo "[q220] $(date) past 03:00 — games skipped"
fi

# ── 2 early pruning speed, quiet box ────────────────────────────────────────────────────────────────────────────────────
speed() {  # dir, reps, binary, knobs arm 0, knobs arm 1
	local d=$1 n=$2 bin=$3 rep a k
	mkdir -p "$d"
	for rep in $(seq 1 $n); do
		for a in 0 1; do
			stopped "speed $d rep $rep arm $a"
			[ $(date +%s) -lt $LATEST ] || { echo "[q220] $(date) past 09:30 — QUEUE 220 DONE"; exit 0; }
			[ $a = 0 ] && k=$4 || k=$5
			timeout 3000 bash "$R" pyrun diagnostics/_ebf_per_iter.py E1=arm$a="$bin" O1="OwnBook:false;Knobs:$k" D1=13 N=40 SEED=1 \
				OUT="$d/speed_r${rep}_arm$a" > "$d/speed_r${rep}_arm$a.log" 2>&1
		done
	done
	bash "$R" pyrun diagnostics/_sv2_speed_read.py DIR="$d" 2>&1 | grep -v '^REGISTERED (#207)' | show
}
stopped "the early pruning speed read"
echo "[q220] == early pruning speed, SEARCH_ARM=1, arm 0 = SV2_PRUNE_EARLY 0, arm 1 = 1 (12 reps)"
speed "$W/pe_speed2" 12 "$ND/build_uci/sv2_pgo1pe" "SEARCH_ARM=1,SV2_PRUNE_EARLY=0" "SEARCH_ARM=1,SV2_PRUNE_EARLY=1"
echo "[q220] $(date) QUEUE 220 DONE"
