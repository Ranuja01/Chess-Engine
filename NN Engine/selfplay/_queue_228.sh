#!/bin/bash
# 2026-11-13 ~13:25, queue #228: S3 root R3, the first games read (owner 11-13: "we are clear to go"; the question: can the
# pre-search be removed cleanly AND the search get faster). Arm A4 of #226 = the owner's cascading tables in place of the
# pre-search / root-razor pair: ENABLE_ROOT_PRESEARCH=0, ENABLE_ROOT_RAZOR=0, ENABLE_PRESEARCH_OFF_SORT=1,
# PRESEARCH_OFF_SORT_MODE=2 (provenance tiers), ENABLE_ROOT_TABLE=1. Fixed depth 10 (#226): WAC 258 vs 249 (+17.7% nodes), STS
# 2086 vs 2037 (+3.2%), quiet +7.4%; the pair is one mechanism (S3-ROOT-AUDIT §10), so it is replaced whole.
# Production binary (g73fpgo) against itself, A4 through option.Knobs. Gate: bench with the knobs = 258 / 43,894,575 (the
# Cython A4 WAC; any other value means the knobs did not reach the engine) and the default 249 / 37,279,382.
# Games 10+0.1, UHO, seed 228, concurrency 3, 2,000 games (~5.3 h). REGISTERED before any game, A4 − today (pentanomial):
#   ≥ +2σ  the cascading-tables root becomes the root candidate (the ship rule's SF18 and Mediocre legs next);
#   ±2σ    LEVEL: removal is clean but not yet a gain; root reductions on top of A4 (complete table + best exemption) next;
#   ≤ −2σ  the pair stays; the R2 options on it (pre-search subset, aspiration) next.
#   Plus 0 time forfeits and 0 disconnects / stalls in either case. Prediction −10 to +10.
# The owner may game today: then the games are stopped with SIGINT to fastchess (finished games kept) and resumed on a free slot.
# Stop: touch /mnt/e/chess_data/q228.stop (before the games). Never edit this file or the binary while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
W=/mnt/e/chess_data/r3
B="$ND/build_uci/chess_engine_uci_g73fpgo"
STOP=/mnt/e/chess_data/q228.stop
A4K="ENABLE_ROOT_PRESEARCH=0,ENABLE_ROOT_RAZOR=0,ENABLE_PRESEARCH_OFF_SORT=1,PRESEARCH_OFF_SORT_MODE=2,ENABLE_ROOT_TABLE=1"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"; mkdir -p "$W"
show() { sed "s/^/[q228]   /"; }
fp() { local k=""; [ -n "$2" ] && k="setoption name Knobs value $2"; printf '%s\nbench\nquit\n' "$k" | "$1" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+'; }
echo "[q228] $(date '+%Y-%m-%d %H:%M:%S') start"
d=$(fp "$B" ""); a=$(fp "$B" "$A4K")
echo "[q228] default (want 249 / 37279382): [$d]"
echo "[q228] A4 (want 258 / 43894575):      [$a]"
if [ "$d" != "solved 249/300 nodes 37279382" ] || [ "$a" != "solved 258/300 nodes 43894575" ]; then
	echo "[q228] ☠️ fingerprints differ — QUEUE 228 ABORTED"; exit 1
fi
[ -e "$STOP" ] && { echo "[q228] $(date) stop file — QUEUE 228 DONE"; exit 0; }
COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
echo "[q228] $(date '+%H:%M') games: A4 (cascading tables, no pre-search, no root razor) vs today's root at 10+0.1, 2,000 games, seed 228"
nice -n 5 "$FC" -engine "cmd=$B" name=a4 option.OwnBook=false "option.Knobs=$A4K" tc=10+0.1 timemargin=250 \
	-engine "cmd=$B" name=root0 option.OwnBook=false tc=10+0.1 timemargin=250 \
	$COMMON -srand 228 -rounds 1000 -games 2 -pgnout file=$W/a4.pgn notation=san nodes=true append=false > $W/a4.log 2>&1
echo "[q228] games: $(grep -E '^Games:' $W/a4.log | tail -1) · warnings $(grep -c 'have option' $W/a4.log) · disconnects/stalls $(grep -ciE 'disconnect|stall|not responsive' $W/a4.log) · time forfeits $(grep -c 'loses on time' $W/a4.pgn)"
grep -E 'Elo:|Ptnml' $W/a4.log | tail -2 | show
timeout 900 bash "$R" pyrun selfplay/fastchess_report.py $W/a4.pgn --p1 a4 2>&1 | sed -n '/vs the field/,$p' | show
echo "[q228] $(date) QUEUE 228 DONE"
