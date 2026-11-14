#!/bin/bash
# 2026-11-14 11:10, queue #236: the rest of #225 (SW5: the v2 interior as the default, g73spgo vs production g73fpgo), cut at
# 1,695 g by a power outage ~06:58. 306 more games (153 pairs), same binaries and settings, a fresh seed (2251). Read pooled with
# sw5.pgn (PGNs as separate args) against #225's registered rule: NOT HARMFUL = above -2 sigma with 0 forfeits / disconnects.
# The owner games from ~13:55: a timer SIGINTs fastchess at 13:40 (finished games kept).
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
W=/mnt/e/chess_data/q225
NEW="$ND/build_uci/chess_engine_uci_g73spgo"
PRODF="$ND/build_uci/chess_engine_uci_g73fpgo"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
fp() { printf 'bench\nquit\n' | "$1" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+'; }
echo "[q236] $(date '+%Y-%m-%d %H:%M:%S') start"
a=$(fp "$NEW"); b=$(fp "$PRODF")
echo "[q236] g73spgo [$a] · g73fpgo [$b] (want 249 / 37279382 both)"
[ "$a" = "solved 249/300 nodes 37279382" ] && [ "$b" = "$a" ] || { echo "[q236] ☠️ fingerprints differ — QUEUE 236 ABORTED"; exit 1; }
( while [ "$(date +%H%M)" -lt 1340 ]; do sleep 30; done; pkill -INT -f "fastchess-linux-x86-64/fastchess" && echo "[q236] $(date '+%H:%M') 13:40 guard: fastchess interrupted" ) &
GUARD=$!
COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
nice -n 5 "$FC" -engine "cmd=$NEW" name=v2def option.OwnBook=false tc=10+0.1 timemargin=250 \
	-engine "cmd=$PRODF" name=prod option.OwnBook=false tc=10+0.1 timemargin=250 \
	$COMMON -srand 2251 -rounds 153 -games 2 -pgnout file=$W/sw5b.pgn notation=san nodes=true append=false > $W/sw5b.log 2>&1
kill $GUARD 2>/dev/null
echo "[q236] games: $(grep -E '^Games:' $W/sw5b.log | tail -1) · warnings $(grep -c 'have option' $W/sw5b.log) · disconnects/stalls $(grep -ciE 'disconnect|stall|not responsive' $W/sw5b.log) · time forfeits $(grep -c 'loses on time' $W/sw5b.pgn)"
echo "[q236] pooled with sw5.pgn:"
timeout 900 bash "$R" pyrun selfplay/fastchess_report.py $W/sw5.pgn $W/sw5b.pgn --p1 v2def 2>&1 | sed -n '/vs the field/,$p' | sed 's/^/[q236]   /'
echo "[q236] $(date) QUEUE 236 DONE"
