#!/bin/bash
# 2026-11-14 ~01:25, queue #234: the rest of #231 (H2 vs today's root), paused at 1,450 games on 11-13 23:37 while the owner
# gamed. 550 more games (275 pairs) on the same binary chess_engine_uci_r3pgo, same settings, a FRESH seed (2341): #231's seed
# with order=random would replay its first openings. Read pooled with #231 (PGNs as separate args, pairs = games / 2) against
# #231's registered rules (header of _queue_231.sh): >= +2 sigma candidate, +-2 sigma LEVEL, <= -2 sigma BEHIND; 0 forfeits.
# Then, only if all 550 finished, writes /mnt/e/chess_data/q225.go so #225 (the SEARCH_ARM=1 default candidate) runs next.
# Stop: SIGINT to fastchess (finished games kept), or touch /mnt/e/chess_data/q234.stop before the games.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
W=/mnt/e/chess_data/r3b
B="$ND/build_uci/chess_engine_uci_r3pgo"
STOP=/mnt/e/chess_data/q234.stop
H2K="ENABLE_ROOT_PRESEARCH=0,ENABLE_ROOT_RAZOR=0,ENABLE_PRESEARCH_OFF_SORT=1,PRESEARCH_OFF_SORT_MODE=2,ENABLE_ROOT_TABLE=1,PLY1_NODE_PRUNE=3"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"
show() { sed "s/^/[q234]   /"; }
fp() { local k=""; [ -n "$2" ] && k="setoption name Knobs value $2"; printf '%s\nbench\nquit\n' "$k" | "$1" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+'; }
echo "[q234] $(date '+%Y-%m-%d %H:%M:%S') start"
d=$(fp "$B" ""); h=$(fp "$B" "$H2K")
echo "[q234] r3pgo default (want 249 / 37279382): [$d] · H2 (want 253 / 30359342): [$h]"
if [ "$d" != "solved 249/300 nodes 37279382" ] || [ "$h" != "solved 253/300 nodes 30359342" ]; then echo "[q234] ☠️ fingerprints differ — QUEUE 234 ABORTED"; exit 1; fi
[ -e "$STOP" ] && { echo "[q234] stop file present — QUEUE 234 DONE"; exit 0; }

COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
echo "[q234] $(date '+%H:%M') games: H2 vs today's root at 10+0.1, 550 games, seed 2341"
nice -n 5 "$FC" -engine "cmd=$B" name=h2 option.OwnBook=false "option.Knobs=$H2K" tc=10+0.1 timemargin=250 \
	-engine "cmd=$B" name=root0 option.OwnBook=false tc=10+0.1 timemargin=250 \
	$COMMON -srand 2341 -rounds 275 -games 2 -pgnout file=$W/h2b.pgn notation=san nodes=true append=false > $W/h2b.log 2>&1
echo "[q234] games: $(grep -E '^Games:' $W/h2b.log | tail -1) · warnings $(grep -c 'have option' $W/h2b.log) · disconnects/stalls $(grep -ciE 'disconnect|stall|not responsive' $W/h2b.log) · time forfeits $(grep -c 'loses on time' $W/h2b.pgn)"
grep -E 'Elo:|Ptnml' $W/h2b.log | tail -2 | show
echo "[q234] pooled with #231 (h2.pgn + h2b.pgn):"
timeout 900 bash "$R" pyrun selfplay/fastchess_report.py $W/h2.pgn $W/h2b.pgn --p1 h2 2>&1 | sed -n '/vs the field/,$p' | show
if grep -qE '^Games: 550,' $W/h2b.log; then
	touch /mnt/e/chess_data/q225.go && echo "[q234] all 550 finished — q225.go written (#225 runs next)"
else
	echo "[q234] fewer than 550 games — q225.go NOT written"
fi
echo "[q234] $(date) QUEUE 234 DONE"
