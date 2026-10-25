#!/bin/bash
# 2026-10-24 night, queue #101: chained after #100. More gen 6 vs Mediocre 0.5 games to tighten the baseline the time-manager
# work will be judged against (fresh seeds; pool with #99's med_60p6 / med_st1 for 200 g each).
# Registered: 60+0.6 within ±60 of level (as #99), st=1 +60 ... +160.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
ENG="$ND/build_uci/chess_engine_uci"
MED="$ND/selfplay/mediocre_uci.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
cd "$ND" || exit 1
echo "[q101] $(date) waiting for queue 100"
until grep -q 'QUEUE 100 DONE' /mnt/e/chess_data/q100.log 2>/dev/null; do sleep 60; done
med() {   # $1 tag, $2 seed, $3 time control args, $4 rounds
  echo "[q101] $(date) $1 ($(( $4 * 2 )) games)"
  nice -n 5 "$FC" -engine cmd="$ENG" name=gen6 option.OwnBook=false $3 timemargin=250 restart=on \
    -engine cmd="$MED" name=mediocre $3 timemargin=250 restart=on \
    -openings file=$O/openings_uho.epd format=epd order=random -srand $2 -repeat -rounds $4 -games 2 -concurrency 3 -recover \
    -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10 \
    -pgnout file=$O/$1.pgn notation=san nodes=true append=false > $O/$1.log 2>&1
  echo "[q101]   $1 done: $(grep -E '^Games:|^Elo:' $O/$1.log | tail -2 | tr '\n' ' ')  disconnects $(grep -ciE 'disconnect|stall|not responsive' $O/$1.log)"
}
med med_60p6_b 1011 tc=60+0.6 50
med med_st1_b  1012 st=1      50
echo "[q101] $(date) QUEUE 101 DONE"
