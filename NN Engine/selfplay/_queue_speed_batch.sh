#!/bin/bash
# 2026-10-22, queue #93 — the speed batch's equal-time games: today's binary (S1-S4, result-identical speed-ups) vs this morning's
# (build_uci/chess_engine_uci_cnt = gen 5 + inert counters). Same search decisions, so any Elo is pure speed. st=1, 400 g, UHO,
# fresh seed, restart=on, concurrency 3, book off. Registered: +10 … +30 Elo (≈ +25-30% nps ≈ +0.3-0.4 ply).
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
NEW="$ND/build_uci/chess_engine_uci"
OLD="$ND/build_uci/chess_engine_uci_cnt"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
EXIT_AT=${EXIT_AT:-11:00}
cd "$ND" || exit 1
echo "[q93] $(date) start"
for b in "$NEW" "$OLD"; do
  FP=$(printf 'bench\nquit\n' | "$b" 2>/dev/null | grep "^bench")
  echo "[q93] $(basename "$b"): $FP"
  echo "$FP" | grep -q "solved 258/300 nodes 47689586" || { echo "[q93] ☠️ BENCH MISMATCH — aborting"; exit 1; }
done
( while true; do sleep 1800; echo "[q93] $(date +%H:%M) mem: $(free -g | awk '/Mem:/{print "used " $3 "G avail " $7 "G"}')"; done ) &
MEMPID=$!
OURS="option.OwnBook=false st=1 timemargin=250 restart=on"
secs=$(( $(date -d "$EXIT_AT" +%s) - $(date +%s) )); [ $secs -lt 0 ] && secs=$(( secs + 86400 ))
echo "[q93] $(date) speed_batch (400 games), up to ${secs}s"
timeout $secs nice -n 5 $FC -engine cmd="$NEW" name=speed_s1s4 $OURS -engine cmd="$OLD" name=gen5_am $OURS \
  -openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -resign movecount=3 score=1000 \
  -draw movenumber=40 movecount=8 score=10 -srand 931 -rounds 200 -games 2 \
  -pgnout file=$O/speed_batch.pgn notation=san nodes=true append=false > $O/speed_batch.log 2>&1
echo "[q93] $(date) done (rc $?): $(grep -E '^Games:|^Elo:' $O/speed_batch.log | tail -2 | tr '\n' ' ') $(grep -c 'not responsive' $O/speed_batch.log) unresponsive"
kill $MEMPID 2>/dev/null
echo "[q93] $(date) QUEUE 93 DONE"
