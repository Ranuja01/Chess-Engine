#!/bin/bash
# 2026-10-26 night, queue #108: chained after #107 (waits for "QUEUE 107 DONE"). Never edit this file while it runs.
# Binary build_uci/chess_engine_uci_c2h. TM = time manager arm E (the gen 6.2 candidate).
#  1 corr2a_10b   TM,CORR2=1 vs TM, 10+0.1, 600 g, fresh seed (pooled with #107's corr2a_10: 1,000 g)
#  2 tm4e_sf11_odds  TM at 60+0.6 vs SF 1.1 at 20+0.2, 200 g (gen 6.1 scored 22.5% in 100 g, #105)
#  3 corr2d_10    TM,CORR2=1,CORR2_QS=0 vs TM, 10+0.1, 400 g (the correction without qsearch)
# -recover; disconnects / forfeits counted; new WSL crash dumps listed.
# Registered: 1 +0 ... +20 (pooled with #107 the same); 2 22% ... 35% (no worse than gen 6.1's 22.5% by more than its noise);
# 3 at or below arm A (qsearch application helps or is neutral).
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
BIN="$ND/build_uci/chess_engine_uci_c2h"
SF1="/mnt/e/chess_data/engines/sf1.1/stockfish-1.1"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
TM=TM_START=175,TM_SINGLE=1,TM_SCALE=7,TM_NORM=95,TM_OPT_PLY=1
cd "$ND" || exit 1
echo "[q108] $(date) waiting for queue 107"
until grep -q 'QUEUE 107 DONE' /mnt/e/chess_data/q107.log 2>/dev/null; do sleep 60; done
START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[q108] $START start"
nodes() { printf "$2bench\nquit\n" | "$1" 2>/dev/null | grep '^bench' | sed -n 's/.* nodes \([0-9]*\).*/\1/p'; }
n2=$(nodes "$BIN" ''); nt=$(nodes "$BIN" "setoption name Knobs value $TM\n"); nc=$(nodes "$BIN" "setoption name Knobs value $TM,CORR2=1\n")
echo "[q108] c2h: v2 $n2 (want 46196950) TM $nt (want 46196950) TM+CORR2 $nc (want 46681752)"
[ "$n2" = 46196950 ] && [ "$nt" = 46196950 ] && [ "$nc" = 46681752 ] || { echo "[q108] ☠️ fingerprints failed — QUEUE 108 ABORTED"; exit 1; }

COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
report() {   # $1 tag
  echo "[q108] $(date) $1 done: $(grep -E '^Games:|^Elo:' $O/$1.log | tail -2 | tr '\n' ' ')"
  echo "[q108]   disconnects/stalls: $(grep -ciE 'disconnect|stall|not responsive' $O/$1.log)  time forfeits: $(grep -c 'loses on time' $O/$1.pgn)"
}
# Preflight: both knob strings reach their engines, no unknown-option warnings. $1 tag, $2 arm knobs, $3 control knobs
preflight() {
  timeout 180 "$FC" -engine cmd="$BIN" name=arm option.OwnBook=false option.Knobs=$2 st=0.05 \
    -engine cmd="$BIN" name=ctl option.OwnBook=false option.Knobs=$3 st=0.05 \
    -openings file=$O/openings_uho.epd format=epd order=random -rounds 1 -games 2 -concurrency 1 \
    -log file=/tmp/q108_pre.engine.log level=info engine=true > /tmp/q108_pre.log 2>&1
  local W K N; W=$(grep -c "have option" /tmp/q108_pre.log); K=$(grep -cF "knobs $2" /tmp/q108_pre.engine.log)
  N=$(grep -oE "knobs [A-Z0-9_=,]+" /tmp/q108_pre.engine.log | grep -cxF "knobs $3")
  echo "[q108] $1 preflight: warnings $W, arm echoes $K, control echoes $N"
  [ "$W" = "0" ] && [ "$K" -ge 1 ] && [ "$N" -ge 1 ] || { echo "[q108] ☠️ $1 PREFLIGHT FAILED — skipped"; return 1; }
}
corr() {   # $1 tag, $2 extra knobs, $3 tc, $4 rounds, $5 seed
  preflight $1 "$TM,$2" "$TM" || return
  echo "[q108] $(date) $1 ($((2 * $4)) games, $3)"
  nice -n 5 "$FC" -engine cmd="$BIN" name=arm option.OwnBook=false option.Knobs=$TM,$2 tc=$3 timemargin=250 restart=on \
    -engine cmd="$BIN" name=ctl option.OwnBook=false option.Knobs=$TM tc=$3 timemargin=250 restart=on $COMMON -srand $5 -rounds $4 -games 2 \
    -pgnout file=$O/$1.pgn notation=san nodes=true append=false > $O/$1.log 2>&1
  report $1
}

corr corr2a_10b CORR2=1 10+0.1 300 1081

# 2: Stockfish 1.1 at time odds (it gets a third of our clock)
SF1OPT="option.Threads=1 option.Ponder=false option.OwnBook=false option.Hash=128"
echo "[q108] $(date) tm4e_sf11_odds (200 games: tm4e 60+0.6, sf1.1 20+0.2)"
nice -n 5 "$FC" -engine cmd="$BIN" name=tm4e option.OwnBook=false option.Knobs=$TM tc=60+0.6 timemargin=250 restart=on \
  -engine cmd="$SF1" name=sf1.1 $SF1OPT tc=20+0.2 timemargin=250 restart=on $COMMON -srand 1082 -rounds 100 -games 2 \
  -pgnout file=$O/tm4e_sf11_odds.pgn notation=san nodes=true append=false > $O/tm4e_sf11_odds.log 2>&1
report tm4e_sf11_odds

corr corr2d_10 CORR2=1,CORR2_QS=0 10+0.1 200 1083

echo "[q108] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | sed 's/^/[q108]   /'
echo "[q108] $(date) QUEUE 108 DONE"
