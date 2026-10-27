#!/bin/bash
# 2026-10-27, queue #110: corrhist quarter-weight confirmation on a fresh seed. Never edit this file while it runs.
# Binary build_uci/chess_engine_uci_c2h (plain build, as #107/#108). TM = gen 6.2's time manager knobs on both sides.
#  1 corr2q_10b TM,CORR2=1,CORR2_WP=88,CORR2_WN=100 vs TM, 10+0.1, 1,000 g, seed 1093 (fresh)
# #109: full weights -8.7 +- 18.4, half -2.4 +- 19.6, quarter +7.6 +- 20.0 (seed 1092), each 1,000 g.
# Registered: -10 ... +10. Rule: point estimate <= 0 parks corrhist; > 0 keeps it alive but a ship still needs the pooled
# self-play result >= +2 sigma, which at this size needs several thousand more games.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
BIN="$ND/build_uci/chess_engine_uci_c2h"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
TM=TM_START=175,TM_SINGLE=1,TM_SCALE=7,TM_NORM=95,TM_OPT_PLY=1
cd "$ND" || exit 1
START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[q110] $START start"
nodes() { printf "$2bench\nquit\n" | "$1" 2>/dev/null | grep '^bench' | sed -n 's/.* nodes \([0-9]*\).*/\1/p'; }
n2=$(nodes "$BIN" ''); nt=$(nodes "$BIN" "setoption name Knobs value $TM\n"); nc=$(nodes "$BIN" "setoption name Knobs value $TM,CORR2=1\n")
nh=$(nodes "$BIN" "setoption name Knobs value $TM,CORR2=1,CORR2_WP=175,CORR2_WN=200\n")
echo "[q110] c2h: v2 $n2 (want 46196950) TM $nt (want 46196950) TM+CORR2 $nc (want 46681752) half $nh (must differ from $nc)"
[ "$n2" = 46196950 ] && [ "$nt" = 46196950 ] && [ "$nc" = 46681752 ] && [ -n "$nh" ] && [ "$nh" != "$nc" ] && [ "$nh" != "$nt" ] \
  || { echo "[q110] ☠️ fingerprints failed — QUEUE 110 ABORTED"; exit 1; }

COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
report() {   # $1 tag
  echo "[q110] $(date) $1 done: $(grep -E '^Games:|^Elo:' $O/$1.log | tail -2 | tr '\n' ' ')"
  echo "[q110]   $(grep -E '^Ptnml' $O/$1.log | tail -1)  disconnects/stalls: $(grep -ciE 'disconnect|stall|not responsive' $O/$1.log)  time forfeits: $(grep -c 'loses on time' $O/$1.pgn)"
}
preflight() {   # $1 tag, $2 arm knobs, $3 control knobs
  timeout 180 "$FC" -engine cmd="$BIN" name=arm option.OwnBook=false option.Knobs=$2 st=0.05 \
    -engine cmd="$BIN" name=ctl option.OwnBook=false option.Knobs=$3 st=0.05 \
    -openings file=$O/openings_uho.epd format=epd order=random -rounds 1 -games 2 -concurrency 1 \
    -log file=/tmp/q110_pre.engine.log level=info engine=true > /tmp/q110_pre.log 2>&1
  local W K N; W=$(grep -c "have option" /tmp/q110_pre.log); K=$(grep -cF "knobs $2" /tmp/q110_pre.engine.log)
  N=$(grep -oE "knobs [A-Z0-9_=,]+" /tmp/q110_pre.engine.log | grep -cxF "knobs $3")
  echo "[q110] $1 preflight: warnings $W, arm echoes $K, control echoes $N"
  [ "$W" = "0" ] && [ "$K" -ge 1 ] && [ "$N" -ge 1 ] || { echo "[q110] ☠️ $1 PREFLIGHT FAILED — skipped"; return 1; }
}
corr() {   # $1 tag, $2 extra knobs, $3 tc, $4 rounds, $5 seed
  preflight $1 "$TM,$2" "$TM" || return
  echo "[q110] $(date) $1 ($((2 * $4)) games, $3)"
  nice -n 5 "$FC" -engine cmd="$BIN" name=arm option.OwnBook=false option.Knobs=$TM,$2 tc=$3 timemargin=250 restart=on \
    -engine cmd="$BIN" name=ctl option.OwnBook=false option.Knobs=$TM tc=$3 timemargin=250 restart=on $COMMON -srand $5 -rounds $4 -games 2 \
    -pgnout file=$O/$1.pgn notation=san nodes=true append=false > $O/$1.log 2>&1
  report $1
}

corr corr2q_10b CORR2=1,CORR2_WP=88,CORR2_WN=100 10+0.1 500 1093

echo "[q110] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | sed 's/^/[q110]   /'
echo "[q110] $(date) QUEUE 110 DONE"
