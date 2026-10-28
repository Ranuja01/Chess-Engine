#!/bin/bash
# 2026-10-28 afternoon, queue #113: plateau check for threat ordering before the gen 6.3 ship. Never edit this file while it runs.
# Control = W100 (the candidate: +20.9 / +21.9 vs gen 6.2 at 10+0.1, +41.9 at 60+0.6, SF18 @1100 53.8% vs 50.4%, Mediocre flat).
#  1 thr_w50_10   TM,THREAT_ORDER_W=50  vs TM,THREAT_ORDER_W=100, 10+0.1, 1,000 g, seed 1131
#  2 thr_w200_10  TM,THREAT_ORDER_W=200 vs TM,THREAT_ORDER_W=100, 10+0.1, 1,000 g, seed 1132
# Registered: W100 is at or near the plateau if both arms read -15 ... +10. An arm at > +10 is re-run on a fresh seed
# before it replaces W100; otherwise W100 ships.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
BIN="$ND/build_uci/chess_engine_uci_thr"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
TM=TM_START=175,TM_SINGLE=1,TM_SCALE=7,TM_NORM=95,TM_OPT_PLY=1
cd "$ND" || exit 1
START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[q113] $START start"
nodes() { printf "$2bench\nquit\n" | "$1" 2>/dev/null | grep '^bench' | sed -n 's/.* nodes \([0-9]*\).*/\1/p'; }
nt=$(nodes "$BIN" "setoption name Knobs value $TM\n"); n100=$(nodes "$BIN" "setoption name Knobs value $TM,THREAT_ORDER_W=100\n")
n50=$(nodes "$BIN" "setoption name Knobs value $TM,THREAT_ORDER_W=50\n"); n200=$(nodes "$BIN" "setoption name Knobs value $TM,THREAT_ORDER_W=200\n")
echo "[q113] thr: TM $nt (want 46196950) W100 $n100 (want 42263170) W50 $n50 W200 $n200 (each must differ from W100)"
[ "$nt" = 46196950 ] && [ "$n100" = 42263170 ] && [ -n "$n50" ] && [ "$n50" != "$n100" ] && [ -n "$n200" ] && [ "$n200" != "$n100" ] \
  || { echo "[q113] ☠️ fingerprints failed — QUEUE 113 ABORTED"; exit 1; }

COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
report() {   # $1 tag
  echo "[q113] $(date) $1 done: $(grep -E '^Games:|^Elo:' $O/$1.log | tail -2 | tr '\n' ' ')"
  echo "[q113]   $(grep -E '^Ptnml' $O/$1.log | tail -1)  disconnects/stalls: $(grep -ciE 'disconnect|stall|not responsive' $O/$1.log)  time forfeits: $(grep -c 'loses on time' $O/$1.pgn)"
}
preflight() {   # $1 tag, $2 arm knobs, $3 control knobs
  rm -f /tmp/q113_pre.engine.log
  timeout 180 "$FC" -engine cmd="$BIN" name=arm option.OwnBook=false option.Knobs=$2 st=0.05 \
    -engine cmd="$BIN" name=ctl option.OwnBook=false option.Knobs=$3 st=0.05 \
    -openings file=$O/openings_uho.epd format=epd order=random -rounds 1 -games 2 -concurrency 1 \
    -log file=/tmp/q113_pre.engine.log level=info engine=true > /tmp/q113_pre.log 2>&1
  local W K N; W=$(grep -c "have option" /tmp/q113_pre.log)
  K=$(grep -oE "knobs [A-Z0-9_=,]+" /tmp/q113_pre.engine.log | grep -cxF "knobs $2")
  N=$(grep -oE "knobs [A-Z0-9_=,]+" /tmp/q113_pre.engine.log | grep -cxF "knobs $3")
  echo "[q113] $1 preflight: warnings $W, arm echoes $K, control echoes $N"
  [ "$W" = "0" ] && [ "$K" -ge 1 ] && [ "$N" -ge 1 ] || { echo "[q113] ☠️ $1 PREFLIGHT FAILED — skipped"; return 1; }
}
arm() {   # $1 tag, $2 W, $3 seed
  local A="$TM,THREAT_ORDER_W=$2" C="$TM,THREAT_ORDER_W=100"
  preflight $1 "$A" "$C" || return
  echo "[q113] $(date) $1 (1000 games, 10+0.1)"
  nice -n 5 "$FC" -engine cmd="$BIN" name=arm option.OwnBook=false option.Knobs=$A tc=10+0.1 timemargin=250 restart=on \
    -engine cmd="$BIN" name=ctl option.OwnBook=false option.Knobs=$C tc=10+0.1 timemargin=250 restart=on $COMMON -srand $3 -rounds 500 -games 2 \
    -pgnout file=$O/$1.pgn notation=san nodes=true append=false > $O/$1.log 2>&1
  report $1
}
arm thr_w50_10 50 1131
arm thr_w200_10 200 1132

echo "[q113] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | sed 's/^/[q113]   /'
echo "[q113] $(date) QUEUE 113 DONE"
