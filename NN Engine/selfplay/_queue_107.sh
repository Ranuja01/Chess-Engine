#!/bin/bash
# 2026-10-26 evening, queue #107 (overnight). Never edit this file while it runs.
# Binary build_uci/chess_engine_uci_c2h (= tm4b + the corrhist correction-size counter; result-identical).
# TM = time manager arm E from #106 (+59.6 +-46 vs gen 6.1, 0 forfeits): TM_START=175,TM_SINGLE=1,TM_SCALE=7,TM_NORM=95,TM_OPT_PLY=1
#  1 med_tm4e     TM vs Mediocre 0.5, 60+0.6, 200 g  (gen 6.1 measured +96.2 [+51.9, +143.8] in #102 under the same conditions)
#  2 corr2a_10    TM,CORR2=1 vs TM, 10+0.1, 400 g
#  3 corr2a_60    TM,CORR2=1 vs TM, 60+0.6, 200 g
#  4 corr2c_10    TM,CORR2=1,CORR2_GLOBAL=1 vs TM, 10+0.1, 400 g  (one slot per colour: does the keying matter?)
# -recover; disconnects / forfeits counted; new WSL crash dumps listed.
# Registered: 1 +70 ... +180, 0 forfeits by us; 2 +0 ... +20; 3 +5 ... +25 (>= 2); 4 -10 ... +5 (below 2).
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
BIN="$ND/build_uci/chess_engine_uci_c2h"
MED="$ND/selfplay/mediocre_uci.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
TM=TM_START=175,TM_SINGLE=1,TM_SCALE=7,TM_NORM=95,TM_OPT_PLY=1
cd "$ND" || exit 1
START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[q107] $START start"
nodes() { printf "$2bench\nquit\n" | "$1" 2>/dev/null | grep '^bench' | sed -n 's/.* nodes \([0-9]*\).*/\1/p'; }
n2=$(nodes "$BIN" ''); n1=$(nodes "$BIN" 'setoption name EvalArm value 0\n'); n5=$(nodes "$BIN" 'setoption name Knobs value V2_SHIP_GEN=5\n')
nt=$(nodes "$BIN" "setoption name Knobs value $TM\n"); nc=$(nodes "$BIN" "setoption name Knobs value $TM,CORR2=1\n")
echo "[q107] c2h: v2 $n2 (want 46196950) v1 $n1 (want 35310778) gen5 $n5 (want 47689586) TM $nt (want 46196950) TM+CORR2 $nc (want 46681752)"
[ "$n2" = 46196950 ] && [ "$n1" = 35310778 ] && [ "$n5" = 47689586 ] && [ "$nt" = 46196950 ] && [ "$nc" = 46681752 ] \
  || { echo "[q107] ☠️ fingerprints failed — QUEUE 107 ABORTED"; exit 1; }

COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
report() {   # $1 tag
  echo "[q107] $(date) $1 done: $(grep -E '^Games:|^Elo:' $O/$1.log | tail -2 | tr '\n' ' ')"
  echo "[q107]   disconnects/stalls: $(grep -ciE 'disconnect|stall|not responsive' $O/$1.log)  time forfeits: $(grep -c 'loses on time' $O/$1.pgn)"
}
# Preflight: both knob strings reach their engines, no unknown-option warnings. $1 tag, $2 arm knobs, $3 control cmd, $4 control knobs ("" = none)
preflight() {
  local copt=""; [ -n "$4" ] && copt="option.Knobs=$4"
  timeout 180 "$FC" -engine cmd="$BIN" name=arm option.OwnBook=false option.Knobs=$2 st=0.05 \
    -engine cmd="$3" name=ctl $copt st=0.05 \
    -openings file=$O/openings_uho.epd format=epd order=random -rounds 1 -games 2 -concurrency 1 \
    -log file=/tmp/q107_pre.engine.log level=info engine=true > /tmp/q107_pre.log 2>&1
  local W K N; W=$(grep -c "have option" /tmp/q107_pre.log); K=$(grep -cF "knobs $2" /tmp/q107_pre.engine.log)
  if [ -n "$4" ]; then N=$(grep -cxF "knobs $4" <(grep -oE "knobs [A-Z0-9_=,]+" /tmp/q107_pre.engine.log)); else N=1; fi
  echo "[q107] $1 preflight: warnings $W, arm echoes $K, control echoes $N"
  [ "$W" = "0" ] && [ "$K" -ge 1 ] && [ "$N" -ge 1 ] || { echo "[q107] ☠️ $1 PREFLIGHT FAILED — skipped"; return 1; }
}

# 1: the time-manager candidate vs Mediocre
if preflight med_tm4e "$TM" "$MED" ""; then
  echo "[q107] $(date) med_tm4e (200 games)"
  nice -n 5 "$FC" -engine cmd="$BIN" name=tm4e option.OwnBook=false option.Knobs=$TM tc=60+0.6 timemargin=250 restart=on \
    -engine cmd="$MED" name=mediocre tc=60+0.6 timemargin=250 restart=on $COMMON -srand 1071 -rounds 100 -games 2 \
    -pgnout file=$O/med_tm4e.pgn notation=san nodes=true append=false > $O/med_tm4e.log 2>&1
  report med_tm4e
fi

# 2-4: corrhist arms against the same binary with the time-manager knobs only
corr() {   # $1 tag, $2 extra knobs, $3 tc, $4 rounds, $5 seed
  preflight $1 "$TM,$2" "$BIN" "$TM" || return
  echo "[q107] $(date) $1 ($((2 * $4)) games, $3)"
  nice -n 5 "$FC" -engine cmd="$BIN" name=arm option.OwnBook=false option.Knobs=$TM,$2 tc=$3 timemargin=250 restart=on \
    -engine cmd="$BIN" name=ctl option.OwnBook=false option.Knobs=$TM tc=$3 timemargin=250 restart=on $COMMON -srand $5 -rounds $4 -games 2 \
    -pgnout file=$O/$1.pgn notation=san nodes=true append=false > $O/$1.log 2>&1
  report $1
}
corr corr2a_10 CORR2=1                10+0.1 200 1072
corr corr2a_60 CORR2=1                60+0.6 100 1073
corr corr2c_10 CORR2=1,CORR2_GLOBAL=1 10+0.1 200 1074

echo "[q107] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | sed 's/^/[q107]   /'
echo "[q107] $(date) QUEUE 107 DONE"
