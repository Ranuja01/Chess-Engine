#!/bin/bash
# 2026-10-28 morning, queue #112: ship checks for threat ordering (THREAT_ORDER_W=100). Never edit this file while it runs.
# Self-play at 10+0.1 vs gen 6.2 (#111): +20.9 +- 18.8 (seed 1111), fresh-seed re-run +21.9 +- 19.3 (seed 1114).
# Binary build_uci/chess_engine_uci_thr (gen 6.2 code + THREAT_ORDER_W); the Cython .so is the same source (built 10-27 17:22).
#  1 med_ord     TM,THREAT_ORDER_W=100 vs Mediocre 0.5, 60+0.6, 200 g, seed 1071 = the openings of #107's gen 6.2 run
#                (+119.1 +- 48.2, 0 forfeits by us)
#  2 thr_ord_60  TM,THREAT_ORDER_W=100 vs TM, 60+0.6, 200 g, seed 1121 (does the gain hold at the longer clock?)
#  3 SF18 gauntlet @1100 nodes, ours at 250k nodes, 500 g each on the same seed: gen 6.2, then W100
# Registered: 1 not below gen 6.2 (point >= +119 expected, +100 ... +170); 2 +5 ... +35; 3 arm - control > -1 sigma,
# expected 0 ... +4 points.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
BIN="$ND/build_uci/chess_engine_uci_thr"
MED="$ND/selfplay/mediocre_uci.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
TM=TM_START=175,TM_SINGLE=1,TM_SCALE=7,TM_NORM=95,TM_OPT_PLY=1
cd "$ND" || exit 1
START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[q112] $START start"
nodes() { printf "$2bench\nquit\n" | "$1" 2>/dev/null | grep '^bench' | sed -n 's/.* nodes \([0-9]*\).*/\1/p'; }
nt=$(nodes "$BIN" "setoption name Knobs value $TM\n"); no=$(nodes "$BIN" "setoption name Knobs value $TM,THREAT_ORDER_W=100\n")
echo "[q112] thr: TM $nt (want 46196950) ord $no (want 42263170)"
[ "$nt" = 46196950 ] && [ "$no" = 42263170 ] || { echo "[q112] ☠️ fingerprints failed — QUEUE 112 ABORTED"; exit 1; }

COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
report() {   # $1 tag
  echo "[q112] $(date) $1 done: $(grep -E '^Games:|^Elo:' $O/$1.log | tail -2 | tr '\n' ' ')"
  echo "[q112]   $(grep -E '^Ptnml' $O/$1.log | tail -1)  disconnects/stalls: $(grep -ciE 'disconnect|stall|not responsive' $O/$1.log)  time forfeits: $(grep -c 'loses on time' $O/$1.pgn)"
}
# $1 tag, $2 arm knobs, $3 control cmd, $4 control knobs ("" = none)
preflight() {
  local copt=""; [ -n "$4" ] && copt="option.Knobs=$4"
  rm -f /tmp/q112_pre.engine.log
  timeout 180 "$FC" -engine cmd="$BIN" name=arm option.OwnBook=false option.Knobs=$2 st=0.05 \
    -engine cmd="$3" name=ctl $copt st=0.05 \
    -openings file=$O/openings_uho.epd format=epd order=random -rounds 1 -games 2 -concurrency 1 \
    -log file=/tmp/q112_pre.engine.log level=info engine=true > /tmp/q112_pre.log 2>&1
  local W K N; W=$(grep -c "have option" /tmp/q112_pre.log); K=$(grep -cF "knobs $2" /tmp/q112_pre.engine.log)
  if [ -n "$4" ]; then N=$(grep -cxF "knobs $4" <(grep -oE "knobs [A-Z0-9_=,]+" /tmp/q112_pre.engine.log)); else N=1; fi
  echo "[q112] $1 preflight: warnings $W, arm echoes $K, control echoes $N"
  [ "$W" = "0" ] && [ "$K" -ge 1 ] && [ "$N" -ge 1 ] || { echo "[q112] ☠️ $1 PREFLIGHT FAILED — skipped"; return 1; }
}

# 1: Mediocre
if preflight med_ord "$TM,THREAT_ORDER_W=100" "$MED" ""; then
  echo "[q112] $(date) med_ord (200 games, 60+0.6)"
  nice -n 5 "$FC" -engine cmd="$BIN" name=ord option.OwnBook=false option.Knobs=$TM,THREAT_ORDER_W=100 tc=60+0.6 timemargin=250 restart=on \
    -engine cmd="$MED" name=mediocre tc=60+0.6 timemargin=250 restart=on $COMMON -srand 1071 -rounds 100 -games 2 \
    -pgnout file=$O/med_ord.pgn notation=san nodes=true append=false > $O/med_ord.log 2>&1
  report med_ord
fi

# 2: self-play at 60+0.6
if preflight thr_ord_60 "$TM,THREAT_ORDER_W=100" "$BIN" "$TM"; then
  echo "[q112] $(date) thr_ord_60 (200 games, 60+0.6)"
  nice -n 5 "$FC" -engine cmd="$BIN" name=arm option.OwnBook=false option.Knobs=$TM,THREAT_ORDER_W=100 tc=60+0.6 timemargin=250 restart=on \
    -engine cmd="$BIN" name=ctl option.OwnBook=false option.Knobs=$TM tc=60+0.6 timemargin=250 restart=on $COMMON -srand 1121 -rounds 100 -games 2 \
    -pgnout file=$O/thr_ord_60.pgn notation=san nodes=true append=false > $O/thr_ord_60.log 2>&1
  report thr_ord_60
fi

# 3: SF18 gauntlet (node-limited, Cython path). Control fingerprint first: the .so must be gen 6.2.
c=$(env V2_PRESET=shipped bash "$R" wac q112_ctl); a=$(env V2_PRESET=shipped THREAT_ORDER_W=100 bash "$R" wac q112_ord)
echo "[q112] cython: ctl $(echo "$c" | grep -E '^SOLVED|^NODES' | tr '\n' ' ') · ord $(echo "$a" | grep -E '^SOLVED|^NODES' | tr '\n' ' ')"
if echo "$c" | grep -q "NODES: 46196950" && echo "$a" | grep -q "NODES: 42263170"; then
  echo "[q112] $(date) gauntlet g1100_g62_s1122"
  bash "$R" gauntlet 500 1100 3 g1100_g62_s1122 1122 V2_PRESET=shipped 2>&1 | grep -E "our score|games in"
  echo "[q112] $(date) gauntlet g1100_ord_s1122"
  bash "$R" gauntlet 500 1100 3 g1100_ord_s1122 1122 V2_PRESET=shipped THREAT_ORDER_W=100 2>&1 | grep -E "our score|games in"
else
  echo "[q112] ☠️ Cython fingerprints failed — gauntlet skipped"
fi

echo "[q112] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | sed 's/^/[q112]   /'
echo "[q112] $(date) QUEUE 112 DONE"
