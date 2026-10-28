#!/bin/bash
# 2026-10-29, queue #115: the crank 2x2 for threats in eval. A truer static eval should tolerate a tighter RFP margin
# (memory a-truer-eval-buys-pruning-headroom). Chained after #114. Never edit this file while it runs.
# Binary build_uci/chess_engine_uci_thr2. Base = TM + THREAT_ORDER_W=100 (the gen 6.3 candidate, RFP_MARGIN 1000).
# d10 (10-28, WDL calibrated): RFP 1000 -> 600 costs the base +1.9% accuracy for -14% median nodes. With threats 0/100 the
# gap to the base widens from -0.6% (0.3 sigma) to -3.7% (2.1 sigma), 50/100 from -2.6% to -3.7% (2.5 sigma); 0/100 at
# RFP 600 is more accurate than the base at 1000 (-1.9%) on 13% fewer nodes.
#  1 crank_r600_10     base + RFP_MARGIN=600               vs base, 10+0.1, 1,000 g, seed 1151
#  2 crank_m0r600_10   base + TH 0/100 + RFP_MARGIN=600    vs base, 10+0.1, 1,000 g, seed 1152
#  3 crank_m50r600_10  base + TH 50/100 + RFP_MARGIN=600   vs base, 10+0.1, 1,000 g, seed 1153
# Registered: 1 -20 ... +10 (a node saving bought with accuracy); 2 and 3 above 1 if the crank channel is real, and above
# #114's thr_m0_10 / thr_m50_10 (the same evals at RFP 1000).
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
BIN="$ND/build_uci/chess_engine_uci_thr2"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
BASE=TM_START=175,TM_SINGLE=1,TM_SCALE=7,TM_NORM=95,TM_OPT_PLY=1,THREAT_ORDER_W=100
TH=THREAT_V2_PCT=100,THREAT_V2_HANGING=1,THREAT_V2_KING=1,THREAT_V2_PAWN_TARGETS=1
M50=$TH,THREAT_V2_MG_PCT=50,THREAT_V2_EG_PCT=100
M0=$TH,THREAT_V2_MG_PCT=0,THREAT_V2_EG_PCT=100
cd "$ND" || exit 1
echo "[q115] $(date) waiting for queue 114"
until grep -q 'QUEUE 114 DONE' /mnt/e/chess_data/q114.log 2>/dev/null; do sleep 60; done
START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[q115] $START start"
nodes() { printf "$2bench\nquit\n" | "$1" 2>/dev/null | grep '^bench' | sed -n 's/.* nodes \([0-9]*\).*/\1/p'; }
n0=$(nodes "$BIN" ''); nb=$(nodes "$BIN" "setoption name Knobs value $BASE\n")
nr=$(nodes "$BIN" "setoption name Knobs value $BASE,RFP_MARGIN=600\n")
nm0=$(nodes "$BIN" "setoption name Knobs value $BASE,$M0,RFP_MARGIN=600\n")
n50=$(nodes "$BIN" "setoption name Knobs value $BASE,$M50,RFP_MARGIN=600\n")
echo "[q115] thr2: v2 $n0 (want 46196950) base $nb (want 42263170) r600 $nr m0r600 $nm0 m50r600 $n50 (each must differ from base)"
[ "$n0" = 46196950 ] && [ "$nb" = 42263170 ] && [ -n "$nr" ] && [ "$nr" != "$nb" ] && [ -n "$nm0" ] && [ "$nm0" != "$nr" ] \
  && [ -n "$n50" ] && [ "$n50" != "$nr" ] || { echo "[q115] ☠️ fingerprints failed — QUEUE 115 ABORTED"; exit 1; }

COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
report() {   # $1 tag
  echo "[q115] $(date) $1 done: $(grep -E '^Games:|^Elo:' $O/$1.log | tail -2 | tr '\n' ' ')"
  echo "[q115]   $(grep -E '^Ptnml' $O/$1.log | tail -1)  disconnects/stalls: $(grep -ciE 'disconnect|stall|not responsive' $O/$1.log)  time forfeits: $(grep -c 'loses on time' $O/$1.pgn)"
}
preflight() {   # $1 tag, $2 arm knobs, $3 control knobs
  rm -f /tmp/q115_pre.engine.log
  timeout 180 "$FC" -engine cmd="$BIN" name=arm option.OwnBook=false option.Knobs=$2 st=0.05 \
    -engine cmd="$BIN" name=ctl option.OwnBook=false option.Knobs=$3 st=0.05 \
    -openings file=$O/openings_uho.epd format=epd order=random -rounds 1 -games 2 -concurrency 1 \
    -log file=/tmp/q115_pre.engine.log level=info engine=true > /tmp/q115_pre.log 2>&1
  local W K N; W=$(grep -c "have option" /tmp/q115_pre.log)
  K=$(grep -oE "knobs [A-Z0-9_=,]+" /tmp/q115_pre.engine.log | grep -cxF "knobs $2")
  N=$(grep -oE "knobs [A-Z0-9_=,]+" /tmp/q115_pre.engine.log | grep -cxF "knobs $3")
  echo "[q115] $1 preflight: warnings $W, arm echoes $K, control echoes $N"
  [ "$W" = "0" ] && [ "$K" -ge 1 ] && [ "$N" -ge 1 ] || { echo "[q115] ☠️ $1 PREFLIGHT FAILED — skipped"; return 1; }
}
arm() {   # $1 tag, $2 extra knobs, $3 tc, $4 rounds, $5 seed
  preflight $1 "$BASE,$2" "$BASE" || return
  echo "[q115] $(date) $1 ($((2 * $4)) games, $3)"
  nice -n 5 "$FC" -engine cmd="$BIN" name=arm option.OwnBook=false option.Knobs=$BASE,$2 tc=$3 timemargin=250 restart=on \
    -engine cmd="$BIN" name=ctl option.OwnBook=false option.Knobs=$BASE tc=$3 timemargin=250 restart=on $COMMON -srand $5 -rounds $4 -games 2 \
    -pgnout file=$O/$1.pgn notation=san nodes=true append=false > $O/$1.log 2>&1
  report $1
}
arm crank_r600_10    RFP_MARGIN=600       10+0.1 500 1151
arm crank_m0r600_10  $M0,RFP_MARGIN=600   10+0.1 500 1152
arm crank_m50r600_10 $M50,RFP_MARGIN=600  10+0.1 500 1153

echo "[q115] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | sed 's/^/[q115]   /'
echo "[q115] $(date) QUEUE 115 DONE"
