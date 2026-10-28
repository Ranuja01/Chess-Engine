#!/bin/bash
# 2026-10-28 evening, queue #114: threats-in-eval re-test step 3 (games), on top of threat ordering. Chained after #113.
# Never edit this file while it runs. Binary build_uci/chess_engine_uci_thr2 (= thr + THREAT_V2_MG_PCT / EG_PCT, 852057b).
# Base = TM + THREAT_ORDER_W=100 (the gen 6.3 candidate). TH = the threats legs as before.
#  1 thr_m50_10  base + TH, MG 50 / EG 100   vs base, 10+0.1, 1,000 g, seed 1141 (best at d10: -2.6% WDL calibrated, 1.25 sigma)
#  2 thr_m0_10   base + TH, MG 0 / EG 100    vs base, 10+0.1, 1,000 g, seed 1142 (best static: -1.8%, 3.4 sigma; d10 -0.6%)
#  3 the better arm if positive: 60+0.6, 200 g, seed 1143; otherwise stop.
# Bench: base 256 / 42,263,170 · m50 252 / 43,040,865 · m0 253 / 43,885,134.
# Registered: 1 -10 ... +15; 2 -10 ... +10. Both <= 0 => threats in eval parks to the full retune (register).
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
echo "[q114] $(date) waiting for queue 113"
until grep -q 'QUEUE 113 DONE' /mnt/e/chess_data/q113.log 2>/dev/null; do sleep 60; done
START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[q114] $START start"
nodes() { printf "$2bench\nquit\n" | "$1" 2>/dev/null | grep '^bench' | sed -n 's/.* nodes \([0-9]*\).*/\1/p'; }
n0=$(nodes "$BIN" ''); nb=$(nodes "$BIN" "setoption name Knobs value $BASE\n")
n50=$(nodes "$BIN" "setoption name Knobs value $BASE,$M50\n"); nm0=$(nodes "$BIN" "setoption name Knobs value $BASE,$M0\n")
echo "[q114] thr2: v2 $n0 (want 46196950) base $nb (want 42263170) m50 $n50 (want 43040865) m0 $nm0 (want 43885134)"
[ "$n0" = 46196950 ] && [ "$nb" = 42263170 ] && [ "$n50" = 43040865 ] && [ "$nm0" = 43885134 ] \
  || { echo "[q114] ☠️ fingerprints failed — QUEUE 114 ABORTED"; exit 1; }

COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
report() {   # $1 tag
  echo "[q114] $(date) $1 done: $(grep -E '^Games:|^Elo:' $O/$1.log | tail -2 | tr '\n' ' ')"
  echo "[q114]   $(grep -E '^Ptnml' $O/$1.log | tail -1)  disconnects/stalls: $(grep -ciE 'disconnect|stall|not responsive' $O/$1.log)  time forfeits: $(grep -c 'loses on time' $O/$1.pgn)"
}
elo() { grep -E '^Elo:' $O/$1.log | tail -1 | awk '{print $2}' | tr -d ','; }
preflight() {   # $1 tag, $2 arm knobs, $3 control knobs
  rm -f /tmp/q114_pre.engine.log
  timeout 180 "$FC" -engine cmd="$BIN" name=arm option.OwnBook=false option.Knobs=$2 st=0.05 \
    -engine cmd="$BIN" name=ctl option.OwnBook=false option.Knobs=$3 st=0.05 \
    -openings file=$O/openings_uho.epd format=epd order=random -rounds 1 -games 2 -concurrency 1 \
    -log file=/tmp/q114_pre.engine.log level=info engine=true > /tmp/q114_pre.log 2>&1
  local W K N; W=$(grep -c "have option" /tmp/q114_pre.log)
  K=$(grep -oE "knobs [A-Z0-9_=,]+" /tmp/q114_pre.engine.log | grep -cxF "knobs $2")
  N=$(grep -oE "knobs [A-Z0-9_=,]+" /tmp/q114_pre.engine.log | grep -cxF "knobs $3")
  echo "[q114] $1 preflight: warnings $W, arm echoes $K, control echoes $N"
  [ "$W" = "0" ] && [ "$K" -ge 1 ] && [ "$N" -ge 1 ] || { echo "[q114] ☠️ $1 PREFLIGHT FAILED — skipped"; return 1; }
}
arm() {   # $1 tag, $2 extra knobs, $3 tc, $4 rounds, $5 seed
  preflight $1 "$BASE,$2" "$BASE" || return
  echo "[q114] $(date) $1 ($((2 * $4)) games, $3)"
  nice -n 5 "$FC" -engine cmd="$BIN" name=arm option.OwnBook=false option.Knobs=$BASE,$2 tc=$3 timemargin=250 restart=on \
    -engine cmd="$BIN" name=ctl option.OwnBook=false option.Knobs=$BASE tc=$3 timemargin=250 restart=on $COMMON -srand $5 -rounds $4 -games 2 \
    -pgnout file=$O/$1.pgn notation=san nodes=true append=false > $O/$1.log 2>&1
  report $1
}
arm thr_m50_10 $M50 10+0.1 500 1141
arm thr_m0_10  $M0  10+0.1 500 1142

best=""; bestv=0
for t in thr_m50_10:$M50 thr_m0_10:$M0; do
  tag=${t%%:*}; k=${t#*:}; v=$(elo $tag)
  [ -n "$v" ] && awk -v a="$v" -v b="$bestv" 'BEGIN{exit !(a > b)}' && { best=$tag; bestk=$k; bestv=$v; }
done
if [ -n "$best" ]; then
  echo "[q114] best positive arm: $best ($bestv) — 60+0.6"
  arm ${best%_10}_60 $bestk 60+0.6 100 1143
else
  echo "[q114] no arm positive — no 60+0.6 run"
fi

echo "[q114] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | sed 's/^/[q114]   /'
echo "[q114] $(date) QUEUE 114 DONE"
