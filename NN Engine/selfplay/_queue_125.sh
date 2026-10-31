#!/bin/bash
# 2026-10-31 late morning, queue #125: follow-through on #124's damp + RFP 500 (+15.3 ± 19.1, crank ≈ 1.5σ over damp alone).
# All vs gen 6.4, 3 games at a time, binary chess_engine_uci_damp for every leg (its defaults = gen 6.4 exactly).
#  1 damp_rfp500_r  damp + RFP_MARGIN 500, 1,000 g UHO 10+0.1, seed 1251 (fresh-seed replication)
#  2 rfp500         RFP_MARGIN 500 alone, 1,000 g UHO 10+0.1, seed 1252 (control: is the +15 the margin, not the crank?)
#  3 damp_rfp500_60 damp + RFP 500, 200 g UHO 60+0.6, seed 1253         } only if leg 1 scores > 0
#  4 damp_rfp400    damp + RFP_MARGIN 400, 1,000 g UHO 10+0.1, seed 1254 } (plateau on the margin side)
# Registered: 1 −5 … +25 · 2 −15 … +10 (RFP 400 alone was −5.2 in #116) · 3 −10 … +40 · 4 −10 … +25.
# Reading: the crank holds if 1 pooled with #124 stays ≥ 1.5σ AND 1 beats 2 by > 1σ. Then the ship checks (SF18, Mediocre).
# Stop between legs: touch /mnt/e/chess_data/q125.stop (the leg in flight finishes). Never edit this file while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
DMP="$ND/build_uci/chess_engine_uci_damp"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
STOP=/mnt/e/chess_data/q125.stop
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
BASE=THREAT_ORDER_W=100
DAMP=$BASE,WIN_V2=1,WIN_V2_DAMP=1,WIN_V2_PHASE_MAX=0,WIN_V2_SKIP_PAWNEND=1,WIN_V2_PASSED=42,WIN_V2_PAWNS=52,WIN_V2_OUTFLANK=42,WIN_V2_INFILT=56,WIN_V2_FLANKS=98,WIN_V2_UNWIN=-202,WIN_V2_BASE=-469
cd "$ND" || exit 1
rm -f "$STOP"
echo "[q125] $(date) waiting for the box to clear"
while pgrep -f "fastchess|vs_sf.py|_queue_124.sh" >/dev/null; do sleep 30; done
START=$(date '+%Y-%m-%d %H:%M:%S')
DEADLINE=$(date -d 'tomorrow 08:00' +%s)
echo "[q125] $START start (deadline $(date -d @$DEADLINE); stop file $STOP)"
nodes() { printf "setoption name Knobs value $2\nbench\nquit\n" | "$1" 2>/dev/null | grep '^bench' | sed -n 's/.* nodes \([0-9]*\).*/\1/p'; }
db=$(nodes "$DMP" $BASE); dd=$(nodes "$DMP" $DAMP); dk=$(nodes "$DMP" $DAMP,RFP_MARGIN=500)
dr=$(nodes "$DMP" $BASE,RFP_MARGIN=500); d4=$(nodes "$DMP" $DAMP,RFP_MARGIN=400)
echo "[q125] damp: base $db (want 36991901) damp $dd (want 35917541) damp+rfp500 $dk (want 37179696) rfp500 $dr damp+rfp400 $d4 (must differ)"
[ "$db" = 36991901 ] && [ "$dd" = 35917541 ] && [ "$dk" = 37179696 ] && [ -n "$dr" ] && [ "$dr" != "$db" ] \
  && [ -n "$d4" ] && [ "$d4" != "$dk" ] \
  || { echo "[q125] ☠️ fingerprints failed — QUEUE 125 ABORTED"; exit 1; }

ADJ="-repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
preflight() {   # $1 binary, $2 arm knobs
  timeout 180 "$FC" -engine cmd="$1" name=arm option.OwnBook=false option.Knobs=$2 st=0.05 \
    -engine cmd="$1" name=ctl option.OwnBook=false option.Knobs=$BASE st=0.05 \
    -openings file=$O/openings_uho.epd format=epd order=random -rounds 1 -games 2 -concurrency 1 \
    -log file=/tmp/q125_pre.engine.log level=info engine=true > /tmp/q125_pre.log 2>&1
  local W K N; W=$(grep -c "have option" /tmp/q125_pre.log); K=$(grep -cF "knobs $2" /tmp/q125_pre.engine.log)
  N=$(grep -oE "knobs [A-Z0-9_=,-]+" /tmp/q125_pre.engine.log | grep -cxF "knobs $BASE")
  echo "[q125] preflight: warnings $W, arm echoes $K, base echoes $N"
  [ "$W" = "0" ] && [ "$K" -ge 1 ] && [ "$N" -ge 1 ]
}
run() {   # $1 tag, $2 binary, $3 arm knobs, $4 tc, $5 seed, $6 rounds, $7 openings, $8 seconds needed
  [ -f "$STOP" ] && { echo "[q125] $(date) skip $1: stop file present"; return; }
  local left=$(( DEADLINE - $(date +%s) ))
  [ $left -ge $8 ] || { echo "[q125] $(date) skip $1: only ${left}s before the deadline (needs $8)"; return; }
  preflight "$2" "$3" || { echo "[q125] ☠️ preflight failed — skipping $1"; return; }
  echo "[q125] $(date) $1 ($(( $6 * 2 )) games, $4)"
  nice -n 5 "$FC" -engine cmd="$2" name=arm option.OwnBook=false option.Knobs=$3 tc=$4 timemargin=250 \
    -engine cmd="$2" name=g64 option.OwnBook=false option.Knobs=$BASE tc=$4 timemargin=250 \
    -openings file=$7 format=epd order=random $ADJ -srand $5 -rounds $6 -games 2 \
    -pgnout file=$O/$1.pgn notation=san nodes=true append=false > $O/$1.log 2>&1
  echo "[q125] $(date) $1 done: $(grep -E '^Games:|^Elo:' $O/$1.log | tail -2 | tr '\n' ' ')"
  echo "[q125]   $(grep -E '^Ptnml' $O/$1.log | tail -1)  disconnects/stalls: $(grep -ciE 'disconnect|stall|not responsive' $O/$1.log)  time forfeits: $(grep -c 'loses on time' $O/$1.pgn)"
}
UHO=$O/openings_uho.epd
run damp_rfp500_r  "$DMP" "$DAMP,RFP_MARGIN=500" 10+0.1 1251 500 $UHO 10200
run rfp500         "$DMP" "$BASE,RFP_MARGIN=500" 10+0.1 1252 500 $UHO 10200
E1=$(grep -E '^Elo:' $O/damp_rfp500_r.log 2>/dev/null | tail -1 | awk '{print $2}')
if awk -v e="${E1:-0}" 'BEGIN{exit !(e > 0)}'; then
  run damp_rfp500_60 "$DMP" "$DAMP,RFP_MARGIN=500" 60+0.6 1253 100 $UHO 12600
  run damp_rfp400    "$DMP" "$DAMP,RFP_MARGIN=400" 10+0.1 1254 500 $UHO 10200
else
  echo "[q125] leg 1 scored ${E1:-none} (not > 0): legs 3-4 dropped as registered"
fi
echo "[q125] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | sed 's/^/[q125]   /'
echo "[q125] $(date) QUEUE 125 DONE"
