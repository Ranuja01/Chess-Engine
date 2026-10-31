#!/bin/bash
# 2026-10-31 evening, queue #126 (replaces #125's legs 3-4). #125: damp + RFP 500 replicated (+20.2 ± 18.8; pooled with #124
# +17.7 ± 13.4), but RFP 500 ALONE scored +13.6 ± 18.3 (registered −15 … +10 ✗), so the crank did not hold: damp on top of
# RFP 500 ≈ +4, unresolved. The candidate is now the margin; this queue judges it and the damp on top of it head to head.
# Binary chess_engine_uci_damp for every leg (its defaults = gen 6.4 exactly), 3 games at a time.
#  1 rfp500_r    RFP_MARGIN 500 vs gen 6.4, 1,000 g UHO 10+0.1, seed 1261 (fresh-seed replication)
#  2 damp_on500  damp + RFP 500 vs RFP 500 (both sides at 500), 1,000 g UHO 10+0.1, seed 1262
#  3 rfp500_60   RFP 500 vs gen 6.4, 200 g UHO 60+0.6, seed 1263 (only if #125's +13.6 pooled with leg 1 is > 0)
# Registered: 1 −5 … +20 · 2 −10 … +15 · 3 −10 … +40. Margin plateau on record (#116, vs 600): 400 −5.2 ± 18.9, 800 −19.8 ± 19.4.
# Stop between legs: touch /mnt/e/chess_data/q126.stop (the leg in flight finishes). Never edit this file while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
DMP="$ND/build_uci/chess_engine_uci_damp"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
STOP=/mnt/e/chess_data/q126.stop
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
BASE=THREAT_ORDER_W=100
R500=$BASE,RFP_MARGIN=500
DAMP=$BASE,WIN_V2=1,WIN_V2_DAMP=1,WIN_V2_PHASE_MAX=0,WIN_V2_SKIP_PAWNEND=1,WIN_V2_PASSED=42,WIN_V2_PAWNS=52,WIN_V2_OUTFLANK=42,WIN_V2_INFILT=56,WIN_V2_FLANKS=98,WIN_V2_UNWIN=-202,WIN_V2_BASE=-469
cd "$ND" || exit 1
rm -f "$STOP"
echo "[q126] $(date) waiting for the box to clear"
while pgrep -f "fastchess|vs_sf.py|_queue_125.sh" >/dev/null; do sleep 30; done
START=$(date '+%Y-%m-%d %H:%M:%S')
DEADLINE=$(date -d 'tomorrow 08:00' +%s)
echo "[q126] $START start (deadline $(date -d @$DEADLINE); stop file $STOP)"
nodes() { printf "setoption name Knobs value $2\nbench\nquit\n" | "$1" 2>/dev/null | grep '^bench' | sed -n 's/.* nodes \([0-9]*\).*/\1/p'; }
db=$(nodes "$DMP" $BASE); dr=$(nodes "$DMP" $R500); dk=$(nodes "$DMP" $DAMP,RFP_MARGIN=500)
echo "[q126] damp binary: base $db (want 36991901) rfp500 $dr (want 36670022) damp+rfp500 $dk (want 37179696)"
[ "$db" = 36991901 ] && [ "$dr" = 36670022 ] && [ "$dk" = 37179696 ] \
  || { echo "[q126] ☠️ fingerprints failed — QUEUE 126 ABORTED"; exit 1; }

ADJ="-repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
preflight() {   # $1 binary, $2 arm knobs, $3 control knobs
  timeout 180 "$FC" -engine cmd="$1" name=arm option.OwnBook=false option.Knobs=$2 st=0.05 \
    -engine cmd="$1" name=ctl option.OwnBook=false option.Knobs=$3 st=0.05 \
    -openings file=$O/openings_uho.epd format=epd order=random -rounds 1 -games 2 -concurrency 1 \
    -log file=/tmp/q126_pre.engine.log level=info engine=true > /tmp/q126_pre.log 2>&1
  local W K N; W=$(grep -c "have option" /tmp/q126_pre.log)
  K=$(grep -oE "knobs [A-Z0-9_=,-]+" /tmp/q126_pre.engine.log | grep -cxF "knobs $2")
  N=$(grep -oE "knobs [A-Z0-9_=,-]+" /tmp/q126_pre.engine.log | grep -cxF "knobs $3")
  echo "[q126] preflight: warnings $W, arm echoes $K, control echoes $N"
  [ "$W" = "0" ] && [ "$K" -ge 1 ] && [ "$N" -ge 1 ]
}
run() {   # $1 tag, $2 binary, $3 arm knobs, $4 control knobs, $5 tc, $6 seed, $7 rounds, $8 openings, $9 seconds needed
  [ -f "$STOP" ] && { echo "[q126] $(date) skip $1: stop file present"; return; }
  local left=$(( DEADLINE - $(date +%s) ))
  [ $left -ge $9 ] || { echo "[q126] $(date) skip $1: only ${left}s before the deadline (needs $9)"; return; }
  preflight "$2" "$3" "$4" || { echo "[q126] ☠️ preflight failed — skipping $1"; return; }
  echo "[q126] $(date) $1 ($(( $7 * 2 )) games, $5)"
  nice -n 5 "$FC" -engine cmd="$2" name=arm option.OwnBook=false option.Knobs=$3 tc=$5 timemargin=250 \
    -engine cmd="$2" name=ctl option.OwnBook=false option.Knobs=$4 tc=$5 timemargin=250 \
    -openings file=$8 format=epd order=random $ADJ -srand $6 -rounds $7 -games 2 \
    -pgnout file=$O/$1.pgn notation=san nodes=true append=false > $O/$1.log 2>&1
  echo "[q126] $(date) $1 done: $(grep -E '^Games:|^Elo:' $O/$1.log | tail -2 | tr '\n' ' ')"
  echo "[q126]   $(grep -E '^Ptnml' $O/$1.log | tail -1)  disconnects/stalls: $(grep -ciE 'disconnect|stall|not responsive' $O/$1.log)  time forfeits: $(grep -c 'loses on time' $O/$1.pgn)"
}
UHO=$O/openings_uho.epd
run rfp500_r   "$DMP" "$R500"                 "$BASE" 10+0.1 1261 500 $UHO 10200
run damp_on500 "$DMP" "$DAMP,RFP_MARGIN=500"  "$R500" 10+0.1 1262 500 $UHO 10200
E1=$(grep -E '^Elo:' $O/rfp500_r.log 2>/dev/null | tail -1 | awk '{print $2}')
if awk -v e="${E1:-x}" 'BEGIN{exit !(e != "x" && (13.56 + e) / 2 > 0)}'; then
  run rfp500_60 "$DMP" "$R500" "$BASE" 60+0.6 1263 100 $UHO 12600
else
  echo "[q126] RFP 500 pooled not > 0 (leg 1: ${E1:-none}): leg 3 dropped as registered"
fi
echo "[q126] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | sed 's/^/[q126]   /'
echo "[q126] $(date) QUEUE 126 DONE"
