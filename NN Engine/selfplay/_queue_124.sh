#!/bin/bash
# 2026-10-30 night, queue #124 (replaces #122's remaining legs and #123; the owner finished early, so it starts now).
# All 10+0.1 unless noted, vs gen 6.4, 3 games at a time. Each leg runs only if it can finish before 12:00 Saturday.
#  1 damp_uho      damp-only drawishness threshold (binary chess_engine_uci_damp), 1,000 g UHO, seed 1231
#  2 passer_eg     ENABLE_PASSER_PRUNE_EXEMPT=1 (true-passer pushes kept out of LMP / LMR; the span-checked version, never gamed),
#                  eg_critical starts (928 g), seed 1241. From the owner's UI games: at game 2 move 11 the shipped search prunes a
#                  ~5-ply d-pawn line at every depth; the exemption finds SF18's Kf1 at d13-15, no selectivity at d11.
#  3 caps_rfp500   endgame caps + RFP_MARGIN 500 (g64pgo), 1,000 g, seed 1212 (crank arm)
#  4 damp_rfp500   damp + RFP_MARGIN 500 (damp binary), 1,000 g, seed 1232 (crank arm)
#  5 caps_60       caps, 60+0.6, 200 g, seed 1213
# Registered: 1 −5 … +15 · 2 −10 … +25 (narrow: ships only with ≥ +2σ here and a UHO no-harm leg) · 3 −5 … +30 · 4 −5 … +25
# (a crank is present if it beats its term alone by > 1σ) · 5 −10 … +40. Damp: both legs ≤ 0 ⇒ park with the caps.
# Never edit this file while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
G64="$ND/build_uci/chess_engine_uci_g64pgo"
DMP="$ND/build_uci/chess_engine_uci_damp"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
BASE=THREAT_ORDER_W=100
CAPS=$BASE,POT_V2_WIN_OCBX=-42,POT_V2_WIN_OCBX_PC=3,POT_V2_WIN_ROOKE=-28,POT_V2_WIN_QNOQ=-27,POT_V2_WIN_QNOQ_MINOR=3,POT_V2_WIN_LONEMINOR=-64
DAMP=$BASE,WIN_V2=1,WIN_V2_DAMP=1,WIN_V2_PHASE_MAX=0,WIN_V2_SKIP_PAWNEND=1,WIN_V2_PASSED=42,WIN_V2_PAWNS=52,WIN_V2_OUTFLANK=42,WIN_V2_INFILT=56,WIN_V2_FLANKS=98,WIN_V2_UNWIN=-202,WIN_V2_BASE=-469
PASS=$BASE,ENABLE_PASSER_PRUNE_EXEMPT=1
cd "$ND" || exit 1
echo "[q124] $(date) waiting for the box to clear"
while pgrep -f "fastchess|vs_sf.py|_queue_122.sh|_queue_123.sh" >/dev/null; do sleep 30; done
START=$(date '+%Y-%m-%d %H:%M:%S')
if [ "$(date +%H)" -ge 12 ]; then DEADLINE=$(date -d 'tomorrow 12:00' +%s); else DEADLINE=$(date -d '12:00' +%s); fi
echo "[q124] $START start (deadline $(date -d @$DEADLINE))"
nodes() { printf "setoption name Knobs value $2\nbench\nquit\n" | "$1" 2>/dev/null | grep '^bench' | sed -n 's/.* nodes \([0-9]*\).*/\1/p'; }
gb=$(nodes "$G64" $BASE); gc=$(nodes "$G64" $CAPS); gk=$(nodes "$G64" $CAPS,RFP_MARGIN=500); gp=$(nodes "$G64" $PASS)
db=$(nodes "$DMP" $BASE); dd=$(nodes "$DMP" $DAMP); dk=$(nodes "$DMP" $DAMP,RFP_MARGIN=500)
echo "[q124] g64pgo: base $gb (want 36991901) caps $gc (want 38998902) caps+rfp500 $gk passer $gp (both must differ)"
echo "[q124] damp:   base $db (want 36991901) damp $dd (want 35917541) damp+rfp500 $dk (must differ)"
[ "$gb" = 36991901 ] && [ "$gc" = 38998902 ] && [ -n "$gk" ] && [ "$gk" != "$gc" ] && [ -n "$gp" ] && [ "$gp" != "$gb" ] \
  && [ "$db" = 36991901 ] && [ "$dd" = 35917541 ] && [ -n "$dk" ] && [ "$dk" != "$dd" ] \
  || { echo "[q124] ☠️ fingerprints failed — QUEUE 124 ABORTED"; exit 1; }

ADJ="-repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
preflight() {   # $1 binary, $2 arm knobs
  timeout 180 "$FC" -engine cmd="$1" name=arm option.OwnBook=false option.Knobs=$2 st=0.05 \
    -engine cmd="$1" name=ctl option.OwnBook=false option.Knobs=$BASE st=0.05 \
    -openings file=$O/openings_uho.epd format=epd order=random -rounds 1 -games 2 -concurrency 1 \
    -log file=/tmp/q124_pre.engine.log level=info engine=true > /tmp/q124_pre.log 2>&1
  local W K N; W=$(grep -c "have option" /tmp/q124_pre.log); K=$(grep -cF "knobs $2" /tmp/q124_pre.engine.log)
  N=$(grep -oE "knobs [A-Z0-9_=,-]+" /tmp/q124_pre.engine.log | grep -cxF "knobs $BASE")
  echo "[q124] preflight: warnings $W, arm echoes $K, base echoes $N"
  [ "$W" = "0" ] && [ "$K" -ge 1 ] && [ "$N" -ge 1 ]
}
run() {   # $1 tag, $2 binary, $3 arm knobs, $4 tc, $5 seed, $6 rounds, $7 openings, $8 seconds needed
  local left=$(( DEADLINE - $(date +%s) ))
  [ $left -ge $8 ] || { echo "[q124] $(date) skip $1: only ${left}s before the deadline (needs $8)"; return; }
  preflight "$2" "$3" || { echo "[q124] ☠️ preflight failed — skipping $1"; return; }
  echo "[q124] $(date) $1 ($(( $6 * 2 )) games, $4)"
  nice -n 5 "$FC" -engine cmd="$2" name=arm option.OwnBook=false option.Knobs=$3 tc=$4 timemargin=250 \
    -engine cmd="$2" name=g64 option.OwnBook=false option.Knobs=$BASE tc=$4 timemargin=250 \
    -openings file=$7 format=epd order=random $ADJ -srand $5 -rounds $6 -games 2 \
    -pgnout file=$O/$1.pgn notation=san nodes=true append=false > $O/$1.log 2>&1
  echo "[q124] $(date) $1 done: $(grep -E '^Games:|^Elo:' $O/$1.log | tail -2 | tr '\n' ' ')"
  echo "[q124]   $(grep -E '^Ptnml' $O/$1.log | tail -1)  disconnects/stalls: $(grep -ciE 'disconnect|stall|not responsive' $O/$1.log)  time forfeits: $(grep -c 'loses on time' $O/$1.pgn)"
}
UHO=$O/openings_uho.epd; EG=$O/eg_critical.epd
run damp_uho    "$DMP" "$DAMP"                 10+0.1 1231 500                   $UHO 10200
run passer_eg   "$G64" "$PASS"                 10+0.1 1241 $(wc -l < $EG)        $EG  9900
run caps_rfp500 "$G64" "$CAPS,RFP_MARGIN=500"  10+0.1 1212 500                   $UHO 10200
run damp_rfp500 "$DMP" "$DAMP,RFP_MARGIN=500"  10+0.1 1232 500                   $UHO 10200
run caps_60     "$G64" "$CAPS"                 60+0.6 1213 100                   $UHO 8100
echo "[q124] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | sed 's/^/[q124]   /'
echo "[q124] $(date) QUEUE 124 DONE"
