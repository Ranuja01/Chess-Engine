#!/bin/bash
# 2026-10-30, queue #123 (Saturday morning, after #122): the damp-only drawishness threshold on gen 6.4. SF11's initiative in its
# damp-only form (can shrink the leader's edge toward a draw, never add), light endgames only (phase256 0), pure pawn endings
# excluded, SF11's weights converted by the endgame pawn (×4.69). Static WDL read: eg-stage MSE 290.8 → 282.4 (−2.9%), egl −0.7%,
# mgl / mg identical; symmetry 0 violations on 6,000 + 5,229. Binary chess_engine_uci_damp (gen 6.4 + the knobs, non-PGO, both sides).
#  1 damp_uho     damp vs gen 6.4, 10+0.1, 1,000 g UHO, seed 1231
#  2 damp_rfp500  damp + RFP_MARGIN 500 vs gen 6.4, 10+0.1, 1,000 g, seed 1232 (crank arm)
# Each leg runs only if it can finish before 12:00 (the owner plays weekend afternoons).
# Registered: 1 −5 … +15 · 2 −5 … +25 (crank present if it beats damp alone by > 1σ). If both ≤ 0, park with the caps.
# Never edit this file while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
BIN="$ND/build_uci/chess_engine_uci_damp"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
BASE=THREAT_ORDER_W=100
DAMP=$BASE,WIN_V2=1,WIN_V2_DAMP=1,WIN_V2_PHASE_MAX=0,WIN_V2_SKIP_PAWNEND=1,WIN_V2_PASSED=42,WIN_V2_PAWNS=52,WIN_V2_OUTFLANK=42,WIN_V2_INFILT=56,WIN_V2_FLANKS=98,WIN_V2_UNWIN=-202,WIN_V2_BASE=-469
CRANK=$DAMP,RFP_MARGIN=500
cd "$ND" || exit 1
echo "[q123] $(date) waiting for queue 122"
until grep -q 'QUEUE 122 DONE' /mnt/e/chess_data/q122.log 2>/dev/null; do sleep 60; done
until [ "$(date +%H%M)" -ge 0005 ] && [ "$(date +%H%M)" -lt 1200 ]; do sleep 60; done
while pgrep -f "fastchess|vs_sf.py|_queue_122.sh" >/dev/null; do sleep 60; done
START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[q123] $START start"
nodes() { printf "$2bench\nquit\n" | "$1" 2>/dev/null | grep '^bench' | sed -n 's/.* nodes \([0-9]*\).*/\1/p'; }
nb=$(nodes "$BIN" "setoption name Knobs value $BASE\n"); nd=$(nodes "$BIN" "setoption name Knobs value $DAMP\n")
nk=$(nodes "$BIN" "setoption name Knobs value $CRANK\n")
echo "[q123] damp binary: base $nb (want 36991901) damp $nd (want 35917541) damp+rfp500 $nk (must differ from damp)"
[ "$nb" = 36991901 ] && [ "$nd" = 35917541 ] && [ -n "$nk" ] && [ "$nk" != "$nd" ] || { echo "[q123] ☠️ fingerprints failed — QUEUE 123 ABORTED"; exit 1; }

COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
preflight() {   # $1 arm knobs
  timeout 180 "$FC" -engine cmd="$BIN" name=arm option.OwnBook=false option.Knobs=$1 st=0.05 \
    -engine cmd="$BIN" name=ctl option.OwnBook=false option.Knobs=$BASE st=0.05 \
    -openings file=$O/openings_uho.epd format=epd order=random -rounds 1 -games 2 -concurrency 1 \
    -log file=/tmp/q123_pre.engine.log level=info engine=true > /tmp/q123_pre.log 2>&1
  local W K N; W=$(grep -c "have option" /tmp/q123_pre.log); K=$(grep -cF "knobs $1" /tmp/q123_pre.engine.log)
  N=$(grep -oE "knobs [A-Z0-9_=,-]+" /tmp/q123_pre.engine.log | grep -cxF "knobs $BASE")
  echo "[q123] preflight: warnings $W, arm echoes $K, base echoes $N"
  [ "$W" = "0" ] && [ "$K" -ge 1 ] && [ "$N" -ge 1 ]
}
run() {   # $1 tag, $2 arm knobs, $3 seed, $4 rounds
  local left=$(( $(date -d '12:00' +%s) - $(date +%s) ))
  [ $left -ge 10200 ] || { echo "[q123] $(date) skip $1: only ${left}s before 12:00"; return; }
  preflight "$2" || { echo "[q123] ☠️ preflight failed — skipping $1"; return; }
  echo "[q123] $(date) $1 ($(( $4 * 2 )) games, 10+0.1)"
  nice -n 5 "$FC" -engine cmd="$BIN" name=arm option.OwnBook=false option.Knobs=$2 tc=10+0.1 timemargin=250 \
    -engine cmd="$BIN" name=g64 option.OwnBook=false option.Knobs=$BASE tc=10+0.1 timemargin=250 $COMMON -srand $3 -rounds $4 -games 2 \
    -pgnout file=$O/$1.pgn notation=san nodes=true append=false > $O/$1.log 2>&1
  echo "[q123] $(date) $1 done: $(grep -E '^Games:|^Elo:' $O/$1.log | tail -2 | tr '\n' ' ')"
  echo "[q123]   $(grep -E '^Ptnml' $O/$1.log | tail -1)  disconnects/stalls: $(grep -ciE 'disconnect|stall|not responsive' $O/$1.log)  time forfeits: $(grep -c 'loses on time' $O/$1.pgn)"
}
run damp_uho "$DAMP" 1231 500
run damp_rfp500 "$CRANK" 1232 500
echo "[q123] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | sed 's/^/[q123]   /'
echo "[q123] $(date) QUEUE 123 DONE"
