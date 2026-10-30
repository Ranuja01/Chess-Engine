#!/bin/bash
# 2026-10-30, queue #118: idle-box external check of the gen 6.4 candidate against Stockfish 1.1 at time odds
# (us 60+0.6, it 20+0.2), the same match as #108 (gen 6.2 21.0%, seed 1082 = the same openings).
#  1 c_sf11_odds    candidate C (threats mg 50 / eg 100 + RFP 600) vs SF 1.1, 200 g
#  2 g63_sf11_odds  gen 6.3 vs SF 1.1, 200 g (the paired control on the same openings)
# Registered: C − gen 6.3 within −3 … +8 points (a stronger opponent than SF18 @1100; not a ship gate).
# Never edit this file while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
BIN="$ND/build_uci/chess_engine_uci_thr2"
SF1="/mnt/e/chess_data/engines/sf1.1/stockfish-1.1"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
BASE=TM_START=175,TM_SINGLE=1,TM_SCALE=7,TM_NORM=95,TM_OPT_PLY=1,THREAT_ORDER_W=100
C=$BASE,THREAT_V2_PCT=100,THREAT_V2_HANGING=1,THREAT_V2_KING=1,THREAT_V2_PAWN_TARGETS=1,THREAT_V2_MG_PCT=50,THREAT_V2_EG_PCT=100,RFP_MARGIN=600
cd "$ND" || exit 1
START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[q118] $START start"
while pgrep -f "fastchess|vs_sf.py" >/dev/null; do sleep 60; done
nodes() { printf "$2bench\nquit\n" | "$1" 2>/dev/null | grep '^bench' | sed -n 's/.* nodes \([0-9]*\).*/\1/p'; }
nb=$(nodes "$BIN" "setoption name Knobs value $BASE\n"); nc=$(nodes "$BIN" "setoption name Knobs value $C\n")
echo "[q118] thr2: base $nb (want 42263170) C $nc (want 36991901)"
[ "$nb" = 42263170 ] && [ "$nc" = 36991901 ] || { echo "[q118] ☠️ fingerprints failed — QUEUE 118 ABORTED"; exit 1; }

COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
SF1OPT="option.Threads=1 option.Ponder=false option.OwnBook=false option.Hash=128"
report() {   # $1 tag
  echo "[q118] $(date) $1 done: $(grep -E '^Games:|^Elo:' $O/$1.log | tail -2 | tr '\n' ' ')"
  echo "[q118]   $(grep -E '^Ptnml' $O/$1.log | tail -1)  disconnects/stalls: $(grep -ciE 'disconnect|stall|not responsive' $O/$1.log)  time forfeits: $(grep -c 'loses on time' $O/$1.pgn)"
}
# Preflight: both knob strings reach the engine, no unknown-option warnings.
preflight() {
  timeout 180 "$FC" -engine cmd="$BIN" name=arm option.OwnBook=false option.Knobs=$C st=0.05 \
    -engine cmd="$BIN" name=ctl option.OwnBook=false option.Knobs=$BASE st=0.05 \
    -openings file=$O/openings_uho.epd format=epd order=random -rounds 1 -games 2 -concurrency 1 \
    -log file=/tmp/q118_pre.engine.log level=info engine=true > /tmp/q118_pre.log 2>&1
  local W K N; W=$(grep -c "have option" /tmp/q118_pre.log); K=$(grep -cF "knobs $C" /tmp/q118_pre.engine.log)
  N=$(grep -oE "knobs [A-Z0-9_=,]+" /tmp/q118_pre.engine.log | grep -cxF "knobs $BASE")
  echo "[q118] preflight: warnings $W, C echoes $K, base echoes $N"
  [ "$W" = "0" ] && [ "$K" -ge 1 ] && [ "$N" -ge 1 ] || { echo "[q118] ☠️ PREFLIGHT FAILED — QUEUE 118 ABORTED"; exit 1; }
}
preflight
odds() {   # $1 tag, $2 knobs
  echo "[q118] $(date) $1 (200 games: ours 60+0.6, sf1.1 20+0.2)"
  nice -n 5 "$FC" -engine cmd="$BIN" name=ours option.OwnBook=false option.Knobs=$2 tc=60+0.6 timemargin=250 restart=on \
    -engine cmd="$SF1" name=sf1.1 $SF1OPT tc=20+0.2 timemargin=250 restart=on $COMMON -srand 1082 -rounds 100 -games 2 \
    -pgnout file=$O/$1.pgn notation=san nodes=true append=false > $O/$1.log 2>&1
  report $1
}
odds c_sf11_odds "$C"
odds g63_sf11_odds "$BASE"
echo "[q118] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | sed 's/^/[q118]   /'
echo "[q118] $(date) QUEUE 118 DONE"
