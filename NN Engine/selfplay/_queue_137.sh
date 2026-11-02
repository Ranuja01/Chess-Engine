#!/bin/bash
# 2026-11-02 afternoon, queue #137: step 1 of dev_notes/active/SEARCH-CORRECTNESS-ITEMS-2026-11-02.md, the two built knobs whose
# match designs are fully registered there (their 10-19 ... 10-22 games were void: Knob_ options never reached the engine).
#  1 A4 FIX_MAX_NULL=1 (the maximizer's null move searches (beta-1, beta), as the minimizer's does) vs gen 6.4 on UHO, 10+0.1,
#    1,000 g, seed 1361. General change. Registered: -5 ... +15. Bench with the knob: 248 / 37,936,030.
#  2 G5 NULL_NPM_GUARD=2 (SF's no-null-without-non-pawn-material rule added to our count rule) vs gen 6.4 on transition.epd,
#    10+0.1, 1,000 g, seed 1362. Fires only with a pawn-only mover. Registered: -10 ... +10.
# Correctness items ship on not harmful (>= -1 sigma); no replicate; any ship is the owner's call.
# Production PGO binary on both sides, knobs through option.Knobs (preflight: 0 "have option" warnings, both echoes).
# Timed legs must finish by 20:45 (no timed games 9 pm - midnight). Stop between legs: touch /mnt/e/chess_data/q137.stop.
# Never edit this file while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
BIN="$ND/build_uci/chess_engine_uci"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
STOP=/mnt/e/chess_data/q137.stop
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
BASE=THREAT_ORDER_W=100
A4=$BASE,FIX_MAX_NULL=1
G5=$BASE,NULL_NPM_GUARD=2
PE_FEN="8/1p3kpp/p1p2p2/8/2P5/1P2PP2/P5PP/6K1 w - - 0 1"
cd "$ND" || exit 1
rm -f "$STOP"
if pgrep -f "fastchess|_queue_13[0-6]\.sh" >/dev/null; then echo "[q137] another queue is running - not starting"; exit 1; fi
START=$(date '+%Y-%m-%d %H:%M:%S')
DEADLINE=$(date -d '2026-11-02 20:45' +%s)
echo "[q137] $START start (timed legs must finish by $(date -d @$DEADLINE); stop file $STOP)"

nodes() { printf "setoption name Knobs value $1\nbench\nquit\n" | "$BIN" 2>/dev/null | grep '^bench' | sed -n 's/.* nodes \([0-9]*\).*/\1/p'; }
pe_nodes() { (printf "setoption name Knobs value $1\nisready\nposition fen $PE_FEN\ngo depth 14\n"; sleep 15; printf "quit\n") \
  | timeout 40 "$BIN" 2>/dev/null \
  | awk '/^info /{for(i=1;i<NF;i++) if($i=="depth" && $(i+1)=="14") for(j=1;j<NF;j++) if($j=="nodes") n=$(j+1)} END{print n}'; }
db=$(nodes $BASE); da=$(nodes $A4); pb=$(pe_nodes $BASE); pg=$(pe_nodes $G5)
echo "[q137] bench: base $db (want 36991901) · A4 $da (want 37936030) · pawn ending d14: base $pb, G5 $pg (must differ)"
[ "$db" = 36991901 ] && [ "$da" = 37936030 ] && [ -n "$pb" ] && [ -n "$pg" ] && [ "$pb" != "$pg" ] \
  || { echo "[q137] ☠️ fingerprints failed — QUEUE 137 ABORTED"; exit 1; }

COMMON="-repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
report() {
  echo "[q137] $(date) $1 done: $(grep -E '^Games:|^Elo:' $O/$1.log | tail -2 | tr '\n' ' ')"
  echo "[q137]   $(grep -E '^Ptnml' $O/$1.log | tail -1)  disconnects/stalls: $(grep -ciE 'disconnect|stall|not responsive' $O/$1.log)  time forfeits: $(grep -c 'loses on time' $O/$1.pgn)"
}
gate() {   # $1 tag, $2 seconds needed
  [ -f "$STOP" ] && { echo "[q137] $(date) skip $1: stop file present"; return 1; }
  local left=$(( DEADLINE - $(date +%s) ))
  [ $left -ge $2 ] || { echo "[q137] $(date) skip $1: only ${left}s before the deadline (needs $2)"; return 1; }
}
preflight() {   # $1 tag, $2 arm knobs, $3 control knobs, $4 openings
  rm -f /tmp/q137_pre.engine.log
  timeout 180 "$FC" -engine cmd="$BIN" name=arm option.OwnBook=false option.Knobs=$2 st=0.05 \
    -engine cmd="$BIN" name=ctl option.OwnBook=false option.Knobs=$3 st=0.05 \
    -openings file=$4 format=epd order=random -rounds 1 -games 2 -concurrency 1 \
    -log file=/tmp/q137_pre.engine.log level=info engine=true > /tmp/q137_pre.log 2>&1
  local W A C; W=$(grep -c "have option" /tmp/q137_pre.log)
  A=$(grep -oE " arm ---> info string knobs [A-Z0-9_=,-]+" /tmp/q137_pre.engine.log | grep -cxF " arm ---> info string knobs $2")
  C=$(grep -oE " ctl ---> info string knobs [A-Z0-9_=,-]+" /tmp/q137_pre.engine.log | grep -cxF " ctl ---> info string knobs $3")
  echo "[q137] $1 preflight: warnings $W, arm echoes $A, control echoes $C"
  [ "$W" = "0" ] && [ "$A" -ge 1 ] && [ "$C" -ge 1 ] || { echo "[q137] ☠️ $1 PREFLIGHT FAILED — skipped"; return 1; }
}
selfplay() {   # $1 tag, $2 arm knobs, $3 control knobs, $4 openings, $5 tc, $6 rounds, $7 seed, $8 seconds needed
  gate $1 $8 && preflight $1 "$2" "$3" $4 || return
  echo "[q137] $(date) $1 ($((2 * $6)) games, $5, $(basename $4))"
  nice -n 5 "$FC" -engine cmd="$BIN" name=arm option.OwnBook=false option.Knobs=$2 tc=$5 timemargin=250 \
    -engine cmd="$BIN" name=ctl option.OwnBook=false option.Knobs=$3 tc=$5 timemargin=250 \
    -openings file=$4 format=epd order=random $COMMON -srand $7 -rounds $6 -games 2 \
    -pgnout file=$O/$1.pgn notation=san nodes=true append=false > $O/$1.log 2>&1
  report $1
}

selfplay a4_uho "$A4" "$BASE" $O/openings_uho.epd 10+0.1 500 1361 10000
selfplay g5_tr "$G5" "$BASE" $O/transition.epd 10+0.1 500 1362 7500

echo "[q137] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | sed 's/^/[q137]   /'
echo "[q137] $(date) QUEUE 137 DONE"
