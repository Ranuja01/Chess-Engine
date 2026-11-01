#!/bin/bash
# 2026-11-01, queue #129: transition item 4, middlegame king safety. Pinned defenders defend only along their pin line
# (KS_V2_PIN_DEF=1, SF's restricted defender maps; never measured before). The shipped BLOCKERS weight is −20 where SF11 has +98,
# read as the symptom of counting pinned defenders as full ones; with the restriction in place BLOCKERS 0 tests that reading.
# Both arms vs gen 6.4 at 10+0.1, 1,000 games, on the pin binary for BOTH sides (it carries a result-identical guard the damp
# binary lacks). Registered: pin −5 … +15 · pin + BLOCKERS 0 −10 … +15. One run under ~+30 is unresolved; the better arm is
# replicated on a fresh seed before anything is called.
# Fingerprints (pin binary): default 249 / 36,991,901 · PIN_DEF=1 250 / 36,176,814 · PIN_DEF=1,BLOCKERS=0 247 / 38,269,511.
# Waits for queue 128. Timed legs start only if they can finish by $1 (a date string, default "today 21:00").
# Stop between legs: touch /mnt/e/chess_data/q129.stop. Never edit this file while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
PIN="$ND/build_uci/chess_engine_uci_pin"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
STOP=/mnt/e/chess_data/q129.stop
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
BASE=THREAT_ORDER_W=100
cd "$ND" || exit 1
rm -f "$STOP"
echo "[q129] $(date) waiting for queue 128 to finish"
while pgrep -f "fastchess|vs_sf.py|_queue_128\.sh" >/dev/null; do sleep 30; done
START=$(date '+%Y-%m-%d %H:%M:%S')
DEADLINE=$(date -d "${1:-today 21:00}" +%s)
echo "[q129] $START start (timed legs must finish by $(date -d @$DEADLINE); stop file $STOP)"

nodes() { printf "setoption name Knobs value $2\nbench\nquit\n" | "$1" 2>/dev/null | grep '^bench' | sed -n 's/.* nodes \([0-9]*\).*/\1/p'; }
db=$(nodes "$PIN" $BASE); dp=$(nodes "$PIN" $BASE,KS_V2_PIN_DEF=1); d0=$(nodes "$PIN" $BASE,KS_V2_PIN_DEF=1,KS_V2_BLOCKERS=0)
echo "[q129] pin binary: base $db (want 36991901) pin $dp (want 36176814) pin_b0 $d0 (want 38269511)"
[ "$db" = 36991901 ] && [ "$dp" = 36176814 ] && [ "$d0" = 38269511 ] || { echo "[q129] ☠️ fingerprints failed — QUEUE 129 ABORTED"; exit 1; }

COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
report() {
  echo "[q129] $(date) $1 done: $(grep -E '^Games:|^Elo:' $O/$1.log | tail -2 | tr '\n' ' ')"
  echo "[q129]   $(grep -E '^Ptnml' $O/$1.log | tail -1)  disconnects/stalls: $(grep -ciE 'disconnect|stall|not responsive' $O/$1.log)  time forfeits: $(grep -c 'loses on time' $O/$1.pgn)"
}
gate() {   # $1 tag, $2 seconds needed
  [ -f "$STOP" ] && { echo "[q129] $(date) skip $1: stop file present"; return 1; }
  local left=$(( DEADLINE - $(date +%s) ))
  [ $left -ge $2 ] || { echo "[q129] $(date) skip $1: only ${left}s before the deadline (needs $2)"; return 1; }
}
preflight() {   # $1 tag, $2 arm knobs, $3 control knobs
  rm -f /tmp/q129_pre.engine.log
  timeout 180 "$FC" -engine cmd="$PIN" name=arm option.OwnBook=false option.Knobs=$2 st=0.05 \
    -engine cmd="$PIN" name=ctl option.OwnBook=false option.Knobs=$3 st=0.05 \
    -openings file=$O/openings_uho.epd format=epd order=random -rounds 1 -games 2 -concurrency 1 \
    -log file=/tmp/q129_pre.engine.log level=info engine=true > /tmp/q129_pre.log 2>&1
  local W A C; W=$(grep -c "have option" /tmp/q129_pre.log)
  A=$(grep -oE " arm ---> info string knobs [A-Z0-9_=,-]+" /tmp/q129_pre.engine.log | grep -cxF " arm ---> info string knobs $2")
  C=$(grep -oE " ctl ---> info string knobs [A-Z0-9_=,-]+" /tmp/q129_pre.engine.log | grep -cxF " ctl ---> info string knobs $3")
  echo "[q129] $1 preflight: warnings $W, arm echoes $A, control echoes $C"
  [ "$W" = "0" ] && [ "$A" -ge 1 ] && [ "$C" -ge 1 ] || { echo "[q129] ☠️ $1 PREFLIGHT FAILED — skipped"; return 1; }
}
selfplay() {   # $1 tag, $2 arm knobs, $3 control knobs, $4 tc, $5 rounds, $6 seed, $7 seconds needed
  gate $1 $7 && preflight $1 "$2" "$3" || return
  echo "[q129] $(date) $1 ($((2 * $5)) games, $4)"
  nice -n 5 "$FC" -engine cmd="$PIN" name=arm option.OwnBook=false option.Knobs=$2 tc=$4 timemargin=250 \
    -engine cmd="$PIN" name=ctl option.OwnBook=false option.Knobs=$3 tc=$4 timemargin=250 $COMMON -srand $6 -rounds $5 -games 2 \
    -pgnout file=$O/$1.pgn notation=san nodes=true append=false > $O/$1.log 2>&1
  report $1
}

selfplay  pin     "$BASE,KS_V2_PIN_DEF=1"                  "$BASE" 10+0.1 500 1291 10200
selfplay  pin_b0  "$BASE,KS_V2_PIN_DEF=1,KS_V2_BLOCKERS=0" "$BASE" 10+0.1 500 1292 10200
echo "[q129] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | sed 's/^/[q129]   /'
echo "[q129] $(date) QUEUE 129 DONE"
