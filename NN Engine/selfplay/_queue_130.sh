#!/bin/bash
# 2026-11-01, queue #130: chained after #129. A single 1,000-game run under ~+30 is unresolved, so the better pinned-defender arm
# of #129 (pin, or pin + BLOCKERS 0) is replicated on a fresh seed, vs gen 6.4 at 10+0.1 on the pin binary — only if it read
# above 0; otherwise nothing runs. Registered: the replicate lands within its first run's ± (regression toward 0 expected);
# pooled ≥ 2σ ⇒ the depth-fit refresh and BLOCKERS refit are worth building; pooled ≤ 0.5σ ⇒ pinned defenders park.
# Timed legs start only if they can finish by Monday 08:00. Stop between legs: touch /mnt/e/chess_data/q130.stop.
# Never edit this file while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
PIN="$ND/build_uci/chess_engine_uci_pin"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
STOP=/mnt/e/chess_data/q130.stop
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
BASE=THREAT_ORDER_W=100
cd "$ND" || exit 1
rm -f "$STOP"
echo "[q130] $(date) waiting for queue 129 to finish"
while pgrep -f "fastchess|vs_sf.py|_queue_129\.sh" >/dev/null; do sleep 30; done
START=$(date '+%Y-%m-%d %H:%M:%S')
DEADLINE=$(date -d 'tomorrow 08:00' +%s); [ "$(date +%H)" -lt 8 ] && DEADLINE=$(date -d 'today 08:00' +%s)
echo "[q130] $START start (timed legs must finish by $(date -d @$DEADLINE); stop file $STOP)"

elo_of() { grep -E '^Elo:' $O/$1.log 2>/dev/null | tail -1 | awk '{gsub(",","",$4); print $2, $4}'; }
pool() { awk '{n++; s+=$1; v+=$2*$2} END{if(n==0){print "0 0 0"; exit} m=s/n; c=sqrt(v)/n; printf "%.2f %.2f %.2f\n", m, c, 1.96*m/c}'; }
EP=$(elo_of pin | awk '{print $1}'); E0=$(elo_of pin_b0 | awk '{print $1}')
echo "[q130] #129: pin $(elo_of pin) · pin_b0 $(elo_of pin_b0)"
[ -n "$EP" ] && [ -n "$E0" ] || { echo "[q130] #129 incomplete — QUEUE 130 ABORTED"; exit 1; }
if awk -v a=$EP -v b=$E0 'BEGIN{exit !(a >= b)}'; then TAG=pin; ARM="$BASE,KS_V2_PIN_DEF=1"; BEST=$EP
else TAG=pin_b0; ARM="$BASE,KS_V2_PIN_DEF=1,KS_V2_BLOCKERS=0"; BEST=$E0; fi
awk -v e=$BEST 'BEGIN{exit !(e > 0)}' || { echo "[q130] better arm $TAG at $BEST ≤ 0: nothing to replicate. $(date) QUEUE 130 DONE"; exit 0; }
echo "[q130] replicating $TAG ($BEST)"

nodes() { printf "setoption name Knobs value $2\nbench\nquit\n" | "$1" 2>/dev/null | grep '^bench' | sed -n 's/.* nodes \([0-9]*\).*/\1/p'; }
db=$(nodes "$PIN" $BASE); dp=$(nodes "$PIN" $BASE,KS_V2_PIN_DEF=1); d0=$(nodes "$PIN" $BASE,KS_V2_PIN_DEF=1,KS_V2_BLOCKERS=0)
echo "[q130] pin binary: base $db (want 36991901) pin $dp (want 36176814) pin_b0 $d0 (want 38269511)"
[ "$db" = 36991901 ] && [ "$dp" = 36176814 ] && [ "$d0" = 38269511 ] || { echo "[q130] ☠️ fingerprints failed — QUEUE 130 ABORTED"; exit 1; }

COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
report() {
  echo "[q130] $(date) $1 done: $(grep -E '^Games:|^Elo:' $O/$1.log | tail -2 | tr '\n' ' ')"
  echo "[q130]   $(grep -E '^Ptnml' $O/$1.log | tail -1)  disconnects/stalls: $(grep -ciE 'disconnect|stall|not responsive' $O/$1.log)  time forfeits: $(grep -c 'loses on time' $O/$1.pgn)"
}
gate() {   # $1 tag, $2 seconds needed
  [ -f "$STOP" ] && { echo "[q130] $(date) skip $1: stop file present"; return 1; }
  local left=$(( DEADLINE - $(date +%s) ))
  [ $left -ge $2 ] || { echo "[q130] $(date) skip $1: only ${left}s before the deadline (needs $2)"; return 1; }
}
preflight() {   # $1 tag, $2 arm knobs, $3 control knobs
  rm -f /tmp/q130_pre.engine.log
  timeout 180 "$FC" -engine cmd="$PIN" name=arm option.OwnBook=false option.Knobs=$2 st=0.05 \
    -engine cmd="$PIN" name=ctl option.OwnBook=false option.Knobs=$3 st=0.05 \
    -openings file=$O/openings_uho.epd format=epd order=random -rounds 1 -games 2 -concurrency 1 \
    -log file=/tmp/q130_pre.engine.log level=info engine=true > /tmp/q130_pre.log 2>&1
  local W A C; W=$(grep -c "have option" /tmp/q130_pre.log)
  A=$(grep -oE " arm ---> info string knobs [A-Z0-9_=,-]+" /tmp/q130_pre.engine.log | grep -cxF " arm ---> info string knobs $2")
  C=$(grep -oE " ctl ---> info string knobs [A-Z0-9_=,-]+" /tmp/q130_pre.engine.log | grep -cxF " ctl ---> info string knobs $3")
  echo "[q130] $1 preflight: warnings $W, arm echoes $A, control echoes $C"
  [ "$W" = "0" ] && [ "$A" -ge 1 ] && [ "$C" -ge 1 ] || { echo "[q130] ☠️ $1 PREFLIGHT FAILED — skipped"; return 1; }
}
selfplay() {   # $1 tag, $2 arm knobs, $3 control knobs, $4 tc, $5 rounds, $6 seed, $7 seconds needed
  gate $1 $7 && preflight $1 "$2" "$3" || return
  echo "[q130] $(date) $1 ($((2 * $5)) games, $4)"
  nice -n 5 "$FC" -engine cmd="$PIN" name=arm option.OwnBook=false option.Knobs=$2 tc=$4 timemargin=250 \
    -engine cmd="$PIN" name=ctl option.OwnBook=false option.Knobs=$3 tc=$4 timemargin=250 $COMMON -srand $6 -rounds $5 -games 2 \
    -pgnout file=$O/$1.pgn notation=san nodes=true append=false > $O/$1.log 2>&1
  report $1
}

selfplay  ${TAG}_r  "$ARM"  "$BASE" 10+0.1 500 1301 10200
P=$( { elo_of $TAG; elo_of ${TAG}_r; } | pool )
echo "[q130] $TAG pooled over two runs: $(echo $P | awk '{printf "%+.1f ± %.1f (z %.2f)", $1, $2, $3}')"
echo "[q130] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | sed 's/^/[q130]   /'
echo "[q130] $(date) QUEUE 130 DONE"
