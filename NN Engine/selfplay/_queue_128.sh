#!/bin/bash
# 2026-11-01, queue #128: an A/A control on the fastchess harness. Gen 6.4 plays itself (the same binary and knobs on both sides)
# with the openings, time control, adjudication and concurrency of every arm in #124-#127. No A/A control has ever been run on any
# of our harnesses, so every ± quoted so far is theoretical. Registered before the run:
#  each 1,000-game run: 0 ± ~19 (95%) · pooled over the two: 0 ± ~13.5; a pooled |Elo| above that ⇒ a slot bias in the harness
#  the two runs, beside RFP 500's three same-config runs (+13.6, +10.4, −17.0; χ² ≈ 6.2 on 2 df), say whether the run-to-run
#  spread is wider than fastchess's ±; 100-game blocks within each run are checked against the pentanomial variance afterwards
# Timed legs start only if they can finish by 21:00 today (the owner's evening games). Stop between legs: touch
# /mnt/e/chess_data/q128.stop. Never edit this file while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
DMP="$ND/build_uci/chess_engine_uci_damp"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
STOP=/mnt/e/chess_data/q128.stop
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
BASE=THREAT_ORDER_W=100
cd "$ND" || exit 1
rm -f "$STOP"
if pgrep -f "fastchess|vs_sf.py|_queue_12[0-7]\.sh" >/dev/null; then echo "[q128] another queue is running — QUEUE 128 ABORTED"; exit 1; fi
START=$(date '+%Y-%m-%d %H:%M:%S')
DEADLINE=$(date -d 'today 21:00' +%s)
echo "[q128] $START start (timed legs must finish by $(date -d @$DEADLINE); stop file $STOP)"

# pooled Elo over runs: each input line "elo ci95"; prints "mean ci95 z"
elo_of() { grep -E '^Elo:' $O/$1.log 2>/dev/null | tail -1 | awk '{gsub(",","",$4); print $2, $4}'; }
pool() { awk '{n++; s+=$1; v+=$2*$2} END{if(n==0){print "0 0 0"; exit} m=s/n; c=sqrt(v)/n; printf "%.2f %.2f %.2f\n", m, c, 1.96*m/c}'; }

nodes() { printf "setoption name Knobs value $2\nbench\nquit\n" | "$1" 2>/dev/null | grep '^bench' | sed -n 's/.* nodes \([0-9]*\).*/\1/p'; }
db=$(nodes "$DMP" $BASE)
echo "[q128] damp binary: base $db (want 36991901)"
[ "$db" = 36991901 ] || { echo "[q128] ☠️ fingerprint failed — QUEUE 128 ABORTED"; exit 1; }

COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
report() {
  echo "[q128] $(date) $1 done: $(grep -E '^Games:|^Elo:' $O/$1.log | tail -2 | tr '\n' ' ')"
  echo "[q128]   $(grep -E '^Ptnml' $O/$1.log | tail -1)  disconnects/stalls: $(grep -ciE 'disconnect|stall|not responsive' $O/$1.log)  time forfeits: $(grep -c 'loses on time' $O/$1.pgn)"
}
gate() {   # $1 tag, $2 seconds needed
  [ -f "$STOP" ] && { echo "[q128] $(date) skip $1: stop file present"; return 1; }
  local left=$(( DEADLINE - $(date +%s) ))
  [ $left -ge $2 ] || { echo "[q128] $(date) skip $1: only ${left}s before the deadline (needs $2)"; return 1; }
}
preflight() {   # $1 tag, $2 knobs (both sides)
  rm -f /tmp/q128_pre.engine.log
  timeout 180 "$FC" -engine cmd="$DMP" name=arm option.OwnBook=false option.Knobs=$2 st=0.05 \
    -engine cmd="$DMP" name=ctl option.OwnBook=false option.Knobs=$2 st=0.05 \
    -openings file=$O/openings_uho.epd format=epd order=random -rounds 1 -games 2 -concurrency 1 \
    -log file=/tmp/q128_pre.engine.log level=info engine=true > /tmp/q128_pre.log 2>&1
  local W A C; W=$(grep -c "have option" /tmp/q128_pre.log)
  A=$(grep -oE " arm ---> info string knobs [A-Z0-9_=,-]+" /tmp/q128_pre.engine.log | grep -cxF " arm ---> info string knobs $2")
  C=$(grep -oE " ctl ---> info string knobs [A-Z0-9_=,-]+" /tmp/q128_pre.engine.log | grep -cxF " ctl ---> info string knobs $2")
  echo "[q128] $1 preflight: warnings $W, arm echoes $A, control echoes $C"
  [ "$W" = "0" ] && [ "$A" -ge 1 ] && [ "$C" -ge 1 ] || { echo "[q128] ☠️ $1 PREFLIGHT FAILED — skipped"; return 1; }
}
selfplay() {   # $1 tag, $2 knobs (both sides), $3 tc, $4 rounds, $5 seed, $6 seconds needed
  gate $1 $6 && preflight $1 "$2" || return
  echo "[q128] $(date) $1 ($((2 * $4)) games, $3)"
  nice -n 5 "$FC" -engine cmd="$DMP" name=arm option.OwnBook=false option.Knobs=$2 tc=$3 timemargin=250 \
    -engine cmd="$DMP" name=ctl option.OwnBook=false option.Knobs=$2 tc=$3 timemargin=250 $COMMON -srand $5 -rounds $4 -games 2 \
    -pgnout file=$O/$1.pgn notation=san nodes=true append=false > $O/$1.log 2>&1
  report $1
}

selfplay  aa1  "$BASE" 10+0.1 500 1281 10200
selfplay  aa2  "$BASE" 10+0.1 500 1282 10200
P=$( { elo_of aa1; elo_of aa2; } | pool )
echo "[q128] A/A pooled over two runs: $(echo $P | awk '{printf "%+.1f ± %.1f (z %.2f)", $1, $2, $3}')"
echo "[q128] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | sed 's/^/[q128]   /'
echo "[q128] $(date) QUEUE 128 DONE"
