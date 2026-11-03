#!/bin/bash
# 2026-11-03 daytime (idle box), queue #140: confirm match for the proposed gen 6.5 = queue #139's candidate without FIX_MDP,
# vs gen 6.4 on UHO at 10+0.1, 1,000 games as five pausable 200-game chunks (same harness as #139). Why without MDP: on the
# fixed-depth benches the candidate without it is cheaper everywhere (UHO d10 -6.5% vs -2.6% with it, WAC +0.5% vs +6.7%, WAC
# bench 252 solved / -0.1% nodes vs 251 / +10.6%), because the TT never stores MDP's mate-range bounds. Registered -10 ... +10;
# passes if Elo >= -1 sigma. Measurement only: shipping is the owner's call.
# Pause: touch /mnt/e/chess_data/q140.pause (waits after the current chunk); stop: touch /mnt/e/chess_data/q140.stop.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
BIN="$ND/build_uci/chess_engine_uci_fix"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
STOP=/mnt/e/chess_data/q140.stop
PAUSE=/mnt/e/chess_data/q140.pause
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
BASE=THREAT_ORDER_W=100
CAND="$BASE,FIX_MAX_NULL=1,NULL_NPM_GUARD=2,FIX_MATE_50=1,FIX_HALFMOVE=1,FIX_REP_KEY=1,FIX_KILLER_SPLICE=1,FIX_MATE_GATE=1"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP" "$PAUSE"
pgrep -f "fastchess" >/dev/null && { echo "[q140] matches are running - not starting"; exit 1; }
START=$(date '+%Y-%m-%d %H:%M:%S')
DEADLINE=$(date -d '2026-11-03 20:45' +%s)
echo "[q140] $START start (timed chunks must finish by $(date -d @$DEADLINE); pause $PAUSE, stop $STOP)"
bench_line() { printf "setoption name Knobs value $1\nbench\nquit\n" | "$BIN" 2>/dev/null | grep '^bench' | tail -1; }
b0=$(bench_line $BASE)
echo "$b0" | grep -q " nodes 36991901 " || { echo "[q140] ☠️ fix binary bench '$b0' (want 36991901) — QUEUE 140 ABORTED"; exit 1; }
b1=$(bench_line $CAND)
echo "[q140] candidate bench: $b1"
echo "$b1" | grep -q " nodes 36960934 " || { echo "[q140] ☠️ candidate bench (want 36960934) — QUEUE 140 ABORTED"; exit 1; }

COMMON="-repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
hold() {   # between chunks: honour the pause file, then the stop file and the deadline; $1 tag, $2 seconds needed
  if [ -e "$PAUSE" ]; then
    echo "[q140] $(date) paused before $1 (remove $PAUSE to resume)"
    while [ -e "$PAUSE" ] && [ ! -e "$STOP" ]; do sleep 30; done
    echo "[q140] $(date) resumed"
  fi
  [ -e "$STOP" ] && { echo "[q140] $(date) skip $1: stop file present"; return 1; }
  local left=$(( DEADLINE - $(date +%s) ))
  [ $left -ge $2 ] || { echo "[q140] $(date) skip $1: only ${left}s before the deadline (needs $2)"; return 1; }
}
preflight() {   # $1 tag, $2 arm knobs, $3 control knobs, $4 openings
  rm -f /tmp/q140_pre.engine.log
  timeout 180 "$FC" -engine cmd="$BIN" name=arm option.OwnBook=false option.Knobs=$2 st=0.05 \
    -engine cmd="$BIN" name=ctl option.OwnBook=false option.Knobs=$3 st=0.05 \
    -openings file=$4 format=epd order=random -rounds 1 -games 2 -concurrency 1 \
    -log file=/tmp/q140_pre.engine.log level=info engine=true > /tmp/q140_pre.log 2>&1
  local W A C; W=$(grep -c "have option" /tmp/q140_pre.log)
  A=$(grep -oE " arm ---> info string knobs [A-Z0-9_=,-]+" /tmp/q140_pre.engine.log | grep -cxF " arm ---> info string knobs $2")
  C=$(grep -oE " ctl ---> info string knobs [A-Z0-9_=,-]+" /tmp/q140_pre.engine.log | grep -cxF " ctl ---> info string knobs $3")
  echo "[q140] $1 preflight: warnings $W, arm echoes $A, control echoes $C"
  [ "$W" = "0" ] && [ "$A" -ge 1 ] && [ "$C" -ge 1 ] || { echo "[q140] ☠️ $1 PREFLIGHT FAILED — skipped"; return 1; }
}
pooled() {   # $1 tag: pooled report over its chunk PGNs
  local f; f=$(ls $O/$1_c[1-5].pgn 2>/dev/null)
  [ -n "$f" ] || return 1
  timeout 600 bash "$R" pyrun selfplay/fastchess_report.py $f --p1 arm 2>&1 | grep -E "vs the field|^  Elo |terminations"
}
match() {   # $1 tag, $2 arm knobs, $3 openings, $4 rounds per chunk, $5 seed base, $6 seconds per chunk
  rm -f $O/$1_c[1-5].pgn $O/$1_c[1-5].log
  hold $1 $6 && preflight $1 "$2" "$BASE" $3 || return
  for c in 1 2 3 4 5; do
    hold "$1 chunk $c" $6 || break
    nice -n 5 "$FC" -engine cmd="$BIN" name=arm option.OwnBook=false option.Knobs=$2 tc=10+0.1 timemargin=250 \
      -engine cmd="$BIN" name=ctl option.OwnBook=false option.Knobs=$BASE tc=10+0.1 timemargin=250 \
      -openings file=$3 format=epd order=random $COMMON -srand $(( $5 * 10 + c )) -rounds $4 -games 2 \
      -pgnout file=$O/$1_c$c.pgn notation=san nodes=true append=false > $O/$1_c$c.log 2>&1
    echo "[q140] $(date) $1 chunk $c: $(grep -E '^Elo:' $O/$1_c$c.log | tail -1)  disconnects/stalls: $(grep -ciE 'disconnect|stall|not responsive' $O/$1_c$c.log)  time forfeits: $(grep -c 'loses on time' $O/$1_c$c.pgn)"
  done
  echo "[q140] $(date) $1 pooled:"; pooled $1 | sed 's/^/[q140]   /'
}
match cand2_uho "$CAND" $O/openings_uho.epd 100 1401 2200

echo "[q140] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | sed 's/^/[q140]   /'
echo "[q140] $(date) QUEUE 140 DONE"
