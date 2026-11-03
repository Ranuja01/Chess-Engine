#!/bin/bash
# 2026-11-02 evening -> 11-03 morning, queue #139: replaces #138's part B (owner 11-02: start now, the 9 pm - midnight rule waived
# for tonight). Correctness items of dev_notes/active/SEARCH-CORRECTNESS-ITEMS-2026-11-02.md, all on build_uci/chess_engine_uci_fix
# (part A of #138 verified it: no knob = gen 6.4's bench 36,991,901), knobs through option.Knobs, 10+0.1.
# Each match runs as 200-game chunks with their own seeds, pooled by fastchess_report.py over the chunk PGNs, so it can be
# PAUSED: touch /mnt/e/chess_data/q139.pause and the queue waits after the current chunk (about 30 min at most) until the file is
# removed. touch /mnt/e/chess_data/q139.stop ends it after the current chunk. Never edit this file while it runs.
#  1 b12_eg   FIX_HALFMOVE=1,FIX_REP_KEY=1 on eg_critical.epd (B1 fires only in long endgames). Registered -10 ... +10
#  2 mdp_uho  FIX_MDP=1 on UHO. Registered -10 ... +10 (bench: +14% nodes on WAC, all from mate positions; UHO -0.7%)
#  3 ks_uho   FIX_KILLER_SPLICE=1 on UHO. Registered -10 ... +15 (UHO -4.5% nodes at d10)
#  4 mg_uho   FIX_MATE_GATE=1 on UHO. Registered -10 ... +10
#  5 cand_uho the gen 6.5 candidate vs gen 6.4 on UHO: A4 + G5 + FIX_MATE_50 (passed / verified 11-02) + each of 1-4 whose pooled
#    result is not harmful (Elo >= -1 sigma). Registered -5 ... +20.
#  6 combined bench, candidate vs gen 6.4 (fixed depth, no timed games): WAC bench nodes / solved / time; d10 nodes per position
#    on 300 UHO and 300 WAC positions (_depth_nodes_probe.py PAIRED=1)
#  7 g5_tr_r  G5 replicate on transition.epd (first run +15.0 +/- 12.8, above its band), chunks while time remains
# Timed chunks must finish by 10:30. Correctness items ship on not harmful; ships are the owner's call.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
BIN="$ND/build_uci/chess_engine_uci_fix"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
STOP=/mnt/e/chess_data/q139.stop
PAUSE=/mnt/e/chess_data/q139.pause
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
BASE=THREAT_ORDER_W=100
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP" "$PAUSE"
echo "[q139] $(date '+%Y-%m-%d %H:%M:%S') waiting for queue 138 to exit"
while pgrep -f "fastchess|_queue_138\.sh" >/dev/null; do sleep 15; done
START=$(date '+%Y-%m-%d %H:%M:%S')
DEADLINE=$(date -d '2026-11-03 10:30' +%s)
echo "[q139] $START start (timed chunks must finish by $(date -d @$DEADLINE); pause $PAUSE, stop $STOP)"
bench_line() { printf "setoption name Knobs value $1\nbench\nquit\n" | "$BIN" 2>/dev/null | grep '^bench' | tail -1; }
b0=$(bench_line $BASE)
echo "$b0" | grep -q " nodes 36991901 " || { echo "[q139] ☠️ fix binary bench '$b0' (want 36991901) — QUEUE 139 ABORTED"; exit 1; }

COMMON="-repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
hold() {   # between chunks: honour the pause file, then the stop file and the deadline; $1 tag, $2 seconds needed
  if [ -e "$PAUSE" ]; then
    echo "[q139] $(date) paused before $1 (remove $PAUSE to resume)"
    while [ -e "$PAUSE" ] && [ ! -e "$STOP" ]; do sleep 30; done
    echo "[q139] $(date) resumed"
  fi
  [ -e "$STOP" ] && { echo "[q139] $(date) skip $1: stop file present"; return 1; }
  local left=$(( DEADLINE - $(date +%s) ))
  [ $left -ge $2 ] || { echo "[q139] $(date) skip $1: only ${left}s before the deadline (needs $2)"; return 1; }
}
preflight() {   # $1 tag, $2 arm knobs, $3 control knobs, $4 openings
  rm -f /tmp/q139_pre.engine.log
  timeout 180 "$FC" -engine cmd="$BIN" name=arm option.OwnBook=false option.Knobs=$2 st=0.05 \
    -engine cmd="$BIN" name=ctl option.OwnBook=false option.Knobs=$3 st=0.05 \
    -openings file=$4 format=epd order=random -rounds 1 -games 2 -concurrency 1 \
    -log file=/tmp/q139_pre.engine.log level=info engine=true > /tmp/q139_pre.log 2>&1
  local W A C; W=$(grep -c "have option" /tmp/q139_pre.log)
  A=$(grep -oE " arm ---> info string knobs [A-Z0-9_=,-]+" /tmp/q139_pre.engine.log | grep -cxF " arm ---> info string knobs $2")
  C=$(grep -oE " ctl ---> info string knobs [A-Z0-9_=,-]+" /tmp/q139_pre.engine.log | grep -cxF " ctl ---> info string knobs $3")
  echo "[q139] $1 preflight: warnings $W, arm echoes $A, control echoes $C"
  [ "$W" = "0" ] && [ "$A" -ge 1 ] && [ "$C" -ge 1 ] || { echo "[q139] ☠️ $1 PREFLIGHT FAILED — skipped"; return 1; }
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
    echo "[q139] $(date) $1 chunk $c: $(grep -E '^Elo:' $O/$1_c$c.log | tail -1)  disconnects/stalls: $(grep -ciE 'disconnect|stall|not responsive' $O/$1_c$c.log)  time forfeits: $(grep -c 'loses on time' $O/$1_c$c.pgn)"
  done
  echo "[q139] $(date) $1 pooled:"; pooled $1 | sed 's/^/[q139]   /'
}
passes() {   # $1 tag: pooled Elo >= -1 sigma (sigma = 95% interval width / 3.92)
  pooled $1 | awk '/^  Elo /{for(i=1;i<=NF;i++){if($i=="Elo") e=$(i+1); if($i=="(95%") lo=$(i+1); if($i=="…") hi=$(i+1)}}
                   END{if(e=="") exit 1; s=(hi-lo)/3.92; exit !(e+0 >= -s)}'
}
EG=$O/eg_critical.epd
UHO=$O/openings_uho.epd
match b12_eg "$BASE,FIX_HALFMOVE=1,FIX_REP_KEY=1" $EG 93 1371 1700
match mdp_uho "$BASE,FIX_MDP=1" $UHO 100 1372 2200
match ks_uho "$BASE,FIX_KILLER_SPLICE=1" $UHO 100 1373 2200
match mg_uho "$BASE,FIX_MATE_GATE=1" $UHO 100 1374 2200

CAND="$BASE,FIX_MAX_NULL=1,NULL_NPM_GUARD=2,FIX_MATE_50=1"
passes b12_eg && CAND="$CAND,FIX_HALFMOVE=1,FIX_REP_KEY=1"
passes mdp_uho && CAND="$CAND,FIX_MDP=1"
passes ks_uho && CAND="$CAND,FIX_KILLER_SPLICE=1"
passes mg_uho && CAND="$CAND,FIX_MATE_GATE=1"
echo "[q139] $(date) gen 6.5 candidate: $CAND"
match cand_uho "$CAND" $UHO 100 1375 2200

echo "[q139] $(date) combined bench, candidate vs gen 6.4 (fixed depth):"
for k in "$BASE" "$CAND" "$BASE"; do echo "[q139]   [$k] $(bench_line $k)"; done
B=build_uci/chess_engine_uci_fix
for s in /mnt/e/chess_data/fastchess/openings_uho.epd diagnostics/suites/wac_uci.txt; do
  timeout 2400 bash "$R" pyrun diagnostics/_depth_nodes_probe.py base=$B@OwnBook=false@Knobs=$BASE cand=$B@OwnBook=false@Knobs=$CAND \
    SUITE=$s DEPTH=10 N=300 PAIRED=1 2>&1 | grep -vE "^\s*$" | sed 's/^/[q139]   /'
done

match g5_tr_r "$BASE,NULL_NPM_GUARD=2" $O/transition.epd 100 1376 2000

echo "[q139] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | sed 's/^/[q139]   /'
echo "[q139] $(date) QUEUE 139 DONE"
