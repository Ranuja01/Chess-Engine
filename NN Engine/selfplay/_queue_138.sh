#!/bin/bash
# 2026-11-02 evening + overnight, queue #138: the step 2 code items of dev_notes/active/SEARCH-CORRECTNESS-ITEMS-2026-11-02.md
# (owner 11-02: "as long as they are as efficient and correct as can be") and step 1's B1 + B2 bit 1. Waits for queue #137.
# Part A, evening (fixed-node / fixed-depth only, no timed games):
#  1 build chess_engine_uci_fix from the edited source; with no knobs its bench must equal gen 6.4's 36,991,901, or the queue aborts
#  2 bench with each new knob alone (nodes, solved, nps): FIX_MDP, FIX_MATE_GATE, FIX_MATE_50, FIX_TT_R50, FIX_KILLER_SPLICE
#  3 FIX_MATE_50 check: Ra8# on the 100th half-move (clock 99 in the FEN) must score as mate with the knob, a draw without it
#  4 per knob: self-play from eg_critical / kp_dense / UHO at 30k nodes, OFF vs ON (_fix_firecount.py, resign at 1000 cp):
#    0 crashes; games diverge where the item fires (MATE_50 / TT_R50 only in eg_critical)
# Part B, timed, from 00:05 (no timed games 9 pm - midnight), legs must finish by 10:30, each 10+0.1 vs gen 6.4 knobs, same binary:
#  b12_eg   FIX_HALFMOVE=1,FIX_REP_KEY=1 on eg_critical.epd (B1 fires only in long endgames), seed 1371. Registered -10 ... +10
#  mdp_uho  FIX_MDP=1 on UHO, seed 1372. Registered -10 ... +10 (a node saver of ~2.6% on UHO, Elo-neutral alone)
#  ks_uho   FIX_KILLER_SPLICE=1 on UHO, seed 1373. Registered -10 ... +15
#  mg_uho   FIX_MATE_GATE=1 on UHO, seed 1374. Registered -10 ... +10
# Correctness items ship on not harmful (>= -1 sigma); no replicates; ships are the owner's call.
# Stop between steps: touch /mnt/e/chess_data/q138.stop. Never edit this file while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
BIN="$ND/build_uci/chess_engine_uci_fix"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
STOP=/mnt/e/chess_data/q138.stop
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
BASE=THREAT_ORDER_W=100
KNOBS="FIX_MDP FIX_MATE_GATE FIX_MATE_50 FIX_TT_R50 FIX_KILLER_SPLICE"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"
echo "[q138] $(date '+%Y-%m-%d %H:%M:%S') waiting for queue 137"
while pgrep -f "fastchess|_queue_137\.sh" >/dev/null; do sleep 30; done
START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[q138] $START start (stop file $STOP)"

# Part A
OUT=chess_engine_uci_fix bash selfplay/build_uci.sh | sed 's/^/[q138]   /'
bench_line() { printf "setoption name Knobs value $1\nbench\nquit\n" | "$BIN" 2>/dev/null | grep '^bench' | tail -1; }
b0=$(bench_line $BASE)
echo "[q138] bench, no new knob: $b0"
echo "$b0" | grep -q " nodes 36991901 " || { echo "[q138] ☠️ default build is not result-identical (want 36991901) — QUEUE 138 ABORTED"; exit 1; }
for k in $KNOBS; do echo "[q138] bench, $k=1: $(bench_line $BASE,$k=1)"; done

uci_go() {   # $1 knobs, $2 fen, $3 depth -> last info score + bestmove
  (printf "setoption name Knobs value $1\nisready\nposition fen $2\ngo depth $3\n"; sleep 8; printf "quit\n") | timeout 30 "$BIN" 2>/dev/null \
    | awk '/^info .* score /{s=$0; sub(/.* score /,"",s); split(s,a," "); sc=a[1]" "a[2]} /^bestmove/{bm=$2} END{print "score " sc " bestmove " bm}'
}
M50="7k/8/6K1/8/8/8/8/R7 w - - 99 120"
echo "[q138] mate on the 100th half-move ($M50), depth 8:"
echo "[q138]   without FIX_MATE_50: $(uci_go $BASE "$M50" 8)"
echo "[q138]   with FIX_MATE_50:    $(uci_go $BASE,FIX_MATE_50=1 "$M50" 8)"
echo "[q138]   same position at clock 0, without: $(uci_go $BASE "7k/8/6K1/8/8/8/8/R7 w - - 0 120" 8)"

for k in $KNOBS; do
  [ -e "$STOP" ] && { echo "[q138] stop file — skip the divergence checks"; break; }
  echo "[q138] $(date) divergence check $k=1 (30k nodes, 30 starts per set, resign at 1000 cp)"
  timeout 2400 bash "$R" pyrun diagnostics/_fix_firecount.py BIN=build_uci/chess_engine_uci_fix "ON=$k=1" N=30 NODES=30000 JOBS=3 ADJ=1000 2>&1 \
    | grep -E "^=== |diverged|CRASH|Traceback|Error" | sed 's/^/[q138]   /'
done
echo "[q138] $(date) part A done"

# Part B
until [ "$(date +%Y%m%d%H%M)" -ge 202611030005 ]; do [ -e "$STOP" ] && break; sleep 60; done
DEADLINE=$(date -d '2026-11-03 10:30' +%s)
echo "[q138] $(date) part B (timed legs must finish by $(date -d @$DEADLINE))"
COMMON="-repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
report() {
  echo "[q138] $(date) $1 done: $(grep -E '^Games:|^Elo:' $O/$1.log | tail -2 | tr '\n' ' ')"
  echo "[q138]   $(grep -E '^Ptnml' $O/$1.log | tail -1)  disconnects/stalls: $(grep -ciE 'disconnect|stall|not responsive' $O/$1.log)  time forfeits: $(grep -c 'loses on time' $O/$1.pgn)"
}
gate() {   # $1 tag, $2 seconds needed
  [ -f "$STOP" ] && { echo "[q138] $(date) skip $1: stop file present"; return 1; }
  local left=$(( DEADLINE - $(date +%s) ))
  [ $left -ge $2 ] || { echo "[q138] $(date) skip $1: only ${left}s before the deadline (needs $2)"; return 1; }
}
preflight() {   # $1 tag, $2 arm knobs, $3 control knobs, $4 openings
  rm -f /tmp/q138_pre.engine.log
  timeout 180 "$FC" -engine cmd="$BIN" name=arm option.OwnBook=false option.Knobs=$2 st=0.05 \
    -engine cmd="$BIN" name=ctl option.OwnBook=false option.Knobs=$3 st=0.05 \
    -openings file=$4 format=epd order=random -rounds 1 -games 2 -concurrency 1 \
    -log file=/tmp/q138_pre.engine.log level=info engine=true > /tmp/q138_pre.log 2>&1
  local W A C; W=$(grep -c "have option" /tmp/q138_pre.log)
  A=$(grep -oE " arm ---> info string knobs [A-Z0-9_=,-]+" /tmp/q138_pre.engine.log | grep -cxF " arm ---> info string knobs $2")
  C=$(grep -oE " ctl ---> info string knobs [A-Z0-9_=,-]+" /tmp/q138_pre.engine.log | grep -cxF " ctl ---> info string knobs $3")
  echo "[q138] $1 preflight: warnings $W, arm echoes $A, control echoes $C"
  [ "$W" = "0" ] && [ "$A" -ge 1 ] && [ "$C" -ge 1 ] || { echo "[q138] ☠️ $1 PREFLIGHT FAILED — skipped"; return 1; }
}
selfplay() {   # $1 tag, $2 arm knobs, $3 control knobs, $4 openings, $5 tc, $6 rounds, $7 seed, $8 seconds needed
  gate $1 $8 && preflight $1 "$2" "$3" $4 || return
  echo "[q138] $(date) $1 ($((2 * $6)) games, $5, $(basename $4))"
  nice -n 5 "$FC" -engine cmd="$BIN" name=arm option.OwnBook=false option.Knobs=$2 tc=$5 timemargin=250 \
    -engine cmd="$BIN" name=ctl option.OwnBook=false option.Knobs=$3 tc=$5 timemargin=250 \
    -openings file=$4 format=epd order=random $COMMON -srand $7 -rounds $6 -games 2 \
    -pgnout file=$O/$1.pgn notation=san nodes=true append=false > $O/$1.log 2>&1
  report $1
}
EG=$O/eg_critical.epd
L=$(grep -c . $EG)
selfplay b12_eg "$BASE,FIX_HALFMOVE=1,FIX_REP_KEY=1" "$BASE" $EG 10+0.1 $L 1371 7500
selfplay mdp_uho "$BASE,FIX_MDP=1" "$BASE" $O/openings_uho.epd 10+0.1 500 1372 10000
selfplay ks_uho "$BASE,FIX_KILLER_SPLICE=1" "$BASE" $O/openings_uho.epd 10+0.1 500 1373 10000
selfplay mg_uho "$BASE,FIX_MATE_GATE=1" "$BASE" $O/openings_uho.epd 10+0.1 500 1374 10000

echo "[q138] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | sed 's/^/[q138]   /'
echo "[q138] $(date) QUEUE 138 DONE"
