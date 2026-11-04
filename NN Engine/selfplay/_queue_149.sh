#!/bin/bash
# 2026-11-04 afternoon, queue #149: TT mate storage (FIX_TT_MATE) with and without mate-distance pruning (FIX_MDP), on the
# plain build chess_engine_uci_ttm (default bench 252 / 36,960,934 = production; dev_notes/active/SEARCH-CORRECTNESS-ITEMS).
#  1 bench (WAC d10, full run) for off / TTM / MDP / both
#  2 targeted: diagnostics/_mate_distance.py MODE=find (WAC positions Stockfish reports as mates: found, exact distance,
#    impossible shorter reports, nodes to the first mate report) and MODE=play (KQK / KRK self-play: mates, plies beyond the
#    shortest)
#  3 d10 nodes per position, WAC and UHO, paired against off (_depth_nodes_probe.py PAIRED=1)
#  4 liveness on UHO, then 1,000 g at 10+0.1 UHO: both = FIX_TT_MATE=1,FIX_MDP=1 vs off, same binary
# Registered: (2) TTM reports no mate shorter than Stockfish's where off reports none, and finds mates in no more nodes
# (median); MDP's +14% WAC bench nodes shrinks with TTM. (4) correctness bar, not harmful: > -1 sigma, expected -5 ... +10.
# Never edit this file while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
B="$ND/build_uci/chess_engine_uci_ttm"
STOP=/mnt/e/chess_data/q149.stop
PAUSE=/mnt/e/chess_data/q149.pause
DEADLINE=$(date -d '2026-11-04 20:45' +%s)
BOTH=FIX_TT_MATE=1,FIX_MDP=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"
START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[q149] $START start"
pgrep -f "fastchess-linux-x86-64/fastchess|vs_sf.py" >/dev/null && { echo "[q149] a match is running — QUEUE 149 ABORTED"; exit 1; }
k() { if [ -z "$1" ]; then printf "bench\nquit\n"; else printf "setoption name Knobs value $1\nbench\nquit\n"; fi | "$B" 2>/dev/null | grep '^bench'; }
d=$(k "")
echo "[q149] off  (want 252 / 36960934): $d"
echo "$d" | grep -q "solved 252/300 nodes 36960934 " || { echo "[q149] ☠️ the build with every knob off is not production — QUEUE 149 ABORTED"; exit 1; }
echo "[q149] TTM:  $(k FIX_TT_MATE=1)"
echo "[q149] MDP:  $(k FIX_MDP=1)"
echo "[q149] both: $(k $BOTH)"

ARMS="off:;ttm:FIX_TT_MATE=1;mdp:FIX_MDP=1;both:FIX_TT_MATE=1,FIX_MDP=1"
echo "[q149] $(date) mate finding on WAC (1M nodes per position)"
bash "$R" pyrun diagnostics/_mate_distance.py BIN="$B" MODE=find ARMS="$ARMS" NODES=1000000 JOBS=2 2>&1 | sed 's/^/[q149]   /'
echo "[q149] $(date) KQK / KRK self-play (200k nodes per move, 60 starts)"
bash "$R" pyrun diagnostics/_mate_distance.py BIN="$B" MODE=play ARMS="$ARMS" NODES=200000 N=60 JOBS=2 2>&1 | sed 's/^/[q149]   /'

for S in wac_uci.txt /mnt/e/chess_data/fastchess/openings_uho.epd; do
  echo "[q149] $(date) d10 nodes per position, paired vs off: $S"
  bash "$R" pyrun diagnostics/_depth_nodes_probe.py off="$B@OwnBook=false" ttm="$B@OwnBook=false@Knobs=FIX_TT_MATE=1" \
    mdp="$B@OwnBook=false@Knobs=FIX_MDP=1" both="$B@OwnBook=false@Knobs=$BOTH" SUITE=$S DEPTH=10 N=300 PAIRED=1 2>&1 \
    | grep -vE '^\s*$' | sed 's/^/[q149]   /'
done

[ -e "$STOP" ] && { echo "[q149] $(date) stop file present — QUEUE 149 DONE (no match)"; exit 0; }
echo "[q149] $(date) liveness on UHO (fixed 30k nodes, 40 starts): $BOTH"
bash "$R" pyrun diagnostics/_fix_firecount.py BIN="$B" N=40 NODES=30000 PLIES=200 SETS=uho:$O/openings_uho.epd ON="FIX_TT_MATE=1;FIX_MDP=1" JOBS=2 2>&1 \
  | grep -E 'diverged|CRASH' | sed 's/^/[q149]   /'
rm -f /tmp/q149_pre.engine.log
timeout 180 "$FC" -engine cmd="$B" name=arm option.OwnBook=false option.Knobs=$BOTH st=0.05 \
  -engine cmd="$B" name=ctl option.OwnBook=false st=0.05 \
  -openings file=$O/openings_uho.epd format=epd order=random -rounds 1 -games 2 -concurrency 1 \
  -log file=/tmp/q149_pre.engine.log level=info engine=true > /tmp/q149_pre.log 2>&1
W=$(grep -c 'have option' /tmp/q149_pre.log); K=$(grep -c "knobs $BOTH" /tmp/q149_pre.engine.log)
echo "[q149] preflight: $(grep -E '^Games:' /tmp/q149_pre.log | tail -1) · warnings $W · arm knob echoes $K"
[ "$W" = "0" ] && [ "$K" -ge 1 ] || { echo "[q149] ☠️ PREFLIGHT FAILED — match skipped. QUEUE 149 DONE"; exit 0; }
hold() {   # $1 tag, $2 seconds needed
  if [ -e "$PAUSE" ]; then
    echo "[q149] $(date) paused before $1 (remove $PAUSE to resume)"
    while [ -e "$PAUSE" ] && [ ! -e "$STOP" ]; do sleep 30; done
    echo "[q149] $(date) resumed"
  fi
  [ -e "$STOP" ] && { echo "[q149] $(date) skip $1: stop file present"; return 1; }
  local left=$(( DEADLINE - $(date +%s) ))
  [ $left -ge $2 ] || { echo "[q149] $(date) skip $1: only ${left}s before the deadline (needs $2)"; return 1; }
}
COMMON="-repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
T=ttm_uho
rm -f $O/${T}_c[1-5].pgn $O/${T}_c[1-5].log
for c in 1 2 3 4 5; do
  hold "$T chunk $c" 1800 || break
  nice -n 5 "$FC" -engine cmd="$B" name=arm option.OwnBook=false option.Knobs=$BOTH tc=10+0.1 timemargin=250 \
    -engine cmd="$B" name=ctl option.OwnBook=false tc=10+0.1 timemargin=250 \
    -openings file=$O/openings_uho.epd format=epd order=random $COMMON -srand $(( 1491 * 10 + c )) -rounds 100 -games 2 \
    -pgnout file=$O/${T}_c$c.pgn notation=san nodes=true append=false > $O/${T}_c$c.log 2>&1
  echo "[q149] $(date) $T chunk $c: $(grep -E '^Elo:' $O/${T}_c$c.log | tail -1)  disconnects/stalls: $(grep -ciE 'disconnect|stall|not responsive' $O/${T}_c$c.log)  time forfeits: $(grep -c 'loses on time' $O/${T}_c$c.pgn)"
done
f=$(ls $O/${T}_c[1-5].pgn 2>/dev/null)
if [ -n "$f" ]; then
  echo "[q149] $(date) $T pooled (arm = $BOTH, ctl = off, same binary):"
  timeout 600 bash "$R" pyrun selfplay/fastchess_report.py $f --p1 arm 2>&1 | sed -n '/vs the field/,$p' | sed 's/^/[q149]   /'
fi
echo "[q149] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | sed 's/^/[q149]   /'
echo "[q149] $(date) QUEUE 149 DONE"
