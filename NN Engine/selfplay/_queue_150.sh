#!/bin/bash
# 2026-11-04 afternoon, queue #150 (owner 11-04: "thats probably worth being correct, lets try it"): the mate-score correctness
# bundle vs production, replacing #149's match (stopped before its first game).
#   bundle = FIX_TT_MATE=1 (mates stored in the TT as distance from the stored position) + FIX_MDP=1 (mate-distance pruning)
#            + ENABLE_NULL_MATE_CLAMP=1 (an unproven null-move mate returns the bound; shipped 08-22 but read at start-up with
#            default false, so off ever since — it alone removes WAC's three impossibly-short mate reports)
#   binary = chess_engine_uci_ttm for both sides (plain build; every knob off = production's 252 / 36,960,934)
# Evidence so far (#149 + the clamp re-run): WAC mates found 64 / exact 64 / shorter 0 for TTM + clamp (production 64 / 61 / 3);
#   d10 nodes: TTM + MDP +2.1% WAC, +0.8% UHO (MDP alone +6.1% / +4.1%); KQK / KRK plies beyond the shortest mate 3.75 vs 4.25.
# Registered: correctness bar, not harmful: > -1 sigma, expected -5 ... +10. A pass makes it a gen 6.6 candidate (owner ships).
# Never edit this file while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
B="$ND/build_uci/chess_engine_uci_ttm"
STOP=/mnt/e/chess_data/q150.stop
PAUSE=/mnt/e/chess_data/q150.pause
DEADLINE=$(date -d '2026-11-04 20:45' +%s)
ARM=FIX_TT_MATE=1,FIX_MDP=1,ENABLE_NULL_MATE_CLAMP=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"
START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[q150] $START start"
pgrep -f "fastchess-linux-x86-64/fastchess|vs_sf.py|_queue_149.sh" >/dev/null && { echo "[q150] a match or #149 is running — QUEUE 150 ABORTED"; exit 1; }
k() { if [ -z "$1" ]; then printf "bench\nquit\n"; else printf "setoption name Knobs value $1\nbench\nquit\n"; fi | "$B" 2>/dev/null | grep '^bench'; }
d=$(k "")
echo "[q150] off    (want 252 / 36960934): $d"
echo "$d" | grep -q "solved 252/300 nodes 36960934 " || { echo "[q150] ☠️ the build with every knob off is not production — QUEUE 150 ABORTED"; exit 1; }
echo "[q150] bundle: $(k $ARM)"
echo "[q150] $(date) liveness on UHO (fixed 30k nodes, 40 starts)"
bash "$R" pyrun diagnostics/_fix_firecount.py BIN="$B" N=40 NODES=30000 PLIES=200 SETS=uho:$O/openings_uho.epd \
  ON="FIX_TT_MATE=1;FIX_MDP=1;ENABLE_NULL_MATE_CLAMP=1" JOBS=2 2>&1 | grep -E 'diverged|CRASH' | sed 's/^/[q150]   /'
rm -f /tmp/q150_pre.engine.log
timeout 180 "$FC" -engine cmd="$B" name=arm option.OwnBook=false option.Knobs=$ARM st=0.05 \
  -engine cmd="$B" name=ctl option.OwnBook=false st=0.05 \
  -openings file=$O/openings_uho.epd format=epd order=random -rounds 1 -games 2 -concurrency 1 \
  -log file=/tmp/q150_pre.engine.log level=info engine=true > /tmp/q150_pre.log 2>&1
W=$(grep -c 'have option' /tmp/q150_pre.log); K=$(grep -c "knobs $ARM" /tmp/q150_pre.engine.log)
echo "[q150] preflight: $(grep -E '^Games:' /tmp/q150_pre.log | tail -1) · warnings $W · arm knob echoes $K"
[ "$W" = "0" ] && [ "$K" -ge 1 ] || { echo "[q150] ☠️ PREFLIGHT FAILED — match skipped. QUEUE 150 DONE"; exit 0; }
hold() {   # $1 tag, $2 seconds needed
  if [ -e "$PAUSE" ]; then
    echo "[q150] $(date) paused before $1 (remove $PAUSE to resume)"
    while [ -e "$PAUSE" ] && [ ! -e "$STOP" ]; do sleep 30; done
    echo "[q150] $(date) resumed"
  fi
  [ -e "$STOP" ] && { echo "[q150] $(date) skip $1: stop file present"; return 1; }
  local left=$(( DEADLINE - $(date +%s) ))
  [ $left -ge $2 ] || { echo "[q150] $(date) skip $1: only ${left}s before the deadline (needs $2)"; return 1; }
}
COMMON="-repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
T=mates_uho
rm -f $O/${T}_c[1-5].pgn $O/${T}_c[1-5].log
for c in 1 2 3 4 5; do
  hold "$T chunk $c" 1800 || break
  nice -n 5 "$FC" -engine cmd="$B" name=arm option.OwnBook=false option.Knobs=$ARM tc=10+0.1 timemargin=250 \
    -engine cmd="$B" name=ctl option.OwnBook=false tc=10+0.1 timemargin=250 \
    -openings file=$O/openings_uho.epd format=epd order=random $COMMON -srand $(( 1501 * 10 + c )) -rounds 100 -games 2 \
    -pgnout file=$O/${T}_c$c.pgn notation=san nodes=true append=false > $O/${T}_c$c.log 2>&1
  echo "[q150] $(date) $T chunk $c: $(grep -E '^Elo:' $O/${T}_c$c.log | tail -1)  disconnects/stalls: $(grep -ciE 'disconnect|stall|not responsive' $O/${T}_c$c.log)  time forfeits: $(grep -c 'loses on time' $O/${T}_c$c.pgn)"
done
f=$(ls $O/${T}_c[1-5].pgn 2>/dev/null)
if [ -n "$f" ]; then
  echo "[q150] $(date) $T pooled (arm = $ARM, ctl = off, same binary):"
  timeout 600 bash "$R" pyrun selfplay/fastchess_report.py $f --p1 arm 2>&1 | sed -n '/vs the field/,$p' | sed 's/^/[q150]   /'
fi
echo "[q150] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | sed 's/^/[q150]   /'
echo "[q150] $(date) QUEUE 150 DONE"
