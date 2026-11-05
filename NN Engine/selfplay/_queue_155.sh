#!/bin/bash
# 2026-11-05 ~02:00, queue #155 — replaces #154 (the Stockfish 1.1 series point, stopped before it started). #152 ended with two
# reads that pull opposite ways: ENABLE_TT_FLAG_FIX on gen 6.6 +34.2 (+15.4 … +53.2) in self-play, and gen 6.6 − gen 6.5 vs
# SF18 @ 1,100 −3.60 pp ± 4.92 (z −1.43, below the registered −1 sigma bar ⇒ gen 6.6 is flagged for the owner). Search changes
# interact, so the flag fix may be repairing something the mate bundle broke; the owner's decision needs the 2x2, and a gain
# ship needs the SF18 check (ship rule: self-play ≥ +2σ, SF18 > −1σ, Mediocre not negative). After #153:
#  1 liveness of ENABLE_TT_FLAG_FIX in the Cython module (the gauntlet's engine): WAC d10 must equal the UCI bench's
#    249 / 38,002,371 on gen 6.6, and UCI == Cython on gen 6.5 + the flag fix.
#  2 SF18 @ 1,100 nodes, seed 1521 (paired with #152's g66 / g65 runs): gen 6.6 + flag fix, then gen 6.5 + flag fix.
#    Registered: gen 6.6 + flag fix − gen 6.6 > −1σ (expected 0 … +4 pp); reported also vs gen 6.5 and flag fix on gen 6.5.
#  3 the missing self-play cell, gen 6.5 + flag fix vs gen 6.5 (gen 6.6's binary, knobs on both sides), 400 g, fresh seeds
#    155111-155112. A first look (±27 at 95%): does the flag fix's gain need the mate bundle?
# Nothing is installed. Deadline 2026-11-05 10:00. Never edit this file while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
PROD="$ND/build_uci/chess_engine_uci_g66pgo"
STOP=/mnt/e/chess_data/q155.stop
PAUSE=/mnt/e/chess_data/q155.pause
DEADLINE=$(date -d '2026-11-05 10:00' +%s)
G65E="FIX_TT_MATE=0 FIX_MDP=0 ENABLE_NULL_MATE_CLAMP=0"
G65K=FIX_TT_MATE=0,FIX_MDP=0,ENABLE_NULL_MATE_CLAMP=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"
echo "[q155] $(date) waiting for queue #153 (stop: $STOP)"
until grep -qE 'QUEUE 153 (DONE|ABORTED)' /mnt/e/chess_data/q153.log 2>/dev/null; do
  [ -e "$STOP" ] && { echo "[q155] stop file present — QUEUE 155 DONE (nothing run)"; exit 0; }
  sleep 30
done
sleep 20
pgrep -f "fastchess-linux-x86-64/fastchess|vs_sf.py" >/dev/null && { echo "[q155] a match is running — QUEUE 155 ABORTED"; exit 1; }
START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[q155] $START start"
cmp -s build_uci/chess_engine_uci "$PROD" && echo "[q155] production == chess_engine_uci_g66pgo" \
  || { echo "[q155] ☠️ production differs from chess_engine_uci_g66pgo — QUEUE 155 ABORTED"; exit 1; }

hold() {   # $1 tag, $2 seconds needed
  if [ -e "$PAUSE" ]; then
    echo "[q155] $(date) paused before $1 (remove $PAUSE to resume)"
    while [ -e "$PAUSE" ] && [ ! -e "$STOP" ]; do sleep 30; done
    echo "[q155] $(date) resumed"
  fi
  [ -e "$STOP" ] && { echo "[q155] $(date) skip $1: stop file present"; return 1; }
  local left=$(( DEADLINE - $(date +%s) ))
  [ $left -ge $2 ] || { echo "[q155] $(date) skip $1: only ${left}s before the deadline (needs $2)"; return 1; }
}
k() { printf "setoption name Knobs value $2\nbench\nquit\n" | "$1" 2>/dev/null | grep '^bench'; }
nodes_of() { grep -oE '(nodes|NODES:) *[0-9]+' | grep -oE '[0-9]+' | head -1; }

# ── 1 liveness in the Cython module ──────────────────────────────────────────────────────────────────────────────────
c66=$(bash "$R" wac fp_g66f V2_PRESET=shipped ENABLE_TT_FLAG_FIX=1 2>&1 | grep -E '^SOLVED|^NODES|^EBF' | tr '\n' ' ')
echo "[q155] Cython gen 6.6 + flag fix (want 249 / 38002371): $c66"
u65=$(k "$PROD" $G65K,ENABLE_TT_FLAG_FIX=1)
c65=$(bash "$R" wac fp_g65f V2_PRESET=shipped $G65E ENABLE_TT_FLAG_FIX=1 2>&1 | grep -E '^SOLVED|^NODES|^EBF' | tr '\n' ' ')
echo "[q155] gen 6.5 + flag fix: UCI $u65 · Cython $c65"
LIVE=0
echo "$c66" | grep -q "Solved 249/300 NODES: 38002371 " && [ -n "$u65" ] \
  && [ "$(echo "$u65" | nodes_of)" = "$(echo "$c65" | nodes_of)" ] && LIVE=1
[ $LIVE = 1 ] && echo "[q155] ⇒ live in Cython, UCI == Cython" || echo "[q155] ☠️ liveness / parity failed — SF18 part skipped"

# ── 2 SF18 @ 1,100 nodes, seed 1521 ─────────────────────────────────────────────────────────────────────────────────
if [ $LIVE = 1 ]; then
  for a in "g66f|V2_PRESET=shipped ENABLE_TT_FLAG_FIX=1" "g65f|V2_PRESET=shipped $G65E ENABLE_TT_FLAG_FIX=1"; do
    hold "SF18 ${a%%|*}" 3000 || break
    echo "[q155] $(date) gauntlet g1100_${a%%|*}_s1521 (500 games)"
    bash "$R" gauntlet 500 1100 3 g1100_${a%%|*}_s1521 1521 ${a#*|} 2>&1 | grep -E 'decided|games in' | sed 's/^/[q155]   /'
  done
  pair() { [ -d selfplay/games/$1 ] && [ -d selfplay/games/$2 ] && { echo "[q155] $3:"; \
    bash "$R" pyrun diagnostics/_gauntlet_pair.py $1:$2 2>&1 | tail -2 | sed 's/^/[q155]   /'; }; }
  pair g1100_g66_s1521 g1100_g66f_s1521 "flag fix on gen 6.6 (registered > -1 sigma)"
  pair g1100_g65_s1521 g1100_g66f_s1521 "gen 6.6 + flag fix vs gen 6.5"
  pair g1100_g65_s1521 g1100_g65f_s1521 "flag fix on gen 6.5"
fi

# ── 3 self-play: gen 6.5 + flag fix vs gen 6.5 ──────────────────────────────────────────────────────────────────────
AK=$G65K,ENABLE_TT_FLAG_FIX=1
rm -f /tmp/q155_pre.engine.log
timeout 180 "$FC" -engine cmd="$PROD" name=arm option.OwnBook=false option.Knobs=$AK st=0.05 \
  -engine cmd="$PROD" name=ctl option.OwnBook=false option.Knobs=$G65K st=0.05 \
  -openings file=$O/openings_uho.epd format=epd order=random -rounds 1 -games 2 -concurrency 1 \
  -log file=/tmp/q155_pre.engine.log level=info engine=true > /tmp/q155_pre.log 2>&1
W=$(grep -c 'have option' /tmp/q155_pre.log)
echo "[q155] preflight 6.5+flag vs 6.5: $(grep -E '^Games:' /tmp/q155_pre.log | tail -1) · warnings $W · arm echoes $(grep -c "knobs $AK" /tmp/q155_pre.engine.log) · ctl echoes $(grep -cE "knobs $G65K\$|knobs $G65K[^,]" /tmp/q155_pre.engine.log)"
if [ "$W" = 0 ]; then
  COMMON="-repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
  T=ttflag65; rm -f $O/${T}_c[1-5].pgn $O/${T}_c[1-5].log
  for c in 1 2; do
    hold "$T chunk $c" 2400 || break
    nice -n 5 "$FC" -engine cmd="$PROD" name=arm option.OwnBook=false option.Knobs=$AK tc=10+0.1 timemargin=250 \
      -engine cmd="$PROD" name=ctl option.OwnBook=false option.Knobs=$G65K tc=10+0.1 timemargin=250 \
      -openings file=$O/openings_uho.epd format=epd order=random $COMMON -srand $(( 155110 + c )) -rounds 100 -games 2 \
      -pgnout file=$O/${T}_c$c.pgn notation=san nodes=true append=false > $O/${T}_c$c.log 2>&1
    echo "[q155] $(date) $T chunk $c: $(grep -E '^Elo:' $O/${T}_c$c.log | tail -1)  disconnects/stalls: $(grep -ciE 'disconnect|stall|not responsive' $O/${T}_c$c.log)  time forfeits: $(grep -c 'loses on time' $O/${T}_c$c.pgn)"
  done
  f=$(ls $O/${T}_c[1-2].pgn 2>/dev/null)
  if [ -n "$f" ]; then
    echo "[q155] $(date) $T pooled:"
    timeout 600 bash "$R" pyrun selfplay/fastchess_report.py $f --p1 arm 2>&1 | sed -n '/vs the field/,$p' | sed 's/^/[q155]   /'
  fi
else
  echo "[q155] ☠️ preflight warnings — self-play cell skipped"
fi

echo "[q155] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | sed 's/^/[q155]   /'
echo "[q155] $(date) QUEUE 155 DONE"
