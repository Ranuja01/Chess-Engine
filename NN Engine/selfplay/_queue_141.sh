#!/bin/bash
# 2026-11-03 afternoon -> evening, queue #141 (owner 11-03: flip the speed batch on, measure it, and get a clear NPS read).
# Speed batch 7 (KS_FAST + REP_FILTER + PAWN_HASH, result-identical) as defaults:
#  1 PGO build chess_engine_uci_g65fpgo; fingerprints (default, the three knobs off, gen 6.4, V2_SHIP_GEN=5, v1, id); install
#    as production only if the default bench is 252 / 36,960,934. The gen 6.5 production binary stays as chess_engine_uci_g65pgo.
#  2 Cython module rebuilt with the same defaults; WAC fingerprints.
#  3 NPS reads: UCI bench pinned to one core, old vs new production, alternating x6; Cython depth_nps_bench (60 midgame FENs,
#    PRESET=LIGHTNING, ~1 s each) with the three knobs off vs on, pinned, alternating x3.
#  4 The speed's Elo at 10+0.1 on UHO: new vs old production, 1,000 games as five pausable 200-game chunks, pooled; the
#    per-engine nps from the PGNs is the real-play NPS read. Registered 0 ... +25. Timed chunks must finish by 20:45.
# Pause: touch /mnt/e/chess_data/q141.pause (waits after the current chunk); stop: touch /mnt/e/chess_data/q141.stop.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
P=/mnt/e/chess_data/pgo_g65f
STOP=/mnt/e/chess_data/q141.stop
PAUSE=/mnt/e/chess_data/q141.pause
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
OLD="$ND/build_uci/chess_engine_uci_g65pgo"
NEW="$ND/build_uci/chess_engine_uci_g65fpgo"
OFF="KS_FAST=0,REP_FILTER=0,PAWN_HASH=0"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP" "$PAUSE"
pgrep -f "fastchess" >/dev/null && { echo "[q141] matches are running - not starting"; exit 1; }
START=$(date '+%Y-%m-%d %H:%M:%S')
DEADLINE=$(date -d '2026-11-03 20:45' +%s)
echo "[q141] $START start (timed chunks must finish by $(date -d @$DEADLINE); pause $PAUSE, stop $STOP)"

# ── 1 PGO build ─────────────────────────────────────────────────────────────────────────────────────────────────────
cmp -s build_uci/chess_engine_uci "$OLD" && echo "[q141] backup ok: production == chess_engine_uci_g65pgo" \
  || { echo "[q141] ☠️ production differs from chess_engine_uci_g65pgo — QUEUE 141 ABORTED"; exit 1; }
rm -rf "$P"; mkdir -p "$P"
OUT=chess_engine_uci_g65fgen PGO=gen PGO_DIR=$P bash selfplay/build_uci.sh || exit 1
printf 'bench\nquit\n' | nice -n 5 build_uci/chess_engine_uci_g65fgen 2>/dev/null | grep '^bench'
nice -n 5 "$FC" -engine cmd="$ND/build_uci/chess_engine_uci_g65fgen" name=a option.OwnBook=false st=0.25 \
  -engine cmd="$ND/build_uci/chess_engine_uci_g65fgen" name=b option.OwnBook=false st=0.25 \
  -openings file=$O/openings_uho.epd format=epd order=random -srand 65 -rounds 4 -games 2 -concurrency 1 \
  > /tmp/pgo_g65f_train.log 2>&1
grep -E '^Games:' /tmp/pgo_g65f_train.log | tail -1
echo "[q141] profile files: $(find "$P" -name '*.gcda' | wc -l)"
OUT=chess_engine_uci_g65fpgo PGO=use PGO_DIR=$P bash selfplay/build_uci.sh || exit 1
k() { printf "setoption name Knobs value $2\nbench\nquit\n" | "$1" 2>/dev/null | grep '^bench'; }
G64="FIX_MAX_NULL=0,NULL_NPM_GUARD=0,FIX_MATE_50=0,FIX_HALFMOVE=0,FIX_REP_KEY=0,FIX_KILLER_SPLICE=0,FIX_MATE_GATE=0"
d=$(printf 'bench\nquit\n' | "$NEW" 2>/dev/null | grep '^bench')
echo "[q141] default (want 252 / 36960934):           $d"
echo "[q141] speed knobs off (same nodes):            $(k "$NEW" $OFF)"
echo "[q141] gen 6.4 knobs (want 249 / 36991901):     $(k "$NEW" $G64)"
echo "[q141] V2_SHIP_GEN=5 (want 258 / 47689586):     $(k "$NEW" V2_SHIP_GEN=5)"
echo "[q141] v1 EvalArm 0 (want 250 / 35310778):      $(printf 'setoption name EvalArm value 0\nbench\nquit\n' | "$NEW" 2>/dev/null | grep '^bench')"
echo "[q141] id: $(printf 'uci\nquit\n' | "$NEW" 2>/dev/null | grep '^id name')"
echo "$d" | grep -q "solved 252/300 nodes 36960934 " || { echo "[q141] ☠️ default bench is not gen 6.5's — production NOT replaced, QUEUE 141 ABORTED"; exit 1; }
cp "$NEW" build_uci/chess_engine_uci && echo "[q141] INSTALLED: build_uci/chess_engine_uci == chess_engine_uci_g65fpgo"

# ── 2 Cython module ─────────────────────────────────────────────────────────────────────────────────────────────────
echo "[q141] $(date) Cython module rebuild"
bash "$R" build 2>&1 | grep -iE 'error|copying' | tail -n 3
echo "[q141] Cython v2 shipped (want 252 / 36960934 / 3.813): $(V2_PRESET=shipped bash "$R" wac fp_v2 V2_PRESET=shipped 2>&1 | grep -E 'SOLVED|NODES|EBF' | tr '\n' ' ')"
echo "[q141] Cython v1 (want 250 / 35310778 / 3.784):         $(bash "$R" wac fp_v1 2>&1 | grep -E 'SOLVED|NODES|EBF' | tr '\n' ' ')"

# ── 3 NPS reads ─────────────────────────────────────────────────────────────────────────────────────────────────────
echo "[q141] $(date) NPS read A: UCI bench pinned to one core, old vs new production, alternating x6"
for r in 1 2 3 4 5 6; do
  echo "[q141]   old $(printf 'bench\nquit\n' | taskset -c 5 "$OLD" 2>/dev/null | grep '^bench')"
  echo "[q141]   new $(printf 'bench\nquit\n' | taskset -c 5 "$NEW" 2>/dev/null | grep '^bench')"
done
echo "[q141] $(date) NPS read B: Cython depth_nps_bench, 60 midgame FENs, PRESET=LIGHTNING, knobs off vs on, pinned, alternating x3"
for r in 1 2 3; do
  echo "[q141]   off $(env PRESET=LIGHTNING USE_OPENING_BOOK=0 V2_PRESET=shipped KS_FAST=0 REP_FILTER=0 PAWN_HASH=0 taskset -c 5 bash "$R" pyrun diagnostics/depth_nps_bench.py --n 60 2>&1 | grep -E 'DEPTH reached|NPS: median' | tr '\n' ' ')"
  echo "[q141]   on  $(env PRESET=LIGHTNING USE_OPENING_BOOK=0 V2_PRESET=shipped taskset -c 5 bash "$R" pyrun diagnostics/depth_nps_bench.py --n 60 2>&1 | grep -E 'DEPTH reached|NPS: median' | tr '\n' ' ')"
done

# ── 4 the speed's Elo ───────────────────────────────────────────────────────────────────────────────────────────────
COMMON="-repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
hold() {   # between chunks: honour the pause file, then the stop file and the deadline; $1 tag, $2 seconds needed
  if [ -e "$PAUSE" ]; then
    echo "[q141] $(date) paused before $1 (remove $PAUSE to resume)"
    while [ -e "$PAUSE" ] && [ ! -e "$STOP" ]; do sleep 30; done
    echo "[q141] $(date) resumed"
  fi
  [ -e "$STOP" ] && { echo "[q141] $(date) skip $1: stop file present"; return 1; }
  local left=$(( DEADLINE - $(date +%s) ))
  [ $left -ge $2 ] || { echo "[q141] $(date) skip $1: only ${left}s before the deadline (needs $2)"; return 1; }
}
T=speed_uho
rm -f $O/${T}_c[1-5].pgn $O/${T}_c[1-5].log
timeout 120 "$FC" -engine cmd="$NEW" name=arm option.OwnBook=false st=0.05 -engine cmd="$OLD" name=ctl option.OwnBook=false st=0.05 \
  -openings file=$O/openings_uho.epd format=epd order=random -rounds 1 -games 2 -concurrency 1 > /tmp/q141_pre.log 2>&1
echo "[q141] preflight: $(grep -E '^Games:' /tmp/q141_pre.log | tail -1) · warnings $(grep -c 'have option' /tmp/q141_pre.log)"
for c in 1 2 3 4 5; do
  hold "$T chunk $c" 2200 || break
  nice -n 5 "$FC" -engine cmd="$NEW" name=arm option.OwnBook=false tc=10+0.1 timemargin=250 \
    -engine cmd="$OLD" name=ctl option.OwnBook=false tc=10+0.1 timemargin=250 \
    -openings file=$O/openings_uho.epd format=epd order=random $COMMON -srand $(( 1411 * 10 + c )) -rounds 100 -games 2 \
    -pgnout file=$O/${T}_c$c.pgn notation=san nodes=true append=false > $O/${T}_c$c.log 2>&1
  echo "[q141] $(date) $T chunk $c: $(grep -E '^Elo:' $O/${T}_c$c.log | tail -1)  disconnects/stalls: $(grep -ciE 'disconnect|stall|not responsive' $O/${T}_c$c.log)  time forfeits: $(grep -c 'loses on time' $O/${T}_c$c.pgn)"
done
f=$(ls $O/${T}_c[1-5].pgn 2>/dev/null)
if [ -n "$f" ]; then
  echo "[q141] $(date) $T pooled (arm = new production, ctl = gen 6.5 production):"
  timeout 600 bash "$R" pyrun selfplay/fastchess_report.py $f --p1 arm 2>&1 | sed -n '/vs the field/,$p' | sed 's/^/[q141]   /'
fi
echo "[q141] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | sed 's/^/[q141]   /'
echo "[q141] $(date) QUEUE 141 DONE"
