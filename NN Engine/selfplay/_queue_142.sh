#!/bin/bash
# 2026-11-03 evening -> 2026-11-04 morning, queue #142 (owner 11-03: flip speed batch 8 on, measure it, write it up).
# Speed batch 8 (SCRATCH_REUSE + PIN_FAST, result-identical) as defaults:
#  1 PGO build chess_engine_uci_g65b8pgo; fingerprints (default, the two knobs off, gen 6.4, V2_SHIP_GEN=5, v1, id); install
#    as production only if the default bench is 252 / 36,960,934. The batch-7 production binary stays as chess_engine_uci_g65fpgo.
#  2 Cython module rebuilt with the same defaults; WAC fingerprints.
#  -- everything below is timed, so it waits for 00:05 (the owner uses the box in the evening), or until the owner says the box
#     is free: touch /mnt/e/chess_data/q142.go starts it at once --
#  3 NPS reads: UCI bench pinned to one core, old vs new production, alternating x6; Cython depth_nps_bench (60 midgame FENs,
#    PRESET=LIGHTNING, ~1 s each) with the two knobs off vs on, pinned, alternating x3.
#  4 The speed's Elo at 10+0.1 on UHO: new vs old production, 1,000 games as five pausable 200-game chunks, pooled; the
#    per-engine nps from the PGNs is the real-play NPS read. Registered 0 ... +20. Timed chunks must finish by 12:00.
# Pause: touch /mnt/e/chess_data/q142.pause (waits after the current chunk); stop: touch /mnt/e/chess_data/q142.stop.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
P=/mnt/e/chess_data/pgo_g65b8
STOP=/mnt/e/chess_data/q142.stop
PAUSE=/mnt/e/chess_data/q142.pause
GO=/mnt/e/chess_data/q142.go
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
OLD="$ND/build_uci/chess_engine_uci_g65fpgo"
NEW="$ND/build_uci/chess_engine_uci_g65b8pgo"
OFF="SCRATCH_REUSE=0,PIN_FAST=0"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP" "$PAUSE" "$GO"
pgrep -f "fastchess" >/dev/null && { echo "[q142] matches are running - not starting"; exit 1; }
START=$(date '+%Y-%m-%d %H:%M:%S')
TIMED_FROM=$(date -d '2026-11-04 00:05' +%s)
DEADLINE=$(date -d '2026-11-04 12:00' +%s)
echo "[q142] $START start (timed work from $(date -d @$TIMED_FROM); timed chunks must finish by $(date -d @$DEADLINE); pause $PAUSE, stop $STOP)"

# ── 1 PGO build ─────────────────────────────────────────────────────────────────────────────────────────────────────
cmp -s build_uci/chess_engine_uci "$OLD" && echo "[q142] backup ok: production == chess_engine_uci_g65fpgo" \
  || { echo "[q142] ☠️ production differs from chess_engine_uci_g65fpgo — QUEUE 142 ABORTED"; exit 1; }
rm -rf "$P"; mkdir -p "$P"
OUT=chess_engine_uci_g65b8gen PGO=gen PGO_DIR=$P bash selfplay/build_uci.sh || exit 1
printf 'bench\nquit\n' | nice -n 5 build_uci/chess_engine_uci_g65b8gen 2>/dev/null | grep '^bench'
nice -n 5 "$FC" -engine cmd="$ND/build_uci/chess_engine_uci_g65b8gen" name=a option.OwnBook=false st=0.25 \
  -engine cmd="$ND/build_uci/chess_engine_uci_g65b8gen" name=b option.OwnBook=false st=0.25 \
  -openings file=$O/openings_uho.epd format=epd order=random -srand 65 -rounds 4 -games 2 -concurrency 1 \
  > /tmp/pgo_g65b8_train.log 2>&1
grep -E '^Games:' /tmp/pgo_g65b8_train.log | tail -1
echo "[q142] profile files: $(find "$P" -name '*.gcda' | wc -l)"
OUT=chess_engine_uci_g65b8pgo PGO=use PGO_DIR=$P bash selfplay/build_uci.sh || exit 1
k() { printf "setoption name Knobs value $2\nbench\nquit\n" | "$1" 2>/dev/null | grep '^bench'; }
G64="FIX_MAX_NULL=0,NULL_NPM_GUARD=0,FIX_MATE_50=0,FIX_HALFMOVE=0,FIX_REP_KEY=0,FIX_KILLER_SPLICE=0,FIX_MATE_GATE=0"
d=$(printf 'bench\nquit\n' | "$NEW" 2>/dev/null | grep '^bench')
echo "[q142] default (want 252 / 36960934):           $d"
echo "[q142] batch-8 knobs off (same nodes):          $(k "$NEW" $OFF)"
echo "[q142] gen 6.4 knobs (want 249 / 36991901):     $(k "$NEW" $G64)"
echo "[q142] V2_SHIP_GEN=5 (want 258 / 47689586):     $(k "$NEW" V2_SHIP_GEN=5)"
echo "[q142] v1 EvalArm 0 (want 250 / 35310778):      $(printf 'setoption name EvalArm value 0\nbench\nquit\n' | "$NEW" 2>/dev/null | grep '^bench')"
echo "[q142] id: $(printf 'uci\nquit\n' | "$NEW" 2>/dev/null | grep '^id name')"
echo "$d" | grep -q "solved 252/300 nodes 36960934 " || { echo "[q142] ☠️ default bench is not gen 6.5's — production NOT replaced, QUEUE 142 ABORTED"; exit 1; }
cp "$NEW" build_uci/chess_engine_uci && echo "[q142] INSTALLED: build_uci/chess_engine_uci == chess_engine_uci_g65b8pgo"

# ── 2 Cython module ─────────────────────────────────────────────────────────────────────────────────────────────────
echo "[q142] $(date) Cython module rebuild"
bash "$R" build 2>&1 | grep -iE 'error|copying' | tail -n 3
echo "[q142] Cython v2 shipped (want 252 / 36960934 / 3.813): $(V2_PRESET=shipped bash "$R" wac fp_v2 V2_PRESET=shipped 2>&1 | grep -E 'SOLVED|NODES|EBF' | tr '\n' ' ')"
echo "[q142] Cython v1 (want 250 / 35310778 / 3.784):         $(bash "$R" wac fp_v1 2>&1 | grep -E 'SOLVED|NODES|EBF' | tr '\n' ' ')"

# ── wait for the timed window ───────────────────────────────────────────────────────────────────────────────────────
if [ "$(date +%s)" -lt "$TIMED_FROM" ]; then
  echo "[q142] $(date) build part done; waiting for $(date -d @$TIMED_FROM) (or $GO) before the timed reads"
  while [ "$(date +%s)" -lt "$TIMED_FROM" ] && [ ! -e "$STOP" ] && [ ! -e "$GO" ]; do sleep 30; done
  [ -e "$GO" ] && echo "[q142] $(date) go file present: starting the timed part early"
fi
[ -e "$STOP" ] && { echo "[q142] $(date) stop file present before the timed part — QUEUE 142 DONE (build and install only)"; exit 0; }

# ── 3 NPS reads ─────────────────────────────────────────────────────────────────────────────────────────────────────
echo "[q142] $(date) NPS read A: UCI bench pinned to one core, old vs new production, alternating x6"
for r in 1 2 3 4 5 6; do
  echo "[q142]   old $(printf 'bench\nquit\n' | taskset -c 5 "$OLD" 2>/dev/null | grep '^bench')"
  echo "[q142]   new $(printf 'bench\nquit\n' | taskset -c 5 "$NEW" 2>/dev/null | grep '^bench')"
done
echo "[q142] $(date) NPS read B: Cython depth_nps_bench, 60 midgame FENs, PRESET=LIGHTNING, knobs off vs on, pinned, alternating x3"
for r in 1 2 3; do
  echo "[q142]   off $(env PRESET=LIGHTNING USE_OPENING_BOOK=0 V2_PRESET=shipped SCRATCH_REUSE=0 PIN_FAST=0 taskset -c 5 bash "$R" pyrun diagnostics/depth_nps_bench.py --n 60 2>&1 | grep -E 'DEPTH reached|NPS: median' | tr '\n' ' ')"
  echo "[q142]   on  $(env PRESET=LIGHTNING USE_OPENING_BOOK=0 V2_PRESET=shipped taskset -c 5 bash "$R" pyrun diagnostics/depth_nps_bench.py --n 60 2>&1 | grep -E 'DEPTH reached|NPS: median' | tr '\n' ' ')"
done

# ── 4 the speed's Elo ───────────────────────────────────────────────────────────────────────────────────────────────
COMMON="-repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
hold() {   # between chunks: honour the pause file, then the stop file and the deadline; $1 tag, $2 seconds needed
  if [ -e "$PAUSE" ]; then
    echo "[q142] $(date) paused before $1 (remove $PAUSE to resume)"
    while [ -e "$PAUSE" ] && [ ! -e "$STOP" ]; do sleep 30; done
    echo "[q142] $(date) resumed"
  fi
  [ -e "$STOP" ] && { echo "[q142] $(date) skip $1: stop file present"; return 1; }
  local left=$(( DEADLINE - $(date +%s) ))
  [ $left -ge $2 ] || { echo "[q142] $(date) skip $1: only ${left}s before the deadline (needs $2)"; return 1; }
}
T=speed8_uho
rm -f $O/${T}_c[1-5].pgn $O/${T}_c[1-5].log
timeout 120 "$FC" -engine cmd="$NEW" name=arm option.OwnBook=false st=0.05 -engine cmd="$OLD" name=ctl option.OwnBook=false st=0.05 \
  -openings file=$O/openings_uho.epd format=epd order=random -rounds 1 -games 2 -concurrency 1 > /tmp/q142_pre.log 2>&1
echo "[q142] preflight: $(grep -E '^Games:' /tmp/q142_pre.log | tail -1) · warnings $(grep -c 'have option' /tmp/q142_pre.log)"
for c in 1 2 3 4 5; do
  hold "$T chunk $c" 2200 || break
  nice -n 5 "$FC" -engine cmd="$NEW" name=arm option.OwnBook=false tc=10+0.1 timemargin=250 \
    -engine cmd="$OLD" name=ctl option.OwnBook=false tc=10+0.1 timemargin=250 \
    -openings file=$O/openings_uho.epd format=epd order=random $COMMON -srand $(( 1421 * 10 + c )) -rounds 100 -games 2 \
    -pgnout file=$O/${T}_c$c.pgn notation=san nodes=true append=false > $O/${T}_c$c.log 2>&1
  echo "[q142] $(date) $T chunk $c: $(grep -E '^Elo:' $O/${T}_c$c.log | tail -1)  disconnects/stalls: $(grep -ciE 'disconnect|stall|not responsive' $O/${T}_c$c.log)  time forfeits: $(grep -c 'loses on time' $O/${T}_c$c.pgn)"
done
f=$(ls $O/${T}_c[1-5].pgn 2>/dev/null)
if [ -n "$f" ]; then
  echo "[q142] $(date) $T pooled (arm = batch-8 production, ctl = batch-7 production):"
  timeout 600 bash "$R" pyrun selfplay/fastchess_report.py $f --p1 arm 2>&1 | sed -n '/vs the field/,$p' | sed 's/^/[q142]   /'
fi
echo "[q142] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | sed 's/^/[q142]   /'
echo "[q142] $(date) QUEUE 142 DONE"
