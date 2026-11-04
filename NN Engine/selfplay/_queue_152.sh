#!/bin/bash
# 2026-11-04 evening, queue #152 (owner 11-04: "lazy pass this, but continue runs now instead of waiting"; the box is free in the
# evening window tonight). After #151:
#  1 gen 6.6 = gen 6.5 + the mate-score bundle (FIX_TT_MATE, FIX_MDP, ENABLE_NULL_MATE_CLAMP in the shipped preset): PGO build
#    chess_engine_uci_g66pgo, fingerprints (default must equal the bundle's 252 / 39,257,155; the three at 0 must give gen 6.5's
#    252 / 36,960,934), install as production (gen 6.5's F15 build stays as chess_engine_uci_f15pgo), Cython rebuild and its
#    fingerprints, pinned alternating bench vs gen 6.5's binary.
#  2 the replicate: gen 6.6 vs gen 6.5 production binaries, 10+0.1 UHO, 1,000 g, fresh seeds 15211-15215. Registered: pooled
#    with #150 (-9.0 +- 19.3) the bundle stays above -1 sigma; expected -5 ... +10.
#  3 ENABLE_TT_FLAG_FIX (shipped 08-22, off ever since): liveness, then 1,000 g on gen 6.6 (arm knob on vs off), seeds
#    15221-15225. Registered: not harmful (> -1 sigma), expected -5 ... +10.
#  4 outside opponent, node-limited: SF18 @ 1,100 nodes vs ours @ 250k, 500 g each for gen 6.6 and gen 6.5 (seed 1521).
#    Registered: gen 6.6 - gen 6.5 > -1 sigma.
# Never edit this file while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
P=/mnt/e/chess_data/pgo_g66
OLD="$ND/build_uci/chess_engine_uci_f15pgo"
NEW="$ND/build_uci/chess_engine_uci_g66pgo"
STOP=/mnt/e/chess_data/q152.stop
PAUSE=/mnt/e/chess_data/q152.pause
DEADLINE=$(date -d '2026-11-05 10:00' +%s)
G65K=FIX_TT_MATE=0,FIX_MDP=0,ENABLE_NULL_MATE_CLAMP=0
G64K=$G65K,FIX_MAX_NULL=0,NULL_NPM_GUARD=0,FIX_MATE_50=0,FIX_HALFMOVE=0,FIX_REP_KEY=0,FIX_KILLER_SPLICE=0,FIX_MATE_GATE=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"
echo "[q152] $(date) waiting for queue #151 (stop: $STOP)"
until grep -qE 'QUEUE 151 (DONE|ABORTED)' /mnt/e/chess_data/q151.log 2>/dev/null; do
  [ -e "$STOP" ] && { echo "[q152] stop file present — QUEUE 152 DONE (nothing run)"; exit 0; }
  sleep 30
done
sleep 20
pgrep -f "fastchess-linux-x86-64/fastchess|vs_sf.py" >/dev/null && { echo "[q152] a match is running — QUEUE 152 ABORTED"; exit 1; }
START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[q152] $START start"
cmp -s build_uci/chess_engine_uci "$OLD" && echo "[q152] backup ok: production == chess_engine_uci_f15pgo" \
  || { echo "[q152] ☠️ production differs from chess_engine_uci_f15pgo — QUEUE 152 ABORTED"; exit 1; }

# ── 1 gen 6.6 build, checks, install ─────────────────────────────────────────────────────────────────────────────────
rm -rf "$P"; mkdir -p "$P"
OUT=chess_engine_uci_g66gen PGO=gen PGO_DIR=$P bash selfplay/build_uci.sh || { echo "[q152] gen build failed — QUEUE 152 ABORTED"; exit 1; }
nice -n 5 "$FC" -engine cmd="$ND/build_uci/chess_engine_uci_g66gen" name=a option.OwnBook=false st=0.25 \
  -engine cmd="$ND/build_uci/chess_engine_uci_g66gen" name=b option.OwnBook=false st=0.25 \
  -openings file=$O/openings_uho.epd format=epd order=random -srand 65 -rounds 4 -games 2 -concurrency 1 > /tmp/pgo_g66_train.log 2>&1
echo "[q152] training: $(grep -E '^Games:' /tmp/pgo_g66_train.log | tail -1) · profile files $(find "$P" -name '*.gcda' | wc -l)"
OUT=chess_engine_uci_g66pgo PGO=use PGO_DIR=$P bash selfplay/build_uci.sh || { echo "[q152] use build failed — QUEUE 152 ABORTED"; exit 1; }
k() { printf "setoption name Knobs value $2\nbench\nquit\n" | "$1" 2>/dev/null | grep '^bench'; }
d=$(printf 'bench\nquit\n' | "$NEW" 2>/dev/null | grep '^bench')
g5=$(k "$NEW" $G65K)
echo "[q152] gen 6.6 default (want 252 / 39257155):  $d"
echo "[q152] three at 0 = gen 6.5 (want 252 / 36960934): $g5"
echo "[q152] gen 6.4 knobs (want 249 / 36991901):  $(k "$NEW" $G64K)"
echo "[q152] V2_SHIP_GEN=5 (want 258 / 47689586):  $(k "$NEW" V2_SHIP_GEN=5)"
echo "[q152] v1 EvalArm 0 (F15 builds 35310775):    $(printf 'setoption name EvalArm value 0\nbench\nquit\n' | "$NEW" 2>/dev/null | grep '^bench')"
echo "[q152] id: $(printf 'uci\nquit\n' | "$NEW" 2>/dev/null | grep '^id name')"
echo "$d" | grep -q "solved 252/300 nodes 39257155 " && echo "$g5" | grep -q "solved 252/300 nodes 36960934 " \
  || { echo "[q152] ☠️ fingerprints differ — production NOT replaced, QUEUE 152 ABORTED"; exit 1; }
cp "$NEW" build_uci/chess_engine_uci && echo "[q152] INSTALLED: build_uci/chess_engine_uci == chess_engine_uci_g66pgo"
echo "[q152] $(date) Cython module rebuild"
bash "$R" build 2>&1 | grep -iE 'error|copying' | tail -n 3 | sed 's/^/[q152]   /'
echo "[q152] Cython v2 shipped (want 252 / 39257155): $(bash "$R" wac fp_g66 V2_PRESET=shipped 2>&1 | grep -E 'SOLVED|NODES|EBF' | tr '\n' ' ')"
echo "[q152] Cython gen 6.5 knobs (want 252 / 36960934): $(bash "$R" wac fp_g65k V2_PRESET=shipped FIX_TT_MATE=0 FIX_MDP=0 ENABLE_NULL_MATE_CLAMP=0 2>&1 | grep -E 'SOLVED|NODES|EBF' | tr '\n' ' ')"
echo "[q152] Cython v1 (want 250 / 35310778): $(bash "$R" wac fp_v1 2>&1 | grep -E 'SOLVED|NODES|EBF' | tr '\n' ' ')"
echo "[q152] $(date) pinned bench, gen 6.5 binary vs gen 6.6 binary, alternating x6"
for r in 1 2 3 4 5 6; do
  echo "[q152]   g65 $(printf 'bench\nquit\n' | taskset -c 5 "$OLD" 2>/dev/null | grep '^bench')"
  echo "[q152]   g66 $(printf 'bench\nquit\n' | taskset -c 5 "$NEW" 2>/dev/null | grep '^bench')"
done

hold() {   # $1 tag, $2 seconds needed
  if [ -e "$PAUSE" ]; then
    echo "[q152] $(date) paused before $1 (remove $PAUSE to resume)"
    while [ -e "$PAUSE" ] && [ ! -e "$STOP" ]; do sleep 30; done
    echo "[q152] $(date) resumed"
  fi
  [ -e "$STOP" ] && { echo "[q152] $(date) skip $1: stop file present"; return 1; }
  local left=$(( DEADLINE - $(date +%s) ))
  [ $left -ge $2 ] || { echo "[q152] $(date) skip $1: only ${left}s before the deadline (needs $2)"; return 1; }
}
COMMON="-repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
match() {   # $1 tag, $2 seed base, $3 arm binary, $4 arm knobs (or -), $5 ctl binary
  local T=$1 c ak=""
  [ "$4" != "-" ] && ak="option.Knobs=$4"
  rm -f $O/${T}_c[1-5].pgn $O/${T}_c[1-5].log
  for c in 1 2 3 4 5; do
    hold "$T chunk $c" 2400 || break
    nice -n 5 "$FC" -engine cmd="$3" name=arm option.OwnBook=false $ak tc=10+0.1 timemargin=250 \
      -engine cmd="$5" name=ctl option.OwnBook=false tc=10+0.1 timemargin=250 \
      -openings file=$O/openings_uho.epd format=epd order=random $COMMON -srand $(( $2 + c )) -rounds 100 -games 2 \
      -pgnout file=$O/${T}_c$c.pgn notation=san nodes=true append=false > $O/${T}_c$c.log 2>&1
    echo "[q152] $(date) $T chunk $c: $(grep -E '^Elo:' $O/${T}_c$c.log | tail -1)  disconnects/stalls: $(grep -ciE 'disconnect|stall|not responsive' $O/${T}_c$c.log)  time forfeits: $(grep -c 'loses on time' $O/${T}_c$c.pgn)"
  done
  local f; f=$(ls $O/${T}_c[1-5].pgn 2>/dev/null)
  if [ -n "$f" ]; then
    echo "[q152] $(date) $T pooled:"
    timeout 600 bash "$R" pyrun selfplay/fastchess_report.py $f --p1 arm 2>&1 | sed -n '/vs the field/,$p' | sed 's/^/[q152]   /'
  fi
}

# ── 2 the replicate ───────────────────────────────────────────────────────────────────────────────────────────────────
timeout 120 "$FC" -engine cmd="$NEW" name=arm option.OwnBook=false st=0.05 -engine cmd="$OLD" name=ctl option.OwnBook=false st=0.05 \
  -openings file=$O/openings_uho.epd format=epd order=random -rounds 1 -games 2 -concurrency 1 > /tmp/q152_pre.log 2>&1
echo "[q152] preflight g66 vs g65: $(grep -E '^Games:' /tmp/q152_pre.log | tail -1) · warnings $(grep -c 'have option' /tmp/q152_pre.log)"
echo "[q152] $(date) == replicate: gen 6.6 (arm) vs gen 6.5 (ctl)"
match g66_rep 152110 "$NEW" - "$OLD"

# ── 3 ENABLE_TT_FLAG_FIX on gen 6.6 ───────────────────────────────────────────────────────────────────────────────────
echo "[q152] $(date) == ENABLE_TT_FLAG_FIX liveness on UHO (fixed 30k nodes, 40 starts)"
bash "$R" pyrun diagnostics/_fix_firecount.py BIN="$NEW" N=40 NODES=30000 PLIES=200 SETS=uho:$O/openings_uho.epd ON="ENABLE_TT_FLAG_FIX=1" JOBS=2 2>&1 \
  | grep -E 'diverged|CRASH' | sed 's/^/[q152]   /'
echo "[q152] ENABLE_TT_FLAG_FIX bench: $(k "$NEW" ENABLE_TT_FLAG_FIX=1)"
rm -f /tmp/q152_pre2.engine.log
timeout 180 "$FC" -engine cmd="$NEW" name=arm option.OwnBook=false option.Knobs=ENABLE_TT_FLAG_FIX=1 st=0.05 \
  -engine cmd="$NEW" name=ctl option.OwnBook=false st=0.05 \
  -openings file=$O/openings_uho.epd format=epd order=random -rounds 1 -games 2 -concurrency 1 \
  -log file=/tmp/q152_pre2.engine.log level=info engine=true > /tmp/q152_pre2.log 2>&1
echo "[q152] preflight flag fix: $(grep -E '^Games:' /tmp/q152_pre2.log | tail -1) · warnings $(grep -c 'have option' /tmp/q152_pre2.log) · knob echoes $(grep -c 'knobs ENABLE_TT_FLAG_FIX=1' /tmp/q152_pre2.engine.log)"
match ttflag_uho 152210 "$NEW" ENABLE_TT_FLAG_FIX=1 "$NEW"

# ── 4 SF18 @ 1,100 nodes, node-limited ────────────────────────────────────────────────────────────────────────────────
for a in "g66|V2_PRESET=shipped" "g65|V2_PRESET=shipped FIX_TT_MATE=0 FIX_MDP=0 ENABLE_NULL_MATE_CLAMP=0"; do
  hold "SF18 ${a%%|*}" 3000 || break
  echo "[q152] $(date) gauntlet g1100_${a%%|*}_s1521 (500 games)"
  bash "$R" gauntlet 500 1100 3 g1100_${a%%|*}_s1521 1521 ${a#*|} 2>&1 | grep -E 'decided|games in' | sed 's/^/[q152]   /'
done
[ -d selfplay/games/g1100_g65_s1521 ] && [ -d selfplay/games/g1100_g66_s1521 ] && \
  bash "$R" pyrun diagnostics/_gauntlet_pair.py g1100_g65_s1521:g1100_g66_s1521 2>&1 | tail -2 | sed 's/^/[q152]   /'
echo "[q152] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | sed 's/^/[q152]   /'
echo "[q152] $(date) QUEUE 152 DONE"
