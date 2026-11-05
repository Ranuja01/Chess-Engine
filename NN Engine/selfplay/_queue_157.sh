#!/bin/bash
# 2026-11-05 ~10:00, queue #157 (owner 11-05: ship gen 6.7 = gen 6.5 + ENABLE_TT_FLAG_FIX, mate bundle off; "be sure to present
# the bench numbers and NPS and whatnot as well"). Evidence: self-play +23.3 (+10.1 … +36.6) on gen 6.6 over 2,000 g and +13.9
# on gen 6.5 (400 g); SF18 +0.50 / +0.30 pp; Mediocre 81.7% vs gen 6.5's 79.6% (#156).
#  1 PGO build chess_engine_uci_g67pgo from the repo sources; fingerprints: default must be 251 / 36,859,850 and
#    ENABLE_TT_FLAG_FIX=0 must give gen 6.5's 252 / 36,960,934; install as production (gen 6.6 stays as
#    chess_engine_uci_g66pgo, gen 6.5 as chess_engine_uci_f15pgo); Cython rebuild and its fingerprints.
#  2 the bench report (one process at a time, quiet box): WAC at LIGHTNING and Cython WAC speed for gen 6.5 and gen 6.7;
#    UCI pinned bench alternating x6 over the gen 6.5 / 6.6 / 6.7 binaries; STS300 d10, full STS d10, STS300 and WAC at
#    249k nodes for gen 6.7 (deterministic; gen 6.5's are on record from #151: 1,908 / 9,620 / 1,890 / 250).
#  3 real-play nps: gen 6.7 production vs gen 6.5's binary, 10+0.1 UHO, 2 x 200 g, fresh seeds 157111-157112 (also a
#    production-binary read of the flag fix; registered not below -1 sigma).
# Stop: touch /mnt/e/chess_data/q157.stop (checked between steps and chunks). Never edit this file or the sources while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
P=/mnt/e/chess_data/pgo_g67
G65BIN="$ND/build_uci/chess_engine_uci_f15pgo"
G66BIN="$ND/build_uci/chess_engine_uci_g66pgo"
NEW="$ND/build_uci/chess_engine_uci_g67pgo"
STOP=/mnt/e/chess_data/q157.stop
BUNDLE=FIX_TT_MATE=1,FIX_MDP=1,ENABLE_NULL_MATE_CLAMP=1
G64K=ENABLE_TT_FLAG_FIX=0,FIX_MAX_NULL=0,NULL_NPM_GUARD=0,FIX_MATE_50=0,FIX_HALFMOVE=0,FIX_REP_KEY=0,FIX_KILLER_SPLICE=0,FIX_MATE_GATE=0
G65="V2_PRESET=shipped ENABLE_TT_FLAG_FIX=0"
G67="V2_PRESET=shipped"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"
show() { sed "s/^/[q157]   /"; }
stopped() { [ -e "$STOP" ] && { echo "[q157] $(date) stop file present before $1 — QUEUE 157 DONE"; exit 0; }; }
pgrep -f "fastchess-linux-x86-64/fastchess|vs_sf.py" >/dev/null && { echo "[q157] a match is running — QUEUE 157 ABORTED"; exit 1; }
START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[q157] $START start"
cmp -s build_uci/chess_engine_uci "$G66BIN" && echo "[q157] backup ok: production == chess_engine_uci_g66pgo" \
  || { echo "[q157] ☠️ production differs from chess_engine_uci_g66pgo — QUEUE 157 ABORTED"; exit 1; }

# ── 1 build, checks, install ──────────────────────────────────────────────────────────────────────────────────────────
rm -rf "$P"; mkdir -p "$P"
OUT=chess_engine_uci_g67gen PGO=gen PGO_DIR=$P bash selfplay/build_uci.sh || { echo "[q157] gen build failed — QUEUE 157 ABORTED"; exit 1; }
nice -n 5 "$FC" -engine cmd="$ND/build_uci/chess_engine_uci_g67gen" name=a option.OwnBook=false st=0.25 \
  -engine cmd="$ND/build_uci/chess_engine_uci_g67gen" name=b option.OwnBook=false st=0.25 \
  -openings file=$O/openings_uho.epd format=epd order=random -srand 67 -rounds 4 -games 2 -concurrency 1 > /tmp/pgo_g67_train.log 2>&1
echo "[q157] training: $(grep -E '^Games:' /tmp/pgo_g67_train.log | tail -1) · profile files $(find "$P" -name '*.gcda' | wc -l)"
OUT=chess_engine_uci_g67pgo PGO=use PGO_DIR=$P bash selfplay/build_uci.sh || { echo "[q157] use build failed — QUEUE 157 ABORTED"; exit 1; }
k() { printf "setoption name Knobs value $2\nbench\nquit\n" | "$1" 2>/dev/null | grep '^bench'; }
d=$(printf 'bench\nquit\n' | "$NEW" 2>/dev/null | grep '^bench')
g5=$(k "$NEW" ENABLE_TT_FLAG_FIX=0)
echo "[q157] gen 6.7 default (want 251 / 36859850):            $d"
echo "[q157] ENABLE_TT_FLAG_FIX=0 = gen 6.5 (want 252 / 36960934): $g5"
echo "[q157] + mate bundle = gen 6.6 + fix (want 249 / 38002371): $(k "$NEW" $BUNDLE)"
echo "[q157] mate bundle, flag fix 0 = gen 6.6 (want 252 / 39257155): $(k "$NEW" $BUNDLE,ENABLE_TT_FLAG_FIX=0)"
echo "[q157] gen 6.4 knobs (want 249 / 36991901):               $(k "$NEW" $G64K)"
echo "[q157] V2_SHIP_GEN=5 (want 258 / 47689586):               $(k "$NEW" V2_SHIP_GEN=5)"
echo "[q157] v1 EvalArm 0 (want 250 / 35310778 or 35310775):     $(printf 'setoption name EvalArm value 0\nbench\nquit\n' | "$NEW" 2>/dev/null | grep '^bench')"
echo "[q157] id: $(printf 'uci\nquit\n' | "$NEW" 2>/dev/null | grep '^id name')"
echo "$d" | grep -q "solved 251/300 nodes 36859850 " && echo "$g5" | grep -q "solved 252/300 nodes 36960934 " \
  || { echo "[q157] ☠️ fingerprints differ — production NOT replaced, QUEUE 157 ABORTED"; exit 1; }
cp "$NEW" build_uci/chess_engine_uci && cmp -s build_uci/chess_engine_uci "$NEW" \
  && echo "[q157] INSTALLED: build_uci/chess_engine_uci == chess_engine_uci_g67pgo"
echo "[q157] $(date) Cython module rebuild"
bash "$R" build 2>&1 | grep -iE 'error|copying' | tail -n 3 | show
echo "[q157] Cython gen 6.7 shipped (want 251 / 36859850): $(bash "$R" wac fp_g67 $G67 2>&1 | grep -E '^SOLVED|^NODES|^EBF' | tr '\n' ' ')"
echo "[q157] Cython gen 6.5 knobs (want 252 / 36960934):   $(bash "$R" wac fp_g65 $G65 2>&1 | grep -E '^SOLVED|^NODES|^EBF' | tr '\n' ' ')"
echo "[q157] Cython v1 (want 250 / 35310778):              $(bash "$R" wac fp_v1 2>&1 | grep -E '^SOLVED|^NODES|^EBF' | tr '\n' ' ')"

# ── 2 the bench report ────────────────────────────────────────────────────────────────────────────────────────────────
stopped "the bench report"
echo "[q157] $(date) == TIMED: WAC at LIGHTNING (depth reached per move)"
for a in "g65|$G65" "g67|$G67"; do echo "[q157] ${a%%|*}:"; bash "$R" wac_timed_depth rect_${a%%|*} ${a#*|} 2>&1 | grep -vE '^\s*$' | show; done
echo "[q157] $(date) == TIMED: Cython WAC speed, pinned core, 3 reps (MAX_NPS = the measurement)"
for a in "g65|$G65" "g67|$G67"; do
  echo "[q157] ${a%%|*}: $(bash "$R" wac_speed recs_${a%%|*} 3 ${a#*|} 2>&1 | grep -E 'NODES:|MAX_NPS|MEDIAN|SPREAD' | tr '\n' ' ')"
done
echo "[q157] $(date) == TIMED: UCI bench, gen 6.5 / 6.6 / 6.7 production binaries, pinned, alternating x6"
for i in 1 2 3 4 5 6; do
  echo "[q157]   g65 $(printf 'bench\nquit\n' | taskset -c 5 "$G65BIN" 2>/dev/null | grep '^bench')"
  echo "[q157]   g66 $(printf 'bench\nquit\n' | taskset -c 5 "$G66BIN" 2>/dev/null | grep '^bench')"
  echo "[q157]   g67 $(printf 'bench\nquit\n' | taskset -c 5 "$NEW" 2>/dev/null | grep '^bench')"
done
stopped "STS"
echo "[q157] $(date) == gen 6.7 STS300 d10 / full STS d10 / equal nodes (249,014 per position)"
echo "[q157] g67 STS300 d10: $(bash "$R" sts rec_g67 $G67 | tail -1)"
echo "[q157] g67 STS full d10: $(bash "$R" sts_full recf_g67 $G67 | grep 'STS score' | tail -1)"
echo "[q157] g67 STS300 @249k: $(bash "$R" sts sts_eqn_g67 $G67 MAX_DEPTH=64 NODE_LIMIT=249014 | tail -1)"
echo "[q157] g67 WAC @249k: $(bash "$R" wac wac_eqn_g67 $G67 MAX_DEPTH=64 NODE_LIMIT=249014 | grep -E '^SOLVED|^NODES' | tr '\n' ' ')"

# ── 3 real-play nps: gen 6.7 production vs gen 6.5's binary ───────────────────────────────────────────────────────────
stopped "the real-play match"
timeout 120 "$FC" -engine cmd="$NEW" name=arm option.OwnBook=false st=0.05 -engine cmd="$G65BIN" name=ctl option.OwnBook=false st=0.05 \
  -openings file=$O/openings_uho.epd format=epd order=random -rounds 1 -games 2 -concurrency 1 > /tmp/q157_pre.log 2>&1
W=$(grep -c 'have option' /tmp/q157_pre.log)
echo "[q157] preflight g67 vs g65: $(grep -E '^Games:' /tmp/q157_pre.log | tail -1) · warnings $W"
if [ "$W" = 0 ]; then
  COMMON="-repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
  T=g67_rp; rm -f $O/${T}_c[1-5].pgn $O/${T}_c[1-5].log
  for c in 1 2; do
    [ -e "$STOP" ] && { echo "[q157] $(date) skip $T chunk $c: stop file present"; break; }
    nice -n 5 "$FC" -engine cmd="$NEW" name=arm option.OwnBook=false tc=10+0.1 timemargin=250 \
      -engine cmd="$G65BIN" name=ctl option.OwnBook=false tc=10+0.1 timemargin=250 \
      -openings file=$O/openings_uho.epd format=epd order=random $COMMON -srand $(( 157110 + c )) -rounds 100 -games 2 \
      -pgnout file=$O/${T}_c$c.pgn notation=san nodes=true append=false > $O/${T}_c$c.log 2>&1
    echo "[q157] $(date) $T chunk $c: $(grep -E '^Elo:' $O/${T}_c$c.log | tail -1)  disconnects/stalls: $(grep -ciE 'disconnect|stall|not responsive' $O/${T}_c$c.log)  time forfeits: $(grep -c 'loses on time' $O/${T}_c$c.pgn)"
  done
  f=$(ls $O/${T}_c[1-2].pgn 2>/dev/null)
  if [ -n "$f" ]; then
    echo "[q157] $(date) $T pooled:"
    timeout 600 bash "$R" pyrun selfplay/fastchess_report.py $f --p1 arm 2>&1 | sed -n '/vs the field/,$p' | show
  fi
else
  echo "[q157] ☠️ preflight warnings — real-play match skipped"
fi
echo "[q157] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | show
echo "[q157] $(date) QUEUE 157 DONE"
