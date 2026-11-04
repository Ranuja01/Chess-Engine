#!/bin/bash
# 2026-11-04 midday, queue #147 (owner 11-04: "we can go for it" — the targeted 50-move test and F15).
# Part 1, F15: the production PGO build again without the toolchain's hardening defaults (stack protector, CET landing pads,
#   stack-clash probes, _FORTIFY_SOURCE, PIE), through build_uci.sh's EXTRA_FLAGS. Codegen only. Checks: every fingerprint
#   unchanged, the hardening really gone from the binary, build-identity games vs production (every set "diverged: 0"),
#   pinned alternating bench vs production. NOT installed: the numbers go to the owner.
#   Registered: bench nodes identical; new faster in the pinned bench, guess 0 ... 3%.
# Part 2, FIX_TT_R50 where the 50-move rule decides: venue r50_crit (diagnostics/_r50_venue.py: positions 22 half-moves
#   before a past fifty-move draw, Stockfish 18 at the real clock says one side is still >= 2 pawns ahead). Run A = arm
#   (FIX_TT_R50=1) vs production, run B = production vs itself (the reference), same positions, 10+0.1, no adjudication of
#   any kind, interleaved per chunk of 100 positions. diagnostics/_r50_split.py splits A into conversion (arm attacking)
#   and defence (arm defending), each against B.
#   Registered: conversion delta > 0, expected 0 ... +10 pp; defence delta <= 0. Adopt if either improves by >= 2 sigma and
#   the other is not worse by more than 1 sigma.
# Never edit this file while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
P=/mnt/e/chess_data/pgo_f15
PROD="$ND/build_uci/chess_engine_uci"
REF="$ND/build_uci/chess_engine_uci_g65b8pgo"
NEW="$ND/build_uci/chess_engine_uci_f15pgo"
HF="-fno-stack-protector -fcf-protection=none -fno-stack-clash-protection -U_FORTIFY_SOURCE -fno-pie -no-pie"
V=$O/r50_crit
VLOG=/mnt/e/chess_data/r50_venue.log
STOP=/mnt/e/chess_data/q147.stop
PAUSE=/mnt/e/chess_data/q147.pause
DEADLINE=$(date -d '2026-11-04 20:30' +%s)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"
START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[q147] $START start"
# the fastchess program itself (the venue builder's arguments contain the fastchess data directory)
pgrep -f "fastchess-linux-x86-64/fastchess|vs_sf.py" >/dev/null && { echo "[q147] a match is running — QUEUE 147 ABORTED"; exit 1; }
cmp -s "$PROD" "$REF" && echo "[q147] production == chess_engine_uci_g65b8pgo" \
  || { echo "[q147] ☠️ production differs from chess_engine_uci_g65b8pgo — QUEUE 147 ABORTED"; exit 1; }

# ── Part 1a: F15 build (may overlap the venue labelling: one compile job) ──────────────────────────────────────────────
echo "[q147] $(date) F15 PGO build, EXTRA_FLAGS=$HF"
rm -rf "$P"; mkdir -p "$P"
OUT=chess_engine_uci_f15gen PGO=gen PGO_DIR=$P EXTRA_FLAGS="$HF" bash selfplay/build_uci.sh || { echo "[q147] F15 gen build failed — part 1 skipped"; F15=0; }
if [ "${F15:-1}" = 1 ]; then
  nice -n 5 "$FC" -engine cmd="$ND/build_uci/chess_engine_uci_f15gen" name=a option.OwnBook=false st=0.25 \
    -engine cmd="$ND/build_uci/chess_engine_uci_f15gen" name=b option.OwnBook=false st=0.25 \
    -openings file=$O/openings_uho.epd format=epd order=random -srand 65 -rounds 4 -games 2 -concurrency 1 \
    > /tmp/pgo_f15_train.log 2>&1
  echo "[q147] training: $(grep -E '^Games:' /tmp/pgo_f15_train.log | tail -1) · profile files $(find "$P" -name '*.gcda' | wc -l)"
  OUT=chess_engine_uci_f15pgo PGO=use PGO_DIR=$P EXTRA_FLAGS="$HF" bash selfplay/build_uci.sh || { echo "[q147] F15 use build failed — part 1 skipped"; F15=0; }
fi
if [ "${F15:-1}" = 1 ]; then
  k() { printf "setoption name Knobs value $2\nbench\nquit\n" | "$1" 2>/dev/null | grep '^bench'; }
  G64="FIX_MAX_NULL=0,NULL_NPM_GUARD=0,FIX_MATE_50=0,FIX_HALFMOVE=0,FIX_REP_KEY=0,FIX_KILLER_SPLICE=0,FIX_MATE_GATE=0"
  d=$(printf 'bench\nquit\n' | "$NEW" 2>/dev/null | grep '^bench')
  echo "[q147] F15 default (want 252 / 36960934):       $d"
  echo "[q147] F15 gen 6.4 knobs (want 249 / 36991901): $(k "$NEW" $G64)"
  echo "[q147] F15 V2_SHIP_GEN=5 (want 258 / 47689586): $(k "$NEW" V2_SHIP_GEN=5)"
  echo "[q147] F15 v1 EvalArm 0 (want 250 / 35310778):  $(printf 'setoption name EvalArm value 0\nbench\nquit\n' | "$NEW" 2>/dev/null | grep '^bench')"
  echo "[q147] F15 id: $(printf 'uci\nquit\n' | "$NEW" 2>/dev/null | grep '^id name')"
  hard() { echo "$(readelf -h "$1" | sed -n 's/.*Type: *\([A-Z]*\).*/\1/p') · endbr64 $(objdump -d --no-show-raw-insn "$1" | grep -c endbr64) · stack_chk $(objdump -T "$1" | grep -c stack_chk) · fortify _chk imports $(objdump -T "$1" | grep -cE '__\w+_chk')"; }
  echo "[q147] hardening, production: $(hard "$PROD")"
  echo "[q147] hardening, F15:        $(hard "$NEW")"
  echo "$d" | grep -q "solved 252/300 nodes 36960934 " || { echo "[q147] ☠️ F15 default bench differs — part 1 reads skipped"; F15=0; }
fi

# ── wait for the venue ─────────────────────────────────────────────────────────────────────────────────────────────
echo "[q147] $(date) waiting for the venue ($VLOG)"
until grep -qE '^venue:|Traceback|Error' "$VLOG" 2>/dev/null; do
  [ -e "$STOP" ] && { echo "[q147] $(date) stop file present — QUEUE 147 DONE"; exit 0; }
  sleep 30
done
while pgrep -f _r50_venue.py >/dev/null; do sleep 10; done
grep -E '^venue:|fifty-move draws read|Traceback|Error' "$VLOG" | sed 's/^/[q147]   /'
[ -s "$V.csv" ] && [ -s "$V.pgn" ] || { echo "[q147] ☠️ no venue files — part 2 skipped"; NOV=1; }

# ── Part 1b: F15 reads (box otherwise idle) ────────────────────────────────────────────────────────────────────────────
if [ "${F15:-1}" = 1 ]; then
  echo "[q147] $(date) F15 build identity: production vs F15, fixed 30k nodes, 40 starts per set, one process per side"
  bash "$R" pyrun diagnostics/_fix_firecount.py BIN="$REF" BIN_ON="$NEW" ON= N=40 NODES=30000 PLIES=200 SIDES=2 JOBS=2 2>&1 \
    | grep -E '===|diverged|CRASH' | sed 's/^/[q147]   /'
  echo "[q147] $(date) F15 pinned bench, production vs F15, alternating x6"
  for r in 1 2 3 4 5 6; do
    echo "[q147]   prod $(printf 'bench\nquit\n' | taskset -c 5 "$REF" 2>/dev/null | grep '^bench')"
    echo "[q147]   f15  $(printf 'bench\nquit\n' | taskset -c 5 "$NEW" 2>/dev/null | grep '^bench')"
  done
fi
[ "${NOV:-0}" = 1 ] && { echo "[q147] $(date) QUEUE 147 DONE"; exit 0; }

# ── Part 2: FIX_TT_R50 on the decided-by-the-rule venue ────────────────────────────────────────────────────────────────
N=$(( $(wc -l < "$V.csv") - 1 ))
[ $N -gt 400 ] && N=400
tail -n +2 "$V.csv" | head -n $N | cut -d, -f2 > "$V.fens"
echo "[q147] $(date) part 2: $N positions; liveness (fixed 30k nodes, 40 starts)"
fc=$(bash "$R" pyrun diagnostics/_fix_firecount.py BIN="$PROD" N=40 NODES=30000 PLIES=120 SETS=r50c:$V.fens ON=FIX_TT_R50=1 JOBS=2 2>&1)
echo "$fc" | grep -E 'diverged|CRASH' | sed 's/^/[q147]   /'
div=$(echo "$fc" | sed -n 's/.*diverged: \([0-9]*\) \/.*/\1/p' | head -1)
[ -n "$div" ] && [ "$div" -gt 0 ] || { echo "[q147] ☠️ FIX_TT_R50 changed no game on the venue — match skipped. QUEUE 147 DONE"; exit 0; }
# chunks of 100 positions as separate opening files
rm -f ${V}_c[0-9]*.pgn
awk -v base="$V" '{ c = int((NR - 1) / 100) + 1
  printf "[Event \"r50_crit %d\"]\n[SetUp \"1\"]\n[FEN \"%s\"]\n\n*\n\n", NR - 1, $0 > (base "_c" c ".pgn") }' "$V.fens"
NC=$(( (N + 99) / 100 ))
rm -f /tmp/q147_pre.engine.log
timeout 180 "$FC" -engine cmd="$PROD" name=arm option.OwnBook=false option.Knobs=FIX_TT_R50=1 st=0.05 \
  -engine cmd="$PROD" name=ctl option.OwnBook=false st=0.05 \
  -openings file=${V}_c1.pgn format=pgn order=sequential -rounds 1 -games 2 -concurrency 1 \
  -log file=/tmp/q147_pre.engine.log level=info engine=true > /tmp/q147_pre.log 2>&1
W=$(grep -c 'have option' /tmp/q147_pre.log); K=$(grep -c 'knobs FIX_TT_R50=1' /tmp/q147_pre.engine.log)
C=$(grep -cE 'position fen .* 78 [0-9]+( moves|$)' /tmp/q147_pre.engine.log)
echo "[q147] preflight: $(grep -E '^Games:' /tmp/q147_pre.log | tail -1) · warnings $W · arm knob echoes $K · position commands with clock 78 $C"
[ "$W" = "0" ] && [ "$K" -ge 1 ] && [ "$C" -ge 1 ] || { echo "[q147] ☠️ PREFLIGHT FAILED — match skipped. QUEUE 147 DONE"; exit 0; }
hold() {   # $1 tag, $2 seconds needed
  if [ -e "$PAUSE" ]; then
    echo "[q147] $(date) paused before $1 (remove $PAUSE to resume)"
    while [ -e "$PAUSE" ] && [ ! -e "$STOP" ]; do sleep 30; done
    echo "[q147] $(date) resumed"
  fi
  [ -e "$STOP" ] && { echo "[q147] $(date) skip $1: stop file present"; return 1; }
  local left=$(( DEADLINE - $(date +%s) ))
  [ $left -ge $2 ] || { echo "[q147] $(date) skip $1: only ${left}s before the deadline (needs $2)"; return 1; }
}
COMMON="-repeat -concurrency 3 -recover"
rm -f $O/r50A_c*.pgn $O/r50B_c*.pgn $O/r50A_c*.log $O/r50B_c*.log
for c in $(seq 1 $NC); do
  hold "chunk $c" 3600 || break
  n=$(grep -c '^\[FEN' ${V}_c$c.pgn)
  nice -n 5 "$FC" -engine cmd="$PROD" name=arm option.OwnBook=false option.Knobs=FIX_TT_R50=1 tc=10+0.1 timemargin=250 \
    -engine cmd="$PROD" name=ctl option.OwnBook=false tc=10+0.1 timemargin=250 \
    -openings file=${V}_c$c.pgn format=pgn order=sequential $COMMON -rounds $n -games 2 \
    -pgnout file=$O/r50A_c$c.pgn notation=san nodes=true append=false > $O/r50A_c$c.log 2>&1
  echo "[q147] $(date) A chunk $c ($n positions): $(grep -E '^Elo:' $O/r50A_c$c.log | tail -1)  disconnects/stalls: $(grep -ciE 'disconnect|stall|not responsive' $O/r50A_c$c.log)  forfeits: $(grep -c 'loses on time' $O/r50A_c$c.pgn)"
  nice -n 5 "$FC" -engine cmd="$PROD" name=ref1 option.OwnBook=false tc=10+0.1 timemargin=250 \
    -engine cmd="$PROD" name=ref2 option.OwnBook=false tc=10+0.1 timemargin=250 \
    -openings file=${V}_c$c.pgn format=pgn order=sequential $COMMON -rounds $n -games 2 \
    -pgnout file=$O/r50B_c$c.pgn notation=san nodes=true append=false > $O/r50B_c$c.log 2>&1
  echo "[q147] $(date) B chunk $c: $(grep -E '^Games:' $O/r50B_c$c.log | tail -1)  disconnects/stalls: $(grep -ciE 'disconnect|stall|not responsive' $O/r50B_c$c.log)  forfeits: $(grep -c 'loses on time' $O/r50B_c$c.pgn)"
  bash "$R" pyrun diagnostics/_r50_split.py VENUE=$V.csv A="$O/r50A_c*.pgn" B="$O/r50B_c*.pgn" 2>&1 | sed 's/^/[q147]     /'
done
echo "[q147] $(date) final split (arm = FIX_TT_R50=1):"
bash "$R" pyrun diagnostics/_r50_split.py VENUE=$V.csv A="$O/r50A_c*.pgn" B="$O/r50B_c*.pgn" 2>&1 | sed 's/^/[q147]   /'
echo "[q147] head-to-head A pooled:"
timeout 600 bash "$R" pyrun selfplay/fastchess_report.py $(ls $O/r50A_c*.pgn) --p1 arm 2>&1 | sed -n '/vs the field/,/terminations/p' | sed 's/^/[q147]   /'
echo "[q147] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | sed 's/^/[q147]   /'
echo "[q147] $(date) QUEUE 147 DONE"
