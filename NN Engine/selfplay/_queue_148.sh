#!/bin/bash
# 2026-11-04 midday, queue #148 (owner 11-04: "we can go for it"; follows #147).
# Part A, two jobs side by side (no timed games yet):
#   - a wider 50-move venue: diagnostics/_r50_venue.py SOURCE=all — the first position at halfmove clock 78 of every past
#     game that reached it, Stockfish 18 at the real clock >= 2 pawns for one side (#147's fifty-move-draws-only venue kept
#     98 positions, too few to resolve the registered effect);
#   - F15 bisection: #147's hardening-off PGO build matched every v2 fingerprint but read v1 (EvalArm 0) 35,310,775 vs
#     35,310,778. Plain (non-PGO) builds — no extra flags, all F15 flags, and each flag alone — with v2 and v1 bench nodes,
#     to name the flag (or show that only the PGO build moves, i.e. fast-math codegen, not the flags as such).
# Part B: #147's part 2 on the wider venue (run A = FIX_TT_R50=1 vs production, run B = production vs itself, interleaved per
#   chunk of 100 positions, 10+0.1, no adjudication), split by diagnostics/_r50_split.py. Registered as in #147: conversion
#   delta > 0 (expected 0 ... +10 pp), defence delta <= 0; adopt if either improves by >= 2 sigma and the other is not worse
#   by more than 1 sigma.
# Never edit this file while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
PROD="$ND/build_uci/chess_engine_uci"
V=$O/r50_crit2
VLOG=/mnt/e/chess_data/r50_venue2.log
STOP=/mnt/e/chess_data/q148.stop
PAUSE=/mnt/e/chess_data/q148.pause
DEADLINE=$(date -d '2026-11-04 20:30' +%s)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"
START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[q148] $START start"
pgrep -f "fastchess-linux-x86-64/fastchess|vs_sf.py|_queue_147.sh" >/dev/null && { echo "[q148] a match or #147 is running — QUEUE 148 ABORTED"; exit 1; }

# ── Part A1: the wider venue, in the background ───────────────────────────────────────────────────────────────────────
nice -n 5 bash "$R" pyrun diagnostics/_r50_venue.py OUT=$V SOURCE=all CLOCK=78 WIN_CP=200 NODES=500000 JOBS=2 MAXPOS=6000 \
  > "$VLOG" 2>&1 &
VPID=$!
echo "[q148] $(date) venue labelling started (pid $VPID, log $VLOG)"

# ── Part A2: F15 bisection, plain builds (node counts only, so the labelling beside them does not matter) ──────────────
nodes() { printf "$2bench\nquit\n" | "$1" 2>/dev/null | sed -n 's/^bench: solved \([0-9]*\)\/300 nodes \([0-9]*\) .*/\1 \/ \2/p'; }
variant() {   # $1 tag, $2 extra flags
  OUT=chess_engine_uci_f15b_$1 EXTRA_FLAGS="$2" bash selfplay/build_uci.sh > /tmp/q148_build_$1.log 2>&1 \
    || { echo "[q148]   $1: BUILD FAILED"; return; }
  local b="$ND/build_uci/chess_engine_uci_f15b_$1"
  echo "[q148]   $(printf '%-14s' $1) v2 $(nodes "$b" "")   v1 $(nodes "$b" "setoption name EvalArm value 0\n")   [$2]"
}
echo "[q148] $(date) F15 bisection (want v2 252 / 36960934, v1 250 / 35310778):"
variant none ""
variant all "-fno-stack-protector -fcf-protection=none -fno-stack-clash-protection -U_FORTIFY_SOURCE -fno-pie -no-pie"
variant ssp "-fno-stack-protector"
variant cet "-fcf-protection=none"
variant clash "-fno-stack-clash-protection"
variant fortify "-U_FORTIFY_SOURCE"
variant nopie "-fno-pie -no-pie"
echo "[q148] $(date) bisection done"

# ── wait for the venue ─────────────────────────────────────────────────────────────────────────────────────────────────
wait $VPID
grep -E '^venue:|read:|sampled|Traceback|Error' "$VLOG" | sed 's/^/[q148]   /'
[ -s "$V.csv" ] && [ -s "$V.pgn" ] || { echo "[q148] ☠️ no venue files — QUEUE 148 DONE"; exit 0; }
[ -e "$STOP" ] && { echo "[q148] $(date) stop file present — QUEUE 148 DONE"; exit 0; }

# ── Part B: FIX_TT_R50 on the wider venue ──────────────────────────────────────────────────────────────────────────────
N=$(( $(wc -l < "$V.csv") - 1 ))
[ $N -gt 400 ] && N=400
tail -n +2 "$V.csv" | head -n $N | cut -d, -f2 > "$V.fens"
echo "[q148] $(date) part B: $N positions; liveness (fixed 30k nodes, 40 starts)"
fc=$(bash "$R" pyrun diagnostics/_fix_firecount.py BIN="$PROD" N=40 NODES=30000 PLIES=120 SETS=r50w:$V.fens ON=FIX_TT_R50=1 JOBS=2 2>&1)
echo "$fc" | grep -E 'diverged|CRASH' | sed 's/^/[q148]   /'
div=$(echo "$fc" | sed -n 's/.*diverged: \([0-9]*\) \/.*/\1/p' | head -1)
[ -n "$div" ] && [ "$div" -gt 0 ] || { echo "[q148] ☠️ FIX_TT_R50 changed no game on the venue — match skipped. QUEUE 148 DONE"; exit 0; }
rm -f ${V}_c[0-9]*.pgn
awk -v base="$V" '{ c = int((NR - 1) / 100) + 1
  printf "[Event \"r50_crit2 %d\"]\n[SetUp \"1\"]\n[FEN \"%s\"]\n\n*\n\n", NR - 1, $0 > (base "_c" c ".pgn") }' "$V.fens"
NC=$(( (N + 99) / 100 ))
rm -f /tmp/q148_pre.engine.log
timeout 180 "$FC" -engine cmd="$PROD" name=arm option.OwnBook=false option.Knobs=FIX_TT_R50=1 st=0.05 \
  -engine cmd="$PROD" name=ctl option.OwnBook=false st=0.05 \
  -openings file=${V}_c1.pgn format=pgn order=sequential -rounds 1 -games 2 -concurrency 1 \
  -log file=/tmp/q148_pre.engine.log level=info engine=true > /tmp/q148_pre.log 2>&1
W=$(grep -c 'have option' /tmp/q148_pre.log); K=$(grep -c 'knobs FIX_TT_R50=1' /tmp/q148_pre.engine.log)
C=$(grep -cE 'position fen .* 78 [0-9]+( moves|$)' /tmp/q148_pre.engine.log)
echo "[q148] preflight: $(grep -E '^Games:' /tmp/q148_pre.log | tail -1) · warnings $W · arm knob echoes $K · position commands with clock 78 $C"
[ "$W" = "0" ] && [ "$K" -ge 1 ] && [ "$C" -ge 1 ] || { echo "[q148] ☠️ PREFLIGHT FAILED — match skipped. QUEUE 148 DONE"; exit 0; }
hold() {   # $1 tag, $2 seconds needed
  if [ -e "$PAUSE" ]; then
    echo "[q148] $(date) paused before $1 (remove $PAUSE to resume)"
    while [ -e "$PAUSE" ] && [ ! -e "$STOP" ]; do sleep 30; done
    echo "[q148] $(date) resumed"
  fi
  [ -e "$STOP" ] && { echo "[q148] $(date) skip $1: stop file present"; return 1; }
  local left=$(( DEADLINE - $(date +%s) ))
  [ $left -ge $2 ] || { echo "[q148] $(date) skip $1: only ${left}s before the deadline (needs $2)"; return 1; }
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
  echo "[q148] $(date) A chunk $c ($n positions): $(grep -E '^Elo:' $O/r50A_c$c.log | tail -1)  disconnects/stalls: $(grep -ciE 'disconnect|stall|not responsive' $O/r50A_c$c.log)  forfeits: $(grep -c 'loses on time' $O/r50A_c$c.pgn)"
  nice -n 5 "$FC" -engine cmd="$PROD" name=ref1 option.OwnBook=false tc=10+0.1 timemargin=250 \
    -engine cmd="$PROD" name=ref2 option.OwnBook=false tc=10+0.1 timemargin=250 \
    -openings file=${V}_c$c.pgn format=pgn order=sequential $COMMON -rounds $n -games 2 \
    -pgnout file=$O/r50B_c$c.pgn notation=san nodes=true append=false > $O/r50B_c$c.log 2>&1
  echo "[q148] $(date) B chunk $c: $(grep -E '^Games:' $O/r50B_c$c.log | tail -1)  disconnects/stalls: $(grep -ciE 'disconnect|stall|not responsive' $O/r50B_c$c.log)  forfeits: $(grep -c 'loses on time' $O/r50B_c$c.pgn)"
  bash "$R" pyrun diagnostics/_r50_split.py VENUE=$V.csv A="$O/r50A_c*.pgn" B="$O/r50B_c*.pgn" 2>&1 | sed 's/^/[q148]     /'
done
echo "[q148] $(date) final split (arm = FIX_TT_R50=1):"
bash "$R" pyrun diagnostics/_r50_split.py VENUE=$V.csv A="$O/r50A_c*.pgn" B="$O/r50B_c*.pgn" 2>&1 | sed 's/^/[q148]   /'
echo "[q148] head-to-head A pooled:"
timeout 600 bash "$R" pyrun selfplay/fastchess_report.py $(ls $O/r50A_c*.pgn) --p1 arm 2>&1 | sed -n '/vs the field/,/terminations/p' | sed 's/^/[q148]   /'
echo "[q148] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | sed 's/^/[q148]   /'
echo "[q148] $(date) QUEUE 148 DONE"
