#!/bin/bash
# 2026-11-05 night, queue #166: THE RETUNE FIT, overnight (owner 11-05: fine to run in idle time, it needs no decisions from
# them). Waits for #165's timed legs to end (its "A2 d10:" line), then runs beside #165's A2 depth pass:
#  1 two hybrid fits on corpus_v3 + feat_g67 (gen 6.7's export), the gen-3 recipe (VAL=1 WIN_AWARE=1 λ 0.03, PST means pinned,
#    piece-value cells and the KD mg legs frozen, mobility out of the depth design); depth target = gen 6.7's d10 pass:
#      N = normal play: own-play static + depth (A, GAUNT, EGX, CUR), KP / VAR stress static at 0.5 as in gens 2/3, the new
#          stress slices (VARG / ODDS / STRIP) left out. Learning curve: 25 / 50 / 100% of the own-play training games.
#      S = N + the new stress slices at 0.5 in both targets (100% only).
#    Both scored on the same held-out own-play games (static and depth, per stage and per slice) and on the stress holds.
#  2 per arm (N at 100%, S): tables from the gen 6.7 engine → closure (the engine realises ≥ 70% of the predicted val gain) →
#    colour + file symmetry 0 violations → UCI bench under the arm's knobs == Cython WAC under the same env, ≠ gen 6.7.
#  3 self-play SCREEN, node-limited (safe any hour): each passing arm vs gen 6.7 at 50k nodes, 2,000 games in 4 chunks of 500
#    (N and S interleaved), seeds 1661 / 1662, UHO openings. Not a ship check (that is timed, plus SF18 / Mediocre).
# Registered predictions: N self-play +0…+20 (round 3, the last refit, was flat; this one has the gen 4-6.7 terms and search
# under it and about twice the depth rows) · S within ±10 of N on normal-play val and better on the VARG / ODDS / STRIP holds
# · learning curve: if 50% → 100% still moves "depth all" val by ≥ 10% of the 100% gain, more depth rows help.
# Stop: touch /mnt/e/chess_data/q166.stop (checked between steps and chunks). Never edit this file while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
D=/mnt/e/chess_data/retune
BIN="$ND/build_uci/chess_engine_uci_g67pgo"
LOGD=/mnt/e/chess_data/q166
STOP=/mnt/e/chess_data/q166.stop
PAR=${PAR:-1}          # 1: the two fits run side by side
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"; mkdir -p "$LOGD"
show() { sed "s/^/[q166]   /"; }
stopped() { [ -e "$STOP" ] && { echo "[q166] $(date) stop file present before $1 — QUEUE 166 DONE"; exit 0; }; }
echo "[q166] $(date) waiting for queue #165's timed legs to end (stop: $STOP)"
until grep -qE 'A2 d10: [0-9]+ jobs|QUEUE 165 (DONE|ABORTED)' /mnt/e/chess_data/q165.log 2>/dev/null; do
  [ -e "$STOP" ] && { echo "[q166] stop file present — QUEUE 166 DONE (nothing run)"; exit 0; }
  sleep 60
done
START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[q166] $START start"
grep -q 'CORPUS EXPORT DONE' /mnt/e/chess_data/q165/export.log 2>/dev/null && [ -s $D/corpus_v3.csv ] \
  || { echo "[q166] ☠️ corpus_v3 or its gen 6.7 export is missing — QUEUE 166 ABORTED"; exit 1; }
cmp -s "$BIN" "$ND/build_uci/chess_engine_uci" && echo "[q166] production == chess_engine_uci_g67pgo" \
  || echo "[q166] ⚠️ production is not the g67pgo binary (the screen still plays g67pgo on both sides)"
FP=$(env V2_PRESET=shipped bash "$R" wac fp_g67_q166 2>&1 | grep -E "^SOLVED|^NODES" | tr '\n' ' ')
echo "$FP" | grep -q "Solved 251/300 NODES: 36859850 " || { echo "[q166] ☠️ Cython is not gen 6.7 ($FP) — QUEUE 166 ABORTED"; exit 1; }
UB=$(printf 'bench\nquit\n' | "$BIN" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+')
[ "$UB" = "solved 251/300 nodes 36859850" ] || { echo "[q166] ☠️ g67pgo bench is '$UB' — QUEUE 166 ABORTED"; exit 1; }
TIME=""; [ -x /usr/bin/time ] && TIME="/usr/bin/time -v"

# ── 1 the fits ────────────────────────────────────────────────────────────────────────────────────────────────────────
COMMON="EXPERIMENT=hybrid MODES=hybrid VAL=1 WIN_AWARE=1 LAMBDAS=0.03 CORPUS=corpus_v3.csv FEAT=feat_g67 EXTRA_SCOPES=1 TRAIN_N=500000 VAL_N=120000"
fitN() { [ -s $D/fit_hybrid_v3n.npz ] || $TIME bash "$R" pyrun diagnostics/_retune_fit.py $COMMON DEPTH_SLICES=A,GAUNT,EGX,CUR \
           SLICE_W=VARG:0,ODDS:0,STRIP:0 LC=0.25,0.5,1 TAG=_v3n > $LOGD/fit_v3n.log 2> $LOGD/fit_v3n.err; }
fitS() { [ -s $D/fit_hybrid_v3s.npz ] || $TIME bash "$R" pyrun diagnostics/_retune_fit.py $COMMON DEPTH_SLICES=A,GAUNT,EGX,CUR,VARG,ODDS,STRIP \
           SLICE_W=VARG:0.5,ODDS:0.5,STRIP:0.5 TAG=_v3s > $LOGD/fit_v3s.log 2> $LOGD/fit_v3s.err; }
echo "[q166] $(date) fits (side by side: $PAR)"
if [ "$PAR" = 1 ]; then fitN & fitS & wait; else fitN; fitS; fi
for t in v3n v3s; do
  echo "[q166] $(date) fit $t: $( [ -s $D/fit_hybrid_$t.npz ] && echo ok || echo '☠️ FAILED') · peak RSS $(grep -E 'Maximum resident' $LOGD/fit_$t.err | grep -oE '[0-9]+$') kB"
  [ -s $D/fit_hybrid_$t.npz ] || grep -vE '^\s+(Command|User|System|Percent|Elapsed|Average|Maximum|Major|Minor|Voluntary|Involuntary|Swaps|File|Socket|Signals|Page|Exit)' $LOGD/fit_$t.err | tail -6 | show
  grep -E "rows:|design:|weight by slice|LC [0-9.]+:|hybrid  λ|^  scope|static all|static (mg|mgl|egl|eg) |depth all|depth (mg|mgl|egl|eg) |val CUR|val GAUNT|val EGX|hold |PSTeg|KAUF|rror|Trace" $LOGD/fit_$t.log | show
done
stopped "the exports"

# ── 2 export → closure → symmetry → UCI == Cython ─────────────────────────────────────────────────────────────────────
PASS=""
for a in v3n:2 v3s:0; do
  t=${a%%:*}; i=${a#*:}
  [ -s $D/fit_hybrid_$t.npz ] || continue
  rm -rf $D/arm_$t
  env V2_PRESET=shipped bash "$R" pyrun diagnostics/_retune_export.py MODE=tables NPZ=fit_hybrid_$t.npz ARM_I=$i OUT=$D/arm_$t > $LOGD/tables_$t.log 2>&1
  grep -E "cells changed|values:|⚠️|EXPORT|rror" $LOGD/tables_$t.log | show
  [ -s $D/arm_$t/env.txt ] || { echo "[q166] $t ✗ no tables — dropped"; continue; }
  ENVL=$(cat $D/arm_$t/env.txt)
  env V2_PRESET=shipped $ENVL bash "$R" pyrun diagnostics/_retune_export.py MODE=verify NPZ=fit_hybrid_$t.npz ARM_I=$i ARMDIR=$t N=3000 \
    FEAT=feat_g67 WIN_AWARE=1 > $LOGD/verify_$t.log 2>&1
  grep -E "skipped|REALISED|^CLOSURE|winnability|signed|rror" $LOGD/verify_$t.log | sed "s/^/[q166] $t /"
  set -- $(grep REALISED $LOGD/verify_$t.log | grep -oE '\(-?[0-9.]+%\)' | tr -d '()%')
  awk -v a="$1" -v b="$2" 'BEGIN{exit !(a < 0 && b/a >= 0.7)}' || { echo "[q166] $t ✗ closure gate (predicted $1% / engine $2%) — dropped"; continue; }
  SYM=$(env V2_PRESET=shipped $ENVL bash "$R" pyrun diagnostics/_eval_symmetry.py N=4000 2>&1 | grep -E "violations")
  echo "[q166] $t symmetry: $(echo "$SYM" | tr '\n' ' ')"
  [ "$(echo "$SYM" | grep -c 'violations 0 ')" = "2" ] || { echo "[q166] $t ✗ symmetry — dropped"; continue; }
  KN=$(tr -s ' \n' ',' < $D/arm_$t/env.txt | sed 's/,$//')
  U=$(printf 'setoption name Knobs value %s\nbench\nquit\n' "$KN" | "$BIN" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+')
  W=$(env V2_PRESET=shipped $ENVL bash "$R" wac fp_$t 2>&1 | grep -E "^SOLVED|^NODES" | tr '\n' ' ' | grep -oE 'Solved [0-9]+/300 NODES: [0-9]+')
  un=$(echo "$U" | grep -oE '[0-9]+$'); wn=$(echo "$W" | grep -oE '[0-9]+$')
  echo "[q166] $t UCI bench under the knobs: $U · Cython WAC: $W"
  [ -n "$un" ] && [ "$un" = "$wn" ] && [ "$un" != 36859850 ] || { echo "[q166] $t ✗ UCI arm ≠ Cython arm (or knobs dead) — dropped"; continue; }
  echo "$KN" > $LOGD/knobs_$t.txt
  PASS="$PASS $t"
done
echo "[q166] $(date) arms for the screen:${PASS:- none}"
[ -n "$PASS" ] || { echo "[q166] QUEUE 166 DONE"; exit 0; }

# ── 3 self-play screen at 50k nodes ──────────────────────────────────────────────────────────────────────────────────
GC="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
for c in 1 2 3 4; do
  for t in $PASS; do
    stopped "$t chunk $c"
    seed=1661; [ $t = v3s ] && seed=1662
    T=rt_${t}_n50k
    nice -n 5 "$FC" -engine "cmd=$BIN" name=arm option.OwnBook=false "option.Knobs=$(cat $LOGD/knobs_$t.txt)" nodes=50000 \
      -engine "cmd=$BIN" name=ctl option.OwnBook=false nodes=50000 $GC -srand $(( seed * 10 + c )) -rounds 250 -games 2 \
      -pgnout file=$O/${T}_c$c.pgn notation=san nodes=true append=false > $O/${T}_c$c.log 2>&1
    echo "[q166] $(date) $T chunk $c: $(grep -E '^Elo:' $O/${T}_c$c.log | tail -1) · warnings $(grep -c 'have option' $O/${T}_c$c.log) · stalls $(grep -ciE 'disconnect|stall|not responsive' $O/${T}_c$c.log)"
  done
done
for t in $PASS; do
  echo "[q166] rt_${t}_n50k pooled:"
  timeout 600 bash "$R" pyrun selfplay/fastchess_report.py $O/rt_${t}_n50k_c[1-4].pgn --p1 arm 2>&1 | sed -n '/vs the field/,/pentanomial/p' | show
done
echo "[q166] new WSL crash dumps since $START:"
find /mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes -newermt "$START" -name '*.dmp' 2>/dev/null | show
echo "[q166] $(date) QUEUE 166 DONE"
