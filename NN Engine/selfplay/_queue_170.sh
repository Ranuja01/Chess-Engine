#!/bin/bash
# 2026-11-06, queue #170: the control #168 turned out to need, added before any of its games were read. #168 varies the own-play rows
# with the stress weight fixed, so a quarter-size fit also has 4 times the stress share of S and a half-size fit 2 times. Its fits show
# this matters: all five stress holds get steadily worse as own-play rows grow (static VAR -2.54 / -1.42 / -0.58, VARG -2.69 / -2.23 /
# -1.82, ODDS -2.44 / -2.33 / -2.13, KP +0.38 / +1.30 / +1.78, STRIP +0.05 / +0.96 / +1.41 at 25 / 50 / 100%), and S beat N by +23 +- 10 on a
# small stress share. So level(100%) - level(25%) mixes two effects, and a flat result could hide a real gain from rows.
#  1 two fits on all the rows at the share of the smaller fits: S x 2 (all stress slices at 1.0, the share of a half) and S x 4 (at 2.0, the
#    share of a quarter). Same recipe, same rows, no new data.
#  2 gates per arm, as in #166.
#  3 50k nodes vs gen 6.7, 4,000 g each, 8 interleaved chunks of 500, fresh seeds. About 4 h. Starts when #168's games end.
# Registered read, replacing #168's level(100%) - level(25%). Rows at equal stress share (bars as agreed with the owner):
#    S x 4 - quarters >= +12 and S x 2 - halves >= +6: the data seems to matter; refit with A2's rows next.
#    S x 4 - quarters under +6: more rows are a ghost.  In between: no dedicated time.
#  Stress share with all the rows: S, S x 2, S x 4. An arm that beats S by +2 sigma (about +14 on 4,000 g each) is a better candidate than
#  S at no data cost and goes to the timed check. Expected: within +-15 of S; the holds should improve and own-play val fall a little.
# Stop: touch /mnt/e/chess_data/q170.stop (checked between steps and chunks). Never edit this file while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
D=/mnt/e/chess_data/retune
BIN="$ND/build_uci/chess_engine_uci_g67pgo"
LOGD=/mnt/e/chess_data/q170
STOP=/mnt/e/chess_data/q170.stop
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"; mkdir -p "$LOGD"
show() { sed "s/^/[q170]   /"; }
stopped() { [ -e "$STOP" ] && { echo "[q170] $(date) stop file present before $1 — QUEUE 170 DONE"; exit 0; }; }
echo "[q170] $(date) waiting for #168's games to end (stop: $STOP)"
until grep -qE 'GAMES OVER|QUEUE 168 (DONE|ABORTED)' /mnt/e/chess_data/q168.log 2>/dev/null; do
  [ -e "$STOP" ] && { echo "[q170] stop file present — QUEUE 170 DONE (nothing run)"; exit 0; }
  sleep 60
done
echo "[q170] $(date) start"

# ── 1 the fits ───────────────────────────────────────────────────────────────────────────────────────────────────────
TIME=""; [ -x /usr/bin/time ] && TIME="/usr/bin/time -v"
COMMON="EXPERIMENT=hybrid MODES=hybrid VAL=1 WIN_AWARE=1 LAMBDAS=0.03 CORPUS=corpus_v3.csv FEAT=feat_g67 EXTRA_SCOPES=1 TRAIN_N=500000 VAL_N=120000 DEPTH_SLICES=A,GAUNT,EGX,CUR,VARG,ODDS,STRIP"
fit() { [ -s $D/fit_hybrid_$1.npz ] || $TIME bash "$R" pyrun diagnostics/_retune_fit.py $COMMON STRESS_W=$2 SLICE_W=VARG:$2,ODDS:$2,STRIP:$2 TAG=_$1 > $LOGD/fit_$1.log 2> $LOGD/fit_$1.err; }
echo "[q170] $(date) fits, side by side"
fit v3x2 1 & fit v3x4 2 & wait
for t in v3x2 v3x4; do
  echo "[q170] $(date) fit $t: $( [ -s $D/fit_hybrid_$t.npz ] && echo ok || echo '☠️ FAILED') · peak RSS $(grep -E 'Maximum resident' $LOGD/fit_$t.err | grep -oE '[0-9]+$') kB"
  grep -E "weight by slice|hybrid  λ|static all|static (mg|mgl|egl|eg) |depth all|depth (mg|mgl|egl|eg) |val CUR|val GAUNT|val EGX|hold |rror|Trace" $LOGD/fit_$t.log | show
done
stopped "the gates"

# ── 2 gates ──────────────────────────────────────────────────────────────────────────────────────────────────────────
PASS=""
for t in v3x2 v3x4; do
  [ -s $D/fit_hybrid_$t.npz ] || continue
  rm -rf $D/arm_$t
  env V2_PRESET=shipped bash "$R" pyrun diagnostics/_retune_export.py MODE=tables NPZ=fit_hybrid_$t.npz ARM_I=0 OUT=$D/arm_$t > $LOGD/tables_$t.log 2>&1
  grep -E "values:|⚠️|rror" $LOGD/tables_$t.log | sed "s/^/[q170] $t /"
  [ -s $D/arm_$t/env.txt ] || { echo "[q170] $t ✗ no tables — dropped"; continue; }
  ENVL=$(cat $D/arm_$t/env.txt)
  env V2_PRESET=shipped $ENVL bash "$R" pyrun diagnostics/_retune_export.py MODE=verify NPZ=fit_hybrid_$t.npz ARM_I=0 ARMDIR=$t N=3000 \
    FEAT=feat_g67 WIN_AWARE=1 > $LOGD/verify_$t.log 2>&1
  grep -E "skipped|REALISED|rror" $LOGD/verify_$t.log | sed "s/^/[q170] $t /"
  set -- $(grep REALISED $LOGD/verify_$t.log | grep -oE '\(-?[0-9.]+%\)' | tr -d '()%')
  awk -v a="$1" -v b="$2" 'BEGIN{exit !(a < 0 && b/a >= 0.7)}' || { echo "[q170] $t ✗ closure gate (predicted $1% / engine $2%) — dropped"; continue; }
  SYM=$(env V2_PRESET=shipped $ENVL bash "$R" pyrun diagnostics/_eval_symmetry.py N=4000 2>&1 | grep -E "violations")
  echo "[q170] $t symmetry: $(echo "$SYM" | tr '\n' ' ')"
  [ "$(echo "$SYM" | grep -c 'violations 0 ')" = "2" ] || { echo "[q170] $t ✗ symmetry — dropped"; continue; }
  KN=$(tr -s ' \n' ',' < $D/arm_$t/env.txt | sed 's/,$//')
  U=$(printf 'setoption name Knobs value %s\nbench\nquit\n' "$KN" | "$BIN" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+')
  W=$(env V2_PRESET=shipped $ENVL bash "$R" wac fp_$t 2>&1 | grep -E "^SOLVED|^NODES" | tr '\n' ' ' | grep -oE 'Solved [0-9]+/300 NODES: [0-9]+')
  un=$(echo "$U" | grep -oE '[0-9]+$'); wn=$(echo "$W" | grep -oE '[0-9]+$')
  echo "[q170] $t UCI bench: $U · Cython WAC: $W"
  [ -n "$un" ] && [ "$un" = "$wn" ] && [ "$un" != 36859850 ] || { echo "[q170] $t ✗ UCI arm ≠ Cython arm (or knobs dead) — dropped"; continue; }
  echo "$KN" > $LOGD/knobs_$t.txt
  PASS="$PASS $t"
done
echo "[q170] $(date) arms for the games:${PASS:- none}"
[ -n "$PASS" ] || { echo "[q170] QUEUE 170 DONE"; exit 0; }

# ── 3 games ──────────────────────────────────────────────────────────────────────────────────────────────────────────
GC="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
n=0
rm -f $O/lcg_v3x*_c*.pgn $O/lcg_v3x*_c*.log
for c in 1 2 3 4 5 6 7 8; do
  for t in $PASS; do
    n=$(( n + 1 ))
    stopped "$t chunk $c"
    T=lcg_$t
    nice -n 5 "$FC" -engine "cmd=$BIN" name=arm option.OwnBook=false "option.Knobs=$(cat $LOGD/knobs_$t.txt)" nodes=50000 \
      -engine "cmd=$BIN" name=ctl option.OwnBook=false nodes=50000 $GC -srand $(( 170000 + n )) -rounds 250 -games 2 \
      -pgnout file=$O/${T}_c$c.pgn notation=san nodes=true append=false > $O/${T}_c$c.log 2>&1
    echo "[q170] $(date) $T chunk $c: $(grep -E '^Elo:' $O/${T}_c$c.log | tail -1) · warnings $(grep -c 'have option' $O/${T}_c$c.log) · stalls $(grep -ciE 'disconnect|stall|not responsive' $O/${T}_c$c.log)"
  done
done
for t in $PASS; do
  echo "[q170] arm $t pooled:"
  timeout 900 bash "$R" pyrun selfplay/fastchess_report.py $O/lcg_${t}_c*.pgn --p1 arm 2>&1 | sed -n '/vs the field/,/pentanomial/p' | show
done
echo "[q170] $(date) QUEUE 170 DONE"
