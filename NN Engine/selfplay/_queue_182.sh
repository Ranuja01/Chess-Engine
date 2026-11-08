#!/bin/bash
# 2026-11-08 afternoon, queue #182 (owner 11-08: "it's not necessarily eval that's the issue ... search may not be correctly keeping
# everything that's needed"): is our passed-pawn weakness (#180) in the eval or in the search? Our search with SF11's static eval
# substituted for ours (ENABLE_ORACLE_EVAL, a persistent UCI pipe; memory sf-oracle-eval-substitution-...) against our own, at the
# same fixed depth, so search, ordering and pruning are held fixed and only the eval changes. Fixed depth, so safe on a weekend.
#  Binary build_uci/chess_engine_uci_orc (the sources of lazy2 plus the oracle memo cleared at ucinewgame; default bench 246 /
#  36,509,830; lazy off). Depths 8 and 10. Suites (5,624): 5 passed-pawn x 300, 18 critical themes x 200, WAC 300, ksattack 224.
#  The oracle runs are split by suite group over cores 3 / 5 / 7; our own runs follow on core 3.
# Registered before it runs:
#   EVAL-BORNE (at least in part) if SF11's eval gains more on the passer group than on the themes, per position, at both depths.
#   SEARCH-BORNE if the passer gain is no larger than the theme gain at both depths. Mixed otherwise.
#   Also reported: the oracle's passer gain as a share of #180's passer gap to SF11 (118 positions at 400k nodes; different regimes,
#   so a scale only).
#   Prediction: the oracle gains a little more per position on passers than on the themes, and closes under half of the gap.
# Stop: touch /mnt/e/chess_data/q182.stop (checked between runs). Never edit this file while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
SF11="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/stockfish_11_linux/stockfish-11-linux/Linux/stockfish_20011801_x64_bmi2"
R="$ND/selfplay/overnight_runner.sh"
BIN="$ND/build_uci/chess_engine_uci_orc"
P=/mnt/e/chess_data/puzzles/passer
TH=/mnt/e/chess_data/puzzles/themes
L=/mnt/e/chess_data/lazy
OD=$L/oracle182
STOP=/mnt/e/chess_data/q182.stop
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"
show() { sed "s/^/[q182]   /"; }
busy() { pgrep -f "fastchess-linux-x86-64/fastchess|vs_sf.py|_depth_residual_pass|_build_regret_set|_retune_fit|_ks_footprint_regret|sts_test|tactical_test|_theme_movetime" >/dev/null; }
echo "[q182] $(date) waiting for a quiet box (stop: $STOP)"
while busy; do [ -e "$STOP" ] && { echo "[q182] stop file while waiting — QUEUE 182 DONE"; exit 0; }; sleep 60; done
START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[q182] $START start"
B0=$(printf 'bench\nquit\n' | "$BIN" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+')
B4=$(printf 'bench 4\nquit\n' | ENABLE_ORACLE_EVAL=1 ORACLE_ENGINE_PATH="$SF11" "$BIN" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+')
echo "[q182] bench default: $B0 · oracle at depth 4: $B4"
[ "$B0" = "solved 246/300 nodes 36509830" ] || { echo "[q182] ☠️ default bench is not gen 7's — QUEUE 182 ABORTED"; exit 1; }
[ "$B4" = "solved 126/300 nodes 935416" ] || { echo "[q182] ☠️ the oracle is not live (depth-4 bench differs from the 11-08 check) — QUEUE 182 ABORTED"; exit 1; }

PS=$(ls $P/*.txt); THS=($(ls $TH/*.txt))
G1="$PS"; G2="${THS[*]:0:9} $L/wac.txt"; G3="${THS[*]:9:9} $L/ksattack.txt"
run() {  # core, arm, depth, suites...
  local core=$1 v=$2 d=$3; shift 3
  [ -e "$STOP" ] && return
  mkdir -p $OD/d$d
  if [ $v = orc ]; then
    ENABLE_ORACLE_EVAL=1 ORACLE_ENGINE_PATH="$SF11" EACH=$OD/d$d/$v bash "$R" pyrun diagnostics/_theme_movetime.py "$BIN" $core d$d "LAZY_V2=0" "$@" >> $OD/d$d/$v.txt 2>&1
  else
    EACH=$OD/d$d/$v bash "$R" pyrun diagnostics/_theme_movetime.py "$BIN" $core d$d "LAZY_V2=0" "$@" >> $OD/d$d/$v.txt 2>&1
  fi
  echo "[q182] $(date) $v depth $d, $# suites done"
}
rm -rf "$OD"; mkdir -p "$OD"
{ run 3 orc 10 $G1; run 3 orc 8 $G1; run 3 ours 10 $G1 $G2 $G3; run 3 ours 8 $G1 $G2 $G3; } &
{ run 5 orc 10 $G2; run 5 orc 8 $G2; } &
{ run 7 orc 10 $G3; run 7 orc 8 $G3; } &
wait
[ -e "$STOP" ] && { echo "[q182] stop file during the runs — QUEUE 182 DONE"; exit 0; }

# ── tables ─────────────────────────────────────────────────────────────────────────────────────────────────────────
SETS="$G1 $G2 $G3"
grp() { case $1 in promotion|advancedPawn_mg|pawnEndgame|rookEndgame_passer|minorQueenEndgame_passer) echo passer ;;
  wac) echo wac ;; ksattack) echo ksattack ;; *) echo themes ;; esac; }
for d in 10 8; do
  echo "[q182] depth $d, solved per suite (ours / ours with SF11's eval):"
  { for f in $SETS; do
      s=$(basename $f .txt)
      printf '%-26s %6s %6s\n' "$s" "$(awk -v s=$s '$1 == s { print $2 }' $OD/d$d/ours.txt 2>/dev/null)" "$(awk -v s=$s '$1 == s { print $2 }' $OD/d$d/orc.txt 2>/dev/null)"
    done
    for g in passer themes wac ksattack; do
      a=0; b=0; n=0
      for f in $SETS; do
        s=$(basename $f .txt); [ "$(grp $s)" = $g ] || continue
        a=$((a + $(awk -v s=$s '$1 == s { v = $2 } END { print v + 0 }' $OD/d$d/ours.txt 2>/dev/null || echo 0)))
        b=$((b + $(awk -v s=$s '$1 == s { v = $2 } END { print v + 0 }' $OD/d$d/orc.txt 2>/dev/null || echo 0)))
        n=$((n + $(grep -c '|' $f)))
      done
      awk -v g=$g -v a=$a -v b=$b -v n=$n 'BEGIN { printf "GROUP %-20s %6d %6d  of %d  gain %+d = %+.2f points\n", g, a, b, n, b - a, (n > 0 ? 100 * (b - a) / n : 0) }'
    done
  } | show
done
pair() {  # depth, group: positions only ours solves, only the oracle solves, z
  local d=$1 g=$2 a=0 b=0 f s x y
  for f in $SETS; do
    s=$(basename $f .txt); [ "$(grp $s)" = $g ] || continue
    read x y < <(paste -d' ' "$OD/d$d/ours/$s.txt" "$OD/d$d/orc/$s.txt" | awk '{ if ($2 == 1 && $5 == 0) a++; if ($2 == 0 && $5 == 1) b++ } END { print a + 0, b + 0 }')
    a=$((a + x)); b=$((b + y))
  done
  awk -v d=$d -v g=$g -v a=$a -v b=$b 'BEGIN { printf "  depth %-3s %-8s only ours %4d · only with SF11 eval %4d · z %+.2f\n", d, g, a, b, (a + b > 0 ? (b - a) / sqrt(a + b) : 0) }'
}
echo "[q182] paired (z > 0: SF11's eval solves more):"
for d in 10 8; do for g in passer themes wac ksattack; do pair $d $g; done; done | show
echo "[q182] $(date) QUEUE 182 DONE"
