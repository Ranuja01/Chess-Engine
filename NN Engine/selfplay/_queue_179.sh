#!/bin/bash
# 2026-11-08 afternoon, queue #179 (owner 11-08 "ok" to the passed-pawn test): does the lazy eval guard for an advanced passer protect
# anything? The literature's most-cited lazy eval hazard is passed pawns; #178's pick (keep_g3) drops that guard. Fixed depth only
# (weekend afternoon, the owner may be on the box), so nothing here is timed.
#  1 five passed-pawn suites x 300 Lichess puzzles, seed 1108 (diagnostics/_build_theme_suites.py GROUP=passer) into $P.
#  2 depth 10 and depth 12, arms off / b1 / keep / keep_g3 / keep_m1000 (binary chess_engine_uci_lazy2, knobs via the Knobs option),
#    per-position results (diagnostics/_theme_movetime.py with EACH), three processes pinned to cores 3 / 5 / 7.
#  3 table of solved counts; paired comparisons on the positions only one arm solves, z = (a - b) / sqrt(a + b).
# Registered before it runs:
#   The guard MATTERS if keep beats keep_g3 at z >= 2 over all 1,500 positions at one depth, with the same sign at the other depth.
#   An arm is HARMFUL on passers if off beats it at z >= 2 at one depth, with the same sign at the other.
#   Prediction: keep vs keep_g3 |z| < 2 at both depths (#178: advancedPawn -2 at depth 10, +3 at equal time, over 200 positions).
# Stop: touch /mnt/e/chess_data/q179.stop (checked between runs). Never edit this file while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
BIN="$ND/build_uci/chess_engine_uci_lazy2"
DB=/mnt/e/chess_data/puzzles/lichess_db_puzzle.csv.zst
P=/mnt/e/chess_data/puzzles/passer
OD=/mnt/e/chess_data/lazy/passer179
STOP=/mnt/e/chess_data/q179.stop
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"
show() { sed "s/^/[q179]   /"; }
busy() { pgrep -f "fastchess-linux-x86-64/fastchess|vs_sf.py|_depth_residual_pass|_build_regret_set|_retune_fit|_ks_footprint_regret|sts_test|tactical_test|_theme_movetime" >/dev/null; }
echo "[q179] $(date) waiting for a quiet box (stop: $STOP)"
while busy; do [ -e "$STOP" ] && { echo "[q179] stop file while waiting — QUEUE 179 DONE"; exit 0; }; sleep 60; done
START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[q179] $START start"

# ── 1 suites ─────────────────────────────────────────────────────────────────────────────────────────────────────────
bash "$R" pyrun diagnostics/_build_theme_suites.py "$DB" "$P" 300 1108 passer 2>&1 | show
N=$(cat $P/*.txt 2>/dev/null | grep -c '|')
[ "$N" -ge 1000 ] || { echo "[q179] ☠️ only $N passer positions built — QUEUE 179 ABORTED"; exit 1; }
echo "[q179] $(date) $N positions in $(ls $P/*.txt | wc -l) suites"

# ── 2 fixed-depth runs ───────────────────────────────────────────────────────────────────────────────────────────────
ARMS="off b1 keep keep_g3 keep_m1000"
kn() {
  case $1 in
    off) echo "LAZY_V2=0" ;;
    b1) echo "LAZY_V2=7" ;;
    keep) echo "LAZY_V2=7,LAZY_V2_KEEP=1" ;;
    keep_g3) echo "LAZY_V2=7,LAZY_V2_KEEP=1,LAZY_V2_GUARDS=3" ;;
    keep_m1000) echo "LAZY_V2=7,LAZY_V2_KEEP=1,LAZY_V2_M=1000,LAZY_V2_M_CHEAP=2000" ;;
  esac
}
run() {  # core, depth, arms...
  local core=$1 d=$2 v; shift 2
  for v in "$@"; do
    [ -e "$STOP" ] && return
    EACH=$OD/d$d/$v bash "$R" pyrun diagnostics/_theme_movetime.py "$BIN" $core d$d "$(kn $v)" $P/*.txt > $OD/d$d/$v.txt 2>&1
  done
}
rm -rf "$OD"; mkdir -p $OD/d10 $OD/d12
# depth 12 costs about four depth-10 runs: lane 7 takes one depth-12 arm and all five depth-10 arms
run 3 12 off keep_m1000 & run 5 12 b1 keep & { run 7 12 keep_g3; run 7 10 $ARMS; } & wait
[ -e "$STOP" ] && { echo "[q179] stop file during the runs — QUEUE 179 DONE"; exit 0; }
echo "[q179] $(date) runs done"

# ── 3 tables and paired comparisons ──────────────────────────────────────────────────────────────────────────────────
pair() {  # depth, A, B: positions only A solves, only B solves, z
  local d=$1 A=$2 B=$3 a=0 b=0 f x y
  for f in $OD/d$d/$A/*.txt; do
    read x y < <(paste -d' ' "$f" "$OD/d$d/$B/$(basename $f)" | awk '{ if ($2 == 1 && $5 == 0) a++; if ($2 == 0 && $5 == 1) b++ } END { print a + 0, b + 0 }')
    a=$((a + x)); b=$((b + y))
  done
  awk -v d=$d -v A=$A -v B=$B -v a=$a -v b=$b 'BEGIN { printf "  depth %-3s %-11s vs %-11s only the first %4d · only the second %4d · z %+.2f\n", d, A, B, a, b, (a + b > 0 ? (a - b) / sqrt(a + b) : 0) }'
}
for d in 10 12; do
  echo "[q179] depth $d, solved per suite (of 300):"
  { printf '%-26s' "suite"; for v in $ARMS; do printf '%12s' $v; done; echo
    for f in $P/*.txt; do
      s=$(basename $f .txt); printf '%-26s' "$s"
      for v in $ARMS; do printf '%12s' "$(awk -v s=$s '$1 == s { print $2 }' $OD/d$d/$v.txt)"; done; echo
    done
    printf '%-26s' "TOTAL"
    for v in $ARMS; do printf '%12s' "$(awk 'NF == 3 && $2 ~ /^[0-9]+$/ && $3 ~ /^[0-9]+$/ { t += $2 } END { print t + 0 }' $OD/d$d/$v.txt)"; done; echo
  } | show
done
echo "[q179] paired (positions only one arm solves):"
for d in 10 12; do
  pair $d keep keep_g3; pair $d off keep_g3; pair $d off keep; pair $d off keep_m1000; pair $d off b1; pair $d keep keep_m1000
done | show
echo "[q179] $(date) QUEUE 179 DONE"
