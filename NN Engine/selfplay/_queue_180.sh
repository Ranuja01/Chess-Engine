#!/bin/bash
# 2026-11-08 afternoon, queue #180 (owner 11-08: "passers are known to be somewhat of a weakness already ... test our passer
# positions against other engines to see if our power has noticeably dropped in these critical situations"). Equal WORK, the
# regime of REFERENCE-BENCH-LADDER.md: every engine gets the same nodes per move. Node-limited, so safe on a weekend afternoon.
#  Engines: ours lazy off (LAZY_V2=0) and the lazy candidate keep_m1000, binary chess_engine_uci_lazy2; SF11 (the hand-written
#  ceiling), SF15.1 classical, SF18 (native ELF, 1 thread, Hash 128). SF1.1 is Windows-only here and is left out.
#  Budgets: 100,000 and 400,000 nodes per move (400k is about our budget per move at 10+0.1).
#  Suites (5,624 positions): 5 passed-pawn suites x 300, the 18 critical-theme suites x 200, WAC 300, ksattack 224.
#  Tool: diagnostics/_theme_movetime.py LIMIT nN, per-position output; three processes pinned to cores 3 / 5 / 7.
# Registered before it runs:
#   Q1 (are passed pawns a relative weakness under search?): ours (lazy off) as a share of SF11's solves, on the passer group vs
#     on the 18 themes. A relative weakness if the passer share is lower at both budgets.
#   Q2 (does lazy eval drop us there?): keep_m1000 vs off, paired, per group and budget. HARMFUL on passers if off is ahead at
#     z >= 2 at one budget with the same sign at the other. Reported against the gap to SF11.
#   Prediction: Q1 yes (pawn endings were found to be a search gap); Q2 not harmful, and the lazy change under a tenth of the gap.
# Stop: touch /mnt/e/chess_data/q180.stop (checked between runs). Never edit this file while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
SFR="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine"
R="$ND/selfplay/overnight_runner.sh"
BIN="$ND/build_uci/chess_engine_uci_lazy2"
P=/mnt/e/chess_data/puzzles/passer
TH=/mnt/e/chess_data/puzzles/themes
L=/mnt/e/chess_data/lazy
OD=$L/ref180
STOP=/mnt/e/chess_data/q180.stop
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"
show() { sed "s/^/[q180]   /"; }
busy() { pgrep -f "fastchess-linux-x86-64/fastchess|vs_sf.py|_depth_residual_pass|_build_regret_set|_retune_fit|_ks_footprint_regret|sts_test|tactical_test|_theme_movetime" >/dev/null; }
echo "[q180] $(date) waiting for a quiet box (stop: $STOP)"
while busy; do [ -e "$STOP" ] && { echo "[q180] stop file while waiting — QUEUE 180 DONE"; exit 0; }; sleep 60; done
START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[q180] $START start"
SETS="$(ls $P/*.txt) $(ls $TH/*.txt) $L/wac.txt $L/ksattack.txt"
N=$(cat $SETS | grep -c '|')
echo "[q180] $N positions in $(echo $SETS | wc -w) suites"

ENGS="off m1000 sf11 sf15c sf18"
bin() { case $1 in off|m1000) echo "$BIN" ;;
  sf11) echo "$SFR/stockfish_11_linux/stockfish-11-linux/Linux/stockfish_20011801_x64_bmi2" ;;
  sf15c) echo "$SFR/stockfish_15_linux/stockfish_15.1_linux_x64/stockfish-ubuntu-20.04-x86-64" ;;
  sf18) echo "$SFR/stockfish_18_linux/stockfish-ubuntu-x86-64-avx2" ;; esac; }
kn() { case $1 in off) echo "LAZY_V2=0" ;; m1000) echo "LAZY_V2=7,LAZY_V2_KEEP=1,LAZY_V2_M=1000,LAZY_V2_M_CHEAP=2000" ;; *) echo "-" ;; esac; }
uo() { case $1 in sf15c) echo "Use NNUE=false|Hash=128|Threads=1" ;; sf*) echo "Hash=128|Threads=1" ;; *) echo "" ;; esac; }
run() {  # core, then engine:nodes pairs
  local core=$1 x e n; shift
  for x in "$@"; do
    [ -e "$STOP" ] && return
    e=${x%:*}; n=${x#*:}; mkdir -p $OD/n$n
    UCI_OPTS="$(uo $e)" EACH=$OD/n$n/$e bash "$R" pyrun diagnostics/_theme_movetime.py "$(bin $e)" $core n$n "$(kn $e)" $SETS \
      > $OD/n$n/$e.txt 2>&1
    echo "[q180] $(date) $e at $n nodes: $(awk 'NF == 3 && $2 ~ /^[0-9]+$/ && $3 ~ /^[0-9]+$/ { s += $2; t += $3 } END { print s + 0 "/" t + 0 }' $OD/n$n/$e.txt)"
  done
}
rm -rf "$OD"; mkdir -p "$OD"
run 3 sf18:400000 sf18:100000 & run 5 sf11:400000 sf15c:400000 sf11:100000 sf15c:100000 & \
  run 7 off:400000 m1000:400000 off:100000 m1000:100000 & wait
[ -e "$STOP" ] && { echo "[q180] stop file during the runs — QUEUE 180 DONE"; exit 0; }

# ── tables ─────────────────────────────────────────────────────────────────────────────────────────────────────────
grp() { case $1 in promotion|advancedPawn_mg|pawnEndgame|rookEndgame_passer|minorQueenEndgame_passer) echo passer ;;
  wac) echo wac ;; ksattack) echo ksattack ;; *) echo themes ;; esac; }
for n in 400000 100000; do
  echo "[q180] $n nodes per move, solved per suite:"
  { printf '%-26s' "suite"; for e in $ENGS; do printf '%8s' $e; done; echo
    for f in $SETS; do
      s=$(basename $f .txt); printf '%-26s' "$s"
      for e in $ENGS; do printf '%8s' "$(awk -v s=$s '$1 == s { print $2 }' $OD/n$n/$e.txt 2>/dev/null)"; done; echo
    done
    for g in passer themes wac ksattack; do
      printf '%-26s' "GROUP $g"
      for e in $ENGS; do
        t=0; for f in $SETS; do s=$(basename $f .txt); [ "$(grp $s)" = $g ] && t=$((t + $(awk -v s=$s '$1 == s { v = $2 } END { print v + 0 }' $OD/n$n/$e.txt 2>/dev/null || echo 0))); done
        printf '%8s' $t
      done; echo
    done
  } | show
done
pair() {  # nodes, A, B, group: positions only A solves, only B solves, z
  local n=$1 A=$2 B=$3 g=$4 a=0 b=0 f s x y
  for f in $SETS; do
    s=$(basename $f .txt); [ "$(grp $s)" = $g ] || continue
    read x y < <(paste -d' ' "$OD/n$n/$A/$s.txt" "$OD/n$n/$B/$s.txt" | awk '{ if ($2 == 1 && $5 == 0) a++; if ($2 == 0 && $5 == 1) b++ } END { print a + 0, b + 0 }')
    a=$((a + x)); b=$((b + y))
  done
  awk -v n=$n -v A=$A -v B=$B -v g=$g -v a=$a -v b=$b 'BEGIN { printf "  %7s nodes %-8s %-5s vs %-5s only the first %4d · only the second %4d · z %+.2f\n", n, g, A, B, a, b, (a + b > 0 ? (a - b) / sqrt(a + b) : 0) }'
}
echo "[q180] paired (positions only one side solves):"
for n in 400000 100000; do for g in passer themes wac ksattack; do pair $n off m1000 $g; done; done | show
echo "[q180] $(date) QUEUE 180 DONE"
