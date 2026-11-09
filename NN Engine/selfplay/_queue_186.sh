#!/bin/bash
# 2026-11-08 evening, queue #186 (owner 11-08: start #185's timed games early and use the freed morning). After queue #185.
#  0 fingerprints (build_uci/chess_engine_uci_lazy3): default 246 / 36,509,830, keep_gate 247 / 39,852,210.
#  1 the 22-set depth-10 grid of #178 (WAC, STS300, ksattack, 400 game positions, 18 theme suites), arms off / b1 / keep / gate /
#    keep_m1000, two replicates (cores rotated, order reversed), three lanes pinned to cores 3 / 5 / 7: time to depth, nodes,
#    lazy rates. The question: is keep_gate faster to depth than lazy off over the wide grid? (#185's pinned WAC bench: keep_gate
#    +3.2% slower than off, keep -1.0%, keep_m1000 -6.6%; on #178's grid keep was -5.2%, so WAC alone may understate.)
#  2 only if #185's SF18 read says the gate PASSES: Mediocre v0.5 (selfplay/mediocre_uci.sh) at 10+0.1, keep_gate and lazy off
#    300 games each, ABBA chunks of 150 (ctl, gate, gate, ctl; seeds 186111 / 186112), concurrency 3, Mediocre with a 1,000 ms time
#    margin (its clock overran by up to 0.2 s at 1+0.05 in the 11-08 smoke test; with the margin both games ended normally).
# Registered before it runs:
#   Grid: a measurement, no pick. Prediction: keep_gate 2-5% less time to depth than off over the 22 sets.
#   Mediocre: the ship rule's leg, keep_gate's score minus lazy off's NOT NEGATIVE (point estimate), z reported beside it.
#     Prediction: within +-1 sigma of 0.
# Limits: no grid replicate starts after 10:00 on 11-09, no Mediocre chunk after 11:00. Nothing is installed.
# Stop: touch /mnt/e/chess_data/q186.stop (checked while waiting, between steps and chunks). Never edit this file while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
L=/mnt/e/chess_data/lazy
G=$L/grid186
TH=/mnt/e/chess_data/puzzles/themes
BIN="$ND/build_uci/chess_engine_uci_lazy3"
MED="$ND/selfplay/mediocre_uci.sh"
KG="LAZY_V2=7,LAZY_V2_KEEP=1,LAZY_V2_GATE=3"
STOP=/mnt/e/chess_data/q186.stop
Q185=/mnt/e/chess_data/q185.log
GRID_LATEST=$(date -d '2026-11-09 10:00' +%s)
MED_LATEST=$(date -d '2026-11-09 11:00' +%s)
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"
show() { sed "s/^/[q186]   /"; }
busy() { pgrep -f "fastchess-linux-x86-64/fastchess|vs_sf.py|_depth_residual_pass|_build_regret_set|_retune_fit|_ks_footprint_regret|sts_test|tactical_test|_theme_movetime|chess_engine_uci_lzprobe2" >/dev/null; }
kn() {
  case $1 in
    off) echo "LAZY_V2=0" ;;
    b1) echo "LAZY_V2=7" ;;
    keep) echo "LAZY_V2=7,LAZY_V2_KEEP=1" ;;
    gate) echo "$KG" ;;
    keep_m1000) echo "LAZY_V2=7,LAZY_V2_KEEP=1,LAZY_V2_M=1000,LAZY_V2_M_CHEAP=2000" ;;
  esac
}
echo "[q186] $(date) waiting for queue 185's end line and a quiet box (stop: $STOP)"
until grep -qE 'QUEUE 185 (DONE|ABORTED)' "$Q185" 2>/dev/null && ! busy; do
  [ -e "$STOP" ] && { echo "[q186] $(date) stop file while waiting — QUEUE 186 DONE (nothing run)"; exit 0; }
  [ $(date +%s) -gt $GRID_LATEST ] && { echo "[q186] $(date) past 10:00 before starting — QUEUE 186 DONE (nothing run)"; exit 0; }
  sleep 60
done
grep -q 'QUEUE 185 ABORTED' "$Q185" && { echo "[q186] ☠️ queue 185 aborted — QUEUE 186 ABORTED"; exit 1; }
sleep 30
START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[q186] $START start"

# ── 0 fingerprints ───────────────────────────────────────────────────────────────────────────────────────────────────
fp() { local k=""; [ -n "$1" ] && k="setoption name Knobs value $1"; printf '%s\nbench\nquit\n' "$k" | "$BIN" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+'; }
F0=$(fp ""); FG=$(fp "$KG")
echo "[q186] bench default: $F0 · keep_gate: $FG"
[ "$F0" = "solved 246/300 nodes 36509830" ] && [ "$FG" = "solved 247/300 nodes 39852210" ] || { echo "[q186] ☠️ fingerprints differ — QUEUE 186 ABORTED"; exit 1; }

# ── 1 the 22-set grid ────────────────────────────────────────────────────────────────────────────────────────────────
ARMS="off b1 keep gate keep_m1000"
SETFILES="$L/wac.txt $L/sts300.txt $L/ksattack.txt $L/games.txt $(ls $TH/*.txt)"
lane() {  # core, replicate, arms...
  local core=$1 rep=$2 v f s; shift 2
  for v in "$@"; do
    [ -e "$STOP" ] && return
    for f in $SETFILES; do
      s=$(basename "$f" .txt)
      printf 'bench %s 10\nquit\n' "$f" | env $(kn $v | tr , ' ') taskset -c $core "$BIN" 2>/dev/null \
        | grep -E '^bench|lazy eval' | tr '\n' ' ' > $G/r$rep/${v}__$s.txt
    done
  done
}
rm -rf $G; mkdir -p $G/r1 $G/r2
if [ $(date +%s) -gt $GRID_LATEST ]; then echo "[q186] past 10:00: no grid"; else
  echo "[q186] $(date) grid replicate 1"
  lane 3 1 off keep_m1000 & lane 5 1 b1 gate & lane 7 1 keep & wait
  if [ -e "$STOP" ] || [ $(date +%s) -gt $GRID_LATEST ]; then echo "[q186] stop file or past 10:00: no replicate 2"; else
    echo "[q186] $(date) grid replicate 2 (cores rotated, order reversed)"
    lane 5 2 keep_m1000 off & lane 7 2 gate b1 & lane 3 2 keep & wait
  fi
  for f in $G/r[12]/*__*.txt; do
    rep=$(basename $(dirname $f)); b=$(basename $f .txt)
    line=$(cat $f)
    bn=$(echo "$line" | sed -nE 's/.*solved ([0-9]+)\/([0-9]+) nodes ([0-9]+) time ([0-9]+) ms.*/\1 \2 \3 \4/p')
    lz=$(echo "$line" | sed -nE 's/.*lazy eval: ([0-9]+) lazy returns of ([0-9]+) evals asked; parent known on ([0-9]+); returns\/asks by site: stand-pat ([0-9]+)\/([0-9]+) rfp ([0-9]+)\/([0-9]+) futility ([0-9]+)\/([0-9]+).*/\1 \2 \3 \4 \5 \6 \7 \8 \9/p')
    echo "$rep ${b%%__*} ${b#*__} ${bn:-x x x x} ${lz:-0 0 0 0 0 0 0 0 0}"
  done > $G/rows.txt
  awk '
    $4 == "x" { bad[$2] = 1; next }
    { r = $1; v = $2; s = $3
      t[v, r] += $7; nd[v, r] += $6
      if (r == "r1") {
        if (s == "wac" || s == "sts300" || s == "ksattack" || s == "games") sol[v, s] = $4; else { th[v] += $4; thn[v] += $5 }
        fi[v] += $8; as[v] += $9; pa[v] += $10
        s1[v, s] = $4
      } else if (s1[v, s] != "" && s1[v, s] != $4) nondet[v] = 1 }
    function pc(a, b) { return b > 0 ? sprintf("%5.1f", 100 * a / b) : "    -" }
    END {
      n = split("off b1 keep gate keep_m1000", A, " ")
      T0 = (t["off", "r1"] + t["off", "r2"]) / 2; N0 = nd["off", "r1"]
      printf "arm          themes   wac  sts  ks games   time_r1  time_r2  time_vs_off  nodes_vs_off  lazy%% parent%%\n"
      for (i = 1; i <= n; i++) { v = A[i]; T = (t[v, "r1"] + t[v, "r2"]) / 2
        printf "%-11s %4d/%d %4d %4d %3d %5d  %8d %8d  %+10.1f%%  %+11.1f%%  %s  %s%s%s\n", v, th[v], thn[v], sol[v, "wac"], sol[v, "sts300"],
          sol[v, "ksattack"], sol[v, "games"], t[v, "r1"], t[v, "r2"], (T0 > 0 ? 100 * (T - T0) / T0 : 0), (N0 > 0 ? 100 * (nd[v, "r1"] - N0) / N0 : 0),
          pc(fi[v], as[v]), pc(pa[v], as[v]), (nondet[v] ? "  <- replicates disagree on a count" : ""), (bad[v] ? "  <- a run failed" : "") }
    }' $G/rows.txt > $G/table.txt
  echo "[q186] $(date) grid table (time in ms summed over the 22 sets; time_vs_off = mean of the two replicates; lazy% = lazy returns of asks):"
  show < $G/table.txt
fi

# ── 2 Mediocre leg ───────────────────────────────────────────────────────────────────────────────────────────────────
if ! grep -qE 'gate +- ctl_pooled.*-> PASS' "$Q185"; then
  echo "[q186] keep_gate did not pass #185's SF18 leg (or it did not run): no Mediocre leg"
elif [ -e "$STOP" ]; then echo "[q186] stop file before the Mediocre leg"; else
  rm -f /tmp/q186_pre.engine.log
  timeout 300 "$FC" -engine "cmd=$BIN" name=ours option.OwnBook=false "option.Knobs=$KG" tc=10+0.1 timemargin=250 \
    -engine "cmd=$MED" name=mediocre tc=10+0.1 timemargin=1000 \
    -openings file=$O/openings_uho.epd format=epd order=random -rounds 1 -games 2 -concurrency 1 \
    -log file=/tmp/q186_pre.engine.log level=info engine=true > /tmp/q186_pre.log 2>&1
  W=$(grep -c 'have option' /tmp/q186_pre.log)
  echo "[q186] Mediocre preflight: $(grep -E '^Games:' /tmp/q186_pre.log | tail -1) · warnings $W · arm echoes $(grep -cF "knobs $KG" /tmp/q186_pre.engine.log) · time losses $(grep -c 'loses on time' /tmp/q186_pre.log)"
  if [ "$W" != 0 ]; then echo "[q186] ☠️ Mediocre preflight warnings: no Mediocre leg"; else
    COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
    T=med_lzg
    chunk() {  # n, arm, seed
      local c=$1 a=$2 seed=$3 k=""
      [ -e "$STOP" ] && { echo "[q186] stop file before Mediocre chunk $c"; return; }
      [ $(date +%s) -gt $MED_LATEST ] && { echo "[q186] $(date) skip Mediocre chunk $c (past 11:00)"; return; }
      [ $a = gate ] && k="option.Knobs=$KG"
      nice -n 5 "$FC" -engine "cmd=$BIN" name=ours option.OwnBook=false $k tc=10+0.1 timemargin=250 \
        -engine "cmd=$MED" name=mediocre tc=10+0.1 timemargin=1000 \
        $COMMON -srand $seed -rounds 75 -games 2 -pgnout file=$O/${T}_c${c}_$a.pgn notation=san append=false > $O/${T}_c${c}_$a.log 2>&1
      echo "[q186] $(date) Mediocre chunk $c ($a, seed $seed): $(grep -E '^Games:' $O/${T}_c${c}_$a.log | tail -1) · disconnects/stalls $(grep -ciE 'disconnect|stall|not responsive' $O/${T}_c${c}_$a.log) · time losses $(grep -c 'loses on time' $O/${T}_c${c}_$a.log)"
    }
    rm -f $O/${T}_c*
    chunk 1 ctl 186111; chunk 2 gate 186111; chunk 3 gate 186112; chunk 4 ctl 186112
    for a in ctl gate; do
      for f in $O/${T}_c*_$a.log; do [ -e "$f" ] && grep -E '^Games:' "$f" | tail -1; done | awk -v a=$a '{
          gsub(/,/, ""); for (i = 1; i <= NF; i++) { if ($i == "Games:") n += $(i+1); if ($i == "Wins:") w += $(i+1); if ($i == "Losses:") l += $(i+1); if ($i == "Draws:") d += $(i+1) } }
        END { s = (n > 0 ? (w + d / 2) / n : 0); v = (n > 0 ? (w + d / 4) / n - s * s : 0); printf "%s %d %d %d %d %.6f %.8f\n", a, n, w, l, d, s, (n > 0 ? v / n : 0) }'
    done > /tmp/q186_read.txt
    echo "[q186] $(date) REGISTERED READ, score vs Mediocre at 10+0.1 (gate minus ctl not negative):"
    awk '{ printf "[q186]   %-4s games %4d  +%d -%d =%d  score %.1f%%  (se %.1f pp)\n", $1, $2, $3, $4, $5, 100 * $6, 100 * sqrt($7) }' /tmp/q186_read.txt
    awk '{ s[$1] = $6; v[$1] = $7 } END { d = s["gate"] - s["ctl"]; se = sqrt(v["gate"] + v["ctl"]); printf "[q186]   gate - ctl %+.2f pp  ± %.2f (95%%)  z %+.2f  -> %s\n", 100 * d, 196 * se, (se > 0 ? d / se : 0), (d >= 0 ? "NOT NEGATIVE" : "NEGATIVE") }' /tmp/q186_read.txt
  fi
fi
echo "[q186] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | show
echo "[q186] $(date) QUEUE 186 DONE"
