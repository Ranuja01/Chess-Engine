#!/bin/bash
# 2026-11-08, queue #178 (owner 11-07: overnight and morning runs): lazy eval build 2 (LAZY_V2_KEEP, per-site margin scales, per-site
# and parent-coverage counters; every new knob defaults to build 1's behaviour). Waits for queue #177's end line and a quiet box.
#  1 build build_uci/chess_engine_uci_lazy2 (no PGO).
#  2 fingerprints: default 246 / 36,509,830 (gen 7); LAZY_V2=7 245 / 37,305,726 (build 1 reproduced); the KEEP arm through the
#    Knobs option and through env must agree (the games pass knobs through the option).
#  3 fixed-depth grid, depth 10, 8 arms x (WAC, STS300, ksattack, 400 game positions, 18 Lichess theme suites), three processes
#    pinned to cores 3 / 5 / 7, two replicates with the cores rotated: solved counts, time to depth, lazy returns by site, parent
#    coverage.
#  4 the registered pick (below), then timed self-play 10+0.1 UHO, candidate vs lazy off, fresh seeds 178111-178114, up to 4 x 400 g,
#    concurrency 3; a chunk starts only before 09:30 on 11-08.
#  5 the critical-situation test at equal time for the candidate (18 themes + WAC + ksattack, 0.25 s per move), only if it can start
#    before 10:00 (weekend: no timed run after 10:30).
# Registered before any of it runs:
#   PICK: among the arms other than off and b1, the one with the least time to depth (all 22 sets, mean of the two replicates) whose
#   theme total is at least b1's - 5 and whose WAC count is at least b1's - 1. None passes: no games.
#   GAMES: same rule as #177: kept and developed at >= -1 sigma vs off, a gain only at +2 sigma.
#   Prediction: KEEP raises the asks with a parent value from about 66% to 85% or more, lowers time to depth by a further 2-5%
#   against b1, and moves the theme total by no more than the noise.
# Nothing is installed. Stop: touch /mnt/e/chess_data/q178.stop (checked while waiting, between steps and between chunks).
# Never edit this file while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
L=/mnt/e/chess_data/lazy
G=$L/grid178
TH=/mnt/e/chess_data/puzzles/themes
BIN="$ND/build_uci/chess_engine_uci_lazy2"
STOP=/mnt/e/chess_data/q178.stop
Q177=/mnt/e/chess_data/q177.log
CHUNK_LATEST=$(date -d '2026-11-08 09:30' +%s)
THEMES_LATEST=$(date -d '2026-11-08 10:00' +%s)
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"
show() { sed "s/^/[q178]   /"; }
busy() { pgrep -f "fastchess-linux-x86-64/fastchess|vs_sf.py|_depth_residual_pass|_build_regret_set|_retune_fit|_ks_footprint_regret|sts_test|tactical_test|_theme_movetime" >/dev/null; }
echo "[q178] $(date) waiting for queue 177's end line and a quiet box (stop: $STOP)"
until grep -qE 'QUEUE 177 (DONE|ABORTED)' "$Q177" 2>/dev/null && ! busy; do
  [ -e "$STOP" ] && { echo "[q178] $(date) stop file while waiting — QUEUE 178 DONE (nothing run)"; exit 0; }
  sleep 60
done
grep -q 'QUEUE 177 ABORTED' "$Q177" && { echo "[q178] ☠️ queue 177 aborted; not building on it — QUEUE 178 ABORTED"; exit 1; }
sleep 30
START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[q178] $START start"

# ── 1 build ──────────────────────────────────────────────────────────────────────────────────────────────────────────
OUT=chess_engine_uci_lazy2 bash selfplay/build_uci.sh 2>&1 | show
[ -x "$BIN" ] && [ "$BIN" -nt selfplay/_queue_178.sh ] || { echo "[q178] ☠️ build failed — QUEUE 178 ABORTED"; exit 1; }

# ── 2 fingerprints ───────────────────────────────────────────────────────────────────────────────────────────────────
fp() { printf '%s\nbench\nquit\n' "$1" | "$BIN" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+'; }
B0=$(fp "")
B7=$(fp "setoption name Knobs value LAZY_V2=7")
BK=$(fp "setoption name Knobs value LAZY_V2=7,LAZY_V2_KEEP=1")
BKE=$(printf 'bench\nquit\n' | env LAZY_V2=7 LAZY_V2_KEEP=1 "$BIN" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+')
echo "[q178] bench default: $B0 · LAZY_V2=7: $B7 · +KEEP (option): $BK · +KEEP (env): $BKE"
[ "$B0" = "solved 246/300 nodes 36509830" ] || { echo "[q178] ☠️ default bench is not gen 7's — QUEUE 178 ABORTED"; exit 1; }
[ "$B7" = "solved 245/300 nodes 37305726" ] || { echo "[q178] ☠️ LAZY_V2=7 does not reproduce build 1 — QUEUE 178 ABORTED"; exit 1; }
[ -n "$BK" ] && [ "$BK" = "$BKE" ] || { echo "[q178] ☠️ the KEEP arm differs between the Knobs option and env — QUEUE 178 ABORTED"; exit 1; }
[ "$BK" != "$B7" ] || echo "[q178] note: KEEP=1 is byte-identical to build 1 on WAC (the fix changes nothing there?)"

# ── 3 fixed-depth grid ───────────────────────────────────────────────────────────────────────────────────────────────
ARMS="off b1 keep keep_c0 keep_site keep_m1000 keep_g3 keep_c5"
kn() {  # arm -> knobs, NAME=V comma-separated
  case $1 in
    off) echo "LAZY_V2=0" ;;
    b1) echo "LAZY_V2=7" ;;
    keep) echo "LAZY_V2=7,LAZY_V2_KEEP=1" ;;
    keep_c0) echo "LAZY_V2=7,LAZY_V2_KEEP=1,LAZY_V2_M_CHEAP=0" ;;
    keep_site) echo "LAZY_V2=7,LAZY_V2_KEEP=1,LAZY_V2_S_QS=133,LAZY_V2_S_FUT=67" ;;
    keep_m1000) echo "LAZY_V2=7,LAZY_V2_KEEP=1,LAZY_V2_M=1000,LAZY_V2_M_CHEAP=2000" ;;
    keep_g3) echo "LAZY_V2=7,LAZY_V2_KEEP=1,LAZY_V2_GUARDS=3" ;;
    keep_c5) echo "LAZY_V2=7,LAZY_V2_KEEP=1,LAZY_V2_M_CHEAP=5000" ;;
  esac
}
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
echo "[q178] $(date) grid replicate 1"
lane 3 1 off keep_c0 keep_g3 & lane 5 1 b1 keep_site keep_c5 & lane 7 1 keep keep_m1000 & wait
echo "[q178] $(date) grid replicate 2 (cores rotated, order reversed)"
lane 5 2 keep_g3 keep_c0 off & lane 7 2 keep_c5 keep_site b1 & lane 3 2 keep_m1000 keep & wait
[ -e "$STOP" ] && { echo "[q178] stop file during the grid — QUEUE 178 DONE"; exit 0; }
for f in $G/r[12]/*__*.txt; do
  rep=$(basename $(dirname $f)); b=$(basename $f .txt)
  line=$(cat $f)
  bn=$(echo "$line" | sed -nE 's/.*solved ([0-9]+)\/([0-9]+) nodes ([0-9]+) time ([0-9]+) ms.*/\1 \2 \3 \4/p')
  lz=$(echo "$line" | sed -nE 's/.*lazy eval: ([0-9]+) lazy returns of ([0-9]+) evals asked; parent known on ([0-9]+); returns\/asks by site: stand-pat ([0-9]+)\/([0-9]+) rfp ([0-9]+)\/([0-9]+) futility ([0-9]+)\/([0-9]+).*/\1 \2 \3 \4 \5 \6 \7 \8 \9/p')
  echo "$rep ${b%%__*} ${b#*__} ${bn:-x x x x} ${lz:-0 0 0 0 0 0 0 0 0}"
done > $G/rows.txt
awk '
  $4 == "x" { bad[$2] = 1; next }
  { r = $1; v = $2; s = $3; arm[v] = 1
    t[v, r] += $7
    if (r == "r1") {
      if (s == "wac" || s == "sts300" || s == "ksattack" || s == "games") sol[v, s] = $4; else { th[v] += $4; thn[v] += $5 }
      fi[v] += $8; as[v] += $9; pa[v] += $10; qf[v] += $11; qa[v] += $12; rf[v] += $13; ra[v] += $14; ff[v] += $15; fa[v] += $16
      s1[v, s] = $4
    } else if (s1[v, s] != "" && s1[v, s] != $4) nondet[v] = 1 }
  function pc(a, b) { return b > 0 ? sprintf("%5.1f", 100 * a / b) : "    -" }
  END {
    n = split("off b1 keep keep_c0 keep_site keep_m1000 keep_g3 keep_c5", A, " ")
    T0 = (t["off", "r1"] + t["off", "r2"]) / 2
    printf "arm          themes   wac  sts  ks games   time_r1  time_r2  vs_off   lazy%% parent%%  sp%%  rfp%%  fut%%\n"
    for (i = 1; i <= n; i++) { v = A[i]; T = (t[v, "r1"] + t[v, "r2"]) / 2; mt[v] = T
      printf "%-11s %4d/%d %4d %4d %3d %5d  %8d %8d  %+5.1f%%  %s  %s  %s %s %s%s%s\n", v, th[v], thn[v], sol[v, "wac"], sol[v, "sts300"],
        sol[v, "ksattack"], sol[v, "games"], t[v, "r1"], t[v, "r2"], (T0 > 0 ? 100 * (T - T0) / T0 : 0), pc(fi[v], as[v]), pc(pa[v], as[v]),
        pc(qf[v], qa[v]), pc(rf[v], ra[v]), pc(ff[v], fa[v]), (nondet[v] ? "  <- replicates disagree on a count" : ""), (bad[v] ? "  <- a run failed" : "") }
    best = ""; for (i = 3; i <= n; i++) { v = A[i]
      if (bad[v] || th[v] < th["b1"] - 5 || sol[v, "wac"] < sol["b1", "wac"] - 1) continue
      if (best == "" || mt[v] < mt[best]) best = v }
    print "CANDIDATE " (best == "" ? "none" : best)
  }' $G/rows.txt > $G/table.txt
echo "[q178] $(date) grid table (time in ms, summed over the 22 sets; lazy% = lazy returns of asks; parent% = asks with a parent value; build 1's d10 themes were off 3282, b1 3275):"
show < $G/table.txt
CAND=$(awk '/^CANDIDATE/{print $2}' $G/table.txt)
[ -n "$CAND" ] && [ "$CAND" != none ] || { echo "[q178] no arm passed the registered pick — no games. QUEUE 178 DONE"; exit 0; }
KN=$(kn $CAND)
echo "[q178] registered pick: $CAND ($KN)"

# ── 4 timed self-play, candidate vs lazy off ─────────────────────────────────────────────────────────────────────────
[ -e "$STOP" ] && { echo "[q178] stop file before the games — QUEUE 178 DONE"; exit 0; }
rm -f /tmp/q178_pre.engine.log
timeout 180 "$FC" -engine "cmd=$BIN" name=arm option.OwnBook=false "option.Knobs=$KN" st=0.05 \
  -engine "cmd=$BIN" name=ctl option.OwnBook=false st=0.05 \
  -openings file=$O/openings_uho.epd format=epd order=random -rounds 1 -games 2 -concurrency 1 \
  -log file=/tmp/q178_pre.engine.log level=info engine=true > /tmp/q178_pre.log 2>&1
W=$(grep -c 'have option' /tmp/q178_pre.log)
echo "[q178] preflight: $(grep -E '^Games:' /tmp/q178_pre.log | tail -1) · warnings $W · arm echoes $(grep -c "knobs $KN" /tmp/q178_pre.engine.log)"
[ "$W" = 0 ] || { echo "[q178] ☠️ preflight warnings — QUEUE 178 ABORTED"; exit 1; }
sleep 20
COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
T=lz2_10p1
rm -f $O/${T}_c[1-4].pgn $O/${T}_c[1-4].log
for c in 1 2 3 4; do
  while busy && [ ! -e "$STOP" ]; do sleep 60; done
  if [ -e "$STOP" ] || [ $(date +%s) -gt $CHUNK_LATEST ]; then echo "[q178] $(date) skip $T chunk $c (stop or past 09:30)"; continue; fi
  nice -n 5 "$FC" -engine "cmd=$BIN" name=arm option.OwnBook=false "option.Knobs=$KN" tc=10+0.1 timemargin=250 \
    -engine "cmd=$BIN" name=ctl option.OwnBook=false tc=10+0.1 timemargin=250 \
    $COMMON -srand $(( 178110 + c )) -rounds 200 -games 2 \
    -pgnout file=$O/${T}_c$c.pgn notation=san nodes=true append=false > $O/${T}_c$c.log 2>&1
  echo "[q178] $(date) $T chunk $c: $(grep -E '^Elo:' $O/${T}_c$c.log | tail -1) · warnings $(grep -c 'have option' $O/${T}_c$c.log) · disconnects/stalls $(grep -ciE 'disconnect|stall|not responsive' $O/${T}_c$c.log) · time forfeits $(grep -c 'loses on time' $O/${T}_c$c.pgn)"
done
f=$(ls $O/${T}_c[1-4].pgn 2>/dev/null)
if [ -n "$f" ]; then
  out=$(timeout 900 bash "$R" pyrun selfplay/fastchess_report.py $f --p1 arm 2>&1 | sed -n '/vs the field/,$p')
  echo "[q178] $(date) REGISTERED READ, $CAND vs lazy off, all chunks (keep at >= -1 sigma, gain at +2 sigma):"
  echo "$out" | show
  echo "$out" | grep -m1 -E 'Elo [-+]?[0-9.]+' | awk '{gsub(/[()]/,""); gsub(/\+/,""); printf "[q178]   z = %.2f\n", $2 / (($6 - $4) / 3.92)}'
fi

# ── 5 critical-situation test at equal time ──────────────────────────────────────────────────────────────────────────
if [ -e "$STOP" ] || [ $(date +%s) -gt $THEMES_LATEST ]; then echo "[q178] $(date) theme test skipped (stop or past 10:00)"; else
  while busy; do sleep 60; done
  X="$L/wac.txt $L/ksattack.txt"
  echo "[q178] $(date) themes at 0.25 s per move, $CAND (core 3) and off (core 5) together"
  bash "$R" pyrun diagnostics/_theme_movetime.py "$BIN" 3 0.25 "$KN" $TH/*.txt $X > $L/themes_mt_$CAND.txt 2>&1 &
  bash "$R" pyrun diagnostics/_theme_movetime.py "$BIN" 5 0.25 - $TH/*.txt $X > $L/themes_mt_off2.txt 2>&1 &
  wait
  echo "[q178] theme                off   $CAND   (of n)"
  join <(sort $L/themes_mt_off2.txt) <(sort $L/themes_mt_$CAND.txt) | awk '{d=$4-$2; flag=(d>14||d<-14)?"  <- flag":""; printf "%-20s %4d %4d  (%d)  %+d%s\n", $1, $2, $4, $3, d, flag}' | show
fi
echo "[q178] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | show
echo "[q178] $(date) QUEUE 178 DONE"
