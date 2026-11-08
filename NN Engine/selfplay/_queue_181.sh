#!/bin/bash
# 2026-11-08, queue #181 (owner 11-08 "ok" to tonight's plan): the lazy eval candidate keep_m1000 (LAZY_V2=7, LAZY_V2_KEEP=1,
# LAZY_V2_M=1000, LAZY_V2_M_CHEAP=2000: the record fix, guards on, narrower margins; chosen over keep_g3 by the passed-pawn test #179)
# against itself with lazy off, on build_uci/chess_engine_uci_lazy2 (no PGO; both arms the same binary). Starts at 00:45 on 11-09
# (the owner's Sunday evening stays free of timed work), or earlier if /mnt/e/chess_data/q181.go exists, and after queue #180;
# no timed chunk starts after 12:00 on 11-09.
#  1 fingerprints: default 246 / 36,509,830 (gen 7), the arm 249 / 37,197,972.
#  2 pinned WAC bench, depth 10, off / arm alternating x 4 on one core: time to depth.
#  3 timed self-play 10+0.1 UHO, arm vs ctl, fresh seeds 181111-181114, 4 x 500 g, concurrency 3.
#  4 the critical-situation test at equal time: 18 Lichess theme suites + 5 passed-pawn suites + WAC + ksattack, 0.25 s per move,
#    arm and ctl at the same time on two cores.
# Registered before any game (same rule as #177 / #178): NOT HARMFUL, and so on to the ship gate (SF18 > -1 sigma, Mediocre not
#   negative), if the pooled games are at or above -1 sigma; a gain is claimed only at +2 sigma. A suite moving by more than 14 is
#   flagged for a look either way. Prediction: 0 to +15 Elo, z 0 to 1.5 (time to depth -8.0% on the #178 grid; nps +4.5%).
# Nothing is installed. Stop: touch /mnt/e/chess_data/q181.stop (checked between steps and chunks). Never edit this file while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
L=/mnt/e/chess_data/lazy
BIN="$ND/build_uci/chess_engine_uci_lazy2"
KN="LAZY_V2=7,LAZY_V2_KEEP=1,LAZY_V2_M=1000,LAZY_V2_M_CHEAP=2000"
STOP=/mnt/e/chess_data/q181.stop
GO=/mnt/e/chess_data/q181.go
EARLIEST=$(date -d '2026-11-09 00:45' +%s)
TIMED_LATEST=$(date -d '2026-11-09 12:00' +%s)
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"
show() { sed "s/^/[q181]   /"; }
busy() { pgrep -f "fastchess-linux-x86-64/fastchess|vs_sf.py|_depth_residual_pass|_build_regret_set|_retune_fit|_ks_footprint_regret|sts_test|tactical_test|_theme_movetime" >/dev/null; }
give_up() { [ -e "$STOP" ] || [ $(date +%s) -gt $TIMED_LATEST ]; }
echo "[q181] $(date) waiting for 00:45 on 11-09 (or $GO), queue 180's end and a quiet box (stop: $STOP)"
until { [ $(date +%s) -ge $EARLIEST ] || [ -e "$GO" ]; } && ! busy; do
  give_up && { echo "[q181] $(date) stop file or past 12:00 before starting — QUEUE 181 DONE (nothing run)"; exit 0; }
  sleep 60
done
sleep 30
START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[q181] $START start"

# ── 1 fingerprints ───────────────────────────────────────────────────────────────────────────────────────────────────
B0=$(printf 'bench\nquit\n' | "$BIN" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+')
BA=$(printf 'setoption name Knobs value %s\nbench\nquit\n' "$KN" | "$BIN" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+')
echo "[q181] bench default: $B0 · arm: $BA"
[ "$B0" = "solved 246/300 nodes 36509830" ] || { echo "[q181] ☠️ default bench is not gen 7's — QUEUE 181 ABORTED"; exit 1; }
[ "$BA" = "solved 249/300 nodes 37197972" ] || { echo "[q181] ☠️ the arm's bench is not 249 / 37,197,972 (knobs not reaching it?) — QUEUE 181 ABORTED"; exit 1; }

# ── 2 pinned bench ───────────────────────────────────────────────────────────────────────────────────────────────────
for i in 1 2 3 4; do
  for arm in off on; do
    k=""; [ $arm = on ] && k="setoption name Knobs value $KN"
    echo "[q181] pinned WAC d10 $arm: $(printf '%s\nbench\nquit\n' "$k" | taskset -c 5 "$BIN" 2>/dev/null | grep -E '^bench' | grep -oE 'time [0-9]+ ms nps [0-9]+')"
  done
done

# ── 3 timed self-play ────────────────────────────────────────────────────────────────────────────────────────────────
rm -f /tmp/q181_pre.engine.log
timeout 180 "$FC" -engine "cmd=$BIN" name=arm option.OwnBook=false "option.Knobs=$KN" st=0.05 \
  -engine "cmd=$BIN" name=ctl option.OwnBook=false st=0.05 \
  -openings file=$O/openings_uho.epd format=epd order=random -rounds 1 -games 2 -concurrency 1 \
  -log file=/tmp/q181_pre.engine.log level=info engine=true > /tmp/q181_pre.log 2>&1
W=$(grep -c 'have option' /tmp/q181_pre.log)
echo "[q181] preflight: $(grep -E '^Games:' /tmp/q181_pre.log | tail -1) · warnings $W · arm echoes $(grep -cF "knobs $KN" /tmp/q181_pre.engine.log)"
[ "$W" = 0 ] || { echo "[q181] ☠️ preflight warnings — QUEUE 181 ABORTED"; exit 1; }
sleep 20
COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
T=lzm_10p1
rm -f $O/${T}_c[1-4].pgn $O/${T}_c[1-4].log
for c in 1 2 3 4; do
  while busy && ! give_up; do sleep 60; done
  give_up && { echo "[q181] $(date) skip $T chunk $c (stop or past 12:00)"; continue; }
  nice -n 5 "$FC" -engine "cmd=$BIN" name=arm option.OwnBook=false "option.Knobs=$KN" tc=10+0.1 timemargin=250 \
    -engine "cmd=$BIN" name=ctl option.OwnBook=false tc=10+0.1 timemargin=250 \
    $COMMON -srand $(( 181110 + c )) -rounds 250 -games 2 \
    -pgnout file=$O/${T}_c$c.pgn notation=san nodes=true append=false > $O/${T}_c$c.log 2>&1
  echo "[q181] $(date) $T chunk $c: $(grep -E '^Elo:' $O/${T}_c$c.log | tail -1) · warnings $(grep -c 'have option' $O/${T}_c$c.log) · disconnects/stalls $(grep -ciE 'disconnect|stall|not responsive' $O/${T}_c$c.log) · time forfeits $(grep -c 'loses on time' $O/${T}_c$c.pgn)"
done
f=$(ls $O/${T}_c[1-4].pgn 2>/dev/null)
if [ -n "$f" ]; then
  out=$(timeout 900 bash "$R" pyrun selfplay/fastchess_report.py $f --p1 arm 2>&1 | sed -n '/vs the field/,$p')
  echo "[q181] $(date) REGISTERED READ, keep_m1000 vs lazy off, all chunks (not harmful at >= -1 sigma, gain at +2 sigma):"
  echo "$out" | show
  echo "$out" | grep -m1 -E 'Elo [-+]?[0-9.]+' | awk '{gsub(/[()]/,""); gsub(/\+/,""); printf "[q181]   z = %.2f\n", $2 / (($6 - $4) / 3.92)}'
fi

# ── 4 critical-situation test at equal time ──────────────────────────────────────────────────────────────────────────
if [ -e "$STOP" ]; then echo "[q181] stop file before the theme test"; else
  while busy; do sleep 60; done
  TH="$(ls /mnt/e/chess_data/puzzles/themes/*.txt) $(ls /mnt/e/chess_data/puzzles/passer/*.txt)"; X="$L/wac.txt $L/ksattack.txt"
  echo "[q181] $(date) themes and passers at 0.25 s per move, arm (core 3) and ctl (core 5) together"
  bash "$R" pyrun diagnostics/_theme_movetime.py "$BIN" 3 0.25 "$KN" $TH $X > $L/themes_mt_m1000.txt 2>&1 &
  bash "$R" pyrun diagnostics/_theme_movetime.py "$BIN" 5 0.25 - $TH $X > $L/themes_mt_off3.txt 2>&1 &
  wait
  echo "[q181] theme                off   arm   (of n)"
  join <(sort $L/themes_mt_off3.txt) <(sort $L/themes_mt_m1000.txt) | awk '{d=$4-$2; flag=(d>14||d<-14)?"  <- flag":""; printf "%-26s %4d %4d  (%d)  %+d%s\n", $1, $2, $4, $3, d, flag; so+=$2; sa+=$4; n+=$3} END {printf "%-26s %4d %4d  (%d)  %+d\n", "TOTAL", so, sa, n, sa-so}' | show
fi
echo "[q181] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | show
echo "[q181] $(date) QUEUE 181 DONE"
