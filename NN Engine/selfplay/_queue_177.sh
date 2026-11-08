#!/bin/bash
# 2026-11-07, queue #177 (owner 11-07: overnight runs once the owner is asleep): the first lazy eval build (LAZY_V2=7, commit 86fe007)
# against itself with lazy off, on build_uci/chess_engine_uci_lazy (no PGO; both arms the same binary). Starts at 00:45 on 11-08, or
# earlier if /mnt/e/chess_data/q177.go exists; no timed chunk starts after 10:30 on 11-08 (weekend afternoons stay free of timed games).
#  1 fingerprints: default 246 / 36,509,830 (gen 7), LAZY_V2=7 245 / 37,305,726.
#  2 pinned WAC bench, depth 10, off / on alternating x 4 on one core: time to depth.
#  3 timed self-play 10+0.1 UHO, arm = LAZY_V2=7 vs ctl, fresh seeds 177111-177114, 4 x 500 g, concurrency 3.
#  4 the critical-situation test at equal time: 18 Lichess theme suites (200 each) + WAC + ksattack, 0.25 s per move, arm and ctl at the
#    same time on two cores.
# Registered before any game (owner 11-07: lazy eval is infrastructure for search v2, judged "not harmful", not live or die):
#   KEEP AND DEVELOP if the pooled timed games are at or above -1 sigma; a gain is claimed only at +2 sigma. Below -1 sigma: the margins
#   and guards are reworked before any default changes. Themes: a theme moving by more than 14 of 200 (about 2 sigma) is flagged for a
#   look either way. Prediction: +5 to +15 Elo (depth 10 reached 7-12% sooner on STS, ksattack and game positions, WAC level), z 0.5-2.
# Nothing is installed. Stop: touch /mnt/e/chess_data/q177.stop (checked between steps and chunks). Never edit this file while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
L=/mnt/e/chess_data/lazy
BIN="$ND/build_uci/chess_engine_uci_lazy"
KN="LAZY_V2=7"
STOP=/mnt/e/chess_data/q177.stop
GO=/mnt/e/chess_data/q177.go
EARLIEST=$(date -d '2026-11-08 00:45' +%s)
TIMED_LATEST=$(date -d '2026-11-08 10:30' +%s)
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"
show() { sed "s/^/[q177]   /"; }
busy() { pgrep -f "fastchess-linux-x86-64/fastchess|vs_sf.py|_depth_residual_pass|_build_regret_set|_retune_fit|_ks_footprint_regret|sts_test|tactical_test|_theme_movetime" >/dev/null; }
give_up() { [ -e "$STOP" ] || [ $(date +%s) -gt $TIMED_LATEST ]; }
echo "[q177] $(date) waiting for 00:45 (or $GO) and a quiet box (stop: $STOP)"
until { [ $(date +%s) -ge $EARLIEST ] || [ -e "$GO" ]; } && ! busy; do
  give_up && { echo "[q177] $(date) stop file or past 10:30 before starting — QUEUE 177 DONE (nothing run)"; exit 0; }
  sleep 60
done
sleep 30
START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[q177] $START start"

# ── 1 fingerprints ───────────────────────────────────────────────────────────────────────────────────────────────────
B0=$(printf 'bench\nquit\n' | "$BIN" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+')
B7=$(printf 'setoption name Knobs value %s\nbench\nquit\n' "$KN" | "$BIN" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+')
echo "[q177] bench default: $B0 · $KN: $B7"
[ "$B0" = "solved 246/300 nodes 36509830" ] || { echo "[q177] ☠️ default bench is not gen 7's — QUEUE 177 ABORTED"; exit 1; }
[ "$B7" = "solved 245/300 nodes 37305726" ] || { echo "[q177] ☠️ the lazy arm's bench is not 245 / 37,305,726 (knob not reaching it?) — QUEUE 177 ABORTED"; exit 1; }

# ── 2 pinned bench ───────────────────────────────────────────────────────────────────────────────────────────────────
for i in 1 2 3 4; do
  for arm in off on; do
    k=""; [ $arm = on ] && k="setoption name Knobs value $KN"
    echo "[q177] pinned WAC d10 $arm: $(printf '%s\nbench\nquit\n' "$k" | taskset -c 5 "$BIN" 2>/dev/null | grep -E '^bench' | grep -oE 'time [0-9]+ ms nps [0-9]+')"
  done
done

# ── 3 timed self-play ────────────────────────────────────────────────────────────────────────────────────────────────
rm -f /tmp/q177_pre.engine.log
timeout 180 "$FC" -engine "cmd=$BIN" name=arm option.OwnBook=false "option.Knobs=$KN" st=0.05 \
  -engine "cmd=$BIN" name=ctl option.OwnBook=false st=0.05 \
  -openings file=$O/openings_uho.epd format=epd order=random -rounds 1 -games 2 -concurrency 1 \
  -log file=/tmp/q177_pre.engine.log level=info engine=true > /tmp/q177_pre.log 2>&1
W=$(grep -c 'have option' /tmp/q177_pre.log)
echo "[q177] preflight: $(grep -E '^Games:' /tmp/q177_pre.log | tail -1) · warnings $W · arm echoes $(grep -c 'knobs LAZY_V2=7' /tmp/q177_pre.engine.log)"
[ "$W" = 0 ] || { echo "[q177] ☠️ preflight warnings — QUEUE 177 ABORTED"; exit 1; }
sleep 20
COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
T=lz7_10p1
rm -f $O/${T}_c[1-4].pgn $O/${T}_c[1-4].log
for c in 1 2 3 4; do
  while busy && ! give_up; do sleep 60; done
  give_up && { echo "[q177] $(date) skip $T chunk $c (stop or past 10:30)"; continue; }
  nice -n 5 "$FC" -engine "cmd=$BIN" name=arm option.OwnBook=false "option.Knobs=$KN" tc=10+0.1 timemargin=250 \
    -engine "cmd=$BIN" name=ctl option.OwnBook=false tc=10+0.1 timemargin=250 \
    $COMMON -srand $(( 177110 + c )) -rounds 250 -games 2 \
    -pgnout file=$O/${T}_c$c.pgn notation=san nodes=true append=false > $O/${T}_c$c.log 2>&1
  echo "[q177] $(date) $T chunk $c: $(grep -E '^Elo:' $O/${T}_c$c.log | tail -1) · warnings $(grep -c 'have option' $O/${T}_c$c.log) · disconnects/stalls $(grep -ciE 'disconnect|stall|not responsive' $O/${T}_c$c.log) · time forfeits $(grep -c 'loses on time' $O/${T}_c$c.pgn)"
done
f=$(ls $O/${T}_c[1-4].pgn 2>/dev/null)
if [ -n "$f" ]; then
  out=$(timeout 900 bash "$R" pyrun selfplay/fastchess_report.py $f --p1 arm 2>&1 | sed -n '/vs the field/,$p')
  echo "[q177] $(date) REGISTERED READ, lazy on vs off, all chunks (keep at >= -1 sigma, gain at +2 sigma):"
  echo "$out" | show
  echo "$out" | grep -m1 -E 'Elo [-+]?[0-9.]+' | awk '{gsub(/[()]/,""); gsub(/\+/,""); printf "[q177]   z = %.2f\n", $2 / (($6 - $4) / 3.92)}'
fi

# ── 4 critical-situation test at equal time ──────────────────────────────────────────────────────────────────────────
if [ -e "$STOP" ]; then echo "[q177] stop file before the theme test"; else
  while busy; do sleep 60; done
  TH=$(ls /mnt/e/chess_data/puzzles/themes/*.txt) ; X="$L/wac.txt $L/ksattack.txt"
  echo "[q177] $(date) themes at 0.25 s per move, arm (core 3) and ctl (core 5) together"
  bash "$R" pyrun diagnostics/_theme_movetime.py "$BIN" 3 0.25 "$KN" $TH $X > $L/themes_mt_on.txt 2>&1 &
  bash "$R" pyrun diagnostics/_theme_movetime.py "$BIN" 5 0.25 - $TH $X > $L/themes_mt_off.txt 2>&1 &
  wait
  echo "[q177] theme                off   on   (of n)"
  join <(sort $L/themes_mt_off.txt) <(sort $L/themes_mt_on.txt) | awk '{d=$4-$2; flag=(d>14||d<-14)?"  <- flag":""; printf "%-20s %4d %4d  (%d)  %+d%s\n", $1, $2, $4, $3, d, flag}' | show
fi
echo "[q177] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | show
echo "[q177] $(date) QUEUE 177 DONE"
