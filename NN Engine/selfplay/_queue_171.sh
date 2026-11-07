#!/bin/bash
# 2026-11-06, queue #171 (owner 11-06: fine to queue overnight): the one extension of the timed check. S came in at +11.1 (-2.7 ... +25.0) over
# 2,000 g at 10+0.1 (#167), under +2 sigma (about +14). Starts when #168 (with the A2 pass), #169 and #170 have ended, never before 00:05.
#  1 which arm: S, unless S x 2 or S x 4 beat S by at least +14 at 50k nodes (4,000 fresh-seed g each, #168 and #170; the larger if both).
#  2 TIMED self-play, 10+0.1 UHO, arm vs ctl on chess_engine_uci_g67pgo, fresh seeds 171111-171114, 4 x 500 g.
# Registered, fixed before these games: the arm must reach +2 sigma on all its timed games combined. For S that is these 2,000 g pooled with
# #167's (about +10 on 4,000 g); for S x 2 or S x 4 these 2,000 g alone (about +14). No second extension. Extending only because S fell just
# short slightly favours it, which is why this happens once. Nothing is installed; a ship needs the owner, and SF18 and Mediocre would still
# have to be run for S x 2 or S x 4 (S has passed both).
# No chunk starts after 10:30 on 11-07 (weekend afternoons stay free of timed games). Stop: touch /mnt/e/chess_data/q171.stop (between
# steps and chunks). Never edit this file while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
BIN="$ND/build_uci/chess_engine_uci_g67pgo"
STOP=/mnt/e/chess_data/q171.stop
EARLIEST=$(date -d '2026-11-07 00:05' +%s)
TIMED_LATEST=$(date -d '2026-11-07 10:30' +%s)
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"
show() { sed "s/^/[q171]   /"; }
busy() { pgrep -f "fastchess-linux-x86-64/fastchess|vs_sf.py|_depth_residual_pass|_build_regret_set|_px_export|_retune_fit|_reference_ceiling|sts_test|tactical_test" >/dev/null; }
give_up() { [ -e "$STOP" ] || [ $(date +%s) -gt $TIMED_LATEST ]; }
over() { grep -qE "QUEUE $1 (DONE|ABORTED)" /mnt/e/chess_data/q$1.log 2>/dev/null; }
echo "[q171] $(date) waiting for #168, #169 and #170, then 00:05 and a quiet box (stop: $STOP)"
until over 168 && over 169 && over 170 && [ $(date +%s) -ge $EARLIEST ] && ! busy; do
  give_up && { echo "[q171] $(date) stop file or past 10:30 before starting — QUEUE 171 DONE (nothing run)"; exit 0; }
  sleep 60
done
sleep 60
START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[q171] $START start"
UB=$(printf 'bench\nquit\n' | "$BIN" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+')
[ "$UB" = "solved 251/300 nodes 36859850" ] || { echo "[q171] ☠️ g67pgo bench is '$UB' — QUEUE 171 ABORTED"; exit 1; }

# ── 1 which arm ──────────────────────────────────────────────────────────────────────────────────────────────────────
elo() {   # pooled Elo of the arm over the PGNs of one tag, or nothing unless all 8 chunks are there
  local f; f=$(ls $O/lcg_$1_c[1-8].pgn 2>/dev/null)
  [ "$(echo "$f" | wc -w)" = 8 ] || return
  timeout 900 bash "$R" pyrun selfplay/fastchess_report.py $f --p1 arm 2>&1 | grep -m1 -oE 'Elo [-+]?[0-9.]+' | awk '{print $2}'
}
ES=$(elo s); E2=$(elo v3x2); E4=$(elo v3x4)
echo "[q171] 50k nodes, 4,000 fresh g each: S ${ES:-?} · S x 2 ${E2:-?} · S x 4 ${E4:-?}"
ARM=v3s; KF=/mnt/e/chess_data/q166/knobs_v3s.txt
if [ -n "$ES" ]; then
  P=$(awk -v s="$ES" -v a="${E2:-}" -v b="${E4:-}" 'BEGIN{ x=-1e9; c="v3s"; if (a!="" && a+0>=s+14) {x=a+0; c="v3x2"} if (b!="" && b+0>=s+14 && b+0>x) {c="v3x4"} print c }')
  if [ "$P" != v3s ] && [ -s /mnt/e/chess_data/q170/knobs_$P.txt ]; then ARM=$P; KF=/mnt/e/chess_data/q170/knobs_$P.txt; fi
fi
KN=$(cat "$KF")
[ -n "$KN" ] || { echo "[q171] ☠️ no knobs for $ARM — QUEUE 171 ABORTED"; exit 1; }
echo "[q171] arm for the timed games: $ARM · bench under its knobs: $(printf 'setoption name Knobs value %s\nbench\nquit\n' "$KN" | "$BIN" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+')"

# ── 2 timed self-play ────────────────────────────────────────────────────────────────────────────────────────────────
timed_ok() { ! give_up && ! busy; }
rm -f /tmp/q171_pre.engine.log
timeout 180 "$FC" -engine "cmd=$BIN" name=arm option.OwnBook=false "option.Knobs=$KN" st=0.05 \
  -engine "cmd=$BIN" name=ctl option.OwnBook=false st=0.05 \
  -openings file=$O/openings_uho.epd format=epd order=random -rounds 1 -games 2 -concurrency 1 \
  -log file=/tmp/q171_pre.engine.log level=info engine=true > /tmp/q171_pre.log 2>&1
W=$(grep -c 'have option' /tmp/q171_pre.log)
echo "[q171] preflight: $(grep -E '^Games:' /tmp/q171_pre.log | tail -1) · warnings $W · arm echoes $(grep -c 'knobs C1_V2_FIT=1' /tmp/q171_pre.engine.log)"
[ "$W" = 0 ] || { echo "[q171] ☠️ preflight warnings — QUEUE 171 ABORTED"; exit 1; }
sleep 20
COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
T=rt_${ARM}_10p1b
rm -f $O/${T}_c[1-4].pgn $O/${T}_c[1-4].log
for c in 1 2 3 4; do
  # Wait for a quiet box here: skipping on busy would lose every remaining chunk at once.
  while busy && ! give_up; do sleep 60; done
  timed_ok || { echo "[q171] $(date) skip $T chunk $c (stop, late or busy)"; continue; }
  nice -n 5 "$FC" -engine "cmd=$BIN" name=arm option.OwnBook=false "option.Knobs=$KN" tc=10+0.1 timemargin=250 \
    -engine "cmd=$BIN" name=ctl option.OwnBook=false tc=10+0.1 timemargin=250 \
    $COMMON -srand $(( 171110 + c )) -rounds 250 -games 2 \
    -pgnout file=$O/${T}_c$c.pgn notation=san nodes=true append=false > $O/${T}_c$c.log 2>&1
  echo "[q171] $(date) $T chunk $c: $(grep -E '^Elo:' $O/${T}_c$c.log | tail -1) · warnings $(grep -c 'have option' $O/${T}_c$c.log) · disconnects/stalls $(grep -ciE 'disconnect|stall|not responsive' $O/${T}_c$c.log) · time forfeits $(grep -c 'loses on time' $O/${T}_c$c.pgn)"
done
f=$(ls $O/${T}_c[1-4].pgn 2>/dev/null)
rep() {
  local out; out=$(timeout 900 bash "$R" pyrun selfplay/fastchess_report.py "$@" --p1 arm 2>&1 | sed -n '/vs the field/,$p')
  echo "$out" | show
  echo "$out" | grep -m1 -E 'Elo [-+]?[0-9.]+' | awk '{gsub(/[()]/,""); gsub(/\+/,""); printf "[q171]   z = %.2f\n", $2 / (($6 - $4) / 3.92)}'
}
if [ -n "$f" ]; then
  echo "[q171] $(date) $T, these games:"; rep $f
  if [ "$ARM" = v3s ]; then
    echo "[q171] $(date) REGISTERED READ, S on all its timed games (with #167), bar +2 sigma:"; rep $f $O/rt_v3s_10p1_c[1-4].pgn
  else
    echo "[q171] REGISTERED READ, $ARM: the games above, bar +2 sigma."
  fi
fi
echo "[q171] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | show
echo "[q171] $(date) QUEUE 171 DONE"
