#!/bin/bash
# 2026-11-08 evening, queue #184 (owner 11-08 "sure" to running the SF18 part of the ship gate tonight): the lazy eval candidate
# keep_m1000 and lazy off, each against SF18 at 1,100 nodes per move, ours at 250,000 nodes (the ship rule's SF18 anchor, memory
# the-sf18-gauntlet-anchor-drifted-too-weak). Node-limited (fastchess `go nodes`, tc=inf; checked 11-08), so safe in the owner's
# evening. Through fastchess on the UCI binary build_uci/chess_engine_uci_lazy2 because the Cython module predates the lazy code.
# Equal nodes remove the lazy speed, so this reads the lazy candidate's accuracy cost alone: a conservative gate.
#  After queue #183. Four chunks of 250 games, ABBA (ctl, arm, arm, ctl), seeds 184111 (chunks 1-2) and 184112 (3-4): each arm plays
#  the same openings. Concurrency 3. No chunk starts after 23:30 on 11-08 (queue #181 starts at 00:45 and waits for a quiet box).
# Registered before it runs: PASS (the SF18 leg of the ship rule) if the arm's score vs SF18 minus the control's is above -1 sigma.
#   Prediction: within +-1 sigma of 0 (at equal nodes the #180 suites put keep_m1000 level with off).
# Stop: touch /mnt/e/chess_data/q184.stop (checked between chunks). Never edit this file while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
SF18="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/stockfish_18_linux/stockfish-ubuntu-x86-64-avx2"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
BIN="$ND/build_uci/chess_engine_uci_lazy2"
KN="LAZY_V2=7,LAZY_V2_KEEP=1,LAZY_V2_M=1000,LAZY_V2_M_CHEAP=2000"
STOP=/mnt/e/chess_data/q184.stop
Q183=/mnt/e/chess_data/q183.log
LATEST=$(date -d '2026-11-08 23:30' +%s)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"
show() { sed "s/^/[q184]   /"; }
busy() { pgrep -f "fastchess-linux-x86-64/fastchess|vs_sf.py|_depth_residual_pass|_build_regret_set|_retune_fit|_ks_footprint_regret|sts_test|tactical_test|_theme_movetime|chess_engine_uci_lzprobe2" >/dev/null; }
echo "[q184] $(date) waiting for queue 183's end line and a quiet box (stop: $STOP)"
until grep -qE 'QUEUE 183 (DONE|ABORTED)' "$Q183" 2>/dev/null && ! busy; do
  [ -e "$STOP" ] && { echo "[q184] stop file while waiting — QUEUE 184 DONE"; exit 0; }
  [ $(date +%s) -gt $LATEST ] && { echo "[q184] past 23:30 before starting — QUEUE 184 DONE (nothing run)"; exit 0; }
  sleep 60
done
START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[q184] $START start"
B0=$(printf 'bench\nquit\n' | "$BIN" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+')
BA=$(printf 'setoption name Knobs value %s\nbench\nquit\n' "$KN" | "$BIN" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+')
echo "[q184] bench default: $B0 · arm: $BA"
[ "$B0" = "solved 246/300 nodes 36509830" ] && [ "$BA" = "solved 249/300 nodes 37197972" ] || { echo "[q184] ☠️ fingerprints differ — QUEUE 184 ABORTED"; exit 1; }
COMMON="-each tc=inf -openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
rm -f /tmp/q184_pre.engine.log
timeout 300 "$FC" -engine "cmd=$BIN" name=ours option.OwnBook=false "option.Knobs=$KN" nodes=250000 \
  -engine "cmd=$SF18" name=sf18 option.Threads=1 option.Hash=16 nodes=1100 \
  -each tc=inf -openings file=$O/openings_uho.epd format=epd order=random -rounds 1 -games 2 -concurrency 1 \
  -log file=/tmp/q184_pre.engine.log level=info engine=true > /tmp/q184_pre.log 2>&1
W=$(grep -c 'have option' /tmp/q184_pre.log)
echo "[q184] preflight: $(grep -E '^Games:' /tmp/q184_pre.log | tail -1) · warnings $W · arm echoes $(grep -cF "knobs $KN" /tmp/q184_pre.engine.log) · go lines $(grep -cE 'go nodes (250000|1100)' /tmp/q184_pre.engine.log)"
[ "$W" = 0 ] || { echo "[q184] ☠️ preflight warnings — QUEUE 184 ABORTED"; exit 1; }
T=g18_lzm
chunk() {  # n, arm (arm|ctl), seed
  local c=$1 a=$2 seed=$3 k=""
  [ -e "$STOP" ] && { echo "[q184] stop file before chunk $c"; return; }
  [ $(date +%s) -gt $LATEST ] && { echo "[q184] $(date) skip chunk $c (past 23:30)"; return; }
  [ $a = arm ] && k="option.Knobs=$KN"
  nice -n 5 "$FC" -engine "cmd=$BIN" name=ours option.OwnBook=false $k nodes=250000 \
    -engine "cmd=$SF18" name=sf18 option.Threads=1 option.Hash=16 nodes=1100 \
    $COMMON -srand $seed -rounds 125 -games 2 -pgnout file=$O/${T}_c${c}_$a.pgn notation=san nodes=true append=false > $O/${T}_c${c}_$a.log 2>&1
  echo "[q184] $(date) chunk $c ($a, seed $seed): $(grep -E '^Games:' $O/${T}_c${c}_$a.log | tail -1) · warnings $(grep -c 'have option' $O/${T}_c${c}_$a.log) · disconnects/stalls $(grep -ciE 'disconnect|stall|not responsive' $O/${T}_c${c}_$a.log)"
}
rm -f $O/${T}_c*
chunk 1 ctl 184111; chunk 2 arm 184111; chunk 3 arm 184112; chunk 4 ctl 184112
echo "[q184] $(date) REGISTERED READ, score vs SF18 @ 1,100 nodes, arm minus control (pass above -1 sigma):"
for a in ctl arm; do
  for f in $O/${T}_c*_$a.log; do [ -e "$f" ] && grep -E '^Games:' "$f" | tail -1; done | awk -v a=$a '{
      gsub(/,/, ""); for (i = 1; i <= NF; i++) { if ($i == "Games:") n += $(i+1); if ($i == "Wins:") w += $(i+1); if ($i == "Losses:") l += $(i+1); if ($i == "Draws:") d += $(i+1) } }
    END { s = (n > 0 ? (w + d / 2) / n : 0); v = (n > 0 ? (w + d / 4) / n - s * s : 0); printf "%s %d %d %d %d %.6f %.8f\n", a, n, w, l, d, s, (n > 0 ? v / n : 0) }'
done > /tmp/q184_read.txt
awk '{ printf "[q184]   %-4s games %4d  +%d -%d =%d  score %.1f%%  (se %.1f pp)\n", $1, $2, $3, $4, $5, 100 * $6, 100 * sqrt($7) }' /tmp/q184_read.txt
awk '{ s[$1] = $6; v[$1] = $7 } END { d = s["arm"] - s["ctl"]; se = sqrt(v["arm"] + v["ctl"]); printf "[q184]   arm - ctl %+.2f pp  ± %.2f (95%%)  z %+.2f  -> %s\n", 100 * d, 196 * se, (se > 0 ? d / se : 0), (se > 0 && d / se > -1 ? "PASS" : "FAIL") }' /tmp/q184_read.txt
echo "[q184] $(date) QUEUE 184 DONE"
