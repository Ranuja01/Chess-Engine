#!/bin/bash
# 2026-11-08 afternoon, queue #183 (owner 11-08: gate lazy eval the way search gates pruning, e.g. after non-quiet moves): the context
# probe. build_uci/chess_engine_uci_lzprobe2 (-DLAZY_PROBE; its records follow LAZY_V2_KEEP; new columns mv = the move that led to
# the position, kz = a capture within 2 squares of a king; default bench 246 / 36,509,830). Lazy itself off: the probe logs what the
# build's rule WOULD do at each site, against the full eval. Depth 10, 1 position in 16 by zobrist, records kept (LAZY_V2_KEEP=1),
# sets WAC / STS300 / ksattack / 400 game positions (step 0b's) + the 5 passed-pawn suites; WAC again with KEEP=0 as the check that
# the fix raises parent coverage. Fixed depth, after queue #182; three processes on cores 3 / 5 / 7.
# Read: diagnostics/_lazy_probe_read.py ONLY_CONTEXT=1 (section 7): fires and misses by site x last move, by depth, and priced gates
# (fires kept vs misses removed) at the candidate's margins (1000 / 2000) and build 1's (1500 / 3000).
# Registered before it runs: a gate is WORTH BUILDING if it removes at least 3x as large a share of the misses as of the fires
#   (e.g. 30% of misses for 10% of fires). Prediction (40-position smoke test, 11-08): stand-pat after a capture qualifies.
# Stop: touch /mnt/e/chess_data/q183.stop (checked between runs). Never edit this file while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
BIN="$ND/build_uci/chess_engine_uci_lzprobe2"
L=/mnt/e/chess_data/lazy
C=$L/c
P=/mnt/e/chess_data/puzzles/passer
STOP=/mnt/e/chess_data/q183.stop
Q182=/mnt/e/chess_data/q182.log
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"
busy() { pgrep -f "fastchess-linux-x86-64/fastchess|vs_sf.py|_depth_residual_pass|_build_regret_set|_retune_fit|_ks_footprint_regret|sts_test|tactical_test|_theme_movetime" >/dev/null; }
echo "[q183] $(date) waiting for queue 182's end line and a quiet box (stop: $STOP)"
until grep -qE 'QUEUE 182 (DONE|ABORTED)' "$Q182" 2>/dev/null && ! busy; do
  [ -e "$STOP" ] && { echo "[q183] stop file while waiting — QUEUE 183 DONE"; exit 0; }
  sleep 60
done
START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[q183] $START start"
B0=$(printf 'bench\nquit\n' | LAZY_V2_KEEP=1 "$BIN" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+')
echo "[q183] probe bench (KEEP=1): $B0"
[ "$B0" = "solved 246/300 nodes 36509830" ] || { echo "[q183] ☠️ the probe build's bench is not gen 7's — QUEUE 183 ABORTED"; exit 1; }
mkdir -p $C
probe() {  # core, keep, tag, suite file
  [ -e "$STOP" ] && return
  printf 'bench %s 10\nquit\n' "$4" | LAZY_V2_KEEP=$2 LAZY_PROBE_OUT=$C/$3.csv LAZY_PROBE_EVERY=16 taskset -c $1 "$BIN" 2>/dev/null \
    | grep -E '^bench' | sed "s/^/[q183]   $3: /" | cut -c1-110
}
cat $P/*.txt | grep '|' > $C/passers_all.txt
{ probe 3 1 lzc_sts300 $L/sts300.txt; probe 3 0 lzc0_wac $L/wac.txt; } &
{ probe 5 1 lzc_games $L/games.txt; probe 5 1 lzc_wac $L/wac.txt; } &
{ probe 7 1 lzc_ksattack $L/ksattack.txt; probe 7 1 lzc_passers $C/passers_all.txt; } &
wait
[ -e "$STOP" ] && { echo "[q183] stop file during the runs — QUEUE 183 DONE"; exit 0; }
echo "[q183] $(date) probe runs done: $(ls -s $C/*.csv | tr '\n' ' ')"
ONLY_CONTEXT=1 /home/ranuja/anaconda3/bin/python -u diagnostics/_lazy_probe_read.py $C/lzc_wac.csv $C/lzc_sts300.csv $C/lzc_ksattack.csv \
  $C/lzc_games.csv $C/lzc_passers.csv > $C/read_context.txt 2>&1
ONLY_CONTEXT=1 /home/ranuja/anaconda3/bin/python -u diagnostics/_lazy_probe_read.py $C/lzc_passers.csv > $C/read_context_passers.txt 2>&1
ONLY_CONTEXT=1 /home/ranuja/anaconda3/bin/python -u diagnostics/_lazy_probe_read.py $C/lzc0_wac.csv 2>&1 | grep -m1 'context probe' > $C/read_keep0_wac.txt
ONLY_CONTEXT=1 /home/ranuja/anaconda3/bin/python -u diagnostics/_lazy_probe_read.py $C/lzc_wac.csv 2>&1 | grep -m1 'context probe' >> $C/read_keep0_wac.txt
echo "[q183] parent coverage on WAC, KEEP=0 then KEEP=1:"; sed 's/^/[q183]   /' $C/read_keep0_wac.txt
echo "[q183] context read (all sets): $C/read_context.txt; passers only: $C/read_context_passers.txt"
sed 's/^/[q183]   /' $C/read_context.txt
echo "[q183] $(date) QUEUE 183 DONE"
