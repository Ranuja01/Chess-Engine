#!/bin/bash
# 2026-11-05 ~10:30, queue #158: a clean repeat of two timed rows of #157's bench report. A disk-size walk of selfplay/games ran
# on Windows from about 10:15 to 10:21 while #157 timed gen 6.7's WAC at LIGHTNING and both Cython WAC speed runs, so those
# rows are disturbed. After #157 (quiet box, one process at a time): WAC at LIGHTNING and Cython WAC speed (pinned, 3 reps)
# for gen 6.5 and gen 6.7, order alternated. The selfplay/games archive waits for this queue. Never edit this file while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
STOP=/mnt/e/chess_data/q158.stop
G65="V2_PRESET=shipped ENABLE_TT_FLAG_FIX=0"
G67="V2_PRESET=shipped"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"
show() { sed "s/^/[q158]   /"; }
echo "[q158] $(date) waiting for queue #157 (stop: $STOP)"
until grep -qE 'QUEUE 157 (DONE|ABORTED)' /mnt/e/chess_data/q157.log 2>/dev/null; do
  [ -e "$STOP" ] && { echo "[q158] stop file present — QUEUE 158 DONE (nothing run)"; exit 0; }
  sleep 30
done
sleep 20
pgrep -f "fastchess-linux-x86-64/fastchess|vs_sf.py" >/dev/null && { echo "[q158] a match is running — QUEUE 158 ABORTED"; exit 1; }
echo "[q158] $(date) start"
echo "[q158] $(date) == TIMED: WAC at LIGHTNING (depth reached per move), gen 6.7 first this time"
for a in "g67|$G67" "g65|$G65"; do echo "[q158] ${a%%|*}:"; bash "$R" wac_timed_depth rect2_${a%%|*} ${a#*|} 2>&1 | grep -vE '^\s*$' | show; done
echo "[q158] $(date) == TIMED: Cython WAC speed, pinned core, 3 reps (MAX_NPS = the measurement)"
for a in "g67|$G67" "g65|$G65"; do
  echo "[q158] ${a%%|*}: $(bash "$R" wac_speed recs2_${a%%|*} 3 ${a#*|} 2>&1 | grep -E 'NODES:|MAX_NPS|MEDIAN|SPREAD' | tr '\n' ' ')"
done
echo "[q158] $(date) QUEUE 158 DONE"
