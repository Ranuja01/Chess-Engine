#!/bin/bash
# 2026-11-04 evening, queue #151 (owner 11-04: "for the gen 6.6 when showcasing the candidate, get the bench numbers and NPS to
# go with it too"): the bench report for gen 6.5 (production) and the gen 6.6 candidate (the mate-score bundle FIX_TT_MATE +
# FIX_MDP + ENABLE_NULL_MATE_CLAMP), after #150's match. Template: _queue_117.sh (gen 6.4's report). ONE process at a time
# (timed benches need a quiet box). The static eval is unchanged since gen 6.4 (6.5 and the bundle are search-only), so the
# static ladder is not re-run. Never edit this file while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
TTM="$ND/build_uci/chess_engine_uci_ttm"
STOP=/mnt/e/chess_data/q151.stop
G65="V2_PRESET=shipped"
CAND="V2_PRESET=shipped FIX_TT_MATE=1 FIX_MDP=1 ENABLE_NULL_MATE_CLAMP=1"
CAND_UCI=FIX_TT_MATE=1,FIX_MDP=1,ENABLE_NULL_MATE_CLAMP=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"
echo "[q151] $(date) waiting for queue #150 (stop: $STOP)"
until grep -qE 'QUEUE 150 (DONE|ABORTED)' /mnt/e/chess_data/q150.log 2>/dev/null; do
  [ -e "$STOP" ] && { echo "[q151] stop file present — QUEUE 151 DONE (nothing run)"; exit 0; }
  sleep 60
done
sleep 30
pgrep -f "fastchess-linux-x86-64/fastchess|vs_sf.py" >/dev/null && { echo "[q151] a match is running — QUEUE 151 ABORTED"; exit 1; }
show() { sed "s/^/[q151]   /"; }
echo "[q151] $(date) start"

echo "[q151] $(date) == Cython module rebuild (carries FIX_TT_MATE, default off)"
bash "$R" build 2>&1 | grep -iE 'error|copying' | tail -n 3 | show
echo "[q151] == fingerprints (WAC d10)"
for a in "g65|$G65" "cand|$CAND"; do echo "[q151] ${a%%|*}: $(bash "$R" wac fp_${a%%|*} ${a#*|} | grep -E '^SOLVED|^NODES|^EBF' | tr '\n' ' ')"; done
echo "[q151] v1: $(bash "$R" wac fp_v1 | grep -E '^SOLVED|^NODES|^EBF' | tr '\n' ' ')"

echo "[q151] $(date) == TIMED: WAC at LIGHTNING (depth reached per move)"
for a in "g65|$G65" "cand|$CAND"; do echo "[q151] ${a%%|*}:"; bash "$R" wac_timed_depth rect_${a%%|*} ${a#*|} 2>&1 | grep -vE '^\s*$' | show; done
echo "[q151] $(date) == TIMED: Cython WAC speed, pinned core, 3 reps (MAX_NPS = the measurement)"
for a in "g65|$G65" "cand|$CAND"; do
  echo "[q151] ${a%%|*}: $(bash "$R" wac_speed recs_${a%%|*} 3 ${a#*|} 2>&1 | grep -E 'NODES:|MAX_NPS|MEDIAN|SPREAD' | tr '\n' ' ')"
done
echo "[q151] $(date) == TIMED: UCI bench, plain build with the bundle off vs on, pinned, alternating x6"
for i in 1 2 3 4 5 6; do
  echo "[q151]   off  $(printf 'bench\nquit\n' | taskset -c 5 "$TTM" 2>/dev/null | grep '^bench')"
  echo "[q151]   cand $(printf "setoption name Knobs value $CAND_UCI\nbench\nquit\n" | taskset -c 5 "$TTM" 2>/dev/null | grep '^bench')"
done

echo "[q151] $(date) == STS300 and full STS at d10"
for a in "g65|$G65" "cand|$CAND"; do
  echo "[q151] ${a%%|*} STS300 d10: $(bash "$R" sts rec_${a%%|*} ${a#*|} | tail -1)"
  echo "[q151] ${a%%|*} STS full d10: $(bash "$R" sts_full recf_${a%%|*} ${a#*|} | grep 'STS score' | tail -1)"
done
echo "[q151] $(date) == EQUAL NODES (249,014 per position; the REFERENCE-BENCH-LADDER instrument)"
for a in "g65|$G65" "cand|$CAND"; do
  echo "[q151] ${a%%|*} STS300 @249k: $(bash "$R" sts sts_eqn_${a%%|*} ${a#*|} MAX_DEPTH=64 NODE_LIMIT=249014 | tail -1)"
  echo "[q151] ${a%%|*} WAC @249k: $(bash "$R" wac wac_eqn_${a%%|*} ${a#*|} MAX_DEPTH=64 NODE_LIMIT=249014 | grep -E '^SOLVED|^NODES' | tr '\n' ' ')"
done
echo "[q151] $(date) QUEUE 151 DONE"
