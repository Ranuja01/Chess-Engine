#!/bin/bash
# 2026-11-06, queue #172 (owner 11-06): data for stage 2 of the retune, criticality. The owner's definition: moves that can shift the game
# negatively and are not obvious, so danger or opportunity has to be judged. Critical = missing the best move costs >= 3 win%, and it is a
# quiet move, a sacrifice, a check that is not simply forced, or the clearly best of several replies to check (diagnostics/_critical_mine.py).
# Measured on the 136,759 slice-A positions with multi-PV labels: 18,321 critical (13.4%); our d10 misses 33% of the quiet ones, 40% of the
# sacrifices, 26% of the checks, 8% of the evasions. Only about 5,800 critical rows that we miss, so this adds two enriched sources:
#  1 mine: Lichess puzzles (quiet / defensive / sacrifice / zugzwang themes, no mate themes, rating 1800-3000, 15,000 per eval stage, 20% held out
#    by puzzle) and every collapse recorded in our own games (both ends, 15% held out by game). Nothing from a bench, a regret set or
#    corpus_v3: STS stays out of training, owner 11-06.
#  2 SF18 d14 multi-PV K=4 labels, 3 processes.
#  3 classify, then gen 6.7's d10 pass (ours1105, with the move) on the critical rows only, 3 processes, then the table again with miss rates.
# Runs beside #168 / #170 (node-limited) and the A2 pass. Everything is resumable. Hard stop 01:00 on 11-07, so the timed games of #171 are
# not delayed. Stop: touch /mnt/e/chess_data/q172.stop (between steps). Nothing here fits or plays. Never edit this file while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
D=/mnt/e/chess_data/retune
LOGD=/mnt/e/chess_data/q172
STOP=/mnt/e/chess_data/q172.stop
CUTOFF=$(date -d '2026-11-07 01:00' +%s)
TAG=ours1105
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"; mkdir -p "$LOGD"
show() { sed "s/^/[q172]   /"; }
late() { [ -e "$STOP" ] || [ $(date +%s) -ge $CUTOFF ]; }
echo "[q172] $(date) start (stop: $STOP)"

# ── 1 mine ───────────────────────────────────────────────────────────────────────────────────────────────────────────
if [ ! -s $D/CR_fens.csv ]; then
  bash "$R" pyrun diagnostics/_critical_mine.py MODE=mine 2>&1 | tee $LOGD/mine.log | grep -E 'excluded|puzzles|collapses|MINE|rror|Trace' | show
fi
[ -s $D/CR_fens.csv ] || { echo "[q172] ☠️ mining failed — QUEUE 172 ABORTED"; exit 1; }
N=$(grep -vc '^fen,' $D/CR_fens.csv)
for i in 0 1 2; do head -1 $D/CR_fens.csv > $D/CR_fens_s$i.csv; done
tail -n +2 $D/CR_fens.csv | awk -v B="$D/CR_fens" '{print >> (B "_s" (NR-1)%3 ".csv")}'
echo "[q172] $(date) $N positions to label"

# ── 2 labels ─────────────────────────────────────────────────────────────────────────────────────────────────────────
t0=$(date +%s)
for i in 0 1 2; do
  ( left=$(( CUTOFF - $(date +%s) )); [ $left -gt 60 ] || exit 0
    timeout $left bash "$R" pyrun diagnostics/_build_regret_set.py IN=$D/CR_fens_s$i.csv K=4 SF_DEPTH=14 OUT=$D/CR_sf18k4_s$i.csv > $LOGD/label_$i.log 2>&1
    echo "[q172] $(date) labels s$i: rc $? · $(grep -vc '^fen,' $D/CR_sf18k4_s$i.csv 2>/dev/null) rows" ) &
done
wait
echo "[q172] $(date) labelled $(cat $D/CR_sf18k4_s*.csv 2>/dev/null | grep -vc '^fen,') / $N in $(( ($(date +%s) - t0) / 60 )) min"

# ── 3 classes and the d10 pass ───────────────────────────────────────────────────────────────────────────────────────
bash "$R" pyrun diagnostics/_critical_mine.py MODE=classify 'GLOB=CR_sf18k4_s*.csv' META=CR_fens.csv OUT=CR_classes.csv 2>&1 | grep -E 'classified|source|critical|\*|rror|Trace' | show
[ -s $D/CR_classes.csv ] || { echo "[q172] ☠️ classification failed — QUEUE 172 ABORTED"; exit 1; }
job() {
  local rc
  while :; do
    timeout 3600 env V2_PRESET=shipped PRESET=LONG_FORMAT MAX_DEPTH=10 USE_OPENING_BOOK=0 RESIGN_THRESHOLD=-99999999 \
      bash "$R" pyrun diagnostics/_depth_residual_pass.py IN=$D/CR_classes.csv OUT=$D/CR_${TAG}_d10.csv SHARD=$1/3 CHUNK=800 MOVE=1 >> "$LOGD/d10_$1.log" 2>&1
    rc=$?
    [ $rc -ne 3 ] && break
    late && { echo "[q172] $(date) d10 s$1 paused at a chunk boundary (stop or cut-off)"; return; }
  done
  echo "[q172] $(date) d10 s$1 done (rc $rc)"
}
if ! late; then
  echo "[q172] $(date) d10 pass on $(grep -vc '^fen,' $D/CR_classes.csv) critical rows"
  job 0 & job 1 & job 2 & wait
fi
bash "$R" pyrun diagnostics/_critical_mine.py MODE=classify 'GLOB=CR_sf18k4_s*.csv' META=CR_fens.csv "D10=CR_${TAG}_d10_s*of3.csv" 2>&1 | grep -E 'classified|source|critical|\*|rror|Trace' | show
echo "[q172] $(date) QUEUE 172 DONE"
