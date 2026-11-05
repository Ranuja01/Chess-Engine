#!/bin/bash
# 2026-11-05 afternoon, queue #162: the overnight tail of the retune data build, after #160 (gen 6.7's d10 pass on A / fitC /
# EGX) AND #161 (the new VARG / ODDS / STRIP slices, SF18-labelled). Gen 6.7's d10 depth pass (tag ours1105, same settings as
# #160) on the three new slices first (40k rows: they join the depth target and the move-agreement checks), then GAUNT (97k,
# our games vs stronger opponents, never had a d10 base). Pool of 3 processes. No new shard after 2026-11-06 09:00; running
# shards stop at their next chunk boundary then (resumable with the same command). Fixed depth, safe any hour.
# Stop: touch /mnt/e/chess_data/q162.stop. Never edit this file while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
D=/mnt/e/chess_data/retune
LOGD=/mnt/e/chess_data/q162
STOP=/mnt/e/chess_data/q162.stop
CUTOFF=$(date -d '2026-11-06 09:00' +%s)
TAG=ours1105
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"; mkdir -p "$LOGD"
echo "[q162] $(date) waiting for queues #160 and #161 (stop: $STOP)"
until grep -qE 'QUEUE 160 (DONE|ABORTED)' /mnt/e/chess_data/q160.log 2>/dev/null && grep -qE 'QUEUE 161 (DONE|ABORTED)' /mnt/e/chess_data/q161.log 2>/dev/null; do
  [ -e "$STOP" ] && { echo "[q162] stop file present — QUEUE 162 DONE (nothing run)"; exit 0; }
  sleep 60
done
sleep 20
pgrep -f "fastchess-linux-x86-64/fastchess|vs_sf.py|_depth_residual_pass|_build_regret_set" >/dev/null && { echo "[q162] a match, depth pass or labeller is running — QUEUE 162 ABORTED"; exit 1; }
START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[q162] $START start"
grep -E 'QUEUE 161 (DONE|ABORTED)|labelled:' /mnt/e/chess_data/q161.log | sed 's/^/[q162]   #161: /'
FP=$(env V2_PRESET=shipped bash "$R" wac fp_g67_q162 2>&1 | grep -E "^SOLVED|^NODES|^EBF" | tr '\n' ' ')
echo "[q162] Cython v2 shipped (want gen 6.7 251 / 36859850): $FP"
echo "$FP" | grep -q "Solved 251/300 NODES: 36859850 " || { echo "[q162] ☠️ FINGERPRINT MISMATCH — QUEUE 162 ABORTED"; exit 1; }

late() { [ -e "$STOP" ] || [ $(date +%s) -ge $CUTOFF ]; }
job() {   # $1 name, $2 IN, $3 OUT, $4 shard k/n
  local t0=$(date +%s) rc
  while :; do
    timeout 3600 env V2_PRESET=shipped PRESET=LONG_FORMAT MAX_DEPTH=10 USE_OPENING_BOOK=0 RESIGN_THRESHOLD=-99999999 \
      bash "$R" pyrun diagnostics/_depth_residual_pass.py IN="$2" OUT="$3" SHARD=$4 CHUNK=800 MOVE=1 >> "$LOGD/$1.log" 2>&1
    rc=$?
    [ $rc -ne 3 ] && break
    late && { echo "[q162] $(date) $1 paused at a chunk boundary (stop or cut-off)"; return; }
  done
  echo "[q162] $(date) $1 done (rc $rc) in $(( ($(date +%s) - t0) / 60 )) min"
}
JOBS=()
for s in VARG ODDS STRIP; do
  for i in 0 1; do
    [ -s $D/${s}_fens_sf18_s$i.csv ] || continue
    for k in 0 1; do JOBS+=("${s}_f${i}_k$k|$D/${s}_fens_sf18_s$i.csv|$D/${s}_${TAG}_d10_f$i.csv|$k/2"); done
  done
done
for i in 0 1 2 3; do
  for k in 0 1 2; do JOBS+=("GAUNT_f${i}_k$k|$D/GAUNT_fens_sf18_s$i.csv|$D/GAUNT_${TAG}_d10_f$i.csv|$k/3"); done
done
echo "[q162] ${#JOBS[@]} jobs"
for j in "${JOBS[@]}"; do
  IFS='|' read -r name in out shard <<< "$j"
  late && { echo "[q162] $(date) not starting $name: stop or cut-off"; continue; }
  while [ $(jobs -rp | wc -l) -ge 3 ]; do wait -n; done
  echo "[q162] $(date) start $name ($shard)"
  job "$name" "$in" "$out" "$shard" &
done
wait
rows() { cat "$@" 2>/dev/null | grep -vc '^fen,'; }
echo "[q162] $(date) rows: VARG $(rows $D/VARG_${TAG}_d10_f*.csv) · ODDS $(rows $D/ODDS_${TAG}_d10_f*.csv) · STRIP $(rows $D/STRIP_${TAG}_d10_f*.csv) · GAUNT $(rows $D/GAUNT_${TAG}_d10_f*.csv) / 96753"
echo "[q162] $(date) QUEUE 162 DONE"
