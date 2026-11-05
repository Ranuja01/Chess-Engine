#!/bin/bash
# 2026-11-05 afternoon, queue #160: the retune's first step (close-out order, owner 11-04; agenda 11-05) — our d10 DEPTH PASS
# of the current ship (gen 6.7, 251 / 36,859,850) on the SF18-labelled fit sets. The newest bases are gen 3 (A, EGX, 10-12/15)
# and gen 1 (fitC, 10-04); none exists for gen 4-6.x. Tag ours1105. Fixed depth (safe any hour, incl. the owner's evening
# games). After #159. At most 3 processes (live server). Jobs, in priority order, from a pool of 3:
#  slice A  (140k, the retune fit's depth target, _retune_fit.py D10_GLOB=)        12 shards  A_ours1105_d10_s*of12.csv
#  fitC mg / eg (14.8k / 19.8k, the per-subsystem fit scripts' OURS=ours1105)     4 shards   (their globs need s*of4)
#  EGX (60k endgame rows, the POT tools)                                           6 shards   EGX_ours1105_d10_s*of6.csv
# Settings as the later passes: RESIGN_THRESHOLD=-99999999 (no silently dropped decisive rows), timeout 3600 per CHUNK=800.
# MOVE=1 also writes our d10 best move (an extra column; every reader takes fen / ours_cp_white by name) for move-agreement checks.
# Resumable: a stop leaves partial shards that the same command completes later. No new shard starts after 2026-11-06
# 07:30, and running shards stop at their next chunk boundary then. Pause before the start: touch /mnt/e/chess_data/q160.pause.
# Stop: touch /mnt/e/chess_data/q160.stop. Never edit this file while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
D=/mnt/e/chess_data/retune
K=diagnostics/ks_sets
LOGD=/mnt/e/chess_data/q160
STOP=/mnt/e/chess_data/q160.stop
PAUSE=/mnt/e/chess_data/q160.pause
CUTOFF=$(date -d '2026-11-06 07:30' +%s)
TAG=ours1105
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"; mkdir -p "$LOGD"
echo "[q160] $(date) waiting for queue #159 (stop: $STOP, pause: $PAUSE)"
until grep -qE 'QUEUE 159 (DONE|ABORTED)' /mnt/e/chess_data/q159.log 2>/dev/null; do
  [ -e "$STOP" ] && { echo "[q160] stop file present — QUEUE 160 DONE (nothing run)"; exit 0; }
  sleep 30
done
if [ -e "$PAUSE" ]; then
  echo "[q160] $(date) paused (remove $PAUSE to start)"
  while [ -e "$PAUSE" ] && [ ! -e "$STOP" ]; do sleep 30; done
fi
[ -e "$STOP" ] && { echo "[q160] stop file present — QUEUE 160 DONE (nothing run)"; exit 0; }
sleep 20
pgrep -f "fastchess-linux-x86-64/fastchess|vs_sf.py|_depth_residual_pass" >/dev/null && { echo "[q160] a match or depth pass is running — QUEUE 160 ABORTED"; exit 1; }
START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[q160] $START start"
FP=$(env V2_PRESET=shipped bash "$R" wac fp_g67_q160 2>&1 | grep -E "^SOLVED|^NODES|^EBF" | tr '\n' ' ')
echo "[q160] Cython v2 shipped (want gen 6.7 251 / 36859850): $FP"
echo "$FP" | grep -q "Solved 251/300 NODES: 36859850 " || { echo "[q160] ☠️ FINGERPRINT MISMATCH — QUEUE 160 ABORTED"; exit 1; }

late() { [ -e "$STOP" ] || [ $(date +%s) -ge $CUTOFF ]; }
job() {   # $1 name, $2 IN, $3 OUT, $4 shard k/n — the CHUNK loop, ending at completion, the stop file or the cut-off
  local t0=$(date +%s) rc
  while :; do
    timeout 3600 env V2_PRESET=shipped PRESET=LONG_FORMAT MAX_DEPTH=10 USE_OPENING_BOOK=0 RESIGN_THRESHOLD=-99999999 \
      bash "$R" pyrun diagnostics/_depth_residual_pass.py IN="$2" OUT="$3" SHARD=$4 CHUNK=800 MOVE=1 >> "$LOGD/$1.log" 2>&1
    rc=$?
    [ $rc -ne 3 ] && break
    late && { echo "[q160] $(date) $1 paused at a chunk boundary (stop or cut-off)"; return; }
  done
  echo "[q160] $(date) $1 done (rc $rc) in $(( ($(date +%s) - t0) / 60 )) min"
}
JOBS=()
for s in $(seq 0 11); do JOBS+=("A_s$s|$D/A_sf18.csv|$D/A_${TAG}_d10.csv|$s/12"); done
for s in 0 1 2 3; do JOBS+=("mg_s$s|ks_sets/fitC_mg_sf18.csv|ks_sets/fitC_mg_${TAG}_d10.csv|$s/4"); done
for s in 0 1 2 3; do JOBS+=("eg_s$s|ks_sets/fitC_eg_sf18.csv|ks_sets/fitC_eg_${TAG}_d10.csv|$s/4"); done
for s in $(seq 0 5); do JOBS+=("EGX_s$s|$D/EGX_sf18.csv|$D/EGX_${TAG}_d10.csv|$s/6"); done
for j in "${JOBS[@]}"; do
  IFS='|' read -r name in out shard <<< "$j"
  late && { echo "[q160] $(date) not starting $name: stop or cut-off"; continue; }
  while [ $(jobs -rp | wc -l) -ge 3 ]; do wait -n; done
  echo "[q160] $(date) start $name ($shard)"
  job "$name" "$in" "$out" "$shard" &
done
wait
rows() { cat "$@" 2>/dev/null | grep -vc '^fen,'; }
echo "[q160] $(date) rows: A $(rows $D/A_${TAG}_d10_s*of12.csv) / 140000 · fitC mg $(rows $K/fitC_mg_${TAG}_d10_s*of4.csv) / 14842 · fitC eg $(rows $K/fitC_eg_${TAG}_d10_s*of4.csv) / 19779 · EGX $(rows $D/EGX_${TAG}_d10_s*of6.csv) / 60000"
echo "[q160] $(date) QUEUE 160 DONE"
