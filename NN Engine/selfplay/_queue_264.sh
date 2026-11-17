#!/bin/bash
# 2026-11-17 ~16:20, queue #264: gen 7.5's d10 search on the 6,000 quiet WDL rows (wdlcal), for the calibration-by-band read
# (`_v1_v2_accuracy.py MODE=bands`; memory ui-games-2026-11-16-eval-compression-lead). Fixed depth only: safe beside the owner's
# games while #262 is paused. 3 engine processes. Gen 7.5's static eval equals gen 7.2's (gen 7.3-7.5 changed the search and
# result-identical eval speed only), so static_g72 stands in for static. Stop: touch /mnt/e/chess_data/q264.stop.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
W=/mnt/e/chess_data/wdlcal
STOP=/mnt/e/chess_data/q264.stop
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"
show() { sed "s/^/[q264]   /"; }
echo "[q264] $(date '+%Y-%m-%d %H:%M:%S') start · Cython module: $(ls -l --time-style='+%m-%d %H:%M' ChessAI.*.so 2>/dev/null | awk '{print $6, $7}')"
pgrep -f "[f]astchess-linux-x86-64/fastchess|[b]uild_uci.sh" >/dev/null && { echo "[q264] ☠️ timed games or a build are running — QUEUE 264 ABORTED"; exit 1; }
echo "[q264] fingerprint: $(env V2_PRESET=shipped bash "$R" wac q264_fp V2_PRESET=shipped 2>&1 | grep -E '^SOLVED|^NODES' | tr '\n' ' ') (want 252 / 35235460)"
for k in 0 1 2; do
	( until env OMP_NUM_THREADS=1 PRESET=LONG_FORMAT MAX_DEPTH=10 USE_OPENING_BOOK=0 V2_PRESET=shipped nice -n 10 bash "$R" pyrun \
	      diagnostics/_depth_residual_pass.py IN=$W/cal_d10_in.csv OUT=$W/d10_g75.csv SHARD=$k/3 CHUNK=1000 > /dev/null 2>&1; \
	      [ $? -ne 3 ]; do [ -e "$STOP" ] && break; done ) &
done
wait
{ head -1 $W/d10_g75_s0of3.csv; tail -q -n +2 $W/d10_g75_s0of3.csv $W/d10_g75_s1of3.csv $W/d10_g75_s2of3.csv; } > $W/d10_g75.csv
echo "[q264] $(date '+%H:%M') d10 rows: $(($(wc -l < $W/d10_g75.csv) - 1))"
for st in mg,mgl,egl,eg mg,mgl; do
	echo "[q264] == bands, stages $st"
	bash "$R" pyrun diagnostics/_v1_v2_accuracy.py MODE=bands ARMS=static_sf11,static_g72,d10_g72,d10_g75 STAGES=$st 2>&1 | show
done
echo "[q264] $(date) QUEUE 264 DONE"
