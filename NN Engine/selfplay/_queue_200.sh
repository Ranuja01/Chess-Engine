#!/bin/bash
# 2026-11-11 ~12:45, queue #200: POT DUE DILIGENCE, steps A-rep and B-powered (owner 11-11: "lets give it the proper due
# diligence before writing it off or benching"). Registered in search v2 plan §7d before this ran. Waits for #199 (gen 7.2).
# Fixed depth only: safe beside the owner's games. ≤ 3 engine processes at a time.
#  1 our d10 search and static eval on the 6,000 quiet WDL rows (wdlcal), production gen 7.2 (tag g72).
#  2 the 1,208 game_regret_set_v2 rows whose d12 pass the 11-11 03:29 reboot cut (d10 is complete), 3 workers.
#  3 the reads: MODE=quiet (A-rep) on g72, and on gen 7's v3s dumps as a cross-check; MODE=depthgain on both sets pooled.
# Stop: touch /mnt/e/chess_data/q200.stop (checked between steps). Never edit this file or the tools while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
W=/mnt/e/chess_data/wdlcal
RB=/mnt/e/chess_data/s0_testB_v2
STOP=/mnt/e/chess_data/q200.stop
LATEST=$(date -d '2026-11-11 20:30' +%s)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"
show() { sed "s/^/[q200]   /"; }
stopped() { [ -e "$STOP" ] && { echo "[q200] $(date) stop file present before $1 — QUEUE 200 DONE"; exit 0; }; }
echo "[q200] $(date '+%Y-%m-%d %H:%M:%S') queued; waiting for #199"
until grep -qE "QUEUE 199 (DONE|ABORTED)" /mnt/e/chess_data/q199.log 2>/dev/null || ! pgrep -f "_queue_199.sh" >/dev/null; do
	[ -e "$STOP" ] && { echo "[q200] stopped while waiting"; exit 0; }
	sleep 60
done
grep -q "QUEUE 199 ABORTED" /mnt/e/chess_data/q199.log && echo "[q200] note: #199 aborted — production is still gen 7.1 (the quiet rows read the same)"
[ $(date +%s) -lt $LATEST ] || { echo "[q200] $(date) past 20:30 — skipped"; exit 0; }
START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[q200] $START start · Cython module: $(ls -l --time-style='+%m-%d %H:%M' ChessAI.*.so 2>/dev/null | awk '{print $6, $7}')"

# ── 1 quiet rows: static + d10 ──────────────────────────────────────────────────────────────────────────────────────────
env V2_PRESET=shipped bash "$R" pyrun diagnostics/_v1_v2_accuracy.py MODE=static TAG=g72 2>&1 | tail -1 | show
stopped "the quiet d10 pass"
if [ ! -s $W/d10_g72.csv ]; then
	for k in 0 1 2; do
		( until env OMP_NUM_THREADS=1 PRESET=LONG_FORMAT MAX_DEPTH=10 USE_OPENING_BOOK=0 V2_PRESET=shipped nice -n 10 bash "$R" pyrun \
		      diagnostics/_depth_residual_pass.py IN=$W/cal_d10_in.csv OUT=$W/d10_g72.csv SHARD=$k/3 CHUNK=1000 > /dev/null 2>&1; \
		      [ $? -ne 3 ]; do [ -e "$STOP" ] && break; done ) &
	done
	wait
	{ head -1 $W/d10_g72_s0of3.csv; tail -q -n +2 $W/d10_g72_s0of3.csv $W/d10_g72_s1of3.csv $W/d10_g72_s2of3.csv; } > $W/d10_g72.csv
fi
echo "[q200] $(date) quiet d10 rows: $(($(wc -l < $W/d10_g72.csv) - 1)) · static rows: $(($(wc -l < $W/static_g72.csv) - 1))"

# ── 2 the missing d12 rows of the second regret set ─────────────────────────────────────────────────────────────────────
stopped "the d12 completion"
echo "[q200] $(date) d12 on the $(($(wc -l < $RB/missing_d12_set.csv) - 1)) missing game_regret_set_v2 rows, 3 workers"
for i in 0 1 2; do
	env WORKER=1 SET=$RB/missing_d12_set.csv DEPTH=12 SLICE=$i/3 SPLIT=all OUT=$RB/d12_s$((i + 3)).csv \
		V2_PRESET=shipped bash "$R" pyrun diagnostics/_ks_footprint_regret.py > $RB/d12_s$((i + 3)).log 2>&1 &
done
wait
echo "[q200] $(date) set 2 d12 rows now: $(cat $RB/d12_s*.csv | wc -l) of 11,940"

# ── 3 the registered reads ──────────────────────────────────────────────────────────────────────────────────────────────
stopped "the reads"
echo "[q200] $(date) == A-rep (registered: KEEP ratio >= 1.25 with bootstrap lower bound > 1.10; predicted 1.10-1.25)"
bash "$R" pyrun diagnostics/_pot_mg_fit.py MODE=quiet D10=$W/d10_g72.csv STATIC=$W/static_g72.csv 2>&1 | grep -vE '^β' | show
echo "[q200] cross-check on gen 7's dumps (d10_v3s, static_v3s):"
bash "$R" pyrun diagnostics/_pot_mg_fit.py MODE=quiet D10=$W/d10_v3s.csv STATIC=$W/static_v3s.csv 2>&1 | grep -E 'rows|ratio|REGISTERED' | show
echo "[q200] $(date) == B-powered (registered: KEEP ratio >= 1.5 at > 2 SE on ALL or CRITICAL; predicted ALL 1.0-1.3)"
bash "$R" pyrun diagnostics/_pot_mg_fit.py MODE=depthgain DIRS=/mnt/e/chess_data/s0_testB,$RB \
	REGSETS=ks_sets/game_regret_set.csv,ks_sets/game_regret_set_v2.csv 2>&1 | show
echo "[q200] set 2 alone (the replication the original test B named):"
bash "$R" pyrun diagnostics/_pot_mg_fit.py MODE=depthgain DIRS=$RB REGSETS=ks_sets/game_regret_set_v2.csv 2>&1 | grep -E 'rows|ALL|CRITICAL|REGISTERED' | show
echo "[q200] $(date) QUEUE 200 DONE"
