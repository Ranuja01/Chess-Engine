#!/bin/bash
# 2026-11-18 ~12:00, queue #272: does STS1500 at EQUAL NODES agree with games on a node-cost change? (fixed work: safe beside
# the owner's games). Chained after #271 (the SORT_KEYED install).
# Why: RFP_THREAT_GATE passed the fixed-depth screen (#269, STS1500 d10 =1 +163, =2 +115) and lost in games (#270, =1 −5.8,
# =2 −30.0): a fixed-depth screen does not charge a change for the nodes it adds. A node budget does. Before using it to screen
# C7 (the verify retune, also node-cost knobs), calibrate it on the gate, whose game results are known.
#  STS1500, MAX_DEPTH=64 at NODE_LIMIT 125,000 / 249,014 / 500,000 (memory node-limit-equal-work-is-step-shaped: one budget is
#  a step function, so three), LONG_FORMAT, no book, V2_PRESET=shipped (gen 7.6). Arms: base, RFP_THREAT_GATE=1, =2, in parallel
#  per budget; paired reads against the base.
# REGISTERED: the screen AGREES with the games if, at >= 2 of 3 budgets, rtg2 < base AND rtg2 < rtg1. Prediction: rtg2 − base
# −120 … +40 and rtg1 − base −60 … +80 at each budget (the d10 reads were +115 / +163).
# Stop: touch /mnt/e/chess_data/q272.stop (between budgets).
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
W=/mnt/e/chess_data/q272
STOP=/mnt/e/chess_data/q272.stop
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"; mkdir -p "$W"
show() { sed "s/^/[q272]   /"; }
stopped() { [ -e "$STOP" ] && { echo "[q272] $(date) stop file before $1 — QUEUE 272 DONE"; exit 0; }; }
echo "[q272] $(date '+%Y-%m-%d %H:%M:%S') queued; waiting for #271"
until grep -q 'QUEUE 271' /mnt/e/chess_data/q271.log 2>/dev/null || ! pgrep -f "[_]queue_271.sh" >/dev/null; do stopped "the start"; sleep 30; done
sleep 20
pgrep -f "[f]astchess-linux|[b]uild_uci.sh|[s]ts_test|[v]algrind" >/dev/null && { echo "[q272] ☠️ another job is running — QUEUE 272 ABORTED"; exit 1; }
echo "[q272] $(date '+%H:%M') start · #271: $(grep -E 'QUEUE 271|SHIPPED|no install' /mnt/e/chess_data/q271.log | head -2 | tr '\n' ' ')"
for N in 125000 249014 500000; do
	stopped "budget $N"
	BASE="MAX_DEPTH=64 NODE_LIMIT=$N USE_OPENING_BOOK=0 PRESET=LONG_FORMAT V2_PRESET=shipped"
	echo "[q272] $(date '+%H:%M') STS1500 @ $N nodes: base, =1, =2 in parallel"
	env $BASE nice -n 10 bash "$R" pyrun diagnostics/sts_test.py STS1-STS15_LAN_v3.epd q272_n${N}_base > "$W/n${N}_base.out" 2> "$W/n${N}_base.err" &
	env $BASE RFP_THREAT_GATE=1 nice -n 10 bash "$R" pyrun diagnostics/sts_test.py STS1-STS15_LAN_v3.epd q272_n${N}_rtg1 > "$W/n${N}_rtg1.out" 2> "$W/n${N}_rtg1.err" &
	env $BASE RFP_THREAT_GATE=2 nice -n 10 bash "$R" pyrun diagnostics/sts_test.py STS1-STS15_LAN_v3.epd q272_n${N}_rtg2 > "$W/n${N}_rtg2.out" 2> "$W/n${N}_rtg2.err" &
	wait
	for t in base rtg1 rtg2; do echo "[q272] $(date '+%H:%M') @$N $t: $(grep -E 'STS score' "$W/n${N}_$t.out" | tail -1)"; done
	for t in rtg1 rtg2; do bash "$R" pyrun diagnostics/_sts_paired.py q272_n${N}_$t q272_n${N}_base 2>&1 | show; done
done
echo "[q272] registered: AGREES with the games if rtg2 < base AND rtg2 < rtg1 at >= 2 of 3 budgets; predictions rtg2 − base −120 … +40, rtg1 − base −60 … +80"
echo "[q272] $(date) QUEUE 272 DONE"
