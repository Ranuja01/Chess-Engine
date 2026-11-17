#!/bin/bash
# 2026-11-16 ~21:35, queue #260: C4 on the gen 7.5 base, the other two arms (fixed work, cut at 00:05; #257's timed games
# resume 00:15). #259: bit 2 (bounded node-exit write) STS1500 +48 (59 / 54, z +0.47), WAC 252 ⇒ passes the screen; the gen 7.5
# base reproduced #249's C2c arm move for move (10,155).
# Arms (same Cython module as #259, no rebuild): c4g = SV2_HIST_ALLNODE_GATE=1 (the gate alone; #249 on gen 7.4: −135, z −1.22)
# and c4 = 3 (gate + bound, S5 audit D1 as a whole). STS1500 d10 in parallel, WAC each, paired reads against q259_full_g75.
# REGISTERED (as #259): an arm goes to games if STS1500 >= 10,085 (base − 70) and WAC >= 249; below 10,085 with sign |z| > 2 =
# a real cost; otherwise unresolved. Prediction: c4g between −150 and 0 (z −2 … 0), as on gen 7.4; c4 within 70 of c4g.
# Stop: touch /mnt/e/chess_data/q260.stop.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
W=/mnt/e/chess_data/q260
STOP=/mnt/e/chess_data/q260.stop
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"; mkdir -p "$W"
show() { sed "s/^/[q260]   /"; }
echo "[q260] $(date '+%Y-%m-%d %H:%M:%S') start"
END=$(date -d "$(date -d tomorrow +%F) 00:05" +%s); [ "$(date +%H)" -lt 12 ] && END=$(date -d 'today 00:05' +%s)
left() { echo $(( END - $(date +%s) )); }
pgrep -f "[f]astchess-linux|[s]ts_test|[t]actical_test" >/dev/null && { echo "[q260] ☠️ another job is running — QUEUE 260 ABORTED"; exit 1; }
BASE="MAX_DEPTH=10 USE_OPENING_BOOK=0 PRESET=LONG_FORMAT V2_PRESET=shipped"
timeout "$(left)" env $BASE SV2_HIST_ALLNODE_GATE=1 nice -n 10 bash "$R" pyrun diagnostics/sts_test.py STS1-STS15_LAN_v3.epd q260_full_c4g > "$W/full_c4g.out" 2> "$W/full_c4g.err" &
P1=$!
timeout "$(left)" env $BASE SV2_HIST_ALLNODE_GATE=3 nice -n 10 bash "$R" pyrun diagnostics/sts_test.py STS1-STS15_LAN_v3.epd q260_full_c4 > "$W/full_c4.out" 2> "$W/full_c4.err" &
P2=$!
wait $P1; r1=$?; wait $P2; r2=$?
pkill -f "[s]ts_test.py STS1-STS15_LAN_v3.epd q260_" 2>/dev/null
echo "[q260] $(date '+%H:%M') c4g: $(grep -E 'STS score' "$W/full_c4g.out" | tail -1) (exit $r1) · c4: $(grep -E 'STS score' "$W/full_c4.out" | tail -1) (exit $r2)"
[ -e "$STOP" ] && { echo "[q260] stop file — QUEUE 260 DONE"; exit 0; }
for a in c4g:1 c4:3; do
	echo "[q260] ${a%%:*} WAC $(bash "$R" wac q260_wac_${a%%:*} V2_PRESET=shipped SV2_HIST_ALLNODE_GATE=${a#*:} 2>&1 | grep -E '^SOLVED|^NODES' | tr '\n' ' ')"
done
echo "[q260] paired reads against the gen 7.5 base (q259_full_g75, 10,155):"
for a in c4g c4; do bash "$R" pyrun diagnostics/_sts_paired.py q260_full_$a q259_full_g75 2>&1 | grep -E '^positions' | sed "s/^/$a: /" | show; done
bash "$R" pyrun diagnostics/_sts_paired.py q260_full_c4 q260_full_c4g 2>&1 | grep -E '^positions' | sed "s/^/c4 vs c4g: /" | show
echo "[q260] $(date) QUEUE 260 DONE"
