#!/bin/bash
# 2026-11-16 ~22:00, queue #259: phase H-C, C4's second half. Fixed work only (the owner games; nothing timed until #257 resumes
# at 00:15, so every run here is cut at 00:05).
# SV2_HIST_ALLNODE_GATE gains bit 2: the end-of-node history bonus for the best quiet goes through the gravity bound
# hist_update instead of a raw add (S5 audit D1). Bit 1 is #249's gate (−135, z −1.22, unresolved). Default 0 = byte-identical.
#  1 Cython rebuild (current gen 7.5 .so backed up to /mnt/e/chess_data/so_backup_g75b/). Gates: default WAC fingerprint 252 /
#    35,235,460 (else restore the .so and stop); SV2_HIST_ALLNODE_GATE=2 must move it (liveness).
#  2 STS1500 (STS1-STS15_LAN_v3.epd, d10, LONG_FORMAT, no book, as #249) in parallel: the gen 7.5 base (q259_full_g75) and
#    bit 2 (q259_full_c4b); WAC for bit 2. Paired reads with diagnostics/_sts_paired.py: c4b vs base, and the base vs #249's
#    C2c arm (q249_full_C2c, the same search on the gen 7.4 module with the C2c knobs: it must reproduce, move for move).
# REGISTERED: bit 2 goes to games if STS1500 >= base − 70 and WAC >= base − 3; below base − 70 with sign |z| > 2 = a real cost;
# otherwise unresolved. Prediction: within ±70 and |z| < 1 (the bonus is at most ~15 a node against MAX_HISTORY 16384, so the
# bound changes values by a few points except on long-lived entries).
# Stop: touch /mnt/e/chess_data/q259.stop (between steps).
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
W=/mnt/e/chess_data/q259
BK=/mnt/e/chess_data/so_backup_g75b
STOP=/mnt/e/chess_data/q259.stop
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"; mkdir -p "$W" "$BK"
show() { sed "s/^/[q259]   /"; }
stopped() { [ -e "$STOP" ] && { echo "[q259] $(date) stop file before $1 — QUEUE 259 DONE"; exit 0; }; }
cyr() { grep -E '^SOLVED|^NODES|^EBF' "$1" | tr '\n' ' '; }
echo "[q259] $(date '+%Y-%m-%d %H:%M:%S') start"
END=$(date -d "$(date -d tomorrow +%F) 00:05" +%s); [ "$(date +%H)" -lt 12 ] && END=$(date -d 'today 00:05' +%s)
left() { echo $(( END - $(date +%s) )); }

# ── 1 build + gates ───────────────────────────────────────────────────────────────────────────────────────────────────────
cp ChessAI.*.so "$BK"/ || { echo "[q259] ☠️ .so backup failed — QUEUE 259 ABORTED"; exit 1; }
nice -n 5 bash "$R" build 2>&1 | grep -iE 'error|copying|FAILED' | tail -n 3 | show
ls ChessAI.*.so >/dev/null 2>&1 || { echo "[q259] ☠️ Cython build left no .so — restoring the gen 7.5 .so — QUEUE 259 ABORTED"; cp "$BK"/ChessAI.*.so .; exit 1; }
echo "[q259] $(date '+%H:%M') Cython build done"
bash "$R" wac q259_fp V2_PRESET=shipped > "$W/fp_default.txt" 2>&1
bash "$R" wac q259_fp_c4b V2_PRESET=shipped SV2_HIST_ALLNODE_GATE=2 > "$W/fp_c4b.txt" 2>&1
echo "[q259] default [$(cyr "$W/fp_default.txt")] (want 252 / 35235460) · c4b [$(cyr "$W/fp_c4b.txt")] (must differ)"
grep -q 'Solved 252/300' "$W/fp_default.txt" && grep -q '^NODES: 35235460' "$W/fp_default.txt" \
	|| { echo "[q259] ☠️ default fingerprint moved — restoring the gen 7.5 .so — QUEUE 259 ABORTED"; cp "$BK"/ChessAI.*.so .; exit 1; }
[ "$(grep -E '^NODES' "$W/fp_c4b.txt")" != "$(grep -E '^NODES' "$W/fp_default.txt")" ] \
	|| { echo "[q259] ☠️ SV2_HIST_ALLNODE_GATE=2 does not move the fingerprint — QUEUE 259 DONE"; exit 1; }

# ── 2 STS1500 base + bit 2, in parallel, cut at 00:05 ────────────────────────────────────────────────────────────────────
stopped "the STS runs"
BASE="MAX_DEPTH=10 USE_OPENING_BOOK=0 PRESET=LONG_FORMAT V2_PRESET=shipped"
echo "[q259] $(date '+%H:%M') STS1500 d10: base + c4b in parallel ($(left) s before the cut)"
timeout "$(left)" env $BASE nice -n 10 bash "$R" pyrun diagnostics/sts_test.py STS1-STS15_LAN_v3.epd q259_full_g75 > "$W/full_g75.out" 2> "$W/full_g75.err" &
P1=$!
timeout "$(left)" env $BASE SV2_HIST_ALLNODE_GATE=2 nice -n 10 bash "$R" pyrun diagnostics/sts_test.py STS1-STS15_LAN_v3.epd q259_full_c4b > "$W/full_c4b.out" 2> "$W/full_c4b.err" &
P2=$!
wait $P1; r1=$?; wait $P2; r2=$?
# timeout signals the runner's bash, not its python child: clear any survivor so nothing runs beside the 00:15 timed games
pkill -f "[s]ts_test.py STS1-STS15_LAN_v3.epd q259_" 2>/dev/null
echo "[q259] $(date '+%H:%M') base: $(grep -E 'STS score' "$W/full_g75.out" | tail -1) (exit $r1) · c4b: $(grep -E 'STS score' "$W/full_c4b.out" | tail -1) (exit $r2)"
[ $r1 = 124 ] || [ $r2 = 124 ] && echo "[q259] ☠️ an STS run hit the 00:05 cut — partial"
stopped "the WAC and reads"
w=$(bash "$R" wac q259_wac_c4b V2_PRESET=shipped SV2_HIST_ALLNODE_GATE=2 2>&1 | grep -E '^SOLVED|^NODES' | tr '\n' ' ')
echo "[q259] c4b WAC $w (base 252 / 35235460)"
echo "[q259] paired reads:"
bash "$R" pyrun diagnostics/_sts_paired.py q259_full_c4b q259_full_g75 2>&1 | show
echo "[q259] reproduction check (gen 7.5 base vs #249's C2c arm, must be identical):"
bash "$R" pyrun diagnostics/_sts_paired.py q259_full_g75 q249_full_C2c 2>&1 | show
echo "[q259] $(date) QUEUE 259 DONE"
