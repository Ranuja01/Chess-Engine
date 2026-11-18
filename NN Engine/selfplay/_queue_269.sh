#!/bin/bash
# 2026-11-17 ~23:55, queue #269: RFP_THREAT_GATE screen (fixed work only: safe beside the owner's games). Chained after #268.
# The gate: no reverse futility while the side to move has a piece attacked by a cheaper enemy piece (1 = the queen only;
# 2 = any piece). From the 11-17 UI game's case study: with ENABLE_RFP=0 our search finds the queen trap's 26...Rdg8 at d12.
#  1 plain UCI test build chess_engine_uci_rtgtest. Gates: default 246 / 35,138,434 (= production); =1 and =2 must move it.
#  2 Cython rebuild (the gen 7.6 .so backed up to /mnt/e/chess_data/so_backup_g76b); WAC default / =1 / =2.
#  3 the quiet-threat set (diagnostics/_quiet_threat_set.py, 4 positions, d12/14/16, 60 s cap): default, =1, =2 in parallel.
#  4 STS1500 d10 (LONG_FORMAT, no book): the gen 7.6 base, =1, =2 in parallel; paired reads against the base.
# REGISTERED: liveness = WAC nodes move for =1 and =2. Quiet set: =1 and =2 pass qt1117_b at >= 2 of 3 depths (default
# passes 0). Screen to games: STS1500 >= base − 70 and WAC solved >= base − 3. Predictions: STS =1 within ±70, =2 −120 … +50;
# WAC nodes =1 +0.5 … +3%, =2 +2 … +8%.
# Stop: touch /mnt/e/chess_data/q269.stop (between steps).
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
W=/mnt/e/chess_data/q269
BK=/mnt/e/chess_data/so_backup_g76b
TB="$ND/build_uci/chess_engine_uci_rtgtest"
STOP=/mnt/e/chess_data/q269.stop
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"; mkdir -p "$W" "$BK"
show() { sed "s/^/[q269]   /"; }
stopped() { [ -e "$STOP" ] && { echo "[q269] $(date) stop file before $1 — QUEUE 269 DONE"; exit 0; }; }
cyr() { grep -E '^SOLVED|^NODES|^EBF' "$1" | tr '\n' ' '; }
fp() { local k=""; [ -n "$2" ] && k="setoption name Knobs value $2"; printf '%s\nbench\nquit\n' "$k" | "$1" 2>&1 | grep -oE 'solved [0-9]+/300 nodes [0-9]+' | head -1; }
echo "[q269] $(date '+%Y-%m-%d %H:%M:%S') queued; waiting for #268"
until grep -q 'QUEUE 268' /mnt/e/chess_data/q268.log 2>/dev/null || ! pgrep -f "[_]queue_268.sh" >/dev/null; do stopped "the start"; sleep 30; done
sleep 20
pgrep -f "[f]astchess-linux|[b]uild_uci.sh|[s]ts_test|[v]algrind" >/dev/null && { echo "[q269] ☠️ another job is running — QUEUE 269 ABORTED"; exit 1; }
echo "[q269] $(date '+%H:%M') start"

# ── 1 UCI test build ──────────────────────────────────────────────────────────────────────────────────────────────────────
OUT=chess_engine_uci_rtgtest nice -n 10 bash selfplay/build_uci.sh > "$W/build.out" 2>&1 \
	|| { echo "[q269] ☠️ UCI build failed — QUEUE 269 ABORTED"; grep -iE 'error' "$W/build.out" | head -8 | show; exit 1; }
a=$(fp "$TB" ""); g1=$(fp "$TB" "RFP_THREAT_GATE=1"); g2=$(fp "$TB" "RFP_THREAT_GATE=2")
echo "[q269] $(date '+%H:%M') rtgtest default [$a] (246 / 35138434) · =1 [$g1] · =2 [$g2] (both must differ)"
[ "$a" = "solved 246/300 nodes 35138434" ] && [ -n "$g1" ] && [ "$g1" != "$a" ] && [ -n "$g2" ] && [ "$g2" != "$a" ] \
	|| { echo "[q269] ☠️ default moved or a gate is dead — QUEUE 269 ABORTED"; exit 1; }

# ── 2 Cython + WAC ────────────────────────────────────────────────────────────────────────────────────────────────────────
stopped "the Cython build"
cp ChessAI.*.so "$BK"/ || { echo "[q269] ☠️ .so backup failed — QUEUE 269 ABORTED"; exit 1; }
nice -n 5 bash "$R" build 2>&1 | grep -iE 'error|copying|FAILED' | tail -n 3 | show
ls ChessAI.*.so >/dev/null 2>&1 || { echo "[q269] ☠️ Cython build left no .so — restoring the gen 7.6 .so — QUEUE 269 ABORTED"; cp "$BK"/ChessAI.*.so .; exit 1; }
echo "[q269] $(date '+%H:%M') Cython build done — the sources are free to edit"
bash "$R" wac q269_fp V2_PRESET=shipped > "$W/fp_default.txt" 2>&1
bash "$R" wac q269_fp_g1 V2_PRESET=shipped RFP_THREAT_GATE=1 > "$W/fp_g1.txt" 2>&1
bash "$R" wac q269_fp_g2 V2_PRESET=shipped RFP_THREAT_GATE=2 > "$W/fp_g2.txt" 2>&1
echo "[q269] WAC default [$(cyr "$W/fp_default.txt")] (246 / 35138434 / 3.698) · =1 [$(cyr "$W/fp_g1.txt")] · =2 [$(cyr "$W/fp_g2.txt")]"
grep -q 'Solved 246/300' "$W/fp_default.txt" && grep -q '^NODES: 35138434' "$W/fp_default.txt" \
	|| { echo "[q269] ☠️ Cython default fingerprint moved — restoring the gen 7.6 .so — QUEUE 269 ABORTED"; cp "$BK"/ChessAI.*.so .; exit 1; }

# ── 3 quiet-threat set ────────────────────────────────────────────────────────────────────────────────────────────────────
stopped "the quiet-threat set"
for k in "" "RFP_THREAT_GATE=1" "RFP_THREAT_GATE=2"; do
	n=${k:-default}; n=${n//=/}
	( bash "$R" pyrun diagnostics/_quiet_threat_set.py BIN="$TB" KNOBS="$k" > "$W/qt_$n.txt" 2>&1 ) &
done
wait
for f in "$W"/qt_*.txt; do echo "[q269] quiet-threat $(basename "$f" .txt):"; grep -E 'qt1117_b|passes by depth' "$f" | show; done

# ── 4 STS1500 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
stopped "the STS runs"
BASE="MAX_DEPTH=10 USE_OPENING_BOOK=0 PRESET=LONG_FORMAT V2_PRESET=shipped"
echo "[q269] $(date '+%H:%M') STS1500 d10: base, =1, =2 in parallel"
env $BASE nice -n 10 bash "$R" pyrun diagnostics/sts_test.py STS1-STS15_LAN_v3.epd q269_full_g76 > "$W/full_g76.out" 2> "$W/full_g76.err" &
env $BASE RFP_THREAT_GATE=1 nice -n 10 bash "$R" pyrun diagnostics/sts_test.py STS1-STS15_LAN_v3.epd q269_full_rtg1 > "$W/full_rtg1.out" 2> "$W/full_rtg1.err" &
env $BASE RFP_THREAT_GATE=2 nice -n 10 bash "$R" pyrun diagnostics/sts_test.py STS1-STS15_LAN_v3.epd q269_full_rtg2 > "$W/full_rtg2.out" 2> "$W/full_rtg2.err" &
wait
for t in g76 rtg1 rtg2; do echo "[q269] $(date '+%H:%M') $t: $(grep -E 'STS score' "$W/full_$t.out" | tail -1)"; done
echo "[q269] paired reads (A vs the gen 7.6 base):"
for t in rtg1 rtg2; do bash "$R" pyrun diagnostics/_sts_paired.py q269_full_$t q269_full_g76 2>&1 | show; done
echo "[q269] registered: games if STS1500 >= base − 70 and WAC solved >= base − 3; predictions STS =1 ±70, =2 −120 … +50"
echo "[q269] $(date) QUEUE 269 DONE"
