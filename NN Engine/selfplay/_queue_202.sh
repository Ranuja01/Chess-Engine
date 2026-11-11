#!/bin/bash
# 2026-11-11 ~13:20, queue #202: POT DUE DILIGENCE, test D (search v2 plan §7d; registered in diagnostics/_pot_shadow.py's
# docstring before any pass ran). Do POT markers separate wrong prunes / reductions? The prune-shadow record now carries the
# node's board and the move (search_engine.cpp log_shadow_event; only under ENABLE_SHADOW_EVENTS), so the markers are computed
# offline with test A's features. Waits for #201. Fixed depth. Test copies only.
#  1 test UCI build (plain) chess_engine_uci_potd: bench must equal gen 7.2 (247 / 39,861,964) — the new fields are log-only.
#  2 test Cython module /mnt/e/chess_data/so_potd.
#  3 the quiet middlegame set: wdlcal rows (quiet, own play) with non-pawn material 10-22 (N, B 1, R 2, Q 4) and move >= 10,
#    first 1,500; shadow pass at d10, SHADOW_N=512, 3 shards; then the registered read.
# Stop: touch /mnt/e/chess_data/q202.stop (checked between steps). Never edit this file, the tools or the sources while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
TB="$ND/build_uci/chess_engine_uci_potd"
SO=/mnt/e/chess_data/so_potd
P=/mnt/e/chess_data/pot_d
STOP=/mnt/e/chess_data/q202.stop
LATEST=$(date -d '2026-11-11 20:30' +%s)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"; mkdir -p "$P"
show() { sed "s/^/[q202]   /"; }
stopped() { [ -e "$STOP" ] && { echo "[q202] $(date) stop file present before $1 — QUEUE 202 DONE"; exit 0; }; }
echo "[q202] $(date '+%Y-%m-%d %H:%M:%S') queued; waiting for #201"
until grep -qE "QUEUE 201 (DONE|ABORTED)|skipped" /mnt/e/chess_data/q201.log 2>/dev/null || ! pgrep -f "_queue_201.sh" >/dev/null; do
	[ -e "$STOP" ] && { echo "[q202] stopped while waiting"; exit 0; }
	sleep 60
done
[ $(date +%s) -lt $LATEST ] || { echo "[q202] $(date) past 20:30 — skipped"; exit 0; }
START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[q202] $START start"

# ── 1 the new fields are log-only ───────────────────────────────────────────────────────────────────────────────────────
rm -f "$TB"
OUT=chess_engine_uci_potd bash selfplay/build_uci.sh > $P/build_uci.out 2>&1
[ -x "$TB" ] || { echo "[q202] ☠️ UCI test build failed — QUEUE 202 ABORTED"; tail -5 build_uci/build.log | show; exit 1; }
fp=$(printf 'bench\nquit\n' | "$TB" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+')
echo "[q202] test UCI build (want 247 / 39861964): $fp"
[ "$fp" = "solved 247/300 nodes 39861964" ] || { echo "[q202] ☠️ the default changed — QUEUE 202 ABORTED"; exit 1; }

# ── 2 the test Cython module ────────────────────────────────────────────────────────────────────────────────────────────
stopped "the Cython test build"
rm -rf "$SO"; mkdir -p "$SO"
bash "$R" pyrun setupAI.py build_ext --build-lib "$SO" --build-temp "$SO/tmp" > "$SO/build.log" 2>&1
ls "$SO"/ChessAI*.so >/dev/null 2>&1 || { echo "[q202] ☠️ Cython test build failed — QUEUE 202 ABORTED"; tail -5 "$SO/build.log" | show; exit 1; }

# ── 3 set, pass, read ───────────────────────────────────────────────────────────────────────────────────────────────────
stopped "the shadow pass"
bash "$R" pyrun diagnostics/_pot_shadow_set.py > $P/set.log 2>&1
echo "[q202] $(tail -1 $P/set.log)"
rm -f $P/events_s*.txt
for k in 0 1 2; do
	env CHESSAI_DIR=$SO V2_PRESET=shipped MAX_DEPTH=10 ENABLE_PRUNE_SHADOW=1 ENABLE_SHADOW_EVENTS=1 SHADOW_N=512 nice -n 10 \
		bash "$R" pyrun diagnostics/_pot_shadow.py MODE=pass IN=$P/mg_quiet.csv OUT=$P/events_s$k.txt SHARD=$k/3 > $P/pass_s$k.log 2>&1 &
done
wait
echo "[q202] $(date) shadow pass done: $(cat $P/events_s*.txt | grep -c 'SHADOWEV') events"
tail -1 $P/pass_s0.log | show
echo "[q202] == TEST D (registered: KEEP pooled AUC >= 0.60 / <= 0.40 on >= 200 wrong AND within strata >= 0.57 / <= 0.43)"
bash "$R" pyrun diagnostics/_pot_shadow.py MODE=auc EVENTS=$P/events_s0.txt,$P/events_s1.txt,$P/events_s2.txt > $P/auc.txt 2>&1
cat $P/auc.txt | show
echo "[q202] $(date) QUEUE 202 DONE"
