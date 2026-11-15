#!/bin/bash
# 2026-11-15 ~12:00, queue #248: timed work while the owner is not gaming (owner 11-15: "no games for now. If i do it will be in
# a few hours in the afternoon"). Waits for light queue #247 to stop (no side runs while anything timed measures).
#  1 Pinned UCI bench, gen 7.4 (g74pgo, production) and gen 7.3 (g73spgo) alternating x6 on one core: the ledger's
#    time / nps row for gen 7.4 (its tree is +0.1% nodes on the bench).
#  2 H-C1 games: PROMOTE (SV2_VERIFY_PROMOTE=1) vs gen 7.4, both on the production binary (the arms differ only by the knob),
#    10+0.1, UHO, seed 2481, up to 1,000 pairs. Stopped with SIGINT when the owner games (touch /mnt/e/chess_data/q248.pause:
#    a watcher sends it); finished games are kept and pooled with a later continuation on a fresh seed.
#    REGISTERED: PROMOTE − gen 7.4 >= +2σ over the full 2,000 = candidate (Mediocre / SF18 legs next); within ±2σ = NEUTRAL
#    (it costs ~3% game nodes, so neutral does not ship); <= −2σ = the full-depth settle costs more than it buys. Prediction
#    0 to +15 (STS1500 +212 vs D7 at d10; bench 251 / 37,302,143; game nodes +3.4% at fixed depth).
# Stop: touch /mnt/e/chess_data/q248.stop (between steps).
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
W=/mnt/e/chess_data/r14a
PROD="$ND/build_uci/chess_engine_uci"
NEW="$ND/build_uci/chess_engine_uci_g74pgo"
OLD="$ND/build_uci/chess_engine_uci_g73spgo"
STOP=/mnt/e/chess_data/q248.stop
PAUSE=/mnt/e/chess_data/q248.pause
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP" "$PAUSE"; mkdir -p "$W"
show() { sed "s/^/[q248]   /"; }
stopped() { [ -e "$STOP" ] && { echo "[q248] $(date) stop file present before $1 — QUEUE 248 DONE"; exit 0; }; }
echo "[q248] $(date '+%Y-%m-%d %H:%M:%S') queued; waiting for #247 to stop"
until grep -qE 'QUEUE 247 (DONE|ABORTED)' /mnt/e/chess_data/q247.log; do stopped "the start"; sleep 20; done
sleep 20
if pgrep -f "[f]astchess-linux-x86-64/fastchess" >/dev/null || pgrep -f "[s]ts_test|[_]ks_footprint|[v]algrind" >/dev/null; then echo "[q248] ☠️ another job is running — QUEUE 248 ABORTED"; exit 1; fi
cmp -s "$PROD" "$NEW" || { echo "[q248] ☠️ production is not g74pgo — QUEUE 248 ABORTED"; exit 1; }
[ -e "$PAUSE" ] && { echo "[q248] pause file already present — QUEUE 248 DONE"; exit 0; }
echo "[q248] $(date '+%H:%M') start"

# ── 1 pinned bench ────────────────────────────────────────────────────────────────────────────────────────────────────────
echo "[q248] == TIMED: UCI bench, gen 7.4 g74pgo and gen 7.3 g73spgo, pinned to core 5, alternating x6"
for i in 1 2 3 4 5 6; do
	stopped "bench pair $i"; [ -e "$PAUSE" ] && break
	echo "[q248]   g74  $(printf 'bench\nquit\n' | taskset -c 5 "$NEW" 2>/dev/null | grep '^bench')"
	echo "[q248]   g73s $(printf 'bench\nquit\n' | taskset -c 5 "$OLD" 2>/dev/null | grep '^bench')"
done
grep -E '^\[q248\]   g7' /mnt/e/chess_data/q248.log 2>/dev/null | awk '{n[$2]++; t[$2]+=$9; p[$2]+=$12} END {for (k in n) printf "[q248] mean %s: %.0f ms, %.0f nps over %d runs\n", k, t[k]/n[k], p[k]/n[k], n[k]}'

# ── 2 games ───────────────────────────────────────────────────────────────────────────────────────────────────────────────
stopped "the games"; [ -e "$PAUSE" ] && { echo "[q248] $(date) paused before the games — QUEUE 248 DONE"; exit 0; }
rm -f /tmp/q248_pre.log
timeout 180 "$FC" -engine "cmd=$PROD" name=prom option.OwnBook=false option.Knobs=SV2_VERIFY_PROMOTE=1 st=0.05 \
	-engine "cmd=$PROD" name=g74 option.OwnBook=false st=0.05 \
	-openings file=$O/openings_uho.epd format=epd order=random -rounds 1 -games 2 -concurrency 1 > /tmp/q248_pre.log 2>&1
WN=$(grep -c 'have option' /tmp/q248_pre.log)
echo "[q248] preflight: $(grep -E '^Games:' /tmp/q248_pre.log | tail -1) · warnings $WN"
[ "$WN" = 0 ] || { echo "[q248] ☠️ preflight warnings — QUEUE 248 ABORTED"; exit 1; }
COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
echo "[q248] $(date '+%H:%M') games PROMOTE vs gen 7.4 (pause: touch $PAUSE)"
( until [ -e "$PAUSE" ]; do sleep 10; done; pkill -INT -f "[f]astchess-linux-x86-64/fastchess" && echo "[q248] $(date '+%H:%M') pause file: games interrupted" ) &
GUARD=$!
nice -n 5 "$FC" -engine "cmd=$PROD" name=prom option.OwnBook=false "option.Knobs=SV2_VERIFY_PROMOTE=1" tc=10+0.1 timemargin=250 \
	-engine "cmd=$PROD" name=g74 option.OwnBook=false tc=10+0.1 timemargin=250 \
	$COMMON -srand 2481 -rounds 1000 -games 2 -pgnout file=$W/prom_a.pgn notation=san nodes=true append=false > $W/prom_a.log 2>&1
kill $GUARD 2>/dev/null
echo "[q248] prom_a: $(grep -E '^Games:' $W/prom_a.log | tail -1) · warnings $(grep -c 'have option' $W/prom_a.log) · disconnects/stalls $(grep -ciE 'disconnect|stall|not responsive' $W/prom_a.log) · time forfeits $(grep -c 'loses on time' $W/prom_a.pgn)"
timeout 900 bash "$R" pyrun selfplay/fastchess_report.py $W/prom_a.pgn --p1 prom 2>&1 | sed -n '/vs the field/,$p' | show
echo "[q248] $(date) QUEUE 248 DONE"
