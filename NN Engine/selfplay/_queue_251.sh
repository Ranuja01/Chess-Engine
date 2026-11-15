#!/bin/bash
# 2026-11-15 ~15:35, queue #251: the owner's quiet window ("i stopped playing and probably wont get back on till around 9").
# Starts when #250 ends. Timed work, nothing else on the box.
#  1 PGO speed candidate chess_engine_uci_e12pgo: today's sources (eval speed batches 1 + 2), profile /mnt/e/chess_data/pgo_e12
#    trained with the batch on (Knobs EVAL_MAT_CACHE=1,EVAL_FAST=15). GATES: bench 250 / 37,320,382 with the batch off and on,
#    and with the verify mode (EVAL_MAT_CACHE=2).
#  2 Pinned bench on core 5, alternating x5: production g74pgo (default), e12pgo with the batch ON, e12pgo with it OFF (the
#    third arm separates the switches' effect from the new build's layout and profile). REGISTERED: the batch is a speed-ship
#    candidate when e12-on is faster than g74 by more than 1% mean time and faster than e12-off; prediction −3 to −6% time
#    (instructions −6.1% for batch 1 on the plain build; batch 2's count in #250).
#  3 PROMOTE vs gen 7.4, the rest of the registered 2,000 games: production binary, 10+0.1, fresh seed 2482, 614 pairs, pooled
#    with r14a/prom_a.pgn as separate files. SIGINT at 20:50 (owner back ~21:00) or on touch /mnt/e/chess_data/q251.pause.
#    REGISTERED (from #248): pooled >= +2σ = candidate (Mediocre / SF18 next); within ±2σ = NEUTRAL; <= −2σ = costs more than it buys.
# Stop: touch /mnt/e/chess_data/q251.stop (between steps).
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
W=/mnt/e/chess_data/r14a
P=/mnt/e/chess_data/pgo_e12
PROD="$ND/build_uci/chess_engine_uci"
G74="$ND/build_uci/chess_engine_uci_g74pgo"
NEW="$ND/build_uci/chess_engine_uci_e12pgo"
K="EVAL_MAT_CACHE=1,EVAL_FAST=15"
STOP=/mnt/e/chess_data/q251.stop
PAUSE=/mnt/e/chess_data/q251.pause
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP" "$PAUSE"; mkdir -p "$W"
show() { sed "s/^/[q251]   /"; }
stopped() { [ -e "$STOP" ] && { echo "[q251] $(date) stop file before $1 — QUEUE 251 DONE"; exit 0; }; }
late() { [ "$(date +%H%M)" -ge 2050 ]; }
fp() { local k=""; [ -n "$2" ] && k="setoption name Knobs value $2"; printf '%s\nbench\nquit\n' "$k" | "$1" 2>&1 | grep -oE 'solved [0-9]+/300 nodes [0-9]+|mismatch.*' | head -1; }
echo "[q251] $(date '+%Y-%m-%d %H:%M:%S') queued; waiting for #250 to end"
until grep -qE 'QUEUE 250 (DONE|ABORTED)|☠' /mnt/e/chess_data/q250.log; do stopped "the start"; sleep 20; done
sleep 20
if pgrep -f "[f]astchess-linux-x86-64/fastchess" >/dev/null || pgrep -f "[s]ts_test|[v]algrind|[_]ks_footprint" >/dev/null; then echo "[q251] ☠️ another job is running — QUEUE 251 ABORTED"; exit 1; fi
cmp -s "$PROD" "$G74" || { echo "[q251] ☠️ production is not g74pgo — QUEUE 251 ABORTED"; exit 1; }
echo "[q251] $(date '+%H:%M') start"
speed_ok=1
grep -q 'a gate failed' /mnt/e/chess_data/q250.log && { echo "[q251] #250 reported a failed gate — speed steps skipped"; speed_ok=0; }

# ── 1 PGO speed candidate ─────────────────────────────────────────────────────────────────────────────────────────────────
if [ $speed_ok = 1 ]; then
	rm -rf "$P"; mkdir -p "$P"
	if OUT=chess_engine_uci_e12gen PGO=gen PGO_DIR=$P bash selfplay/build_uci.sh > "$W/e12gen.out" 2>&1; then
		nice -n 5 "$FC" -engine cmd="$ND/build_uci/chess_engine_uci_e12gen" name=a option.OwnBook=false option.Knobs=$K st=0.25 \
			-engine cmd="$ND/build_uci/chess_engine_uci_e12gen" name=b option.OwnBook=false option.Knobs=$K st=0.25 \
			-openings file=$O/openings_uho.epd format=epd order=random -srand 751 -rounds 4 -games 2 -concurrency 1 > "$W/e12train.log" 2>&1
		OUT=chess_engine_uci_e12pgo PGO=use PGO_DIR=$P bash selfplay/build_uci.sh > "$W/e12use.out" 2>&1 || speed_ok=0
	else speed_ok=0; fi
	[ $speed_ok = 1 ] || echo "[q251] ☠️ PGO build failed — speed steps skipped"
fi
if [ $speed_ok = 1 ]; then
	echo "[q251] $(date '+%H:%M') build done — the sources are free to edit"
	a=$(fp "$NEW" ""); b=$(fp "$NEW" "$K"); v=$(fp "$NEW" "EVAL_MAT_CACHE=2,EVAL_FAST=15")
	echo "[q251] e12pgo off [$a] · on [$b] · verify [$v] (want 250 / 37320382 each)"
	[ "$a" = "solved 250/300 nodes 37320382" ] && [ "$b" = "$a" ] && [ "$v" = "$a" ] || { echo "[q251] ☠️ gates failed — speed steps skipped"; speed_ok=0; }
fi

# ── 2 pinned bench ────────────────────────────────────────────────────────────────────────────────────────────────────────
if [ $speed_ok = 1 ]; then
	echo "[q251] == TIMED: pinned bench x5, g74 default · e12 batch on · e12 batch off"
	for i in 1 2 3 4 5; do
		stopped "bench round $i"; [ -e "$PAUSE" ] && break
		echo "[q251]   g74   $(printf 'bench\nquit\n' | taskset -c 5 "$G74" 2>/dev/null | grep '^bench')"
		echo "[q251]   e12on $(printf 'setoption name Knobs value %s\nbench\nquit\n' "$K" | taskset -c 5 "$NEW" 2>/dev/null | grep '^bench')"
		echo "[q251]   e12of $(printf 'bench\nquit\n' | taskset -c 5 "$NEW" 2>/dev/null | grep '^bench')"
	done
	grep -E '^\[q251\]   (g74|e12on|e12of) ' /mnt/e/chess_data/q251.log | awk '{n[$2]++; t[$2]+=$9; p[$2]+=$12} END {for (k in n) printf "[q251] mean %s: %.0f ms, %.0f nps over %d runs\n", k, t[k]/n[k], p[k]/n[k], n[k]}'
fi

# ── 3 PROMOTE, the rest ───────────────────────────────────────────────────────────────────────────────────────────────────
stopped "the games"
{ [ -e "$PAUSE" ] || late; } && { echo "[q251] $(date) paused or past 20:50 before the games — QUEUE 251 DONE"; exit 0; }
COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
echo "[q251] $(date '+%H:%M') games PROMOTE vs gen 7.4, seed 2482 (SIGINT at 20:50 or on $PAUSE)"
( until [ -e "$PAUSE" ] || late; do sleep 15; done; pkill -INT -f "[f]astchess-linux-x86-64/fastchess" && echo "[q251] $(date '+%H:%M') guard: games interrupted" ) &
GUARD=$!
nice -n 5 "$FC" -engine "cmd=$PROD" name=prom option.OwnBook=false "option.Knobs=SV2_VERIFY_PROMOTE=1" tc=10+0.1 timemargin=250 \
	-engine "cmd=$PROD" name=g74 option.OwnBook=false tc=10+0.1 timemargin=250 \
	$COMMON -srand 2482 -rounds 614 -games 2 -pgnout file=$W/prom_b.pgn notation=san nodes=true append=false > $W/prom_b.log 2>&1
kill $GUARD 2>/dev/null
echo "[q251] prom_b: $(grep -E '^Games:' $W/prom_b.log | tail -1) · warnings $(grep -c 'have option' $W/prom_b.log) · disconnects/stalls $(grep -ciE 'disconnect|stall|not responsive' $W/prom_b.log) · time forfeits $(grep -c 'loses on time' $W/prom_b.pgn)"
echo "[q251] result tags: prom_a $(grep -c '^\[Result' $W/prom_a.pgn) ($(grep -c '^\[Result "\*"' $W/prom_a.pgn) unfinished) · prom_b $(grep -c '^\[Result' $W/prom_b.pgn) ($(grep -c '^\[Result "\*"' $W/prom_b.pgn) unfinished)"
timeout 900 bash "$R" pyrun selfplay/fastchess_report.py $W/prom_a.pgn $W/prom_b.pgn --p1 prom 2>&1 | sed -n '/vs the field/,$p' | show
echo "[q251] $(date) QUEUE 251 DONE"
