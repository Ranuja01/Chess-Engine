#!/bin/bash
# 2026-11-11 ~01:30, queue #198: MOPUP_V2 BUILD AND GATE (owner 11-11: "lets have a cheap technique rule. As long as it
# adds negligible time in general"; tablebases "can be done whenever"). Waits for #197. Builds TEST copies only: the
# production binary build_uci/chess_engine_uci and the UI's live Cython module are not touched. Nothing is installed.
# MOPUP_V2 (eval_v2.cpp mopup_value_mp): pawnless, bare king vs mating material -> material + kings close + the lone king to
# the edge, or for K+B+N vs K to a corner of the bishop's colour; SF15.1's shapes and ratios at MOPUP_V2 percent of its scale.
#  1 UCI test binary build_uci/chess_engine_uci_mopup (plain build; results equal a PGO build's).
#  2 fingerprints on it: default (MOPUP_V2=0) must be 247 / 39,852,210; MOPUP_V2=100 printed.
#  3 cost: bench time, MOPUP_V2=0 vs 100 on that binary, alternating x 4, pinned to one core.
#  4 the #195 probe re-run with MOPUP_V2=100 at 1M and 100k nodes, all 120 positions (OUT /mnt/e/chess_data/mopup3).
#  5 a Cython TEST module in /mnt/e/chess_data/so_mopup, and the mirror test with MOPUP_V2=100 on the 120 bare-king positions
#    (where the rule fires) and on the default corpus (N=800).
# REGISTERED (before any data):
#   default bench byte-identical (247 / 39,852,210); with MOPUP_V2=100 the WAC bench nodes change by < 0.5% or not at all.
#   cost: MOPUP_V2=100 bench time within +-1% of MOPUP_V2=0 (inside the spread).
#   probe: KBNK >= 28/30 at 1M and >= 25/30 at 100k (v2 1/30 at both in #195); KBBK >= 29/30 at 100k (15); KQK and KRK
#   30/30 at both budgets, KQK mean moves at 100k nearer DTM than #195's 15 (DTM 5).
#   mirror test: 0 violations on both sets.
#   READ: all of these = MOPUP_V2=100 is a ship candidate for the owner (a narrow item, judged where it fires: games
#   cannot see it, since self-play adjudicates long before a bare king). Any miss = report, no ship.
# Stop: touch /mnt/e/chess_data/q198.stop (checked between steps). Never edit this file while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
BIN="$ND/build_uci/chess_engine_uci"
MB="$ND/build_uci/chess_engine_uci_mopup"
MOUT=/mnt/e/chess_data/mopup3
SO=/mnt/e/chess_data/so_mopup
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
STOP=/mnt/e/chess_data/q198.stop
LATEST=$(date -d '2026-11-11 07:30' +%s)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"
echo "[q198] $(date '+%Y-%m-%d %H:%M:%S') queued; waiting for #197"
until grep -q "QUEUE 197 DONE" /mnt/e/chess_data/q197.log 2>/dev/null || ! pgrep -f "_queue_197.sh" >/dev/null; do
	[ -e "$STOP" ] && { echo "[q198] stopped while waiting"; exit 0; }
	sleep 60
done
while pgrep -f "fastchess-linux-x86-64/fastchess|vs_sf.py|_ks_footprint_regret|_case_triage|_mopup_conversion" >/dev/null; do sleep 30; done
[ $(date +%s) -lt $LATEST ] || { echo "[q198] $(date) past 07:30 — skipped (the owner may be using the box)"; exit 0; }
START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[q198] $START start"
fp=$(printf "bench\nquit\n" | "$BIN" 2>/dev/null | grep '^bench')
echo "[q198] production (want 247 / 39852210): $fp"
echo "$fp" | grep -q "solved 247/300 nodes 39852210 " || { echo "[q198] ☠️ production fingerprint failed — QUEUE 198 ABORTED"; exit 1; }

# ── 1 the UCI test binary ───────────────────────────────────────────────────────────────────────────────────────────────
rm -f "$MB"
echo "[q198] $(date) build: build_uci/chess_engine_uci_mopup"
OUT=chess_engine_uci_mopup bash selfplay/build_uci.sh > /tmp/q198_build.out 2>&1
rc=$?
sed 's/^/[q198]   /' /tmp/q198_build.out
[ $rc -eq 0 ] && [ -x "$MB" ] || { echo "[q198] ☠️ build failed — QUEUE 198 ABORTED"; exit 1; }

# ── 2 fingerprints ──────────────────────────────────────────────────────────────────────────────────────────────────────
f0=$(printf "bench\nquit\n" | "$MB" 2>/dev/null | grep '^bench')
f1=$(printf "bench\nquit\n" | MOPUP_V2=100 "$MB" 2>/dev/null | grep '^bench')
echo "[q198] test binary, default (want 247 / 39852210): $f0"
echo "[q198] test binary, MOPUP_V2=100:                  $f1"
echo "$f0" | grep -q "solved 247/300 nodes 39852210 " || { echo "[q198] ☠️ not byte-identical at the default — QUEUE 198 ABORTED"; exit 1; }

# ── 3 cost: the same binary, the knob off and on, alternating, pinned ───────────────────────────────────────────────────
[ -e "$STOP" ] && { echo "[q198] stopped"; exit 0; }
for i in 1 2 3 4; do
	for m in 0 100; do
		t=$(printf "bench\nquit\n" | MOPUP_V2=$m taskset -c 5 "$MB" 2>/dev/null | grep '^bench')
		echo "[q198]   cost rep $i MOPUP_V2=$m: $t"
	done
done

# ── 4 the conversion probe with the rule on ─────────────────────────────────────────────────────────────────────────────
[ -e "$STOP" ] && { echo "[q198] stopped"; exit 0; }
echo "[q198] $(date) conversion probe, MOPUP_V2=100: 2 arms x 120 positions, 3 threads"
bash "$R" pyrun diagnostics/_mopup_conversion.py MODE=play THREADS=3 DEF_NODES=200000 OUT=$MOUT \
	ARMS=mop_1m:ours_mop:1000000,mop_100k:ours_mop:100000 > $MOUT.log 2>&1
grep -vE '^ +[0-9]+ / [0-9]+ +last:' $MOUT.log | sed 's/^/[q198]   /'

# ── 5 Cython test module + the mirror test where the rule fires ─────────────────────────────────────────────────────────
[ -e "$STOP" ] && { echo "[q198] stopped"; exit 0; }
rm -rf "$SO"; mkdir -p "$SO"
echo "[q198] $(date) Cython test module -> $SO (the live module is not touched)"
bash "$R" pyrun setupAI.py build_ext --build-lib "$SO" --build-temp "$SO/tmp" > "$SO/build.log" 2>&1
if ls "$SO"/ChessAI*.so >/dev/null 2>&1; then
	for set in "IN=ks_sets/mopup_positions.csv N=120" "N=800"; do
		bash "$R" pyrun diagnostics/_eval_symmetry.py CHESSAI_DIR=$SO V2_PRESET=shipped MOPUP_V2=100 $set > $SO/sym.log 2>&1
		echo "[q198]   mirror test ($set):"
		grep -E 'tested|clean|off by' $SO/sym.log | sed 's/^/[q198]     /'
	done
else
	echo "[q198] ☠️ Cython test build failed (see $SO/build.log) — mirror test skipped"
	tail -5 "$SO/build.log" | sed 's/^/[q198]   /'
fi
echo "[q198] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | sed 's/^/[q198]   /'
echo "[q198] $(date) QUEUE 198 DONE"
