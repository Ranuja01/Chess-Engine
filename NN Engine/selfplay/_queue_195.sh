#!/bin/bash
# 2026-11-10 ~20:00, queue #195: BARE-KING CONVERSION PROBE (owner 11-10: "certain mate types (like bishop and knight)
# are unknown by our eval"). Waits for #194 (checkpoint #3). Node-limited throughout (load-independent).
# diagnostics/_mopup_conversion.py: 120 tablebase-won positions (30 each KQK, KRK, KBBK, KBNK; White to move), the
# attacker White against Stockfish 18 defending at 200k nodes per move; ends on mate, stalemate, a captured piece,
# threefold, or the 50-move rule.
# Arms: v2 (production gen 7.1) at 1M and 100k nodes per move, v1 (EvalArm 0, the generic mate drive) at 1M, Stockfish 11
# (the SF KXK / KBNK handlers) at 1M and 100k as the control.
# REGISTERED (before any full-run data; the smoke test was 4 games per arm at 50k):
#   KQK, KRK: every arm >= 29/30 at 1M (search alone finds them).
#   KBBK: v2 >= 25/30 at 1M.
#   KBNK: SF11 >= 28/30 at both budgets. Prediction v2 at 1M 9-18/30; v1 at 1M no better than v2.
#   READ: v2 KBNK at 1M <= 20/30 with SF11 >= 28/30 = a real technique defect ⇒ propose a narrow KBNK replacement value
#   (the tier2b pattern: discard material, drive the king to the bishop's corner, kings close). v2 failing KQK or KRK more
#   than once at 1M = a generic mop-up defect (it matters more: those endings are far more common).
# Stop: touch /mnt/e/chess_data/q195.stop (checked while waiting). Never edit this file while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
BIN="$ND/build_uci/chess_engine_uci"
OUT=/mnt/e/chess_data/mopup
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
STOP=/mnt/e/chess_data/q195.stop
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"
echo "[q195] $(date '+%Y-%m-%d %H:%M:%S') queued; waiting for #194"
until grep -q "QUEUE 194 DONE" /mnt/e/chess_data/q194.log 2>/dev/null || ! pgrep -f "_queue_194.sh" >/dev/null; do
	[ -e "$STOP" ] && { echo "[q195] stopped while waiting"; exit 0; }
	sleep 60
done
while pgrep -f "fastchess-linux-x86-64/fastchess|vs_sf.py|_ks_footprint_regret|_case_triage" >/dev/null; do sleep 30; done
[ -e "$STOP" ] && { echo "[q195] stopped"; exit 0; }
START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[q195] $START start"
fp=$(printf "bench\nquit\n" | "$BIN" 2>/dev/null | grep '^bench')
echo "[q195] production (want 247 / 39852210): $fp"
echo "$fp" | grep -q "solved 247/300 nodes 39852210 " || { echo "[q195] ☠️ fingerprint failed — QUEUE 195 ABORTED"; exit 1; }

echo "[q195] $(date) conversion probe: 5 arms x 120 positions, 3 threads"
bash "$R" pyrun diagnostics/_mopup_conversion.py MODE=play THREADS=3 DEF_NODES=200000 OUT=$OUT \
  ARMS=v2_1m:ours:1000000,v2_100k:ours:100000,v1_1m:ours_v1:1000000,sf11_1m:sf11:1000000,sf11_100k:sf11:100000 2>&1 | sed 's/^/[q195]   /'
echo "[q195] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | sed 's/^/[q195]   /'
echo "[q195] $(date) QUEUE 195 DONE"
