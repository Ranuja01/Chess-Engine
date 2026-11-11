#!/bin/bash
# 2026-11-11 ~00:45, queue #197: BARE-KING TECHNIQUE, SEARCH OR KNOWLEDGE? (owner 11-11: research "whether its just deep
# search that helps them or eval knowledge"). Waits for #196. Node-limited throughout (load-independent).
# Source read first (11-11):
#  - SF 1.1 (2008), SF11 and SF15.1 carry hand-written handlers in endgame.cpp: KXK = PushToEdges + PushClose, KBNK =
#    PushClose + PushToCorners flipped to the bishop's colour.
#  - SF16 / SF17 / SF18 have no endgame.cpp and no such code. Reckless (CCRL #2, local source) has none either. Both lean
#    on NNUE plus Syzygy tablebases, and no tablebases are configured here.
# Same 60 KBNK + KBBK positions as #195 (ks_sets/mopup_positions.csv), SF18 defending at 200k nodes, so the #195 arms (v2
# 1M / 100k, v1 1M, SF11 1M / 100k) are directly comparable. Arms:
#   sf11_10k   SF11 at 10k nodes: the handler with almost no search (KNOWLEDGE ALONE).
#   sf1.1_1m   SF 1.1 at 1M: the 2008 handler with an old search.
#   sf18_100k / sf18_1m   SF18 attacking (separate process from the defender): no handler, NNUE, no tablebases (LEARNED).
#   v2_10m     production at 10M nodes, 10x #195's top budget (SEARCH ALONE, no technique term). Played last.
# REGISTERED (before any data):
#   KBNK: sf11_10k >= 25/30 · sf1.1_1m >= 25/30 · sf18_100k >= 25/30, sf18_1m >= 28/30 · v2_10m <= 6/30.
#   KBBK: v2_10m >= 28/30 (a shorter mate; search should get there).
#   READ: v2_10m KBNK <= 6/30 with sf11_10k >= 25/30 => the gap is KNOWLEDGE, and the depth search v2 adds will not close
#   it. v2_10m KBNK >= 20/30 => depth does much of it, and the case for a technique term weakens to "speed of conversion".
#   SF18 converting without a handler says the modern engines carry it in a learned eval, not in search depth alone.
# Stop: touch /mnt/e/chess_data/q197.stop (checked while waiting). Never edit this file while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
BIN="$ND/build_uci/chess_engine_uci"
OUT=/mnt/e/chess_data/mopup2
SMOKE=/mnt/e/chess_data/mopup2_smoke
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
STOP=/mnt/e/chess_data/q197.stop
LATEST=$(date -d '2026-11-11 19:30' +%s)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"
echo "[q197] $(date '+%Y-%m-%d %H:%M:%S') queued; waiting for #198 (relaunched after the 11-11 03:29 reboot)"
until grep -qE "QUEUE 198 (DONE|ABORTED)|skipped|stopped" /mnt/e/chess_data/q198.log 2>/dev/null || ! pgrep -f "_queue_198.sh" >/dev/null; do
	[ -e "$STOP" ] && { echo "[q197] stopped while waiting"; exit 0; }
	sleep 60
done
while pgrep -f "fastchess-linux-x86-64/fastchess|vs_sf.py|_ks_footprint_regret|_case_triage|_mopup_conversion" >/dev/null; do sleep 30; done
[ $(date +%s) -lt $LATEST ] || { echo "[q197] $(date) past 19:30 — skipped (the owner may be using the box)"; exit 0; }
START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[q197] $START start"
fp=$(printf "bench\nquit\n" | "$BIN" 2>/dev/null | grep '^bench')
echo "[q197] production (want 247 / 39852210): $fp"
echo "$fp" | grep -q "solved 247/300 nodes 39852210 " || { echo "[q197] ☠️ fingerprint failed — QUEUE 197 ABORTED"; exit 1; }

# smoke: the two engines #195 did not use, one KBNK game each at 10k nodes
rm -rf "$SMOKE"
timeout 600 bash "$R" pyrun diagnostics/_mopup_conversion.py MODE=play THREADS=2 DEF_NODES=20000 CLS=KBNK LIMIT=1 OUT=$SMOKE \
	ARMS=s_sf1.1:sf1.1:10000,s_sf18:sf18_att:10000 2>&1 | tail -6 | sed 's/^/[q197]   smoke /'
n=$(grep -cE '^s_sf(1\.1|18),' $SMOKE/games.csv 2>/dev/null)
bad=$(grep -cE ',nomove,' $SMOKE/games.csv 2>/dev/null)
[ "$n" = "2" ] && [ "$bad" = "0" ] || { echo "[q197] ☠️ smoke failed (rows $n, nomove $bad) — QUEUE 197 ABORTED"; exit 1; }

echo "[q197] $(date) search-or-knowledge probe: 5 arms x 60 positions (KBNK, KBBK), 3 threads"
bash "$R" pyrun diagnostics/_mopup_conversion.py MODE=play THREADS=3 DEF_NODES=200000 CLS=KBNK,KBBK OUT=$OUT \
	ARMS=sf11_10k:sf11:10000,sf1.1_1m:sf1.1:1000000,sf18_100k:sf18_att:100000,sf18_1m:sf18_att:1000000,v2_10m:ours:10000000 \
	2>&1 | sed 's/^/[q197]   /'
echo "[q197] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | sed 's/^/[q197]   /'
echo "[q197] $(date) QUEUE 197 DONE"
