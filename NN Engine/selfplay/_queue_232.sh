#!/bin/bash
# 2026-11-13 ~21:00, queue #232: fixed-depth screens, safe while the owner games (node counts do not depend on load; one
# engine process at a time). Starts on /mnt/e/chess_data/q232.go (written when the owner games and the timed games pause) or
# when queue #225 has ended, whichever comes first.
#  1 Plain UCI test build chess_engine_uci_r4test: default bench 249 / 37,279,382 must not move (the new counters count only).
#  2 Cython module rebuilt (production backed up in /mnt/e/chess_data/so_backup_g73f/ and restored at the end), gated.
#  3 Arms at depth 10 on WAC, STS300, the quiet corpus (100) and the v2 game set (100), with [root_r1], [sv2_lmr], [sv2_null]:
#      S0 defaults + ENABLE_LMR_COUNTERS=1 (S5 Stack 0: how deep the reductions really go, how often VERIFY re-searches,
#         the null child's remaining depth; node counts must equal the defaults' #230 Z0)
#      S2 H2 + ENABLE_LMR_COUNTERS=1 (the same on the pre-search-free root)
#      L1 H2 + ENABLE_ROOT_LMR=1 ROOT_LMR_EXEMPT_BEST=1 ROOT_LMR_DIV=12 (at most about 3 plies, SF19's range)
#      L2 H2 + ENABLE_ROOT_LMR=1 ROOT_LMR_EXEMPT_BEST=1 ROOT_LMR_DIV=16
#      L3 H2 + ENABLE_ROOT_LMR=1 ROOT_LMR_EXEMPT_BEST=1 ROOT_LMR_DIV=12 ROOT_LMR_MIN_IDX=5
#    and S0 / S2 again on WAC at depth 12 (the S5 audit's second depth).
#    REGISTERED directions: S0 / S2 node counts equal Z0 / H2 exactly; L1-L3 nodes 5-20% below H2 with WAC within 4 solves
#    of H2's 253 (C4's DIV 4 collapsed to 197). Screens only.
# Stop: touch /mnt/e/chess_data/q232.stop (between runs). Never edit this file while it runs; never edit the sources before
# step 2's "module built".
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
W=/mnt/e/chess_data/r4a
BK=/mnt/e/chess_data/so_backup_g73f
STOP=/mnt/e/chess_data/q232.stop
GO=/mnt/e/chess_data/q232.go
GAMECSV=/mnt/e/chess_data/r1a/game_v2_stratum.csv
T="$ND/build_uci/chess_engine_uci_r4test"
H2E="ENABLE_ROOT_PRESEARCH=0 ENABLE_ROOT_RAZOR=0 ENABLE_PRESEARCH_OFF_SORT=1 PRESEARCH_OFF_SORT_MODE=2 ENABLE_ROOT_TABLE=1 PLY1_NODE_PRUNE=3"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"; mkdir -p "$W"
show() { sed "s/^/[q232]   /"; }
stopped() { [ -e "$STOP" ] && { echo "[q232] $(date) stop file present before $1 — QUEUE 232 DONE"; exit 0; }; }
fp() { local k=""; [ -n "$2" ] && k="setoption name Knobs value $2"; printf '%s\nbench\nquit\n' "$k" | "$1" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+'; }
echo "[q232] $(date '+%Y-%m-%d %H:%M:%S') queued; waiting for q232.go or the end of #225"
until [ -e "$GO" ] || grep -qE 'QUEUE 225 (DONE|ABORTED)' /mnt/e/chess_data/q225.log; do stopped "the start"; sleep 30; done
echo "[q232] $(date '+%Y-%m-%d %H:%M:%S') start"

# ── 1 test binary ─────────────────────────────────────────────────────────────────────────────────────────────────────
OUT=chess_engine_uci_r4test bash selfplay/build_uci.sh > "$W/build.out" 2>&1 || { echo "[q232] ☠️ test build failed — QUEUE 232 ABORTED"; tail -5 "$W/build.out" | show; exit 1; }
d=$(fp "$T" "")
echo "[q232] test default (want 249 / 37279382): [$d] · ENABLE_LMR_COUNTERS=1: [$(fp "$T" ENABLE_LMR_COUNTERS=1)]"
[ "$d" = "solved 249/300 nodes 37279382" ] || { echo "[q232] ☠️ the new code moved the default — QUEUE 232 ABORTED"; exit 1; }

# ── 2 module ──────────────────────────────────────────────────────────────────────────────────────────────────────────
cmp -s ChessAI.*.so "$BK"/ChessAI.*.so && echo "[q232] production module matches the backup" || { echo "[q232] ☠️ the module differs from $BK — QUEUE 232 ABORTED"; exit 1; }
bash "$R" build 2>&1 | grep -iE 'error|copying|FAILED' | tail -n 3 | show
ls ChessAI.*.so >/dev/null 2>&1 || { echo "[q232] ☠️ no .so after the build — restoring"; cp "$BK"/ChessAI.*.so .; exit 1; }
echo "[q232] $(date '+%H:%M') module built — the sources are free to edit"
f=$(bash "$R" wac r4a_fp V2_PRESET=shipped 2>&1 | grep -E '^SOLVED|^NODES' | tr '\n' ' ')
echo "[q232] module shipped (want 249 / 37279382): $f"
echo "$f" | grep -q 'Solved 249/300 NODES: 37279382' || { echo "[q232] ☠️ the module moved a node — restoring production"; cp "$BK"/ChessAI.*.so .; exit 1; }

# ── 3 screens ─────────────────────────────────────────────────────────────────────────────────────────────────────────
BASE="MAX_DEPTH=10 USE_OPENING_BOOK=0 PRESET=LONG_FORMAT V2_PRESET=shipped"
declare -A ARM
ARM[S0]="ENABLE_LMR_COUNTERS=1"
ARM[S2]="$H2E ENABLE_LMR_COUNTERS=1"
ARM[L1]="$H2E ENABLE_ROOT_LMR=1 ROOT_LMR_EXEMPT_BEST=1 ROOT_LMR_DIV=12"
ARM[L2]="$H2E ENABLE_ROOT_LMR=1 ROOT_LMR_EXEMPT_BEST=1 ROOT_LMR_DIV=16"
ARM[L3]="$H2E ENABLE_ROOT_LMR=1 ROOT_LMR_EXEMPT_BEST=1 ROOT_LMR_DIV=12 ROOT_LMR_MIN_IDX=5"
sum_nodes() { grep -hoE '\(nodes=[0-9]+' "$1" | grep -oE '[0-9]+' | awk '{s+=$1} END{print s+0}'; }
read_run() {
	local lab=$1 e=$2
	echo "[q232]   $lab nodes=$(sum_nodes "$e")"
	echo "[q232]     $(grep -E '^\[root_r1\]' "$e" | tail -1 | grep -oE 'late_nodes=[0-9]+ late_faillow_nodes=[0-9]+ ply1_rfp_cuts=[0-9]+ ply1_null=[0-9/]+')"
	echo "[q232]     $(grep -E '^\[root_lmr\]' "$e" | tail -1)"
	grep -E '^\[sv2_lmr\]|^\[sv2_null\]' "$e" | tail -2 | sed 's/^/[q232]     /'
}
for a in S0 S2 L1 L2 L3; do
	k="${ARM[$a]}"
	stopped "$a"
	echo "[q232] $(date '+%H:%M') == $a $k"
	echo "[q232]   WAC: $(bash "$R" wac r4a_$a V2_PRESET=shipped $k 2>&1 | grep -E '^SOLVED|^NODES' | tr '\n' ' ')"
	cp /tmp/wac_r4a_$a.err "$W/wac_$a.err" 2>/dev/null; read_run WAC "$W/wac_$a.err"
	env $BASE $k bash "$R" pyrun diagnostics/sts_test.py sts300.epd r4a_$a > "$W/sts_$a.out" 2> "$W/sts_$a.err"
	echo "[q232]   STS: $(grep -E 'STS score' "$W/sts_$a.out" | tail -1)"; read_run STS "$W/sts_$a.err"
	bash "$R" pyrun diagnostics/depth_nps_bench.py $BASE $k --n 100 > "$W/quiet_$a.out" 2> "$W/quiet_$a.err"
	echo "[q232]   quiet: $(grep -E 'NPS: median' "$W/quiet_$a.out" | grep -oE 'median nodes [0-9,]+')"; read_run quiet "$W/quiet_$a.err"
	bash "$R" pyrun diagnostics/depth_nps_bench.py $BASE $k --n 100 --corpus "$GAMECSV" > "$W/game_$a.out" 2> "$W/game_$a.err"
	echo "[q232]   game: $(grep -E 'NPS: median' "$W/game_$a.out" | grep -oE 'median nodes [0-9,]+')"; read_run game "$W/game_$a.err"
done
for a in S0 S2; do
	k="${ARM[$a]}"
	stopped "$a d12"
	echo "[q232] $(date '+%H:%M') == $a at depth 12 (WAC)"
	echo "[q232]   WAC d12: $(bash "$R" wac r4a_${a}_d12 V2_PRESET=shipped $k MAX_DEPTH=12 2>&1 | grep -E '^SOLVED|^NODES' | tr '\n' ' ')"
	cp /tmp/wac_r4a_${a}_d12.err "$W/wac_${a}_d12.err" 2>/dev/null; read_run "WAC d12" "$W/wac_${a}_d12.err"
done

# ── 4 restore production ──────────────────────────────────────────────────────────────────────────────────────────────
cp "$BK"/ChessAI.*.so . && echo "[q232] production module restored: $(bash "$R" wac r4a_fprest V2_PRESET=shipped 2>&1 | grep -E '^SOLVED|^NODES' | tr '\n' ' ')"
echo "[q232] $(date) QUEUE 232 DONE"
