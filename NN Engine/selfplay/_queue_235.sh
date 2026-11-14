#!/bin/bash
# 2026-11-14 ~03:20, queue #235: root reductions with a near-miss rule on H2, fixed depth (one engine process at a time; safe
# while the owner games). Waits for the end of queue #225 (its timed games), then:
#  1 Plain UCI test build chess_engine_uci_r6test. Gates: default bench 249 / 37,279,382; H2 253 / 30,359,342; L1 (#232:
#    H2 + ENABLE_ROOT_LMR=1 ROOT_LMR_EXEMPT_BEST=1 ROOT_LMR_DIV=12) 233 / 32,001,785 with the new knob unset and at 0.
#  2 Cython module rebuilt (production backed up in /mnt/e/chess_data/so_backup_g73f/ and restored at the end), gated.
#  3 Depth 10 on WAC, STS300, the quiet corpus (100) and the v2 game set (100), with [root_r1] and [root_lmr]:
#      V1 L1 + ROOT_LMR_VERIFY_MARGIN=1000   (a reduced root scout failing low by under a pawn is re-searched at full depth)
#      V2 L1 + ROOT_LMR_VERIFY_MARGIN=3000
#      V3 L1 + ROOT_LMR_VERIFY_MARGIN=16000  (the interior's margin: nearly every reduced fail low re-searched; the control)
#      P1 H2 + ENABLE_ROOT_LMR=1 ROOT_LMR_EXEMPT_BEST=1 ROOT_LMR_DIV=12 ROOT_LMR_MIN_IDX=8 (SF19's root shape: about 0 plies
#         before move ~8; no verify)
#    References, not re-run: H2 = WAC 253 / 30,359,342, STS 2083 / 78,670,692, quiet 24,508,082, game 18,598,455 (#232 S2);
#    L1 = WAC 233 / 32,001,785, STS 2058 / 62,458,251, quiet 18,819,631, game 13,914,113 (#232).
#    REGISTERED (screens only): an arm PASSES if WAC >= 249 (within 4 of H2's 253, #232's veto) AND its nodes are below H2's
#    on at least 2 of STS / quiet / game. Predictions: V3 recovers WAC to >= 249 at nodes >= H2's (if it does NOT recover WAC,
#    L1's lost solves come from the reduced searches' side effects on the tree, not from the reduced moves' own scores);
#    V1 / V2 lie between L1 and V3 in WAC and in nodes, monotone in the margin; P1 no direction. A passing arm goes to the
#    owner as a games candidate against H2 (timed: never while the owner games).
# Stop: touch /mnt/e/chess_data/q235.stop (between runs). Never edit this file while it runs; never edit the sources before
# step 2's "module built".
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
W=/mnt/e/chess_data/r6a
BK=/mnt/e/chess_data/so_backup_g73f
STOP=/mnt/e/chess_data/q235.stop
GAMECSV=/mnt/e/chess_data/r1a/game_v2_stratum.csv
T="$ND/build_uci/chess_engine_uci_r6test"
H2E="ENABLE_ROOT_PRESEARCH=0 ENABLE_ROOT_RAZOR=0 ENABLE_PRESEARCH_OFF_SORT=1 PRESEARCH_OFF_SORT_MODE=2 ENABLE_ROOT_TABLE=1 PLY1_NODE_PRUNE=3"
L1E="$H2E ENABLE_ROOT_LMR=1 ROOT_LMR_EXEMPT_BEST=1 ROOT_LMR_DIV=12"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"; mkdir -p "$W"
show() { sed "s/^/[q235]   /"; }
stopped() { [ -e "$STOP" ] && { echo "[q235] $(date) stop file present before $1 — QUEUE 235 DONE"; exit 0; }; }
fp() { local k=""; [ -n "$2" ] && k="setoption name Knobs value $2"; printf '%s\nbench\nquit\n' "$k" | "$1" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+'; }
echo "[q235] $(date '+%Y-%m-%d %H:%M:%S') queued; waiting for the end of #225"
until grep -qE 'QUEUE 225 (DONE|ABORTED)|☠' /mnt/e/chess_data/q225.log; do stopped "the start"; sleep 30; done
sleep 60
if pgrep -f "fastchess-linux-x86-64/fastchess" >/dev/null; then echo "[q235] ☠️ a fastchess process is still running — QUEUE 235 ABORTED"; exit 1; fi
echo "[q235] $(date '+%Y-%m-%d %H:%M:%S') start"

# ── 1 test binary ─────────────────────────────────────────────────────────────────────────────────────────────────────
OUT=chess_engine_uci_r6test bash selfplay/build_uci.sh > "$W/build.out" 2>&1 || { echo "[q235] ☠️ test build failed — QUEUE 235 ABORTED"; tail -5 "$W/build.out" | show; exit 1; }
d=$(fp "$T" ""); h=$(fp "$T" "${H2E// /,}"); l=$(fp "$T" "${L1E// /,}"); l0=$(fp "$T" "${L1E// /,},ROOT_LMR_VERIFY_MARGIN=0")
echo "[q235] test default (want 249 / 37279382): [$d] · H2 (want 253 / 30359342): [$h]"
echo "[q235] L1 (want 233 / 32001785): [$l] · L1 + ROOT_LMR_VERIFY_MARGIN=0 (the same): [$l0]"
if [ "$d" != "solved 249/300 nodes 37279382" ] || [ "$h" != "solved 253/300 nodes 30359342" ] || [ "$l" != "solved 233/300 nodes 32001785" ] || [ "$l0" != "$l" ]; then
	echo "[q235] ☠️ the new code moved a node — QUEUE 235 ABORTED"; exit 1
fi

# ── 2 module ──────────────────────────────────────────────────────────────────────────────────────────────────────────
cmp -s ChessAI.*.so "$BK"/ChessAI.*.so && echo "[q235] production module matches the backup" || { echo "[q235] ☠️ the module differs from $BK — QUEUE 235 ABORTED"; exit 1; }
bash "$R" build 2>&1 | grep -iE 'error|copying|FAILED' | tail -n 3 | show
ls ChessAI.*.so >/dev/null 2>&1 || { echo "[q235] ☠️ no .so after the build — restoring"; cp "$BK"/ChessAI.*.so .; exit 1; }
echo "[q235] $(date '+%H:%M') module built — the sources are free to edit"
f=$(bash "$R" wac r6a_fp V2_PRESET=shipped 2>&1 | grep -E '^SOLVED|^NODES' | tr '\n' ' ')
echo "[q235] module shipped (want 249 / 37279382): $f"
echo "$f" | grep -q 'Solved 249/300 NODES: 37279382' || { echo "[q235] ☠️ the module moved a node — restoring production"; cp "$BK"/ChessAI.*.so .; exit 1; }

# ── 3 screens ─────────────────────────────────────────────────────────────────────────────────────────────────────────
BASE="MAX_DEPTH=10 USE_OPENING_BOOK=0 PRESET=LONG_FORMAT V2_PRESET=shipped"
declare -A ARM
ARM[V1]="$L1E ROOT_LMR_VERIFY_MARGIN=1000"
ARM[V2]="$L1E ROOT_LMR_VERIFY_MARGIN=3000"
ARM[V3]="$L1E ROOT_LMR_VERIFY_MARGIN=16000"
ARM[P1]="$H2E ENABLE_ROOT_LMR=1 ROOT_LMR_EXEMPT_BEST=1 ROOT_LMR_DIV=12 ROOT_LMR_MIN_IDX=8"
sum_nodes() { grep -hoE '\(nodes=[0-9]+' "$1" | grep -oE '[0-9]+' | awk '{s+=$1} END{print s+0}'; }
read_run() {
	local lab=$1 e=$2
	echo "[q235]   $lab nodes=$(sum_nodes "$e")"
	echo "[q235]     $(grep -E '^\[root_r1\]' "$e" | tail -1 | grep -oE 'late_nodes=[0-9]+ late_faillow_nodes=[0-9]+ ply1_rfp_cuts=[0-9]+ ply1_null=[0-9/]+')"
	echo "[q235]     $(grep -E '^\[root_lmr\]' "$e" | tail -1)"
}
for a in V1 V2 V3 P1; do
	k="${ARM[$a]}"
	stopped "$a"
	echo "[q235] $(date '+%H:%M') == $a $k"
	echo "[q235]   WAC: $(bash "$R" wac r6a_$a V2_PRESET=shipped $k 2>&1 | grep -E '^SOLVED|^NODES' | tr '\n' ' ')"
	cp /tmp/wac_r6a_$a.err "$W/wac_$a.err" 2>/dev/null; read_run WAC "$W/wac_$a.err"
	env $BASE $k bash "$R" pyrun diagnostics/sts_test.py sts300.epd r6a_$a > "$W/sts_$a.out" 2> "$W/sts_$a.err"
	echo "[q235]   STS: $(grep -E 'STS score' "$W/sts_$a.out" | tail -1)"; read_run STS "$W/sts_$a.err"
	bash "$R" pyrun diagnostics/depth_nps_bench.py $BASE $k --n 100 > "$W/quiet_$a.out" 2> "$W/quiet_$a.err"
	echo "[q235]   quiet: $(grep -E 'NPS: median' "$W/quiet_$a.out" | grep -oE 'median nodes [0-9,]+')"; read_run quiet "$W/quiet_$a.err"
	bash "$R" pyrun diagnostics/depth_nps_bench.py $BASE $k --n 100 --corpus "$GAMECSV" > "$W/game_$a.out" 2> "$W/game_$a.err"
	echo "[q235]   game: $(grep -E 'NPS: median' "$W/game_$a.out" | grep -oE 'median nodes [0-9,]+')"; read_run game "$W/game_$a.err"
done

# ── 4 restore production ──────────────────────────────────────────────────────────────────────────────────────────────
cp "$BK"/ChessAI.*.so . && echo "[q235] production module restored: $(bash "$R" wac r6a_fprest V2_PRESET=shipped 2>&1 | grep -E '^SOLVED|^NODES' | tr '\n' ' ')"
echo "[q235] $(date) QUEUE 235 DONE"
