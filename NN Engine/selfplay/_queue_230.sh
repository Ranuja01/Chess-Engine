#!/bin/bash
# 2026-11-13 ~17:30, queue #230 (replaces #229, stopped before it started): S3 root R2 screens (owner 11-13: "understand the
# dependencies and how much we can improve"; "figure out the logic issue of what we're forgetting"). Never rip a part out
# alone: replace a mechanism with its consumers' new suppliers. Starts after #228's games (timed).
# New knobs (default off, byte-identical; search_engine.h): ROOT_RAZOR_REDUCE (the razor's trigger, reduce instead of skip,
# no root table), ASPIRATION_RECENTER 1 / 2 (a failed window re-opens around the returned fail-soft score; 2 = Stockfish's shape),
# PLY1_NODE_PRUNE (null move = 1, reverse futility = 2 at the reply node of a null-window root move: the step the giants' roots
# rely on and ours lacks, S3-ROOT-WITHOUT-PRESEARCH H1), ROOT_TABLE_REPLY_SCORES (defect D1: the table keeps reply scores).
# New counters in [root_r1]: late_nodes / late_faillow_nodes (nodes under root moves after index 0, and under those that
# failed low = the ceiling of what a cheap dismissal can save), ply1_rfp_cuts, ply1_null cuts / tries.
#  1 Plain UCI test build chess_engine_uci_r3test: bench default 249 / 37,279,382 (must not move), and each new knob's bench
#    printed (a changed node count proves it live).
#  2 Cython module rebuilt (production backed up in /mnt/e/chess_data/so_backup_g73f/), gated 249 / 37,279,382 and v1.
#  3 Fixed depth 10 screens on WAC, STS300, the quiet corpus (100) and the v2 game set (100), with the [root_r1] counters:
#      C1 ROOT_RAZOR_REDUCE=1 · C2 ASPIRATION_RECENTER=1 · C3 ASPIRATION_RECENTER=2
#      C4 A4 + ENABLE_ROOT_LMR=1 ROOT_LMR_EXEMPT_BEST=1 (the cascading-tables root with late root moves reduced)
#      C5 A4 + ASPIRATION_RECENTER=2
#      Z0 defaults and Z4 A4 again (the new late-node counters on both baselines)
#      H1 A4 + PLY1_NODE_PRUNE=1 · H2 A4 + PLY1_NODE_PRUNE=3 · H3 defaults + PLY1_NODE_PRUNE=1 · D1 A4 + ROOT_TABLE_REPLY_SCORES=1
#    Baselines: #227 B0 (defaults) and #226 A4, same positions.
#    REGISTERED directions: C1 nodes above A0 (+3 to +15%) with accuracy up; C2 / C3 nodes −2 to −8% vs A0, accuracy level;
#    C4 nodes 5-15% below A4, accuracy between A0 and A4; C5 nodes 2-6% below A4; H1 nodes 10-25% below A4 with WAC ≥ 255
#    and STS within ±30 of A4; H2 below H1; H3 nodes 5-15% below A0, accuracy level; D1 nodes 0-3% below A4. Screens only.
#  4 Production module restored and checked.
# Stop: touch /mnt/e/chess_data/q230.stop (between runs). Never edit this file while it runs; never edit the sources before
# step 2's "module built".
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
W=/mnt/e/chess_data/r3a
BK=/mnt/e/chess_data/so_backup_g73f
STOP=/mnt/e/chess_data/q230.stop
GAMECSV=/mnt/e/chess_data/r1a/game_v2_stratum.csv
T="$ND/build_uci/chess_engine_uci_r3test"
A4E="ENABLE_ROOT_PRESEARCH=0 ENABLE_ROOT_RAZOR=0 ENABLE_PRESEARCH_OFF_SORT=1 PRESEARCH_OFF_SORT_MODE=2 ENABLE_ROOT_TABLE=1"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"; mkdir -p "$W"
show() { sed "s/^/[q230]   /"; }
stopped() { [ -e "$STOP" ] && { echo "[q230] $(date) stop file present before $1 — QUEUE 230 DONE"; exit 0; }; }
fp() { local k=""; [ -n "$2" ] && k="setoption name Knobs value $2"; printf '%s\nbench\nquit\n' "$k" | "$1" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+'; }
echo "[q230] $(date '+%Y-%m-%d %H:%M:%S') queued; waiting for #228"
until grep -qE 'QUEUE 228 (DONE|ABORTED)' /mnt/e/chess_data/q228.log; do stopped "the start"; sleep 30; done
echo "[q230] $(date '+%Y-%m-%d %H:%M:%S') start"

# ── 1 test binary ─────────────────────────────────────────────────────────────────────────────────────────────────────
OUT=chess_engine_uci_r3test bash selfplay/build_uci.sh > "$W/build.out" 2>&1 || { echo "[q230] ☠️ test build failed — QUEUE 230 ABORTED"; tail -5 "$W/build.out" | show; exit 1; }
d=$(fp "$T" "")
echo "[q230] test default (want 249 / 37279382): [$d]"
[ "$d" = "solved 249/300 nodes 37279382" ] || { echo "[q230] ☠️ the new code moved the default — QUEUE 230 ABORTED"; exit 1; }
echo "[q230] ROOT_RAZOR_REDUCE=1:    [$(fp "$T" ROOT_RAZOR_REDUCE=1)]"
echo "[q230] ASPIRATION_RECENTER=1:  [$(fp "$T" ASPIRATION_RECENTER=1)]"
echo "[q230] ASPIRATION_RECENTER=2:  [$(fp "$T" ASPIRATION_RECENTER=2)]"
echo "[q230] PLY1_NODE_PRUNE=1:      [$(fp "$T" PLY1_NODE_PRUNE=1)]"
echo "[q230] PLY1_NODE_PRUNE=3:      [$(fp "$T" PLY1_NODE_PRUNE=3)]"
echo "[q230] A4 + REPLY_SCORES:      [$(fp "$T" ENABLE_ROOT_PRESEARCH=0,ENABLE_ROOT_RAZOR=0,ENABLE_PRESEARCH_OFF_SORT=1,PRESEARCH_OFF_SORT_MODE=2,ENABLE_ROOT_TABLE=1,ROOT_TABLE_REPLY_SCORES=1)] (A4 alone 258 / 43894575)"

# ── 2 module ──────────────────────────────────────────────────────────────────────────────────────────────────────────
cmp -s ChessAI.*.so "$BK"/ChessAI.*.so && echo "[q230] production module matches the backup" || { echo "[q230] ☠️ the module differs from $BK — QUEUE 230 ABORTED"; exit 1; }
bash "$R" build 2>&1 | grep -iE 'error|copying|FAILED' | tail -n 3 | show
ls ChessAI.*.so >/dev/null 2>&1 || { echo "[q230] ☠️ no .so after the build — restoring"; cp "$BK"/ChessAI.*.so .; exit 1; }
echo "[q230] $(date '+%H:%M') module built — the sources are free to edit"
f=$(bash "$R" wac r3a_fp V2_PRESET=shipped 2>&1 | grep -E '^SOLVED|^NODES' | tr '\n' ' ')
v=$(bash "$R" wac r3a_fpv1 2>&1 | grep -E '^SOLVED|^NODES' | tr '\n' ' ')
echo "[q230] module shipped (want 249 / 37279382): $f · v1 (want 250 / 35310778): $v"
if ! { echo "$f" | grep -q 'Solved 249/300 NODES: 37279382' && echo "$v" | grep -q 'Solved 250/300 NODES: 35310778'; }; then
	echo "[q230] ☠️ the module moved a node — restoring production"; cp "$BK"/ChessAI.*.so .; exit 1
fi

# ── 3 screens ─────────────────────────────────────────────────────────────────────────────────────────────────────────
BASE="MAX_DEPTH=10 USE_OPENING_BOOK=0 PRESET=LONG_FORMAT V2_PRESET=shipped"
declare -A ARM
ARM[C1]="ROOT_RAZOR_REDUCE=1"
ARM[C2]="ASPIRATION_RECENTER=1"
ARM[C3]="ASPIRATION_RECENTER=2"
ARM[C4]="$A4E ENABLE_ROOT_LMR=1 ROOT_LMR_EXEMPT_BEST=1"
ARM[C5]="$A4E ASPIRATION_RECENTER=2"
ARM[Z0]=""
ARM[Z4]="$A4E"
ARM[H1]="$A4E PLY1_NODE_PRUNE=1"
ARM[H2]="$A4E PLY1_NODE_PRUNE=3"
ARM[H3]="PLY1_NODE_PRUNE=1"
ARM[D1]="$A4E ROOT_TABLE_REPLY_SCORES=1"
sum_nodes() { grep -hoE '\(nodes=[0-9]+' "$1" | grep -oE '[0-9]+' | awk '{s+=$1} END{print s+0}'; }
read_run() {
	local lab=$1 e=$2 tot pre
	tot=$(sum_nodes "$e"); pre=$(grep -E '^\[node_split\]' "$e" | tail -1 | grep -oE 'presearch=[0-9]+' | grep -oE '[0-9]+')
	echo "[q230]   $lab nodes=$tot presearch=$pre"
	echo "[q230]     $(grep -E '^\[root_r1\]' "$e" | tail -1)"
	echo "[q230]     $(grep -E '^\[aspiration\]' "$e" | tail -1) $(grep -E '^\[root_lmr\]' "$e" | tail -1) $(grep -E '^\[root_table\]' "$e" | tail -1 | grep -oE 'hybrid_reduced=[0-9]+')"
}
for a in Z0 Z4 H1 H3 H2 D1 C4 C1 C2 C3 C5; do
	k="${ARM[$a]}"
	stopped "$a"
	echo "[q230] $(date '+%H:%M') == $a $k"
	echo "[q230]   WAC: $(bash "$R" wac r3a_$a V2_PRESET=shipped $k 2>&1 | grep -E '^SOLVED|^NODES' | tr '\n' ' ')"
	cp /tmp/wac_r3a_$a.err "$W/wac_$a.err" 2>/dev/null; read_run WAC "$W/wac_$a.err"
	env $BASE $k bash "$R" pyrun diagnostics/sts_test.py sts300.epd r3a_$a > "$W/sts_$a.out" 2> "$W/sts_$a.err"
	echo "[q230]   STS: $(grep -E 'STS score' "$W/sts_$a.out" | tail -1)"; read_run STS "$W/sts_$a.err"
	bash "$R" pyrun diagnostics/depth_nps_bench.py $BASE $k --n 100 > "$W/quiet_$a.out" 2> "$W/quiet_$a.err"
	echo "[q230]   quiet: $(grep -E 'NPS: median' "$W/quiet_$a.out" | grep -oE 'median nodes [0-9,]+')"; read_run quiet "$W/quiet_$a.err"
	bash "$R" pyrun diagnostics/depth_nps_bench.py $BASE $k --n 100 --corpus "$GAMECSV" > "$W/game_$a.out" 2> "$W/game_$a.err"
	echo "[q230]   game: $(grep -E 'NPS: median' "$W/game_$a.out" | grep -oE 'median nodes [0-9,]+')"; read_run game "$W/game_$a.err"
done

# ── 4 restore production ──────────────────────────────────────────────────────────────────────────────────────────────
cp "$BK"/ChessAI.*.so . && echo "[q230] production module restored: $(bash "$R" wac r3a_fprest V2_PRESET=shipped 2>&1 | grep -E '^SOLVED|^NODES' | tr '\n' ' ')"
echo "[q230] $(date) QUEUE 230 DONE"
