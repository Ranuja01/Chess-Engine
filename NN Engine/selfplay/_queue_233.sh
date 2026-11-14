#!/bin/bash
# 2026-11-14 ~00:45, queue #233: S5 Stack 0 follow-up counters, fixed depth (safe while the owner games; one engine process
# at a time). Starts at once.
#  1 Plain UCI test build chess_engine_uci_r5test: default bench 249 / 37,279,382 must not move, nor with ENABLE_LMR_COUNTERS=1
#    (the new counters count only).
#  2 Cython module rebuilt (production backed up in /mnt/e/chess_data/so_backup_g73f/ and restored at the end), gated.
#  3 Depth 10, with [sv2_lmr], [sv2_null] and the new [sv2_verify] line:
#      S0 defaults + ENABLE_LMR_COUNTERS=1 on WAC, STS300, the quiet corpus (100) and the v2 game set (100)
#      S2 H2 + ENABLE_LMR_COUNTERS=1 on WAC and the game set
#    [sv2_verify]: nodes under near-miss (VERIFY) re-searches (outermost only, so a share of the run's nodes), near-miss
#    re-searches of moves that were NOT reduced (their first search was already full depth, the re-search is 2 plies
#    shallower and replaces its score), and reduced scouts that beat the node's bound for the side to move and are kept
#    without a re-search (the root side's are A1).
#    REGISTERED: node counts equal #232's S0 / S2 exactly. No direction registered for the counters (first measurement).
# Stop: touch /mnt/e/chess_data/q233.stop (between runs). Never edit this file while it runs; never edit the sources before
# step 2's "module built".
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
W=/mnt/e/chess_data/r5a
BK=/mnt/e/chess_data/so_backup_g73f
STOP=/mnt/e/chess_data/q233.stop
GAMECSV=/mnt/e/chess_data/r1a/game_v2_stratum.csv
T="$ND/build_uci/chess_engine_uci_r5test"
H2E="ENABLE_ROOT_PRESEARCH=0 ENABLE_ROOT_RAZOR=0 ENABLE_PRESEARCH_OFF_SORT=1 PRESEARCH_OFF_SORT_MODE=2 ENABLE_ROOT_TABLE=1 PLY1_NODE_PRUNE=3"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"; mkdir -p "$W"
show() { sed "s/^/[q233]   /"; }
stopped() { [ -e "$STOP" ] && { echo "[q233] $(date) stop file present before $1 — QUEUE 233 DONE"; exit 0; }; }
fp() { local k=""; [ -n "$2" ] && k="setoption name Knobs value $2"; printf '%s\nbench\nquit\n' "$k" | "$1" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+'; }
echo "[q233] $(date '+%Y-%m-%d %H:%M:%S') start"

# ── 1 test binary ─────────────────────────────────────────────────────────────────────────────────────────────────────
OUT=chess_engine_uci_r5test bash selfplay/build_uci.sh > "$W/build.out" 2>&1 || { echo "[q233] ☠️ test build failed — QUEUE 233 ABORTED"; tail -5 "$W/build.out" | show; exit 1; }
d=$(fp "$T" ""); c=$(fp "$T" ENABLE_LMR_COUNTERS=1)
echo "[q233] test default (want 249 / 37279382): [$d] · ENABLE_LMR_COUNTERS=1: [$c]"
[ "$d" = "solved 249/300 nodes 37279382" ] && [ "$c" = "$d" ] || { echo "[q233] ☠️ the new code moved a node — QUEUE 233 ABORTED"; exit 1; }

# ── 2 module ──────────────────────────────────────────────────────────────────────────────────────────────────────────
cmp -s ChessAI.*.so "$BK"/ChessAI.*.so && echo "[q233] production module matches the backup" || { echo "[q233] ☠️ the module differs from $BK — QUEUE 233 ABORTED"; exit 1; }
bash "$R" build 2>&1 | grep -iE 'error|copying|FAILED' | tail -n 3 | show
ls ChessAI.*.so >/dev/null 2>&1 || { echo "[q233] ☠️ no .so after the build — restoring"; cp "$BK"/ChessAI.*.so .; exit 1; }
echo "[q233] $(date '+%H:%M') module built — the sources are free to edit"
f=$(bash "$R" wac r5a_fp V2_PRESET=shipped 2>&1 | grep -E '^SOLVED|^NODES' | tr '\n' ' ')
echo "[q233] module shipped (want 249 / 37279382): $f"
echo "$f" | grep -q 'Solved 249/300 NODES: 37279382' || { echo "[q233] ☠️ the module moved a node — restoring production"; cp "$BK"/ChessAI.*.so .; exit 1; }

# ── 3 runs ────────────────────────────────────────────────────────────────────────────────────────────────────────────
BASE="MAX_DEPTH=10 USE_OPENING_BOOK=0 PRESET=LONG_FORMAT V2_PRESET=shipped"
sum_nodes() { grep -hoE '\(nodes=[0-9]+' "$1" | grep -oE '[0-9]+' | awk '{s+=$1} END{print s+0}'; }
read_run() {
	echo "[q233]   $1 nodes=$(sum_nodes "$2")"
	grep -E '^\[sv2_lmr\]|^\[sv2_null\]|^\[sv2_verify\]' "$2" | tail -3 | sed 's/^/[q233]     /'
}
run_set() {
	local a=$1 set=$2 k=$3
	stopped "$a $set"
	case $set in
		WAC)
			echo "[q233]   $a WAC: $(bash "$R" wac r5a_$a V2_PRESET=shipped $k 2>&1 | grep -E '^SOLVED|^NODES' | tr '\n' ' ')"
			cp /tmp/wac_r5a_$a.err "$W/wac_$a.err" 2>/dev/null; read_run "$a WAC" "$W/wac_$a.err" ;;
		STS)
			env $BASE $k bash "$R" pyrun diagnostics/sts_test.py sts300.epd r5a_$a > "$W/sts_$a.out" 2> "$W/sts_$a.err"
			echo "[q233]   $a STS: $(grep -E 'STS score' "$W/sts_$a.out" | tail -1)"; read_run "$a STS" "$W/sts_$a.err" ;;
		quiet)
			bash "$R" pyrun diagnostics/depth_nps_bench.py $BASE $k --n 100 > "$W/quiet_$a.out" 2> "$W/quiet_$a.err"
			read_run "$a quiet" "$W/quiet_$a.err" ;;
		game)
			bash "$R" pyrun diagnostics/depth_nps_bench.py $BASE $k --n 100 --corpus "$GAMECSV" > "$W/game_$a.out" 2> "$W/game_$a.err"
			read_run "$a game" "$W/game_$a.err" ;;
	esac
}
echo "[q233] $(date '+%H:%M') == S0 ENABLE_LMR_COUNTERS=1"
for s in WAC STS quiet game; do run_set S0 $s "ENABLE_LMR_COUNTERS=1"; done
echo "[q233] $(date '+%H:%M') == S2 H2 + ENABLE_LMR_COUNTERS=1"
for s in WAC game; do run_set S2 $s "$H2E ENABLE_LMR_COUNTERS=1"; done

# ── 4 restore production ──────────────────────────────────────────────────────────────────────────────────────────────
cp "$BK"/ChessAI.*.so . && echo "[q233] production module restored: $(bash "$R" wac r5a_fprest V2_PRESET=shipped 2>&1 | grep -E '^SOLVED|^NODES' | tr '\n' ' ')"
echo "[q233] $(date) QUEUE 233 DONE"
