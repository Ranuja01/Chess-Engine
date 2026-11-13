#!/bin/bash
# 2026-11-13 ~13:10, queue #227: S3 root R1, phase B — the new root counters ([root_r1] line, search_engine.cpp; counting only,
# bench unchanged 249 / 37,279,382 on the test binary). Starts when #226 has ended (it uses the Cython module).
#  1 Back up the production Cython module (/mnt/e/chess_data/so_backup_g73f/), rebuild it with the counters, gate: WAC d10
#    249 / 37,279,382 under V2_PRESET=shipped and v1 250 / 35,310,778 (the counters must not move a node).
#  2 Arms B0 defaults, B2 ENABLE_ROOT_RAZOR=0, B3 ENABLE_PRESEARCH_SUBSET=1, each on WAC, STS300, the quiet corpus (100) and
#    the v2 game set (100; a copy with a stratum column, /mnt/e/chess_data/r1a/game_v2_stratum.csv), depth 10, LONG_FORMAT.
#    Read per run: [root_r1] (iterations, best-move changes, the winning entry's score source idx0 / main / pre-search /
#    synthetic, root moves searched and raising alpha by source, searches whose played move had been razored, nodes in
#    aspiration passes and in failed ones with the pre-search's share), plus the summed nodes and the pre-search's total.
#  3 Restore the production module from the backup and check its fingerprint.
# No predictions: counting only. Stop: touch /mnt/e/chess_data/q227.stop (between runs). Never edit this file while it runs;
# never edit the sources before step 1's build has finished ("module built").
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
W=/mnt/e/chess_data/r1b
BK=/mnt/e/chess_data/so_backup_g73f
STOP=/mnt/e/chess_data/q227.stop
GAMECSV=/mnt/e/chess_data/r1a/game_v2_stratum.csv
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"; mkdir -p "$W" "$BK"
show() { sed "s/^/[q227]   /"; }
stopped() { [ -e "$STOP" ] && { echo "[q227] $(date) stop file present before $1 — QUEUE 227 DONE"; exit 0; }; }
echo "[q227] $(date '+%Y-%m-%d %H:%M:%S') queued; waiting for #226"
until grep -q 'QUEUE 226 DONE' /mnt/e/chess_data/q226.log; do stopped "the start"; sleep 30; done
echo "[q227] $(date '+%Y-%m-%d %H:%M:%S') start"

# ── 1 module with the counters ────────────────────────────────────────────────────────────────────────────────────────
cp ChessAI.*.so "$BK/" && echo "[q227] production module backed up to $BK" || { echo "[q227] ☠️ backup failed — QUEUE 227 ABORTED"; exit 1; }
bash "$R" build 2>&1 | grep -iE 'error|copying|FAILED' | tail -n 3 | show
ls ChessAI.*.so >/dev/null 2>&1 || { echo "[q227] ☠️ no .so after the build — restoring"; cp "$BK"/ChessAI.*.so .; exit 1; }
echo "[q227] $(date '+%H:%M') module built — the sources are free to edit"
f=$(bash "$R" wac r1b_fp V2_PRESET=shipped 2>&1 | grep -E '^SOLVED|^NODES' | tr '\n' ' ')
v=$(bash "$R" wac r1b_fpv1 2>&1 | grep -E '^SOLVED|^NODES' | tr '\n' ' ')
echo "[q227] counters module shipped (want 249 / 37279382): $f"
echo "[q227] counters module v1 (want 250 / 35310778):      $v"
if ! { echo "$f" | grep -q 'Solved 249/300 NODES: 37279382' && echo "$v" | grep -q 'Solved 250/300 NODES: 35310778'; }; then
	echo "[q227] ☠️ the counters moved a node — restoring production"; cp "$BK"/ChessAI.*.so .; exit 1
fi

# ── 2 arms ────────────────────────────────────────────────────────────────────────────────────────────────────────────
BASE="MAX_DEPTH=10 USE_OPENING_BOOK=0 PRESET=LONG_FORMAT V2_PRESET=shipped"
declare -A ARM
ARM[B0]=""
ARM[B2]="ENABLE_ROOT_RAZOR=0"
ARM[B3]="ENABLE_PRESEARCH_SUBSET=1"
sum_nodes() { grep -hoP '\(nodes=\K[0-9]+' "$1" | awk '{s+=$1} END{print s+0}'; }
read_run() {  # label, err file
	local lab=$1 e=$2 tot pre
	tot=$(sum_nodes "$e"); pre=$(grep -E '^\[node_split\]' "$e" | tail -1 | grep -oE 'presearch=[0-9]+' | grep -oE '[0-9]+')
	echo "[q227]   $lab nodes=$tot presearch=$pre share=$(awk -v p="$pre" -v t="$tot" 'BEGIN{ if (t>0) printf "%.1f%%", 100*p/t }')"
	echo "[q227]     $(grep -E '^\[root_r1\]' "$e" | tail -1)"
	echo "[q227]     $(grep -E '^\[razor_audit\]' "$e" | tail -1) $(grep -E '^\[aspiration\]' "$e" | tail -1)"
}
for a in B0 B2 B3; do
	k="${ARM[$a]}"
	stopped "$a"
	echo "[q227] $(date '+%H:%M') == $a ${k:-(defaults)}"
	echo "[q227]   WAC: $(bash "$R" wac r1b_$a V2_PRESET=shipped $k 2>&1 | grep -E '^SOLVED|^NODES' | tr '\n' ' ')"
	cp /tmp/wac_r1b_$a.err "$W/wac_$a.err" 2>/dev/null; read_run WAC "$W/wac_$a.err"
	env $BASE $k bash "$R" pyrun diagnostics/sts_test.py sts300.epd r1b_$a > "$W/sts_$a.out" 2> "$W/sts_$a.err"
	echo "[q227]   STS: $(grep -E 'STS score' "$W/sts_$a.out" | tail -1)"; read_run STS "$W/sts_$a.err"
	bash "$R" pyrun diagnostics/depth_nps_bench.py $BASE $k --n 100 > "$W/quiet_$a.out" 2> "$W/quiet_$a.err"
	echo "[q227]   quiet: $(grep -E 'NPS: median' "$W/quiet_$a.out" | grep -oE 'median nodes [0-9,]+')"; read_run quiet "$W/quiet_$a.err"
	bash "$R" pyrun diagnostics/depth_nps_bench.py $BASE $k --n 100 --corpus "$GAMECSV" > "$W/game_$a.out" 2> "$W/game_$a.err"
	echo "[q227]   game: $(grep -E 'NPS: median' "$W/game_$a.out" | grep -oE 'median nodes [0-9,]+')"; read_run game "$W/game_$a.err"
done

# ── 3 restore production ──────────────────────────────────────────────────────────────────────────────────────────────
cp "$BK"/ChessAI.*.so . && echo "[q227] production module restored: $(bash "$R" wac r1b_fprest V2_PRESET=shipped 2>&1 | grep -E '^SOLVED|^NODES' | tr '\n' ' ')"
echo "[q227] $(date) QUEUE 227 DONE"
