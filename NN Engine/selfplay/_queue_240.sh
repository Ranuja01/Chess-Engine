#!/bin/bash
# 2026-11-14 ~16:40, queue #240: light fixed-depth work for the owner's gaming (no timing; never runs once q238.go exists).
#  1 Cython module rebuilt from today's sources (production SW5 module backed up to /mnt/e/chess_data/so_backup_q240/,
#    restored at the end), gated 249 / 37,279,382 at default.
#  2 D7 move quality where it changes the move: _ks_footprint_regret.py at depth 7, base = production (V2_PRESET=shipped),
#    arms D7 (SV2_VERIFY_DEEPER=1) and neutral (ASPIRATION_DELTA=300), on game_regret_set then game_regret_set_v2.
#    REGISTERED: D7's changed-move win% >= neutral + 2pp on BOTH sets = it picks better moves where it differs (supports a
#    positive games read); <= neutral - 2pp on both = worse moves (then a negative games read is the productive-bug case);
#    otherwise no resolvable move-quality signal at d7 and the games decide alone.
#  3 Root LMR near-miss, the parked variant: V4 = L1 + ROOT_LMR_VERIFY_MARGIN=3000 + ROOT_LMR_VERIFY_RED=1 (the verify
#    re-search one ply short of full depth). Reference rows from #235: V2 (RED=0) WAC 251, STS 2052, nodes vs H2
#    +2.5% / -3.2% / -2.0%; L1 WAC 233. REGISTERED: a games candidate only if WAC >= 251 AND quiet + game nodes save >= 5%
#    vs H2 (H2 rerun here as the reference).
# Stop: touch /mnt/e/chess_data/q240.stop (between runs). Never edit this file or the sources before step 1's "module built".
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
W=/mnt/e/chess_data/r9a
BK=/mnt/e/chess_data/so_backup_q240
STOP=/mnt/e/chess_data/q240.stop
GO=/mnt/e/chess_data/q238.go
GAMECSV=/mnt/e/chess_data/r1a/game_v2_stratum.csv
H2E="ENABLE_ROOT_PRESEARCH=0 ENABLE_ROOT_RAZOR=0 ENABLE_PRESEARCH_OFF_SORT=1 PRESEARCH_OFF_SORT_MODE=2 ENABLE_ROOT_TABLE=1 PLY1_NODE_PRUNE=3"
L1E="$H2E ENABLE_ROOT_LMR=1 ROOT_LMR_EXEMPT_BEST=1 ROOT_LMR_DIV=12"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"; mkdir -p "$W" "$BK"
show() { sed "s/^/[q240]   /"; }
restore() { cp "$BK"/ChessAI.*.so . && echo "[q240] production module restored: $(bash "$R" wac r9a_fprest V2_PRESET=shipped 2>&1 | grep -E '^SOLVED|^NODES' | tr '\n' ' ')"; }
stopped() {
	[ -e "$STOP" ] && { echo "[q240] $(date) stop file before $1"; restore; echo "[q240] QUEUE 240 DONE"; exit 0; }
	[ -e "$GO" ] && { echo "[q240] $(date) q238.go present (timed games next) before $1"; restore; echo "[q240] QUEUE 240 DONE"; exit 0; }
}
echo "[q240] $(date '+%Y-%m-%d %H:%M:%S') start"
[ -e "$GO" ] && { echo "[q240] q238.go already present — QUEUE 240 DONE"; exit 0; }
if pgrep -x fastchess-linux-x86-64 >/dev/null || pgrep -f "build_uci/chess_engine_uci" >/dev/null; then echo "[q240] ☠️ games are running — QUEUE 240 ABORTED"; exit 1; fi

# ── 1 module ──────────────────────────────────────────────────────────────────────────────────────────────────────────
cp ChessAI.*.so "$BK"/ && echo "[q240] production module backed up to $BK" || { echo "[q240] ☠️ backup failed — QUEUE 240 ABORTED"; exit 1; }
cmp -s ChessAI.*.so /mnt/e/chess_data/so_backup_sw5/ChessAI.*.so && echo "[q240] (it matches so_backup_sw5)" || echo "[q240] note: it differs from so_backup_sw5"
bash "$R" build 2>&1 | grep -iE 'error|copying|FAILED' | tail -n 3 | show
ls ChessAI.*.so >/dev/null 2>&1 || { echo "[q240] ☠️ no .so after the build — restoring"; cp "$BK"/ChessAI.*.so .; exit 1; }
echo "[q240] $(date '+%H:%M') module built — the sources are free to edit"
f=$(bash "$R" wac r9a_fp V2_PRESET=shipped 2>&1 | grep -E '^SOLVED|^NODES' | tr '\n' ' ')
echo "[q240] module shipped (want 249 / 37279382): $f"
echo "$f" | grep -q 'Solved 249/300 NODES: 37279382' || { echo "[q240] ☠️ the module moved a node"; restore; exit 1; }
f=$(bash "$R" wac r9a_fpd7 V2_PRESET=shipped SV2_VERIFY_DEEPER=1 2>&1 | grep -E '^SOLVED|^NODES' | tr '\n' ' ')
echo "[q240] D7 live (want 250 / 37320382): $f"
echo "$f" | grep -q 'Solved 250/300 NODES: 37320382' || { echo "[q240] ☠️ D7 is not live in the module"; restore; exit 1; }

# ── 2 D7 footprint regret ─────────────────────────────────────────────────────────────────────────────────────────────
for s in game_regret_set game_regret_set_v2; do
	stopped "$s"
	echo "[q240] $(date '+%H:%M') == $s"
	env SET=ks_sets/$s.csv DEPTH=7 JOBS=2 BY_PHASE=0 BY_PS=0 BASE_KNOBS="V2_PRESET=shipped" \
		CAND_KNOBS="SV2_VERIFY_DEEPER=1;ASPIRATION_DELTA=300" CAND_NAME="D7;neutral" DUMP=$W/fp_$s.csv \
		nice -n 10 bash "$R" pyrun diagnostics/_ks_footprint_regret.py > "$W/fp_$s.out" 2>&1
	grep -E '^\s+(candidate|D7|neutral|cr4)' "$W/fp_$s.out" | head -12 | show
done

# ── 3 root LMR V4 screen (H2 reference first) ─────────────────────────────────────────────────────────────────────────
BASE="MAX_DEPTH=10 USE_OPENING_BOOK=0 PRESET=LONG_FORMAT V2_PRESET=shipped"
declare -A ARM
ARM[H2]="$H2E"
ARM[V4]="$L1E ROOT_LMR_VERIFY_MARGIN=3000 ROOT_LMR_VERIFY_RED=1"
sum_nodes() { grep -hoE '\(nodes=[0-9]+' "$1" | grep -oE '[0-9]+' | awk '{s+=$1} END{print s+0}'; }
for a in H2 V4; do
	k="${ARM[$a]}"
	stopped "$a"
	echo "[q240] $(date '+%H:%M') == $a $k"
	echo "[q240]   WAC: $(nice -n 10 bash "$R" wac r9a_$a V2_PRESET=shipped $k 2>&1 | grep -E '^SOLVED|^NODES' | tr '\n' ' ')"
	cp /tmp/wac_r9a_$a.err "$W/wac_$a.err" 2>/dev/null; echo "[q240]     $(grep -E '^\[root_lmr\]' "$W/wac_$a.err" | tail -1)"
	env $BASE $k nice -n 10 bash "$R" pyrun diagnostics/sts_test.py sts300.epd r9a_$a > "$W/sts_$a.out" 2> "$W/sts_$a.err"
	echo "[q240]   STS: $(grep -E 'STS score' "$W/sts_$a.out" | tail -1) nodes=$(sum_nodes "$W/sts_$a.err")"
	nice -n 10 bash "$R" pyrun diagnostics/depth_nps_bench.py $BASE $k --n 100 > "$W/quiet_$a.out" 2> "$W/quiet_$a.err"
	echo "[q240]   quiet nodes=$(sum_nodes "$W/quiet_$a.err")"
	nice -n 10 bash "$R" pyrun diagnostics/depth_nps_bench.py $BASE $k --n 100 --corpus "$GAMECSV" > "$W/game_$a.out" 2> "$W/game_$a.err"
	echo "[q240]   game nodes=$(sum_nodes "$W/game_$a.err")"
done

restore
echo "[q240] $(date) QUEUE 240 DONE"
