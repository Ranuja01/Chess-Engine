#!/bin/bash
# 2026-11-14 ~21:55, queue #243: fixed-depth screens of the shallow-overrules-deep switches on a D7 base (light, for the
# owner's gaming; no timing; never runs once q241.go exists — stops between runs and restores the module).
#  1 Cython module rebuilt from today's sources (production SW5 module backed up to /mnt/e/chess_data/so_backup_q243/, restored
#    at the end), gated: default 249 / 37,279,382, D7 250 / 37,320,382.
#  2 d10 screens (WAC / STS / quiet nodes / game nodes), arms: S0 (production), D7, D7=2, D7+PROMOTE, D7+STATIC_CUT_STORE,
#    D7+HIST_ALLNODE_GATE, D7+MALUS_SEARCHED_ONLY, D7+A1 (FIX_RESEARCH). Read each against D7 (the games candidate).
#    REGISTERED: a switch is a games candidate on D7 when WAC >= D7's and STS >= D7's - 30 (the STS noise floor), OR when it
#    saves >= 5% quiet + game nodes with WAC within 2 and STS within 30. Prediction: D7=2 passes on nodes; A1 passes on WAC
#    at a large node cost (games decide that one); the history pair moves nothing resolvable.
#  3 d7 footprint regret, base = D7: arms D7=2 and neutral (ASPIRATION_DELTA=300 on D7), game_regret_set then _v2.
#    REGISTERED: D7=2's changed-move win% within 2pp of neutral on either set = no resolvable move signal.
# Stop: touch /mnt/e/chess_data/q243.stop (between runs).
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
W=/mnt/e/chess_data/r11a
BK=/mnt/e/chess_data/so_backup_q243
STOP=/mnt/e/chess_data/q243.stop
GO=/mnt/e/chess_data/q241.go
GAMECSV=/mnt/e/chess_data/r1a/game_v2_stratum.csv
D7="SV2_VERIFY_DEEPER=1"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"; mkdir -p "$W" "$BK"
show() { sed "s/^/[q243]   /"; }
restore() { cp "$BK"/ChessAI.*.so . && echo "[q243] production module restored: $(bash "$R" wac r11a_fprest V2_PRESET=shipped 2>&1 | grep -E '^SOLVED|^NODES' | tr '\n' ' ')"; }
stopped() {
	[ -e "$STOP" ] && { echo "[q243] $(date) stop file before $1"; restore; echo "[q243] QUEUE 243 DONE"; exit 0; }
	[ -e "$GO" ] && { echo "[q243] $(date) q241.go present (timed games next) before $1"; restore; echo "[q243] QUEUE 243 DONE"; exit 0; }
}
echo "[q243] $(date '+%Y-%m-%d %H:%M:%S') start"
[ -e "$GO" ] && { echo "[q243] q241.go already present — QUEUE 243 DONE"; exit 0; }
if pgrep -x fastchess-linux-x86-64 >/dev/null; then echo "[q243] ☠️ games are running — QUEUE 243 ABORTED"; exit 1; fi

# ── 1 module ──────────────────────────────────────────────────────────────────────────────────────────────────────────
cp ChessAI.*.so "$BK"/ && echo "[q243] production module backed up to $BK" || { echo "[q243] ☠️ backup failed — QUEUE 243 ABORTED"; exit 1; }
cmp -s ChessAI.*.so /mnt/e/chess_data/so_backup_sw5/ChessAI.*.so && echo "[q243] (it matches so_backup_sw5)" || echo "[q243] note: it differs from so_backup_sw5"
nice -n 10 bash "$R" build 2>&1 | grep -iE 'error|copying|FAILED' | tail -n 3 | show
ls ChessAI.*.so >/dev/null 2>&1 || { echo "[q243] ☠️ no .so after the build — restoring"; cp "$BK"/ChessAI.*.so .; exit 1; }
echo "[q243] $(date '+%H:%M') module built — the sources are free to edit"
f=$(bash "$R" wac r11a_fp V2_PRESET=shipped 2>&1 | grep -E '^SOLVED|^NODES' | tr '\n' ' ')
echo "[q243] module shipped (want 249 / 37279382): $f"
echo "$f" | grep -q 'Solved 249/300 NODES: 37279382' || { echo "[q243] ☠️ the module moved a node"; restore; exit 1; }

# ── 2 d10 screens ─────────────────────────────────────────────────────────────────────────────────────────────────────
BASE="MAX_DEPTH=10 USE_OPENING_BOOK=0 PRESET=LONG_FORMAT V2_PRESET=shipped"
declare -A ARM
ARM[S0]=""
ARM[D7]="$D7"
ARM[D72]="SV2_VERIFY_DEEPER=2"
ARM[PROM]="$D7 SV2_VERIFY_PROMOTE=1"
ARM[SCS]="$D7 SV2_STATIC_CUT_STORE=1"
ARM[HAG]="$D7 SV2_HIST_ALLNODE_GATE=1"
ARM[MSO]="$D7 SV2_MALUS_SEARCHED_ONLY=1"
ARM[A1]="$D7 FIX_RESEARCH=1"
sum_nodes() { grep -hoE '\(nodes=[0-9]+' "$1" | grep -oE '[0-9]+' | awk '{s+=$1} END{print s+0}'; }
for a in S0 D7 D72 PROM SCS HAG MSO A1; do
	k="${ARM[$a]}"
	stopped "$a"
	echo "[q243] $(date '+%H:%M') == $a $k"
	echo "[q243]   WAC: $(nice -n 10 bash "$R" wac r11a_$a V2_PRESET=shipped $k 2>&1 | grep -E '^SOLVED|^NODES' | tr '\n' ' ')"
	env $BASE $k nice -n 10 bash "$R" pyrun diagnostics/sts_test.py sts300.epd r11a_$a > "$W/sts_$a.out" 2> "$W/sts_$a.err"
	echo "[q243]   STS: $(grep -E 'STS score' "$W/sts_$a.out" | tail -1) nodes=$(sum_nodes "$W/sts_$a.err")"
	nice -n 10 bash "$R" pyrun diagnostics/depth_nps_bench.py $BASE $k --n 100 > "$W/quiet_$a.out" 2> "$W/quiet_$a.err"
	echo "[q243]   quiet nodes=$(sum_nodes "$W/quiet_$a.err")"
	nice -n 10 bash "$R" pyrun diagnostics/depth_nps_bench.py $BASE $k --n 100 --corpus "$GAMECSV" > "$W/game_$a.out" 2> "$W/game_$a.err"
	echo "[q243]   game nodes=$(sum_nodes "$W/game_$a.err")"
done

# ── 3 D7=2 footprint regret on a D7 base ──────────────────────────────────────────────────────────────────────────────
for s in game_regret_set game_regret_set_v2; do
	stopped "$s"
	echo "[q243] $(date '+%H:%M') == regret $s"
	env SET=ks_sets/$s.csv DEPTH=7 JOBS=2 BY_PHASE=0 BY_PS=0 BASE_KNOBS="V2_PRESET=shipped $D7" \
		CAND_KNOBS="SV2_VERIFY_DEEPER=2;$D7 ASPIRATION_DELTA=300" CAND_NAME="D7=2;neutral" DUMP=$W/fp_$s.csv \
		nice -n 10 bash "$R" pyrun diagnostics/_ks_footprint_regret.py > "$W/fp_$s.out" 2>&1
	grep -E '^\s+(candidate|D7=2|neutral|cr4)' "$W/fp_$s.out" | head -12 | show
done

restore
echo "[q243] $(date) QUEUE 243 DONE"
