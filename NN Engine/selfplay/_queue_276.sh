#!/bin/bash
# 2026-11-18 ~16:50, queue #276: size the stand-pat seed's second defect (owner 11-18: "sure" to counting it first).
# In search v2's qsearch without ENABLE_QSTANDPAT_SEED, `best` starts at the mated sentinel. When the stand-pat raised alpha
# (it was inside the window) and every capture searched scored below it, the node returns the best capture's score, below
# its own stand-pat (q_short). When that score is at or below the entry alpha (q_short_flip), the node reports a fail low the
# stand-pat contradicts, so the parent takes a fail high (a cutoff) it should not. #273 counted only the other case (q_phantom).
# Runs inside #274's SF18 legs (node-limited, so a build and fixed-depth benches beside them change no result); never beside
# timed games.
#  1 plain UCI test build chess_engine_uci_cnt18b from the current sources (count-only counters q_short / q_short_flip).
#    GATES: default 246 / 35,138,434; seed 254 / 35,922,538; v1 250 / 35,310,775-778.
#  2 q_phantom / q_short / q_short_flip over the WAC bench and game12 d10, default and with the seed (the seed must read 0 0 0).
# REGISTERED: q_short_flip "material" if >= 0.1% of qsearch visits on game12 (the bar #273 used for q_phantom).
# Prediction: q_short 2 … 15% of qsearch visits, q_short_flip 0.1 … 3%.
# Stop: touch /mnt/e/chess_data/q276.stop. Never edit the sources before "build done".
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
W=/mnt/e/chess_data/q276
G=/mnt/e/chess_data/sv2_speed/game12.txt
NEW="$ND/build_uci/chess_engine_uci_cnt18b"
STOP=/mnt/e/chess_data/q276.stop
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"; mkdir -p "$W"
show() { sed "s/^/[q276]   /"; }
stopped() { [ -e "$STOP" ] && { echo "[q276] $(date) stop file before $1 — QUEUE 276 DONE"; exit 0; }; }
fp() { local k=""; [ -n "$2" ] && k="setoption name Knobs value $2"; printf '%s\nbench\nquit\n' "$k" | "$1" 2>&1 | grep -oE 'solved [0-9]+/300 nodes [0-9]+' | head -1; }
qc() {  # binary, knobs, bench command → summed counters over every search's stats line
	local k=""; [ -n "$2" ] && k="setoption name Knobs value $2"
	printf 'setoption name Debug value true\n%s\n%s\nquit\n' "$k" "$3" | "$1" 2>&1 | grep -oE 'q_phantom=[0-9]+ q_short=[0-9]+ q_short_flip=[0-9]+ qnodes=[0-9]+' \
		| awk -F'[= ]' '{p += $2; s += $4; f += $6; q += $8} END {d = (q ? q : 1); printf "phantom %d (%.3f%%) · short %d (%.3f%%) · short_flip %d (%.3f%%) of %d qsearch visits over %d searches", p, 100 * p / d, s, 100 * s / d, f, 100 * f / d, q, NR}'
}
echo "[q276] $(date '+%Y-%m-%d %H:%M:%S') queued; waiting for #274's SF18 legs"
until grep -q 'SF18 legs (node-limited)' /mnt/e/chess_data/q274.log 2>/dev/null; do
	stopped "the start"
	pgrep -f "[_]queue_274.sh" >/dev/null || { echo "[q276] ☠️ #274 ended before its SF18 legs — QUEUE 276 ABORTED (not run)"; exit 1; }
	sleep 30
done
grep -q 'QUEUE 274' /mnt/e/chess_data/q274.log && { echo "[q276] ☠️ #274 already finished — QUEUE 276 ABORTED (not run)"; exit 1; }
pgrep -f "[b]uild_uci.sh|[s]ts_test|[v]algrind" >/dev/null && { echo "[q276] ☠️ another build or diagnostic is running — QUEUE 276 ABORTED"; exit 1; }
echo "[q276] $(date '+%H:%M') start (beside #274's node-limited SF18 legs)"

# ── 1 build + gates ──────────────────────────────────────────────────────────────────────────────────────────────────────
OUT=chess_engine_uci_cnt18b nice -n 10 bash selfplay/build_uci.sh > "$W/build.out" 2>&1 \
	|| { echo "[q276] ☠️ UCI build failed — QUEUE 276 ABORTED"; grep -iE 'error' "$W/build.out" | head -8 | show; exit 1; }
echo "[q276] $(date '+%H:%M') build done — the sources are free to edit"
a=$(fp "$NEW" ""); sd=$(fp "$NEW" "ENABLE_QSTANDPAT_SEED=1")
v1=$(printf 'setoption name EvalArm value 0\nbench\nquit\n' | "$NEW" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+')
echo "[q276] cnt18b default [$a] (246 / 35138434) · seed [$sd] (254 / 35922538) · v1 [$v1] (250 / 35310775-778)"
[ "$a" = "solved 246/300 nodes 35138434" ] && [ "$sd" = "solved 254/300 nodes 35922538" ] && echo "$v1" | grep -qE 'solved 250/300 nodes 3531077[5-8]' \
	|| { echo "[q276] ☠️ a gate failed — QUEUE 276 ABORTED"; exit 1; }
echo "[q276] ✅ every gate equal"

# ── 2 counts ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
stopped "the counts"
echo "[q276] WAC bench default:   $(qc "$NEW" "" bench)"
echo "[q276] WAC bench seed:      $(qc "$NEW" "ENABLE_QSTANDPAT_SEED=1" bench) (must be 0 / 0 / 0)"
echo "[q276] game12 d10 default:  $(qc "$NEW" "" "bench $G 10")"
echo "[q276] game12 d10 seed:     $(qc "$NEW" "ENABLE_QSTANDPAT_SEED=1" "bench $G 10") (must be 0 / 0 / 0)"
echo "[q276] registered: q_short_flip material if >= 0.1% of qsearch visits on game12; prediction short 2 … 15%, short_flip 0.1 … 3%"
echo "[q276] $(date) QUEUE 276 DONE"
