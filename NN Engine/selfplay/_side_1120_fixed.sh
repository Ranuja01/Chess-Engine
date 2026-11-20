#!/bin/bash
# 2026-11-20 ~16:30: fixed-depth side checks from the sweep-3 reports (dev_notes/active/SWEEP3-*-2026-11-20.md). They run only
# inside the 20:50-00:15 evening block, when the queue (#283 or #284) plays node-limited judge legs, and never while a timed
# self-play game runs (checked before each run). Each setting is its own process: knobs latch at init. Production binary
# g77pgo, `bench <file> <depth>` (LONG_FORMAT preset, search tables cleared per position, fixed depth). stderr on /tmp.
# REGISTERED before any run:
#  A  S2-33 H1 (mate scores are never stored in the TT while FIX_TT_MATE=0): FEN 47 of UI game 56 (47.f5 to play), depth 22.
#     Default ≈ the game's first-mate iteration (172M nodes, mate in 12); FIX_TT_MATE=1 < 10M (the S2-33 agent's prediction).
#     A saving under 3x refutes H1 as the main cause.
#  B  FEN 46 (46.Kxg6 to play), depth 26, the iteration that ran 149 s in the game: default capped at 420 s; FIX_TT_MATE=1
#     predicted to finish inside 60 s if H1 holds.
#  C  The LMR schedule (LMR_SHAPE 1, LMR_PRODUCT_DIV 64) looks saturated: the table exceeds the rem − 2 clamp at every
#     remaining depth ≥ 3 and move ≥ 2, so a reduced scout's child starts at remaining depth 1 before the history, improving
#     and capture-chain adjustments. game12 d10 with ENABLE_LMR_COUNTERS=1: prediction, the child_rem_hist mode is bucket 1
#     and ≥ 50% of reduced scouts sit at child remaining depth ≤ 1.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
G77="$ND/build_uci/chess_engine_uci_g77pgo"
G12=/mnt/e/chess_data/sv2_speed/game12.txt
OUT=/mnt/e/chess_data/side_1120

mkdir -p "$OUT"; cd "$ND" || exit 1
echo '8/8/6K1/6Pp/5P1P/1kP5/8/8 w - - 1 47 | f4f5' > "$OUT/fen47.txt"
echo '8/6K1/6p1/6Pp/2k2P1P/2P5/8/8 w - - 1 46 | g7g6' > "$OUT/fen46.txt"
timed_running() { pgrep -f '[f]astchess.*tc=10[+]0[.]1' >/dev/null; }
in_block() { local m; m=$((10#$(date +%H) * 60 + 10#$(date +%M))); [ $m -ge $((20 * 60 + 55)) ] && [ $m -lt $((23 * 60 + 40)) ]; }
echo "[side] $(date '+%Y-%m-%d %H:%M:%S') queued; waiting for the evening block (20:55-23:40) with no timed game running"
until in_block && ! timed_running; do
	[ "$(date +%H)" = "00" ] && { echo "[side] $(date '+%H:%M') the block passed without a free slot — nothing run"; exit 0; }
	sleep 60
done
run() {  # label, knobs, file, depth, timeout (s)
	local lab=$1 k=$2 f=$3 d=$4 to=$5 t0 out
	if timed_running || ! in_block; then echo "[side] $(date '+%H:%M') $lab skipped: a timed game is running or the block ended"; return; fi
	t0=$(date +%s)
	out=$( { [ -n "$k" ] && echo "setoption name Knobs value $k"; echo "bench $f $d"; echo quit; } \
		| timeout "$to" "$G77" 2> "/tmp/side1120_$lab.err" | grep -E '^bench:' | tr '\n' ' ')
	echo "[side] $(date '+%H:%M') $lab knobs [${k:-none}] depth $d: ${out:-TIMED OUT at ${to} s} · $(( $(date +%s) - t0 )) s"
	grep -E '^\[(aspiration|sv2_lmr|sv2_null)\]' "/tmp/side1120_$lab.err" | tail -n 3 | sed "s/^/[side]   $lab /"
	cp "/tmp/side1120_$lab.err" "$OUT/$lab.err"
}
run A_default "" "$OUT/fen47.txt" 22 420
run A_ttmate "FIX_TT_MATE=1" "$OUT/fen47.txt" 22 420
run B_ttmate "FIX_TT_MATE=1" "$OUT/fen46.txt" 26 420
run B_default "" "$OUT/fen46.txt" 26 420
run C_lmr "ENABLE_LMR_COUNTERS=1" "$G12" 10 300
echo "[side] $(date) SIDE 1120 DONE"
