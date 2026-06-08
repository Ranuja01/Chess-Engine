#!/usr/bin/env bash
# Fixed-depth color-bias sweep — does the identical-engine White/Black win-rate FLIP with search
# depth parity? (2026-06-05 finding: ship_lightning ~d10 went Black-65%, ship_blitz ~d12 White-80%,
# same openings flipping winner.) Removing the time variable isolates the effect to depth.
#
# Fixed depth via env, NO rebuild: PRESET=LONG_FORMAT gives huge per-move time budgets, and
# MAX_DEPTH=N caps iterative deepening, so each move is a clean depth-N search that exits on the
# DEPTH cap, not time. Both engines run the same config => any color skew is a real asymmetry.
# Run UN-annotated (we only need who won per depth -> summary.csv); analyze with depth_color_summary.py.
#
# Usage (run in WSL, from NN Engine/selfplay/):
#   ./depth_color_sweep.sh                 # depths "8 9 10 11 12 13", 12 openings x both colors
#   DEPTHS="8 10 12" ./depth_color_sweep.sh
#   OPENINGS=openings.txt ./depth_color_sweep.sh   # use the full 44-opening set instead
#
# d12-13 are the slow arm (overnight batch). Results land in games/depth_<N>/.

set -u
cd "$(dirname "$0")"

DEPTHS="${DEPTHS:-8 9 10 11 12 13}"
OPENINGS="${OPENINGS:-openings_sweep.txt}"
SEED="${SEED:-0}"
MAX_PLIES="${MAX_PLIES:-400}"

NOPEN=$(grep -vcE '^[[:space:]]*(#|$)' "$OPENINGS")
GAMES="${GAMES:-$((2 * NOPEN))}"   # both colors of each opening (deterministic at fixed depth => also a determinism check)

echo "[sweep] openings=$OPENINGS ($NOPEN) games/depth=$GAMES depths=[$DEPTHS] seed=$SEED"

for D in $DEPTHS; do
	TAG="depth_${D}"
	echo "[sweep] === depth $D -> games/$TAG ==="
	START=$(date +%s)
	python tournament.py \
		--preset LONG_FORMAT \
		--p1-config "MAX_DEPTH=$D" \
		--p2-config "MAX_DEPTH=$D" \
		--p1-label shipA --p2-label shipB \
		--openings "$OPENINGS" \
		--games "$GAMES" \
		--seed "$SEED" \
		--max-plies "$MAX_PLIES" \
		--adjudicate-draw --sf-movetime 0.3 \
		--tag "$TAG" \
		--quiet
	echo "[sweep] depth $D done in $(( $(date +%s) - START ))s"
done

echo "[sweep] all depths done. Analyze with:  python depth_color_summary.py"
