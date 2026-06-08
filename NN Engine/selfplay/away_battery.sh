#!/usr/bin/env bash
# "Away" battery — a self-contained ~4h run (fits a 4-5h window incl. Stockfish time) that firms up
# the depth-flip COLOR study with bigger n than the original 10-20 games. Identical-engine ship-vs-ship
# (any color skew from SF-equal openings is a real asymmetry), DRAW adjudication only (won games grind
# to the engine's own -15p resign so conversion technique + the loser's defense stay observable), and
# post-hoc Stockfish annotation (so sf_cp / sf_static_cp / eval_breakdown land in the data AND the SF
# time is part of the budget). Ordered FAST-FIRST so a time overrun only costs the last block.
#
# Run in WSL from NN Engine/ (survives logout):
#   nohup ./selfplay/away_battery.sh > selfplay/away_battery.log 2>&1 &
#   tail -f selfplay/away_battery.log
#
# Tune counts (defaults shown) or skip a block with 0:
#   LIGHT_GAMES=48 BLITZ_GAMES=16 STD_GAMES=2 SF_MOVETIME=0.3 SEED=7 ./selfplay/away_battery.sh
#
# Analyze when back:
#   python selfplay/tournament_diag.py --tag away_lightning away_blitz away_standard
#   python diagnostics/eval_symmetry.py --tag away_blitz --sample 600
set -u
cd "$(dirname "$0")/.."   # -> NN Engine/

LIGHT_GAMES="${LIGHT_GAMES:-48}"
BLITZ_GAMES="${BLITZ_GAMES:-16}"
STD_GAMES="${STD_GAMES:-2}"
SF_MOVETIME="${SF_MOVETIME:-0.3}"
SEED="${SEED:-7}"
PREFIX="${PREFIX:-away}"   # output tags are <PREFIX>_lightning/_blitz/_standard; change it to avoid clobbering a prior run

echo "[away] start $(date)  prefix=$PREFIX light=$LIGHT_GAMES blitz=$BLITZ_GAMES std=$STD_GAMES sf=${SF_MOVETIME}s seed=$SEED"
T_ALL=$(date +%s)

run() {  # tc-suffix preset games
	local tag="${PREFIX}_$1" preset="$2" games="$3"
	[ "$games" -le 0 ] && { echo "=== [$(date +%T)] $tag SKIPPED (0 games) ==="; return; }
	echo "=== [$(date +%T)] $tag: $games games @ $preset (draw-adjudicated, annotated) ==="
	local t0=$(date +%s)
	python selfplay/tournament.py \
		--p1-label shipA --p2-label shipB \
		--preset "$preset" --games "$games" --seed "$SEED" \
		--openings selfplay/openings.txt \
		--adjudicate-draw --sf-movetime "$SF_MOVETIME" \
		--annotate \
		--tag "$tag" --quiet
	echo "=== [$(date +%T)] $tag done in $(( $(date +%s) - t0 ))s ==="
}

run lightning LIGHTNING "$LIGHT_GAMES"
run blitz     BLITZ     "$BLITZ_GAMES"
run standard  STANDARD  "$STD_GAMES"

echo "[away] battery complete in $(( $(date +%s) - T_ALL ))s ($(date))"
echo "[away] analyze:"
echo "  python selfplay/tournament_diag.py --tag ${PREFIX}_lightning ${PREFIX}_blitz ${PREFIX}_standard"
echo "  python diagnostics/eval_symmetry.py --tag ${PREFIX}_standard --sample 600"
