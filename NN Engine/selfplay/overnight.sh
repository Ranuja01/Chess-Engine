#!/usr/bin/env bash
# Overnight self-play batch. Run from NN Engine/ in WSL:
#     bash selfplay/overnight.sh 2>&1 | tee selfplay/games/overnight.log
#
# Goal: a weakness-discovery corpus of the SHIPPED engine playing itself across three time controls
# (the analysis.csv is the morning payload), plus one big-effect strength sanity (stack on vs off).
# Games run with the Stockfish arbiter OFF (full engine speed); SF analysis is a post-hoc pass at the
# end. Fast controls run first, so an interrupted night still leaves the most-data runs complete.
set -u
cd "$(dirname "$0")/.."   # -> NN Engine/

# Stockfish for the post-hoc analysis pass (edit if your binary lives elsewhere):
export STOCKFISH_PATH="${STOCKFISH_PATH:-/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/stockfish/stockfish-windows-x86-64-avx2.exe}"

SEED=1
# Games per time control. Games run LONG (~110-140 plies to resignation), so these are conservative
# so the night actually finishes — BUMP THEM UP if your first runs are fast (it's --resume-able, so
# re-running the same command just adds more games). STANDARD is slowest, so it gets the fewest.
declare -A GAMES=( [LIGHTNING]=20 [BLITZ]=10 [STANDARD]=5 )

STACK_OFF="VERIFY_MARGIN=0 CHECK_EXTENSION=0 REPETITION_THRESHOLD=3 ASPIRATION_DELTA=0 HONEST_ROOT_TT=0"
TAGS=()

run() {  # preset tag p1label p1cfg p2label p2cfg
  local preset=$1 tag=$2 p1l=$3 p1c=$4 p2l=$5 p2c=$6
  TAGS+=("$tag")
  echo "=== $(date +%H:%M)  $tag : $p1l vs $p2l @ $preset (${GAMES[$preset]} games) ==="
  python selfplay/tournament.py --preset "$preset" --games "${GAMES[$preset]}" --seed "$SEED" \
    --quiet --tag "$tag" --resume \
    --p1-label "$p1l" --p1-config "$p1c" --p2-label "$p2l" --p2-config "$p2c"
}

# High-value + fast first, so an interrupted night still has the core data. (corpus = shipped vs
# shipped for weakness-mining; stack = full strength stack vs old baseline for the strength sanity.)
run LIGHTNING ship_lightning  ship    "" ship2    ""           # corpus, fast
run LIGHTNING stack_lightning stackON "" stackOFF "$STACK_OFF" # strength sanity, fast
run BLITZ     ship_blitz      ship    "" ship2    ""           # cross-TC corpus
run STANDARD  ship_standard   ship    "" ship2    ""           # cross-TC corpus (the eval-vs-depth key)
run BLITZ     stack_blitz     stackON "" stackOFF "$STACK_OFF" # strength sanity (lowest priority)

# ---- post-hoc Stockfish analysis: ranks each tournament's games by engine-vs-SF divergence ----
echo "=== $(date +%H:%M)  post-hoc Stockfish analysis ==="
for tag in "${TAGS[@]}"; do
  echo "--- annotate $tag ---"
  python selfplay/annotate.py --tag "$tag" --sf-movetime 0.5
done

echo "=== $(date +%H:%M)  DONE ==="
echo "Standings: selfplay/games/*/tournament.json   |   Weakness ranking: selfplay/games/*/analysis.csv"
