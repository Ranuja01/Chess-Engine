#!/usr/bin/env bash
#
# Overnight A/B: depth-adaptive NULLMOVE_PROGRESSIVE (off vs on) across time controls.
# p1 = nmp_off (NULLMOVE_PROGRESSIVE=0), p2 = nmp_on (NULLMOVE_PROGRESSIVE=1); colors swap each game.
#
# PREREQ (do before launching):
#   1) Rebuild with the depth-adaptive gate:  python setupAI.py build_ext --inplace --force
#   2) Anchor BOTH A/B arms to the original locked baseline at shallow depth (criteria a). Both must
#      reproduce 254,973,405 nodes at d10 -- the OFF arm proves "=0 == pre-all-changes baseline", the ON
#      arm proves "the depth-adaptive gate doesn't disturb shallow search":
#        for v in 0 1; do
#          NULLMOVE_PROGRESSIVE=$v PRESET=LONG_FORMAT MAX_DEPTH=10 python diagnostics/tactical_test.py wac.epd nmp_gate_$v
#          awk -F, -v v=$v 'NR>1{n+=$8} END{printf "NULLMOVE_PROGRESSIVE=%s  nodes=%d (want 254,973,405)\n", v, n}' diagnostics/results/tactical_results_nmp_gate_$v.csv
#        done
#      -> BOTH must print exactly 254,973,405. If either is off, stop and fix before launching.
#
# RUN (from "NN Engine/"):  bash selfplay/nmp_overnight.sh
# Tune counts:              BLITZ_GAMES=80 LIGHTNING_GAMES=40 STANDARD_GAMES=8 bash selfplay/nmp_overnight.sh
#
# Est. wall-clock (old-tournament timings x1.2): LIGHTNING ~2.1min/game, BLITZ ~6.8min/game, STANDARD ~27min/game.
# Defaults below (~9h):     BLITZ 50 (~5.7h) + LIGHTNING 30 (~1h) + STANDARD 6 (~2.7h).
#
# Morning analysis = TELEMETRY FIRST (per-move depth/nodes/engine_time/eval from the JSONL, averaged per
# arm), then W/L/D + Elo (coarse at these N). Results land in selfplay/games/nmp_*/ (tournament.json +
# summary.csv + per-game game.jsonl).

set -u
cd "$(dirname "$0")/.."   # -> NN Engine/

OFF='NULLMOVE_PROGRESSIVE=0'
ON='NULLMOVE_PROGRESSIVE=1'

BLITZ_GAMES=${BLITZ_GAMES:-50}
LIGHTNING_GAMES=${LIGHTNING_GAMES:-30}
STANDARD_GAMES=${STANDARD_GAMES:-6}
SEED=${SEED:-1}

run() {  # preset games tag
    local preset="$1" games="$2" tag="$3"
    echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] START $tag : preset=$preset games=$games ==="
    python selfplay/tournament.py \
        --p1-config "$OFF" --p2-config "$ON" \
        --p1-label nmp_off --p2-label nmp_on \
        --preset "$preset" --games "$games" --seed "$SEED" \
        --quiet --resume --tag "$tag"
    echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] DONE  $tag ==="
    echo
}

echo "### nmp_overnight: depth-adaptive NULLMOVE_PROGRESSIVE A/B  (off vs on) ###"
echo "### blitz=$BLITZ_GAMES lightning=$LIGHTNING_GAMES standard=$STANDARD_GAMES seed=$SEED ###"
echo

# Primary signal first (BLITZ), then no-regression confirm (LIGHTNING), then deep observation (STANDARD).
run BLITZ     "$BLITZ_GAMES"     nmp_blitz
run LIGHTNING "$LIGHTNING_GAMES" nmp_lightning
run STANDARD  "$STANDARD_GAMES"  nmp_standard

echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] ALL DONE. Results: selfplay/games/nmp_{blitz,lightning,standard}/ ==="
