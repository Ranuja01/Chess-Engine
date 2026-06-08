#!/usr/bin/env bash
#
# Overnight battery: depth-adaptive NULLMOVE_PROGRESSIVE (off vs on) A/B, STANDARD-WEIGHTED, then auto-annotate.
# p1 = nmp_off (NULLMOVE_PROGRESSIVE=0), p2 = nmp_on (NULLMOVE_PROGRESSIVE=1); colors swap each game.
#
# WHY standard-weighted: NULLMOVE_PROGRESSIVE only fires at genuine depth >=12, which only STANDARD reaches
# (~d12-14). LIGHTNING (~d9-11) / BLITZ (~d10-11) leave it INERT -> they are the no-regression check, STANDARD
# is the actual test. Annotation adds SF static (NNUE) + our static term breakdown per move (eval calibration),
# for free on every game regardless of arm.
#
# NO REBUILD NEEDED: the current build already has NULLMOVE_PROGRESSIVE + ev_breakdown (WAC d10 = 254,973,405).
#
# RUN (from "NN Engine/"):  bash selfplay/overnight_battery.sh
# Tune:   STANDARD_GAMES=18 LIGHTNING_GAMES=30 BLITZ_GAMES=12 bash selfplay/overnight_battery.sh
#
# Combined budget on last night's real timings (LIGHTNING ~1.6min, BLITZ ~5.4min, STANDARD ~22min per game;
# annotation ~= total_plies * SF_MOVETIME * 1.1): defaults below ~= 7.3h elapsed (incl. annotation).

set -u
cd "$(dirname "$0")/.."   # -> NN Engine/

OFF='NULLMOVE_PROGRESSIVE=0'
ON='NULLMOVE_PROGRESSIVE=1'

STANDARD_GAMES=${STANDARD_GAMES:-14}
LIGHTNING_GAMES=${LIGHTNING_GAMES:-24}
BLITZ_GAMES=${BLITZ_GAMES:-12}
SEED=${SEED:-1}
SF_MOVETIME=${SF_MOVETIME:-0.3}

stamp() { date '+%Y-%m-%d %H:%M:%S'; }

run() {  # preset games tag
    local preset="$1" games="$2" tag="$3"
    [ "$games" -gt 0 ] || { echo "=== [$(stamp)] SKIP $tag (games=0) ==="; return 0; }
    local t0=$SECONDS
    echo "=== [$(stamp)] PLAY  $tag : preset=$preset games=$games ==="
    python selfplay/tournament.py \
        --p1-config "$OFF" --p2-config "$ON" \
        --p1-label nmp_off --p2-label nmp_on \
        --preset "$preset" --games "$games" --seed "$SEED" \
        --quiet --resume --tag "$tag"
    local dt=$((SECONDS - t0))
    # Measured wall-clock for future sizing. NB: with --resume, dt/games understates if games were skipped.
    echo "=== [$(stamp)] DONE  $tag in ${dt}s ($((dt / games))s/game wall over $games target games) ==="
    echo
}

annotate() {  # tag
    local tag="$1" dir="selfplay/games/$1"
    [ -d "$dir" ] || return 0
    local t0=$SECONDS
    echo "=== [$(stamp)] ANNOTATE $tag ==="
    python selfplay/annotate.py --tag "$tag" --sf-movetime "$SF_MOVETIME" \
        || echo "!!! annotate $tag failed (games are safe; rerun: python selfplay/annotate.py --tag $tag)"
    echo "=== [$(stamp)] ANNOTATE $tag done in $((SECONDS - t0))s (per-position ms in the [annotate] timing line) ==="
    echo
}

echo "### overnight_battery: NULLMOVE_PROGRESSIVE A/B (off vs on), STANDARD-weighted ###"
echo "### standard=$STANDARD_GAMES lightning=$LIGHTNING_GAMES blitz=$BLITZ_GAMES seed=$SEED sf_movetime=$SF_MOVETIME ###"
echo

BATTERY_START=$SECONDS

# Phase 1 -- play all games, fast-first (cheap regression checks before the slow depth signal).
run LIGHTNING "$LIGHTNING_GAMES" nmp_lightning
run BLITZ     "$BLITZ_GAMES"     nmp_blitz
run STANDARD  "$STANDARD_GAMES"  nmp_standard

# Phase 2 -- annotate everything (SF search + SF static + our eval breakdown). Tail step; non-fatal; games
# are already saved, so a window overrun loses only annotation (re-runnable in the morning).
echo "### [$(stamp)] PHASE 2: annotation ###"
echo
annotate nmp_lightning
annotate nmp_blitz
annotate nmp_standard

echo "=== [$(stamp)] BATTERY COMPLETE in $(( (SECONDS - BATTERY_START) / 60 ))min ($((SECONDS - BATTERY_START))s). Results: selfplay/games/nmp_{lightning,blitz,standard}/ ==="
