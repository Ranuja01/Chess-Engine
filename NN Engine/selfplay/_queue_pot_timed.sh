#!/bin/bash
# 2026-10-16 night, queue #75 (v2) — TARGETED checks of POT set A on BALANCED endgame starts, chained after #74.
# ☠️ v1 of this queue used kp_all_fens (random pawn-dominated starts) — #74's targeted match on them was UNREADABLE: 513 of 600
# opening pairs split 1-1 by colour (170 resigned on ply 1), only 87 discriminated (POT 51.1% there). Replaced by
# selfplay/eg_balanced_fens.txt (diagnostics/_make_balanced_eg_starts.py): 300 pure-pawn + 300 one-minor endings that SF18 d14
# rates within ±80 cp, ≤ 2 per source game.
#  1. NODE-LIMITED @50k, 1,200 games (600 starts × both colours), seed 265 — starts as soon as #74 is done.
#     PREDICTION (registered): +10…+40, and far more discriminating pairs than #74's 87/600.
#  2. TIMED (LIGHTNING, ~1 s/move), 600 games, seed 266 — only 00:00-08:59 (owner plays 9pm-midnight). The race costs −22% NPS
#     inside pure pawn endings and node-limited games cannot charge it. PREDICTION: positive, 10-30% below step 1.
# ≤ 3 processes. ⚠️ Machine suspend corrupts timed games.
A="POT_V2_RACE=1 POT_V2_WIN_PAWNEND=64 POT_V2_WIN_ONEMINOR=16 POT_V2_WIN_RACE=-12 POT_V2_RACE_W_BEST=137 POT_V2_RACE_W_PROMO=217 POT_V2_RACE_W_TEMPI=202 POT_V2_RACE_W_DECISIVE=51"
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
OPEN=selfplay/eg_balanced_fens.txt
until grep -qE "QUEUE 74 DONE|☠️" /mnt/e/chess_data/q74_pot_games.log 2>/dev/null; do sleep 120; done
cd "$ND" || exit 1
echo "[q75] $(date) start"
FP=$(env V2_PRESET=shipped bash "$R" wac fp_v2 2>&1 | grep -E "SOLVED|NODES" | tr '\n' ' ')
echo "$FP" | grep -q "259/300.*47869841" || { echo "[q75] ☠️ FINGERPRINT MISMATCH (expected gen 3) — aborting"; exit 1; }
match() {   # $1 tag, $2 games, $3 seed, $4 preset, $5 node-limit config suffix
  bash "$R" pyrun selfplay/tournament.py --p1-label $1 --p1-config "V2_PRESET=shipped $A MAX_DEPTH=64 $5" \
    --p2-label shipped --p2-config "V2_PRESET=shipped MAX_DEPTH=64 $5" --games $2 --concurrency 3 \
    --preset $4 --max-plies 400 --openings $OPEN --seed $3 --tag $1 --quiet
  echo "[q75] $(date) $1: $(grep -oE '"elo(_margin)?": [-0-9.]+' selfplay/games/$1/tournament.json | tr '\n' ' ')"
}
match potA_egbal265 1200 265 LONG_FORMAT NODE_LIMIT=50000
until [ "$(date +%H)" -lt 9 ]; do sleep 300; done
match potA_egbal_timed266 600 266 LIGHTNING ""
echo "[q75] $(date) QUEUE 75 DONE"
