#!/bin/bash
# 2026-10-16 night, queue #75 — TIMED targeted check of POT set A (chained after #74). Waits for "QUEUE 74 DONE" (or a ☠️ abort)
# AND for 00:00-08:59 (owner plays 9pm-midnight; timed games must not share the box with them).
# WHY: the race costs −22% NPS inside pure pawn endings (#73) and node-limited games cannot charge it. 600 games at LIGHTNING
# (the owner's UI preset, ~1 s/move) from the pawn-dominated starts — exactly where the cost and the gain both live.
# PREDICTION (registered): still positive at equal time, smaller than #74's targeted node-limited result (by ~10-30%).
# ≤ 3 processes. ⚠️ Machine suspend corrupts timed games (memory machine-suspend-corrupts-fixed-time-games).
A="POT_V2_RACE=1 POT_V2_WIN_PAWNEND=64 POT_V2_WIN_ONEMINOR=16 POT_V2_WIN_RACE=-12 POT_V2_RACE_W_BEST=137 POT_V2_RACE_W_PROMO=217 POT_V2_RACE_W_TEMPI=202 POT_V2_RACE_W_DECISIVE=51"
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
until grep -qE "QUEUE 74 DONE|☠️" /mnt/e/chess_data/q74_pot_games.log 2>/dev/null; do sleep 120; done
until [ "$(date +%H)" -lt 9 ]; do sleep 300; done
cd "$ND" || exit 1
echo "[q75] $(date) start"
FP=$(env V2_PRESET=shipped bash "$R" wac fp_v2 2>&1 | grep -E "SOLVED|NODES" | tr '\n' ' ')
echo "$FP" | grep -q "259/300.*47869841" || { echo "[q75] ☠️ FINGERPRINT MISMATCH (expected gen 3) — aborting"; exit 1; }
tag=potA_kp_timed264
bash "$R" pyrun selfplay/tournament.py --p1-label $tag --p1-config "V2_PRESET=shipped $A MAX_DEPTH=64" \
  --p2-label shipped --p2-config "V2_PRESET=shipped MAX_DEPTH=64" --games 600 --concurrency 3 \
  --preset LIGHTNING --max-plies 400 --openings selfplay/kp_all_fens.txt --seed 264 --tag $tag --quiet
echo "[q75] $(date) TIMED $tag: $(grep -oE '"elo(_margin)?": [-0-9.]+' selfplay/games/$tag/tournament.json | tr '\n' ' ')"
echo "[q75] $(date) QUEUE 75 DONE"
