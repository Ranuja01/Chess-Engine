#!/bin/bash
# 2026-10-12, queue #58 — retune round 2, the THIRD-OPPONENT leg for both arms on the gen-2 base.
#   g2l003 (no values, λ 0.03): self-play +31.7 ± 17.9 · SF18 pooled 53.05% vs ship 52.5% (≈ +4 ± 14)
#   g2v003 (values + PST pin):  self-play +22.8 ± 17.9 · SF18 pooled 53.85% vs 52.5% (≈ +9 ± 14)
# Mediocre v0.5 @ ~1 s/move, 200 g per arm, fresh seed 218, concurrency 4 (owner OK'd 10-11; weekday afternoon, done before
# the 9pm window). PREDICTIONS (registered): both within ±25 of the ship, g2v003 ≥ g2l003. Must stop before 21:00.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
D=/mnt/e/chess_data/retune
cd "$ND" || exit 1
echo "[q58] $(date) start"
for arm in ship g2l003 g2v003; do
  [ "$(date +%H)" -ge 20 ] && { echo "[q58] $(date) $arm SKIPPED — would run into the owner's 9pm window"; continue; }
  K=""; [ "$arm" != ship ] && K=$(cat $D/arm_$arm/env.txt)
  bash "$R" pyrun selfplay/vs_sf.py --our-label $arm --our-config "V2_PRESET=shipped $K MAX_DEPTH=64 USE_OPENING_BOOK=0" \
    --preset LIGHTNING --sf-path "$ND/selfplay/mediocre_uci.sh" --opponent-raw --sf-elo 0 --sf-movetime 1.0 \
    --games 200 --concurrency 4 --openings selfplay/openings_uho.txt --seed 218 --tag mediocre_g2_${arm}_s218 --adjudicate-draw --quiet
  echo "[q58] $(date) mediocre $arm done"
done
echo "[q58] $(date) QUEUE 58 DONE"
