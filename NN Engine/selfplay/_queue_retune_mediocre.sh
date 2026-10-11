#!/bin/bash
# 2026-10-11, queue #52 — STRUCTURAL RETUNE, the THIRD-OPPONENT leg of the owner's ship rule for hybrid λ0.1 (arm_h01t).
# So far: self-play +56.6 ± 17.9 (#50) · SF18 @1000 pooled 53.1% vs ship 50.0% ≈ +22 ± 14 (#51, seeds 212/213).
# Rule: Mediocre v0.5 not negative, ≥ 200 games per arm (50 g = ±54 Elo, inconclusive). TIMED (~1 s/move) ⇒ owner-approved
# window (10-11 late morning, concurrency 4 OK; owner will say if the machine is needed back). Fresh seed 214, ship first.
# PREDICTION (registered): h01t − ship ≈ +10…+40 (external SF18 read +22).
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
D=/mnt/e/chess_data/retune
cd "$ND" || exit 1
echo "[q52] $(date) start"
FP=$(env V2_PRESET=shipped bash "$R" wac fp_v2 2>&1 | grep -E "SOLVED|NODES" | tr '\n' ' ')
echo "[q52] v2 shipped: $FP"
echo "$FP" | grep -q "254/300.*50622239" || { echo "[q52] ☠️ FINGERPRINT MISMATCH — aborting"; exit 1; }
for arm in ship h01t; do
  K=""; [ "$arm" = h01t ] && K=$(cat $D/arm_h01t/env.txt)
  bash "$R" pyrun selfplay/vs_sf.py --our-label $arm --our-config "V2_PRESET=shipped $K MAX_DEPTH=64 USE_OPENING_BOOK=0" \
    --preset LIGHTNING --sf-path "$ND/selfplay/mediocre_uci.sh" --opponent-raw --sf-elo 0 --sf-movetime 1.0 \
    --games 200 --concurrency 4 --openings selfplay/openings_uho.txt --seed 214 --tag mediocre_rt_${arm}_s214 --adjudicate-draw --quiet
  echo "[q52] $(date) mediocre $arm done: $(grep -h 'our score' selfplay/games/mediocre_rt_${arm}_s214/*.log 2>/dev/null | tail -1)"
done
echo "[q52] $(date) QUEUE 52 DONE"
