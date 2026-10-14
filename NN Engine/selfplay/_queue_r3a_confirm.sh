#!/bin/bash
# 2026-10-14, queue #64 — R3A whole, SF18 @1000 on the fresh seeds 224/225 of #63 (its gen-3 baselines already played there:
# 54.1% / 55.8%), so R3A gets a 4-seed pooled SF18 read (222-225). #63: neither half reproduced R3A's −14 (values-only +3,
# cells-only +4, self-play +10 / +12) ⇒ R3A's seed-222/223 read looks like −1σ noise. PREDICTION (registered): R3A ≈ +0…+8 on
# 224/225; 4-seed pooled ≈ −5…+0. Node-limited (safe any hour). ≤ 4 engines. Nothing ships without the owner + Mediocre.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
D=/mnt/e/chess_data/retune
cd "$ND" || exit 1
echo "[q64] $(date) start"
FP=$(env V2_PRESET=shipped bash "$R" wac fp_v2 2>&1 | grep -E "SOLVED|NODES" | tr '\n' ' ')
echo "$FP" | grep -q "259/300.*47869841" || { echo "[q64] ☠️ FINGERPRINT MISMATCH (expected gen 3) — aborting"; exit 1; }
for s in 224 225; do
  bash "$R" gauntlet 500 1000 4 g1000_r3a_s$s $s V2_PRESET=shipped $(cat $D/arm_r3a/env.txt)
  echo "[q64] $(date) SF18 seed $s done"
done
echo "[q64] $(date) QUEUE 64 DONE"
