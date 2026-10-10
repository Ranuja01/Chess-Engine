#!/bin/bash
# 2026-10-10, queue #50 — STRUCTURAL RETUNE, first games SCREEN (owner away, autonomous; screening only — nothing ships).
# Arms = the HYBRID fit (_retune_fit.py EXPERIMENT=hybrid: static win% on quiet SF18 labels + depth target on 137k d10 rows,
# flat bands, stress 0.5, piece-value cells frozen, KD mg frozen, isolated-pawn cells file-tied), exported as engine tables
# (_retune_export.py): h01 = λ 0.1 (moves the ship's static eval ~27 cp rms; val static −9.3% / depth −5.6%), h1 = λ 1
# (~16 cp; −4.3% / −2.1%). Closure (engine vs fit, held-out quiet rows) checked before launch.
# PREDICTIONS (registered before the run): h01 self-play +0…+20, h1 +0…+12 (accuracy gains have not become Elo before:
# memory real-d10-accuracy-gains-did-not-become-elo — a null here would not surprise).
# Self-play @50k nodes, 2,000 games per arm vs the ship, fresh seed 211. Node-limited (safe in the owner's evening).
# ≤ 4 engines. No build. Guards: fingerprint → per-arm colour + file symmetry (0 violations each) → games.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
D=/mnt/e/chess_data/retune
cd "$ND" || exit 1
echo "[q50] $(date) start"
FP=$(env V2_PRESET=shipped bash "$R" wac fp_v2 2>&1 | grep -E "SOLVED|NODES" | tr '\n' ' ')
echo "[q50] v2 shipped: $FP"
echo "$FP" | grep -q "254/300.*50622239" || { echo "[q50] ☠️ FINGERPRINT MISMATCH — aborting"; exit 1; }
for arm in h01t h1t; do
  ENVL=$(cat $D/arm_$arm/env.txt)
  SYM=$(env V2_PRESET=shipped $ENVL bash "$R" pyrun diagnostics/_eval_symmetry.py N=4000 2>&1 | grep -E "violations")
  echo "[q50] $arm symmetry: $(echo "$SYM" | tr '\n' ' ')"
  [ "$(echo "$SYM" | grep -c 'violations 0 ')" = "2" ] || { echo "[q50] ☠️ $arm symmetry FAILED — aborting"; exit 1; }
done
for arm in h01t h1t; do
  ENVL=$(cat $D/arm_$arm/env.txt)
  tag=retune_${arm}_sp211
  bash "$R" pyrun selfplay/tournament.py --p1-label $tag --p1-config "V2_PRESET=shipped $ENVL MAX_DEPTH=64 NODE_LIMIT=50000" \
    --p2-label shipped --p2-config "V2_PRESET=shipped MAX_DEPTH=64 NODE_LIMIT=50000" --games 2000 --concurrency 4 \
    --preset LONG_FORMAT --max-plies 400 --openings selfplay/openings_uho.txt --seed 211 --tag $tag --quiet
  echo "[q50] $(date) $tag: $(grep -o '"elo": [-0-9.]*' selfplay/games/$tag/tournament.json)"
done
echo "[q50] $(date) QUEUE 50 DONE"
