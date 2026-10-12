#!/bin/bash
# 2026-10-12, queue #56 — retune round 2 on the GEN-2 base: self-play screen of two λ 0.03 hybrid arms vs ship gen 2.
#   g2l003 = no value knobs (static −6.0% / depth −3.8% val; moves gen 2 ~22 cp; engine realises −3.7% of −4.3% predicted)
#   g2v003 = V2_VAL_* piece values free + PST means pinned (−5.0% / −3.1%; ~18 cp; realises −3.1% of −3.8%) — material moves
#            out of the Kaufman products into explicit per-phase values (R mg −136 / eg +127, minors eg −64…−70, Q eg −68).
# PREDICTIONS (registered): g2l003 +5…+25, g2v003 +0…+20 (round-2 gains on a better base should be smaller than round 1).
# Waits for queue #55 (≤ 4 engines). Self-play @50k, 2,000 g per arm, fresh seed 215. Guards: gen-2 fingerprint + symmetry.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
D=/mnt/e/chess_data/retune
until grep -qE "QUEUE 55 DONE|☠️" /mnt/e/chess_data/q55_bench_gen2.log 2>/dev/null; do sleep 120; done
cd "$ND" || exit 1
echo "[q56] $(date) start"
FP=$(env V2_PRESET=shipped bash "$R" wac fp_v2 2>&1 | grep -E "SOLVED|NODES" | tr '\n' ' ')
echo "[q56] v2 shipped: $FP"
echo "$FP" | grep -q "256/300.*47473313" || { echo "[q56] ☠️ FINGERPRINT MISMATCH — aborting"; exit 1; }
for arm in g2l003 g2v003; do
  ENVL=$(cat $D/arm_$arm/env.txt)
  SYM=$(env V2_PRESET=shipped $ENVL bash "$R" pyrun diagnostics/_eval_symmetry.py N=4000 2>&1 | grep -E "violations")
  echo "[q56] $arm symmetry: $(echo "$SYM" | tr '\n' ' ')"
  [ "$(echo "$SYM" | grep -c 'violations 0 ')" = "2" ] || { echo "[q56] ☠️ $arm symmetry FAILED — aborting"; exit 1; }
done
for arm in g2l003 g2v003; do
  ENVL=$(cat $D/arm_$arm/env.txt)
  tag=retune2_${arm}_sp215
  bash "$R" pyrun selfplay/tournament.py --p1-label $tag --p1-config "V2_PRESET=shipped $ENVL MAX_DEPTH=64 NODE_LIMIT=50000" \
    --p2-label shipped --p2-config "V2_PRESET=shipped MAX_DEPTH=64 NODE_LIMIT=50000" --games 2000 --concurrency 4 \
    --preset LONG_FORMAT --max-plies 400 --openings selfplay/openings_uho.txt --seed 215 --tag $tag --quiet
  echo "[q56] $(date) $tag: $(grep -o '"elo": [-0-9.]*' selfplay/games/$tag/tournament.json)"
done
echo "[q56] $(date) QUEUE 56 DONE"
