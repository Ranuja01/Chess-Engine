#!/bin/bash
# 2026-10-13, queue #61 — RETUNE ROUND 3 on the gen-3 base, end to end (owner away ~1.5 days; autonomous; ship = owner's call).
# Waits for #60 (cap dumps on v1 + GAUNT slice labelled). Arms, both WIN_AWARE (rows scaled by the winnability factor m) + the 4
# winnability CAPS + V2_VAL values (PST pinned), λ 0.03, depth target = gen-3 d10 pass (A_ours1012):
#   R3A = corpus_v2 (+ GAUNT: games vs SF18 / Mediocre)   R3B = corpus_v1 (no GAUNT) — the slice's own contribution
# Gates per arm: closure REALISED ≥ 70% of the predicted gain · colour + file symmetry 0 · then self-play @50k 2,000 g vs gen 3
# (seed 221) · SF18 @1000 (seeds 222/223, gen-3 baseline per seed) for arms ≥ +18 self-play (~2σ). Mediocre is TIMED ⇒ scheduled
# by hand in a daytime window. PREDICTIONS (registered): R3A +0…+20, R3B +0…+15 self-play; caps fire on ~10-20% of endgames.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
D=/mnt/e/chess_data/retune
until grep -qE "QUEUE 60 DONE|☠️" /mnt/e/chess_data/q60_round3_data.log 2>/dev/null; do sleep 120; done
grep -q "☠️" /mnt/e/chess_data/q60_round3_data.log && { echo "[q61] ☠️ queue #60 aborted"; exit 1; }
cd "$ND" || exit 1
echo "[q61] $(date) start"
FP=$(env V2_PRESET=shipped bash "$R" wac fp_v2 2>&1 | grep -E "SOLVED|NODES" | tr '\n' ' ')
echo "$FP" | grep -q "259/300.*47869841" || { echo "[q61] ☠️ FINGERPRINT MISMATCH (expected gen 3) — aborting"; exit 1; }
# (1) corpus v2 + its gen-3 exports (base + caps)
bash "$R" pyrun diagnostics/_corpus_assemble.py MODE=assemble OUT=corpus_v2.csv 2>&1 | grep -E "kept|GAUNT|ASSEMBLE|rror" | sed 's/^/[q61] /'
env V2_PRESET=shipped bash "$R" pyrun diagnostics/_px_export.py IN=$D/corpus_v2.csv OUT=$D/feat_gen3v2.npz 2>&1 | grep -E "CORPUS EXPORT DONE"
for cap in "QNOQ:POT_V2_WIN_QNOQ=-27 POT_V2_WIN_QNOQ_MINOR=3" "OCBX:POT_V2_WIN_OCBX=-17 POT_V2_WIN_OCBX_PC=1" \
           "ROOKE:POT_V2_WIN_ROOKE=-28" "LONEMINOR:POT_V2_WIN_LONEMINOR=-64"; do
  name="${cap%%:*}"; knobs="${cap#*:}"
  env V2_PRESET=shipped $knobs bash "$R" pyrun diagnostics/_px_export.py IN=$D/corpus_v2.csv OUT=$D/feat_gen3v2cap_$name.npz 2>&1 | grep -E "CORPUS EXPORT DONE"
done
echo "[q61] $(date) corpus_v2 exports done"
# (2) the two fits, in parallel (python only)
COMMON="EXPERIMENT=hybrid MODES=hybrid VAL=1 WIN_AWARE=1 CAPS=QNOQ,OCBX,ROOKE,LONEMINOR LAMBDAS=0.03"
bash "$R" pyrun diagnostics/_retune_fit.py $COMMON CORPUS=corpus_v2.csv FEAT=feat_gen3v2 'D10_GLOB=A_ours1012_d10_s*of4.csv' TAG=_r3a > $D/q61_fit_r3a.log 2>&1 &
bash "$R" pyrun diagnostics/_retune_fit.py $COMMON CORPUS=corpus_v1.csv FEAT=feat_gen3 'D10_GLOB=A_ours1012_d10_s*of4.csv' TAG=_r3b > $D/q61_fit_r3b.log 2>&1 &
wait
grep -hE "cap |WIN_AWARE|hybrid  λ|static all|depth all|static eg |depth eg |share" $D/q61_fit_r3a.log $D/q61_fit_r3b.log | sed 's/^/[q61] /'
# (3) per arm: export → closure (REALISED ≥ 70%) → symmetry → self-play
PASS=""
for arm in r3a r3b; do
  FEAT=feat_gen3v2; [ $arm = r3b ] && FEAT=feat_gen3
  env V2_PRESET=shipped bash "$R" pyrun diagnostics/_retune_export.py MODE=tables NPZ=fit_hybrid_$arm.npz ARM_I=0 OUT=$D/arm_$arm 2>&1 | grep -E "values:|caps:|EXPORT"
  ENVL=$(cat $D/arm_$arm/env.txt)
  REAL=$(env V2_PRESET=shipped $ENVL bash "$R" pyrun diagnostics/_retune_export.py MODE=verify NPZ=fit_hybrid_$arm.npz ARM_I=0 ARMDIR=$arm N=3000 \
         FEAT=$FEAT WIN_AWARE=1 2>&1 | grep -E "REALISED|^CLOSURE|winnability")
  echo "$REAL" | sed "s/^/[q61] $arm /"
  set -- $(echo "$REAL" | grep REALISED | grep -oE '\(-?[0-9.]+%\)' | tr -d '()%')
  if ! awk -v a="$1" -v b="$2" 'BEGIN{exit !(a < 0 && b/a >= 0.7)}'; then echo "[q61] $arm ✗ closure gate (pred $1% / engine $2%) — dropped"; continue; fi
  SYM=$(env V2_PRESET=shipped $ENVL bash "$R" pyrun diagnostics/_eval_symmetry.py N=4000 2>&1 | grep -E "violations")
  echo "[q61] $arm symmetry: $(echo "$SYM" | tr '\n' ' ')"
  [ "$(echo "$SYM" | grep -c 'violations 0 ')" = "2" ] || { echo "[q61] $arm ✗ symmetry — dropped"; continue; }
  tag=retune3_${arm}_sp221
  bash "$R" pyrun selfplay/tournament.py --p1-label $tag --p1-config "V2_PRESET=shipped $ENVL MAX_DEPTH=64 NODE_LIMIT=50000" \
    --p2-label shipped --p2-config "V2_PRESET=shipped MAX_DEPTH=64 NODE_LIMIT=50000" --games 2000 --concurrency 4 \
    --preset LONG_FORMAT --max-plies 400 --openings selfplay/openings_uho.txt --seed 221 --tag $tag --quiet
  ELO=$(grep -o '"elo": [-0-9.]*' selfplay/games/$tag/tournament.json | awk '{print $2}')
  echo "[q61] $(date) $tag: elo $ELO"
  awk -v e="$ELO" 'BEGIN{exit !(e >= 18)}' && PASS="$PASS $arm"
done
echo "[q61] self-play passers (≥ +18):$PASS"
# (4) SF18 for the passers
if [ -n "$PASS" ]; then
  for s in 222 223; do
    bash "$R" gauntlet 500 1000 4 g1000_g3ship_s$s $s V2_PRESET=shipped
    for arm in $PASS; do
      bash "$R" gauntlet 500 1000 4 g1000_${arm}_s$s $s V2_PRESET=shipped $(cat $D/arm_$arm/env.txt)
    done
    echo "[q61] $(date) SF18 seed $s done"
  done
fi
echo "[q61] $(date) QUEUE 61 DONE"
