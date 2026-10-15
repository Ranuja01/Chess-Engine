#!/bin/bash
# 2026-10-15, queue #68 — GATE KS_V2_HALF=818 (shipped 772), the only knob SPSA #65 replicated (cold starts → 814 / 822).
# Run BEFORE POT's fit: if it ships it joins the frozen base POT is fitted on. Existing knobs only (no build).
# Owner rule: self-play ≥ +2σ (elo ≥ elo_margin, the 95% half-width) → SF18 @1100 (new anchor, between @1000 54.0% and
# @1500 37.3%) pooled over 2 fresh seeds > −1σ → Mediocre ≥ 200 g (daytime, separate). SF18 runs ONLY if self-play passes.
# PREDICTION (registered): a 6% move of a KS curve midpoint is small ⇒ self-play +0…+8, most likely BELOW the +2σ bar
# (~±18 at 2,000 g) ⇒ "unresolved, not shipped". Node-limited (safe any hour). ≤ 4 engines.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
K="KS_V2_HALF=818"
cd "$ND" || exit 1
echo "[q68] $(date) start"
FP=$(env V2_PRESET=shipped bash "$R" wac fp_v2 2>&1 | grep -E "SOLVED|NODES" | tr '\n' ' ')
echo "[q68] base: $FP"
echo "$FP" | grep -q "259/300.*47869841" || { echo "[q68] ☠️ FINGERPRINT MISMATCH (expected gen 3) — aborting"; exit 1; }
# Liveness: the knob must MOVE the fingerprint (byte-identical = silent fallback).
FPK=$(env V2_PRESET=shipped $K bash "$R" wac fp_ks818 2>&1 | grep -E "SOLVED|NODES" | tr '\n' ' ')
echo "[q68] $K: $FPK"
echo "$FPK" | grep -q "47869841" && { echo "[q68] ☠️ $K is byte-identical to gen 3 — knob not live, aborting"; exit 1; }
SYM=$(env V2_PRESET=shipped $K bash "$R" pyrun diagnostics/_eval_symmetry.py N=4000 2>&1 | grep -E "violations")
echo "[q68] symmetry: $(echo "$SYM" | tr '\n' ' ')"
[ "$(echo "$SYM" | grep -c 'violations 0 ')" = "2" ] || { echo "[q68] ☠️ symmetry FAILED — aborting"; exit 1; }
tag=ks818_sp240
bash "$R" pyrun selfplay/tournament.py --p1-label $tag --p1-config "V2_PRESET=shipped $K MAX_DEPTH=64 NODE_LIMIT=50000" \
  --p2-label shipped --p2-config "V2_PRESET=shipped MAX_DEPTH=64 NODE_LIMIT=50000" --games 2000 --concurrency 4 \
  --preset LONG_FORMAT --max-plies 400 --openings selfplay/openings_uho.txt --seed 240 --tag $tag --quiet
J=selfplay/games/$tag/tournament.json
ELO=$(grep -oP '"elo": \K[-0-9.]+' $J); MAR=$(grep -oP '"elo_margin": \K[-0-9.]+' $J)
echo "[q68] $(date) self-play $tag: elo $ELO ± $MAR"
if ! awk -v e="$ELO" -v m="$MAR" 'BEGIN{exit !(e != "" && e >= m)}'; then
  echo "[q68] self-play below +2σ ⇒ not shipped, SF18 skipped"; echo "[q68] $(date) QUEUE 68 DONE"; exit 0
fi
for s in 241 242; do
  bash "$R" gauntlet 500 1100 4 g1100_g3ship_s$s $s V2_PRESET=shipped
  bash "$R" gauntlet 500 1100 4 g1100_ks818_s$s $s V2_PRESET=shipped $K
  echo "[q68] $(date) SF18 @1100 seed $s done"
done
echo "[q68] $(date) QUEUE 68 DONE"
