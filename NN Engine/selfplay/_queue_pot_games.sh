#!/bin/bash
# 2026-10-16 night, queue #74 — FIRST GAMES for POT set A (pure-pawn-ending race + class cells + winnability feed), vs gen 3.
# ≤ 3 processes (live server on the box). Node-limited @50k (safe any hour). ⚠️ Node-limited games CANNOT charge the race's
# per-node cost (−22% NPS inside pure pawn endings, #73) ⇒ a TIMED check is owed before any ship.
#  1. guards: v1 + gen-3 fingerprints, symmetry with set A on.
#  2. TARGETED: 1,200 games from the 600 pawn-dominated starts (kp_fens + kp_dense, both colours), seed 260.
#     PREDICTION (registered): clearly positive, +20…+60 (the change acts in exactly these positions).
#  3. STANDARD: 2,000 UHO games, seed 261. PREDICTION: +0…+10 (pure pawn endings are rare in normal games).
#  4. SF18 @1100 (2 seeds × 500, gen-3 baseline per seed) IF standard ≥ +2σ, OR targeted ≥ +2σ AND standard > 0.
A="POT_V2_RACE=1 POT_V2_WIN_PAWNEND=64 POT_V2_WIN_ONEMINOR=16 POT_V2_WIN_RACE=-12 POT_V2_RACE_W_BEST=137 POT_V2_RACE_W_PROMO=217 POT_V2_RACE_W_TEMPI=202 POT_V2_RACE_W_DECISIVE=51"
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
cd "$ND" || exit 1
echo "[q74] $(date) start"
FP1=$(bash "$R" wac fp_v1 2>&1 | grep -E "SOLVED|NODES" | tr '\n' ' ')
echo "$FP1" | grep -q "250/300.*35310778" || { echo "[q74] ☠️ v1 FINGERPRINT MOVED — aborting"; exit 1; }
FP3=$(env V2_PRESET=shipped bash "$R" wac fp_v2 2>&1 | grep -E "SOLVED|NODES" | tr '\n' ' ')
echo "$FP3" | grep -q "259/300.*47869841" || { echo "[q74] ☠️ gen-3 FINGERPRINT MOVED — aborting"; exit 1; }
SYM=$(env V2_PRESET=shipped $A bash "$R" pyrun diagnostics/_eval_symmetry.py N=4000 2>&1 | grep -E "violations")
echo "[q74] symmetry (set A): $(echo "$SYM" | tr '\n' ' ')"
[ "$(echo "$SYM" | grep -c 'violations 0 ')" = "2" ] || { echo "[q74] ☠️ symmetry FAILED — aborting"; exit 1; }
grep -hv '^#' selfplay/kp_fens.txt selfplay/kp_dense_fens.txt > selfplay/kp_all_fens.txt
echo "[q74] targeted openings: $(grep -c / selfplay/kp_all_fens.txt)"

run_match() {   # $1 tag, $2 openings, $3 games, $4 seed
  bash "$R" pyrun selfplay/tournament.py --p1-label $1 --p1-config "V2_PRESET=shipped $A MAX_DEPTH=64 NODE_LIMIT=50000" \
    --p2-label shipped --p2-config "V2_PRESET=shipped MAX_DEPTH=64 NODE_LIMIT=50000" --games $3 --concurrency 3 \
    --preset LONG_FORMAT --max-plies 400 --openings $2 --seed $4 --tag $1 --quiet
}
elo() { grep -oP "\"$2\": \K[-0-9.]+" selfplay/games/$1/tournament.json; }
run_match potA_kp260 selfplay/kp_all_fens.txt 1200 260
KE=$(elo potA_kp260 elo); KM=$(elo potA_kp260 elo_margin)
echo "[q74] $(date) TARGETED potA_kp260: elo $KE ± $KM"
run_match potA_sp261 selfplay/openings_uho.txt 2000 261
SE=$(elo potA_sp261 elo); SM=$(elo potA_sp261 elo_margin)
echo "[q74] $(date) STANDARD potA_sp261: elo $SE ± $SM"
if awk -v se="$SE" -v sm="$SM" -v ke="$KE" -v km="$KM" 'BEGIN{exit !((se >= sm) || (ke >= km && se > 0))}'; then
  for s in 262 263; do
    bash "$R" gauntlet 500 1100 3 g1100_g3ship_s$s $s V2_PRESET=shipped
    bash "$R" gauntlet 500 1100 3 g1100_potA_s$s $s V2_PRESET=shipped $A
    echo "[q74] $(date) SF18 @1100 seed $s done"
  done
else
  echo "[q74] SF18 skipped (standard below +2σ and the targeted/standard condition not met)"
fi
echo "[q74] $(date) QUEUE 74 DONE"
