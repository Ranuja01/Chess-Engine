#!/bin/bash
# 2026-10-16, queue #72 — POT knobs at DEPTH (stage 2 of the knob fit; memory static-only-fit-worsens-the-depth-target).
# d10 search on the 2,313 pawn-only rows (pe/pe_in.csv: 313 OWN-game rows = held out from the fit · 1,972 EGX rows = in the
# fit's data) for four candidates vs #69's base: A full static fit · B half · C class cells only · D race only.
# PREDICTION (registered): the static fit OVERSHOOTS at depth — A's bias goes positive (overconfident) on "own" and its MSE
# is worse than B's; C carries most of the bias fix; D ≈ small. ≤ 3 processes. Fixed depth (safe any hour).
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
P=/mnt/e/chess_data/pe
cd "$ND" || exit 1
echo "[q72] $(date) start"
FP=$(env V2_PRESET=shipped bash "$R" wac fp_v2 2>&1 | grep -E "SOLVED|NODES" | tr '\n' ' ')
echo "$FP" | grep -q "259/300.*47869841" || { echo "[q72] ☠️ FINGERPRINT MISMATCH (expected gen 3) — aborting"; exit 1; }
depth_arm() {   # $1 = arm name, $2... = knobs; 3 shards, chunked (run_one leaks)
  arm=$1; shift
  for s in 0 1 2; do
    ( until timeout 5400 env PRESET=LONG_FORMAT MAX_DEPTH=10 USE_OPENING_BOOK=0 V2_PRESET=shipped "$@" bash "$R" pyrun \
          diagnostics/_depth_residual_pass.py IN=$P/pe_in.csv OUT=$P/d10_$arm.csv SHARD=$s/3 CHUNK=500 > /dev/null 2>&1; \
        [ $? -ne 3 ]; do :; done ) &
  done
  wait
  echo "[q72] $(date) arm $arm: $(cat $P/d10_${arm}_s*of3.csv 2>/dev/null | grep -vc '^fen') rows"
}
RACE="POT_V2_RACE=1 POT_V2_RACE_W_BEST=137 POT_V2_RACE_W_PROMO=217 POT_V2_RACE_W_TEMPI=202 POT_V2_RACE_W_DECISIVE=51"
# ☠️ 10-16 run 1: the amplified pawn-ending evals crossed RESIGN_THRESHOLD (−15,000 mp) ⇒ the engine RESIGNED (no move, no
# eval) on 113 decisive rows of arm A, which the depth pass silently skipped. Re-run with resignation disabled (suffix r).
NR="RESIGN_THRESHOLD=-99999999"
if [ "${STAGE:-r}" = "r" ]; then
  depth_arm potAr $NR $RACE POT_V2_WIN_PAWNEND=64 POT_V2_WIN_ONEMINOR=16 POT_V2_WIN_RACE=-12
  depth_arm potBr $NR POT_V2_RACE=1 POT_V2_RACE_W_BEST=68 POT_V2_RACE_W_PROMO=108 POT_V2_RACE_W_TEMPI=101 POT_V2_RACE_W_DECISIVE=25 \
                 POT_V2_WIN_PAWNEND=32 POT_V2_WIN_ONEMINOR=8 POT_V2_WIN_RACE=-6
  depth_arm potCr $NR POT_V2_WIN_PAWNEND=64 POT_V2_WIN_ONEMINOR=16
  bash "$R" pyrun diagnostics/_pe_arms.py MODE=read ARMS=base,potAr,potBr,potCr,potD,d14,sf11d10
else
  depth_arm potA $RACE POT_V2_WIN_PAWNEND=64 POT_V2_WIN_ONEMINOR=16 POT_V2_WIN_RACE=-12
  depth_arm potB POT_V2_RACE=1 POT_V2_RACE_W_BEST=68 POT_V2_RACE_W_PROMO=108 POT_V2_RACE_W_TEMPI=101 POT_V2_RACE_W_DECISIVE=25 \
                 POT_V2_WIN_PAWNEND=32 POT_V2_WIN_ONEMINOR=8 POT_V2_WIN_RACE=-6
  depth_arm potC POT_V2_WIN_PAWNEND=64 POT_V2_WIN_ONEMINOR=16
  depth_arm potD $RACE
  bash "$R" pyrun diagnostics/_pe_arms.py MODE=read ARMS=base,potA,potB,potC,potD,d12,d14,sf11d10
fi
echo "[q72] $(date) QUEUE 72 DONE"
