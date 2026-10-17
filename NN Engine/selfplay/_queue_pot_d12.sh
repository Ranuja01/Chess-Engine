#!/bin/bash
# 2026-10-17, queue #76 — POT at DEPTH 12 (calibration check behind the timed result) + the cells-only candidate.
# Set A was fitted / checked at d10 (bias +0.09); LIGHTNING games search ~d12-14 where our base bias is smaller (d12 −5.8,
# d14 −4.2) ⇒ a d10-tuned correction may OVERSHOOT deeper. Arms on the 2,313 pawn rows (OWN 313 held out), resign disabled:
#   d10: C99 (cells-only refit PAWNEND 99 / ONEMINOR 16)
#   d12: A · C64 (cells 64/16, no race) · C99      vs the existing base d12 arm ("d12").
# PREDICTION (registered): at d12 set A's bias is POSITIVE (+1…+3, overconfident) on own rows; C99 overshoots more than C64.
# ≤ 3 processes. Fixed depth (safe any hour).
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
P=/mnt/e/chess_data/pe
cd "$ND" || exit 1
echo "[q76] $(date) start"
FP=$(env V2_PRESET=shipped bash "$R" wac fp_v2 2>&1 | grep -E "SOLVED|NODES" | tr '\n' ' ')
echo "$FP" | grep -q "259/300.*47869841" || { echo "[q76] ☠️ FINGERPRINT MISMATCH (expected gen 3) — aborting"; exit 1; }
NR="RESIGN_THRESHOLD=-99999999"
A="POT_V2_RACE=1 POT_V2_WIN_PAWNEND=64 POT_V2_WIN_ONEMINOR=16 POT_V2_WIN_RACE=-12 POT_V2_RACE_W_BEST=137 POT_V2_RACE_W_PROMO=217 POT_V2_RACE_W_TEMPI=202 POT_V2_RACE_W_DECISIVE=51"
depth_arm() {   # $1 depth, $2 arm name, $3... knobs
  dd=$1; arm=$2; shift 2
  for s in 0 1 2; do
    ( until timeout 5400 env PRESET=LONG_FORMAT MAX_DEPTH=$dd USE_OPENING_BOOK=0 V2_PRESET=shipped $NR "$@" bash "$R" pyrun \
          diagnostics/_depth_residual_pass.py IN=$P/pe_in.csv OUT=$P/d10_$arm.csv SHARD=$s/3 CHUNK=500 > /dev/null 2>&1; \
        [ $? -ne 3 ]; do :; done ) &
  done
  wait
  echo "[q76] $(date) arm $arm (d$dd): $(cat $P/d10_${arm}_s*of3.csv 2>/dev/null | grep -vc '^fen') rows"
}
depth_arm 10 potC99 POT_V2_WIN_PAWNEND=99 POT_V2_WIN_ONEMINOR=16
depth_arm 12 potA12 $A
depth_arm 12 potC64_12 POT_V2_WIN_PAWNEND=64 POT_V2_WIN_ONEMINOR=16
depth_arm 12 potC99_12 POT_V2_WIN_PAWNEND=99 POT_V2_WIN_ONEMINOR=16
echo "[q76] d10 arms:"
bash "$R" pyrun diagnostics/_pe_arms.py MODE=read ARMS=base,potAr,potCr,potC99
echo "[q76] d12 arms:"
bash "$R" pyrun diagnostics/_pe_arms.py MODE=read ARMS=d12,potA12,potC64_12,potC99_12
echo "[q76] $(date) QUEUE 76 DONE"
