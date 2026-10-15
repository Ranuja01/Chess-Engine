#!/bin/bash
# 2026-10-15, queue #69 — PAWN-ENDING SEARCH RULES + POT data (POT brief §5). ≤ 3 processes at any time (owner: the box also
# hosts a live server — memory no-side-runs-while-a-queue-saturates-the-box). Fixed-depth / node-limited only (safe any hour).
#  1. BUILD with the new search knobs (NULL_NPM_GUARD / PE_PRUNE_GATE / PE_PASSER_EXT, all default = today) → fingerprints
#     v1 250 / 35,310,778 and gen 3 259 / 47,869,841 must be UNCHANGED (knobs at 0 ⇒ byte-identical), + symmetry.
#  2. d10 search per arm on the pawn-ending rows (313 own + 2,000 EGX, SF18 d14): base · g1 (npm == 0) · g2 (legacy OR npm == 0)
#     · gate (no shallow pruning, pawns only) · ext (passed-push extension) · all (g1 + gate + ext) → _pe_arms.py MODE=read.
#     PREDICTION (registered): base bias ≈ −8 on "own"; g1 and gate each move it toward 0 (≥ +2 pts), ext ≈ 0; "all" nearest 0
#     (≈ −3); changed-rate > 0 for every arm (else it did not fire); nodes up on pawn rows (+10…+60%) — cost read separately.
#  (3-4, the long data runs — multi-PV labels + endgame trajectories — are queue #70 `_queue_pot_data.sh`, OVERNIGHT: owner
#   10-15 "long runs overnight, the ones you may need to ask me about while I'm awake".)
# Nothing ships from this queue; games come after the read-out, with the owner.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
P=/mnt/e/chess_data/pe
mkdir -p "$P"
cd "$ND" || exit 1
echo "[q69] $(date) start"
B=$(bash "$R" build 2>&1)
echo "$B" | tail -3
echo "$B" | grep -q "BUILD FAILED" && { echo "[q69] ☠️ BUILD FAILED — aborting"; exit 1; }
FP1=$(bash "$R" wac fp_v1 2>&1 | grep -E "SOLVED|NODES" | tr '\n' ' ')
echo "[q69] v1: $FP1"
echo "$FP1" | grep -q "250/300.*35310778" || { echo "[q69] ☠️ v1 FINGERPRINT MOVED — aborting"; exit 1; }
FP3=$(env V2_PRESET=shipped bash "$R" wac fp_v2 2>&1 | grep -E "SOLVED|NODES" | tr '\n' ' ')
echo "[q69] gen 3: $FP3"
echo "$FP3" | grep -q "259/300.*47869841" || { echo "[q69] ☠️ gen-3 FINGERPRINT MOVED — aborting"; exit 1; }
SYM=$(env V2_PRESET=shipped bash "$R" pyrun diagnostics/_eval_symmetry.py N=4000 2>&1 | grep -E "violations")
echo "[q69] symmetry: $(echo "$SYM" | tr '\n' ' ')"
[ "$(echo "$SYM" | grep -c 'violations 0 ')" = "2" ] || { echo "[q69] ☠️ symmetry FAILED — aborting"; exit 1; }
bash "$R" pyrun diagnostics/_pe_arms.py MODE=prep

arm_env() {
  case "$1" in
    base) echo "" ;;
    g1)   echo "NULL_NPM_GUARD=1" ;;
    g2)   echo "NULL_NPM_GUARD=2" ;;
    gate) echo "PE_PRUNE_GATE=1" ;;
    ext)  echo "PE_PASSER_EXT=3" ;;
    all)  echo "NULL_NPM_GUARD=1 PE_PRUNE_GATE=1 PE_PASSER_EXT=3" ;;
  esac
}
for arm in base g1 g2 gate ext all; do
  K=$(arm_env $arm)
  for s in 0 1 2; do
    ( until env PRESET=LONG_FORMAT MAX_DEPTH=10 USE_OPENING_BOOK=0 V2_PRESET=shipped $K bash "$R" pyrun \
          diagnostics/_depth_residual_pass.py IN=$P/pe_in.csv OUT=$P/d10_$arm.csv SHARD=$s/3 CHUNK=500 > /dev/null 2>&1; \
        [ $? -ne 3 ]; do :; done ) &
  done
  wait
  echo "[q69] $(date) d10 arm $arm done ($K): $(cat $P/d10_${arm}_s*of3.csv | grep -vc '^fen') rows"
done
bash "$R" pyrun diagnostics/_pe_arms.py MODE=read ARMS=base,g1,g2,gate,ext,all
echo "[q69] $(date) QUEUE 69 DONE"
