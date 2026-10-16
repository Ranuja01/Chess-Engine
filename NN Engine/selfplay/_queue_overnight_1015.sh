#!/bin/bash
# 2026-10-15 night, queue #71 — the overnight chain (owner OK'd). ≤ 3 processes at any time (live server on the box).
# Node-limited / fixed-depth only. Each stage independent: a failure is logged and the chain moves on.
#  1. CAUSE CHECKS on the 2,313 pawn-only rows (pe/pe_in.csv; #69 left base d10 + 5 arms):
#     (a) our d10 with the winnability scale OFF (POT_V2_WIN=0)  — is the drawishness the scale over-damping?
#         PREDICTION (registered): bias −7.3 → ≈ −5 (the scale explains some, not most).
#     (b) bias BY DEPTH: ours d6 / d8 / d12 / d14 (d10 = base) and SF11 d6 … d14 on the same rows — horizon (gap closes with
#         depth) vs constant eval offset. PREDICTION: ours improves slowly (d14 ≈ −5.5), SF11 near 0 from d10.
#  2. queue #70 (_queue_pot_data.sh): SF18 multi-PV on 6,000 light endgames + endgame trajectory slice.
#  3. NULL_NPM_GUARD=1 self-play @50k vs gen 3, 2,000 g, concurrency 3, seed 250 (a general correctness fix: −2% nodes on
#     pawn rows; node-limited games credit node savings). PREDICTION: +0…+8 (likely below +2σ ≈ 18).
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
P=/mnt/e/chess_data/pe
cd "$ND" || exit 1
echo "[q71] $(date) start"
FP=$(env V2_PRESET=shipped bash "$R" wac fp_v2 2>&1 | grep -E "SOLVED|NODES" | tr '\n' ' ')
echo "$FP" | grep -q "259/300.*47869841" || { echo "[q71] ☠️ FINGERPRINT MISMATCH (expected gen 3) — aborting"; exit 1; }

depth_arm() {   # $1 = arm name, $2 = depth, $3... = extra knobs; 3 shards, chunked (run_one leaks), each chunk capped at 90 min
  arm=$1; dd=$2; shift 2
  for s in 0 1 2; do
    ( until timeout 5400 env PRESET=LONG_FORMAT MAX_DEPTH=$dd USE_OPENING_BOOK=0 V2_PRESET=shipped "$@" bash "$R" pyrun \
          diagnostics/_depth_residual_pass.py IN=$P/pe_in.csv OUT=$P/d10_$arm.csv SHARD=$s/3 CHUNK=500 > /dev/null 2>&1; \
        [ $? -ne 3 ]; do :; done ) &
  done
  wait
  echo "[q71] $(date) arm $arm (d$dd $*): $(cat $P/d10_${arm}_s*of3.csv 2>/dev/null | grep -vc '^fen') rows"
}
depth_arm nowin 10 POT_V2_WIN=0
depth_arm d6 6
depth_arm d8 8
depth_arm d12 12
depth_arm d14 14
( timeout 7200 bash "$R" pyrun diagnostics/_pe_arms.py MODE=sf11 DEPTHS=6,8 ) &
( timeout 7200 bash "$R" pyrun diagnostics/_pe_arms.py MODE=sf11 DEPTHS=10,12 ) &
( timeout 7200 bash "$R" pyrun diagnostics/_pe_arms.py MODE=sf11 DEPTHS=14 ) &
wait
bash "$R" pyrun diagnostics/_pe_arms.py MODE=read ARMS=base,nowin,d6,d8,d12,d14,sf11d6,sf11d8,sf11d10,sf11d12,sf11d14
echo "[q71] $(date) stage 1 done"

bash selfplay/_queue_pot_data.sh
echo "[q71] $(date) stage 2 (#70) done"

FPK=$(env V2_PRESET=shipped NULL_NPM_GUARD=1 bash "$R" wac fp_npm 2>&1 | grep -E "SOLVED|NODES" | tr '\n' ' ')
echo "[q71] NULL_NPM_GUARD=1 WAC: $FPK"
if echo "$FPK" | grep -q "47869841"; then
  echo "[q71] ☠️ NULL_NPM_GUARD=1 byte-identical on WAC — not live, self-play skipped"
else
  tag=npm1_sp250
  bash "$R" pyrun selfplay/tournament.py --p1-label $tag --p1-config "V2_PRESET=shipped NULL_NPM_GUARD=1 MAX_DEPTH=64 NODE_LIMIT=50000" \
    --p2-label shipped --p2-config "V2_PRESET=shipped MAX_DEPTH=64 NODE_LIMIT=50000" --games 2000 --concurrency 3 \
    --preset LONG_FORMAT --max-plies 400 --openings selfplay/openings_uho.txt --seed 250 --tag $tag --quiet
  echo "[q71] $(date) self-play $tag: $(grep -oE '"elo(_margin)?": [-0-9.]+' selfplay/games/$tag/tournament.json | tr '\n' ' ')"
fi
echo "[q71] $(date) QUEUE 71 DONE"
