#!/bin/bash
# 2026-10-02, queue #10: full KS tune step 2 — KS attack knobs + KS-B cells fitted JOINTLY on the depth target
# (`_ks_depth_fit.py KS_LAMBDA=1e-2`: val −4.07% vs −3.08% KS-B alone; engine closure OK, max 4.9 mp). Per-part rule:
# gated NEXT TO queue #9's KS-B-alone arm, on the SAME seeds 60/61 ⇒ it pairs with q9's baselines (g800_ship_s60/61).
# Waits for queue #9. symmetry → SF18 @800 → self-play 2,000 @50k.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
Q9LOG="/mnt/c/Users/Kumodth/AppData/Local/Temp/claude/c--Users-Kumodth-OneDrive-Desktop-Programming-Chess-Engine-Chess-Engine/97a258a9-0c6e-4a2c-8ae5-fa3a9a6ce239/scratchpad/q9.log"
TX=/mnt/e/chess_data/texel
cd "$ND" || exit 1
echo "[q10] $(date) waiting for queue #9"
until grep -qE "QUEUE 9 DONE|aborting" "$Q9LOG" 2>/dev/null; do sleep 300; done
J="$(cat $TX/ks_depth_L1e-2_ks.txt) KSB_V2=1 KSB_V2_FILE=$TX/ks_depth_L1e-2_ksb.txt"
SYM=$(env V2_PRESET=shipped $J bash "$R" pyrun diagnostics/_eval_symmetry.py N=4000 2>&1 | grep -E "violations")
echo "[q10] symmetry: $(echo "$SYM" | tr '\n' ' ')"
if [ "$(echo "$SYM" | grep -c 'violations 0 ')" != "2" ]; then echo "[q10] ☠️ symmetry failed — aborting"; exit 1; fi
for s in 60 61; do
  [ -f "$ND/selfplay/games/g800_ship_s$s/results.csv" ] || bash "$R" gauntlet 500 800 4 g800_ship_s$s $s V2_PRESET=shipped
  bash "$R" gauntlet 500 800 4 g800_ksjoint_s$s $s V2_PRESET=shipped $J
done
bash "$R" pyrun selfplay/tournament.py --p1-label ksjoint --p1-config "V2_PRESET=shipped $J MAX_DEPTH=64 NODE_LIMIT=50000" \
  --p2-label shipped --p2-config "V2_PRESET=shipped MAX_DEPTH=64 NODE_LIMIT=50000" --games 2000 --concurrency 4 \
  --preset LONG_FORMAT --max-plies 400 --openings selfplay/openings_uho.txt --seed 63 --tag ksjoint_selfplay --quiet
echo "[q10] $(date) QUEUE 10 DONE"
