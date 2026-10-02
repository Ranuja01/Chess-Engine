#!/bin/bash
# 2026-10-02, queue #9: KS-B (shelter + storm) cells fitted on the DEPTH target (`_ksb_depth_fit.py`, val −3.08%) —
# step 1 of the full KS tune (triangulation: structural KS is the consistent miss). Waits for queue #8 (re-gate @800).
#   symmetry with KS-B on (abort on violations) → SF18 @800 (the recalibrated anchor), fresh seeds 60/61, fresh
#   baselines → self-play 2,000 @50k.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
Q8LOG="/mnt/c/Users/Kumodth/AppData/Local/Temp/claude/c--Users-Kumodth-OneDrive-Desktop-Programming-Chess-Engine-Chess-Engine/97a258a9-0c6e-4a2c-8ae5-fa3a9a6ce239/scratchpad/q8.log"
cd "$ND" || exit 1
echo "[q9] $(date) waiting for queue #8"
until grep -q "QUEUE 8 DONE" "$Q8LOG" 2>/dev/null; do sleep 300; done
KSB="KSB_V2=1 KSB_V2_FILE=/mnt/e/chess_data/texel/ksb_depth.txt"
SYM=$(env V2_PRESET=shipped $KSB bash "$R" pyrun diagnostics/_eval_symmetry.py N=4000 2>&1 | grep -E "violations|c3\]|☠")
echo "[q9] symmetry: $(echo "$SYM" | tr '\n' ' ')"
if [ "$(echo "$SYM" | grep -c 'violations 0 ')" != "2" ]; then echo "[q9] ☠️ symmetry failed — aborting"; exit 1; fi
for s in 60 61; do
  bash "$R" gauntlet 500 800 4 g800_ship_s$s $s V2_PRESET=shipped
  bash "$R" gauntlet 500 800 4 g800_ksb_s$s $s V2_PRESET=shipped $KSB
done
bash "$R" pyrun selfplay/tournament.py --p1-label ksb --p1-config "V2_PRESET=shipped $KSB MAX_DEPTH=64 NODE_LIMIT=50000" \
  --p2-label shipped --p2-config "V2_PRESET=shipped MAX_DEPTH=64 NODE_LIMIT=50000" --games 2000 --concurrency 4 \
  --preset LONG_FORMAT --max-plies 400 --openings selfplay/openings_uho.txt --seed 62 --tag ksb_selfplay --quiet
echo "[q9] $(date) QUEUE 9 DONE"
