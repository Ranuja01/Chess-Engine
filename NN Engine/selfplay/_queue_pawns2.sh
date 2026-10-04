#!/bin/bash
# 2026-10-04, queue #16: the PAWN gates, re-run after the FILE-MIRROR fix (queue #14's per-file isolated values were fitted
# untied ⇒ 2,435 / 3,170 file-mirror violations ⇒ gates correctly skipped). Refit with a=h, b=g, c=f, d=e tied (joint val
# −1.26%). Waits for queue #15 (PX prep: build + guards). closure + BOTH symmetry checks → SF18 @800 seeds 70/71
# (STRUCT · PASSER · JOINT vs a shared baseline) → self-play JOINT 2,000.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
Q15LOG="/mnt/c/Users/Kumodth/AppData/Local/Temp/claude/c--Users-Kumodth-OneDrive-Desktop-Programming-Chess-Engine-Chess-Engine/97a258a9-0c6e-4a2c-8ae5-fa3a9a6ce239/scratchpad/q15.log"
TX=/mnt/e/chess_data/texel
cd "$ND" || exit 1
echo "[q16] $(date) waiting for queue #15"
until grep -qE "QUEUE 15 DONE|aborting" "$Q15LOG" 2>/dev/null; do sleep 300; done
if grep -q "aborting" "$Q15LOG"; then echo "[q16] queue #15 aborted (build guard) — not running"; exit 1; fi
J="C1_V2_FIT=1 C1_V2_FILE=$TX/pawn_depth_joint.txt"
env V2_PRESET=shipped $J bash "$R" pyrun diagnostics/_texel_feature_pass.py IN=$TX/fitC_stage1.csv.gz LIMIT=20000 OUT=/tmp/q16_c1.npz > /tmp/q16_clo.txt 2>&1
grep -E "pawn_struct|v2_passers" /tmp/q16_clo.txt | sed 's/^/[q16] closure /'
BAD=$(grep -E "^ *(pawn_struct|v2_passers) " /tmp/q16_clo.txt | awk '{ if ($NF+0 > 10 || $NF+0 < -10) print }' | wc -l)
NLINES=$(grep -cE "^ *(pawn_struct|v2_passers) " /tmp/q16_clo.txt)
[ "$NLINES" = "2" ] || BAD=99
SYM=$(env V2_PRESET=shipped $J bash "$R" pyrun diagnostics/_eval_symmetry.py N=4000 2>&1 | grep -E "violations")
echo "[q16] symmetry: $(echo "$SYM" | tr '\n' ' ')"
if [ "$BAD" != "0" ] || [ "$(echo "$SYM" | grep -c 'violations 0 ')" != "2" ]; then
  echo "[q16] ☠️ closure/symmetry failed — gates skipped"; exit 1
fi
for s in 70 71; do
  bash "$R" gauntlet 500 800 4 g800_ship1003_s$s $s V2_PRESET=shipped
  for arm in struct passer joint; do
    bash "$R" gauntlet 500 800 4 g800_pawn${arm}_s$s $s V2_PRESET=shipped C1_V2_FIT=1 C1_V2_FILE=$TX/pawn_depth_$arm.txt
  done
done
bash "$R" pyrun selfplay/tournament.py --p1-label pawnjoint --p1-config "V2_PRESET=shipped $J MAX_DEPTH=64 NODE_LIMIT=50000" \
  --p2-label shipped --p2-config "V2_PRESET=shipped MAX_DEPTH=64 NODE_LIMIT=50000" --games 2000 --concurrency 4 \
  --preset LONG_FORMAT --max-plies 400 --openings selfplay/openings_uho.txt --seed 72 --tag pawnjoint_selfplay --quiet
echo "[q16] $(date) QUEUE 16 DONE"
