#!/bin/bash
# 2026-10-03, queue #14: PAWN STRUCTURE + PASSERS on the DEPTH target, nested on the 10-03 ship. Waits for queue #13.
#   1 depth pass of the CURRENT shipped engine (memory-bounded: CHUNK restarts, `unattended-jobs-must-have-bounded-memory`)
#   2 refit `_pawn_depth_fit.py` (arms STRUCT / PASSER / JOINT, λ 1e-2) on that base
#   3 closure (feature pass under C1_V2_FIT=1: engine pawn_struct / v2_passers == Σ count × fitted θ) + symmetry
#   4 SF18 @800 (calibrated), fresh seeds 70/71, shared baseline per seed: STRUCT · PASSER · JOINT; self-play JOINT 2,000
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
Q13LOG="/mnt/c/Users/Kumodth/AppData/Local/Temp/claude/c--Users-Kumodth-OneDrive-Desktop-Programming-Chess-Engine-Chess-Engine/97a258a9-0c6e-4a2c-8ae5-fa3a9a6ce239/scratchpad/q13.log"
TX=/mnt/e/chess_data/texel
cd "$ND" || exit 1
echo "[q14] $(date) waiting for queue #13"
until grep -q "QUEUE 13 DONE" "$Q13LOG" 2>/dev/null; do sleep 300; done
# 1 — depth pass, 4 shards, each restarted every 800 rows (bounded memory)
for s in 0 1 2 3; do
  ( until env V2_PRESET=shipped PRESET=LONG_FORMAT MAX_DEPTH=10 USE_OPENING_BOOK=0 bash "$R" pyrun \
        diagnostics/_depth_residual_pass.py IN=ks_sets/fitC_mg_sf18.csv OUT=ks_sets/fitC_mg_ours1003_d10.csv \
        SHARD=$s/4 CHUNK=800 > /tmp/q14_depth_$s.txt 2>&1; [ $? -ne 3 ]; do :; done ) &
done
wait
echo "[q14] $(date) depth pass rows: $(cat ks_sets/fitC_mg_ours1003_d10_s*of4.csv | wc -l)"
# 2 — refit on the new base
bash "$R" pyrun diagnostics/_pawn_depth_fit.py OURS=fitC_mg_ours1003_d10 TAG=pawn_depth 2>&1 | sed 's/^/[q14] /'
# 3 — closure + symmetry for the JOINT arm
J="C1_V2_FIT=1 C1_V2_FILE=$TX/pawn_depth_joint.txt"
env V2_PRESET=shipped $J bash "$R" pyrun diagnostics/_texel_feature_pass.py IN=/mnt/e/chess_data/texel/fitC_stage1.csv.gz LIMIT=20000 OUT=/tmp/q14_c1.npz > /tmp/q14_clo.txt 2>&1
grep -E "pawn_struct|v2_passers" /tmp/q14_clo.txt | sed 's/^/[q14] closure /'
BAD=$(grep -E "^ *(pawn_struct|v2_passers) " /tmp/q14_clo.txt | awk '{ if ($NF+0 > 10 || $NF+0 < -10) print }' | wc -l)
NLINES=$(grep -cE "^ *(pawn_struct|v2_passers) " /tmp/q14_clo.txt)   # a silent tool must not read as a pass
[ "$NLINES" = "2" ] || BAD=99
SYM=$(env V2_PRESET=shipped $J bash "$R" pyrun diagnostics/_eval_symmetry.py N=4000 2>&1 | grep -E "violations")
echo "[q14] symmetry: $(echo "$SYM" | tr '\n' ' ')"
if [ "$BAD" != "0" ] || [ "$(echo "$SYM" | grep -c 'violations 0 ')" != "2" ]; then
  echo "[q14] ☠️ closure/symmetry failed — gates skipped"; exit 1
fi
# 4 — gates per part
for s in 70 71; do
  bash "$R" gauntlet 500 800 4 g800_ship1003_s$s $s V2_PRESET=shipped
  for arm in struct passer joint; do
    bash "$R" gauntlet 500 800 4 g800_pawn${arm}_s$s $s V2_PRESET=shipped C1_V2_FIT=1 C1_V2_FILE=$TX/pawn_depth_$arm.txt
  done
done
bash "$R" pyrun selfplay/tournament.py --p1-label pawnjoint --p1-config "V2_PRESET=shipped $J MAX_DEPTH=64 NODE_LIMIT=50000" \
  --p2-label shipped --p2-config "V2_PRESET=shipped MAX_DEPTH=64 NODE_LIMIT=50000" --games 2000 --concurrency 4 \
  --preset LONG_FORMAT --max-plies 400 --openings selfplay/openings_uho.txt --seed 72 --tag pawnjoint_selfplay --quiet
echo "[q14] $(date) QUEUE 14 DONE"
