#!/bin/bash
# 2026-10-01, queue #6: NARROW MATERIAL CLASSES (MCL_V2, C3 doc §18g) — after queue #5 (which builds the code).
#   1 closure: engine total(on) − total(off) == Python model, exact (POT_V2_WIN=0 during the dumps: the eg scale factor
#     multiplies the total and would distort the difference) + colour/file symmetry — skip the gates on failure
#   2 SF18 @250k, fresh seeds 54 + 55, fresh baselines: ALL four classes, and the QUEEN class alone (per-part rule)
#   3 self-play 2,000 @50k, all classes vs shipped (the gauntlet-opponent question, C3 §18f)
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
Q5LOG="/mnt/c/Users/Kumodth/AppData/Local/Temp/claude/c--Users-Kumodth-OneDrive-Desktop-Programming-Chess-Engine-Chess-Engine/97a258a9-0c6e-4a2c-8ae5-fa3a9a6ce239/scratchpad/q5.log"
cd "$ND" || exit 1
echo "[q6] $(date) waiting for queue #5"
until grep -qE "QUEUE 5 DONE|aborting" "$Q5LOG" 2>/dev/null; do sleep 300; done
if grep -q "aborting" "$Q5LOG"; then echo "[q6] queue #5 aborted — not running"; exit 1; fi
QV="MCL_V2_Q0=-430 MCL_V2_QR=80 MCL_V2_QM=-190 MCL_V2_QP=20"
ALL="MCL_V2=1 $QV MCL_V2_R2M=-170 MCL_V2_MP0=320 MCL_V2_MPP=20 MCL_V2_PAIR=200"
env V2_PRESET=shipped POT_V2_WIN=0 $ALL bash "$R" pyrun diagnostics/_material_class_fit.py MODE=dump OUT=/tmp/mcl_on.csv
env V2_PRESET=shipped POT_V2_WIN=0 MCL_V2=0 bash "$R" pyrun diagnostics/_material_class_fit.py MODE=dump OUT=/tmp/mcl_off.csv
env $ALL bash "$R" pyrun diagnostics/_material_class_fit.py MODE=closure ON=/tmp/mcl_on.csv OFF=/tmp/mcl_off.csv \
  | sed 's/^/[q6] /'
OK=${PIPESTATUS[0]}
SYM=$(env V2_PRESET=shipped $ALL bash "$R" pyrun diagnostics/_eval_symmetry.py N=4000 2>&1 | grep -E "violations")
echo "[q6] symmetry: $(echo "$SYM" | tr '\n' ' ')"
[ "$(echo "$SYM" | grep -c 'violations 0 ')" = "2" ] || OK=1
if [ "$OK" != "0" ]; then echo "[q6] ☠️ MCL checks failed — gates skipped"; exit 1; fi
for s in 54 55; do
  bash "$R" gauntlet 500 400 4 gauntlet_ship7_s$s $s V2_PRESET=shipped
  bash "$R" gauntlet 500 400 4 gauntlet_mclall_s$s $s V2_PRESET=shipped $ALL
  bash "$R" gauntlet 500 400 4 gauntlet_mclq_s$s $s V2_PRESET=shipped MCL_V2=1 $QV
done
bash "$R" pyrun selfplay/tournament.py --p1-label mclall --p1-config "V2_PRESET=shipped $ALL MAX_DEPTH=64 NODE_LIMIT=50000" \
  --p2-label shipped --p2-config "V2_PRESET=shipped MAX_DEPTH=64 NODE_LIMIT=50000" --games 2000 --concurrency 4 \
  --preset LONG_FORMAT --max-plies 400 --openings selfplay/openings_uho.txt --seed 56 --tag mclall_selfplay --quiet
echo "[q6] $(date) QUEUE 6 DONE"
