#!/bin/bash
# 2026-10-06, queue #27 (owner decisions 10-06): (A) COMBINED CONFIRMATION of the two borderline arms — KFL C3-b (+12.8 ± 7.6)
# and PST depth re-fit (+12.0 ± 7.6) TOGETHER vs the ship, both instruments, fresh seeds (parts can cancel); ship both if the
# pair holds ≳ 2σ. (B) KPROT bigger SF18 read — +27 on SF18 @800 but −2.3 self-play: two more fresh SF seeds on the same
# shared baseline to see whether the external-engine gain is real. C3 doc §20c.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
O=/mnt/e/chess_data/texel/revival
cd "$ND" || exit 1
COMBO="KFL_V2=1 KFL_V2_FILE=$O/kfl_depth.txt PST_V2_FILE=$O/pst_depth.txt"
KPROT="KPROT_V2=1 KPROT_V2_FILE=$O/kprot_depth.txt"
echo "[q27] $(date) start"
FP=$(env V2_PRESET=shipped bash "$R" wac fp_v2 2>&1 | grep -E "SOLVED|NODES" | tr '\n' ' ')
echo "[q27] v2 shipped: $FP"
echo "$FP" | grep -q "254/300.*50622239" || { echo "[q27] ☠️ FINGERPRINT MISMATCH — aborting"; exit 1; }
SYM=$(env V2_PRESET=shipped $COMBO bash "$R" pyrun diagnostics/_eval_symmetry.py N=4000 2>&1 | grep -E "violations")
echo "[q27] combo symmetry: $(echo "$SYM" | tr '\n' ' ')"
[ "$(echo "$SYM" | grep -c 'violations 0 ')" = "2" ] || { echo "[q27] ☠️ combo symmetry FAILED — aborting"; exit 1; }
for s in 84 85; do
  bash "$R" gauntlet 500 800 4 g800_ship1004c_s$s $s V2_PRESET=shipped
  bash "$R" gauntlet 500 800 4 g800_combo_kflpst_s$s $s V2_PRESET=shipped $COMBO
  bash "$R" gauntlet 500 800 4 g800_rev_kprot_s$s $s V2_PRESET=shipped $KPROT
  echo "[q27] $(date) seed $s done"
done
bash "$R" pyrun selfplay/tournament.py --p1-label combo_kflpst --p1-config "V2_PRESET=shipped $COMBO MAX_DEPTH=64 NODE_LIMIT=50000" \
  --p2-label shipped --p2-config "V2_PRESET=shipped MAX_DEPTH=64 NODE_LIMIT=50000" --games 2000 --concurrency 4 \
  --preset LONG_FORMAT --max-plies 400 --openings selfplay/openings_uho.txt --seed 86 --tag combo_kflpst_selfplay --quiet
echo "[q27] $(date) combo self-play done: $(grep -o '"elo": [-0-9.]*' selfplay/games/combo_kflpst_selfplay/tournament.json)"
echo "[q27] $(date) QUEUE 27 DONE"
