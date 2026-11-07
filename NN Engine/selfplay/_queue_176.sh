#!/bin/bash
# 2026-11-07, queue #176 (owner 11-07: "we can continue with our work"; a weekend evening, so fixed depth and no timed games): does the
# d7 move-regret ruler rank the retune arms the way the games did? It is the candidate objective for stage 3 of the retune (d7 regret).
# Corpus error did not rank these arms (#174), so stage 3 needs an objective that does, and these arms already have game results.
# Runs after #175 on a quiet box. Base gen 6.7 (V2_PRESET=shipped V2_SHIP_GEN=6). Arms, games vs gen 6.7 at 50k nodes (2,000 g each
# unless noted): S +15.4 (4,000 g; +33.5 at 200k, +16.0 timed) · cr1 +13.6 · cr2 +17.4 (about +2 at 200k) · all2 +2.3 · all0 -9.4 ·
# cr3 -13.7. Plus three neutral arms for the null: ASPIRATION_DELTA=300, ASPIRATION_DELTA=800, EVAL_NOISE_SIGMA=30.
# Both disjoint game sets, game_regret_set (15,000) and game_regret_set_v2 (11,940), d7, 3 workers, one base pass per set. Then each arm
# against the three neutral arms on the positions both changed (_paired_null.py).
# Registered before any number is read:
#   PASS if on BOTH sets the top group (S, cr1, cr2) all read above the bottom group (all0, cr3) on the paired win%, and the gap between
#   the groups' means is larger than the spread of the three nulls.
#   FAIL if all0 or cr3 reads in the top half of the six arms on both sets.
#   Anything else is UNRESOLVED: the arms are closer together than the ruler can separate.
#   My prediction: unresolved 60% · pass 25% · fail 15%. The arms differ by about 1% in eval error; the ruler resolves about 2pp of win%.
# Stop: touch /mnt/e/chess_data/q176.stop (checked between sets and arms). Never edit this file while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
LOGD=/mnt/e/chess_data/q176
STOP=/mnt/e/chess_data/q176.stop
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"; mkdir -p "$LOGD"
stopped() { [ -e "$STOP" ] && { echo "[q176] $(date) stop file present before $1 — QUEUE 176 DONE"; exit 0; }; }
busy() { pgrep -f "fastchess-linux-x86-64/fastchess|vs_sf.py|_retune_fit|_depth_residual_pass|_build_regret_set|sts_test|tactical_test" >/dev/null; }

echo "[q176] $(date) start; waiting for #175 and a quiet box"
until grep -qE "QUEUE 175 (DONE|ABORTED)" /mnt/e/chess_data/q175.log 2>/dev/null && ! busy; do stopped "the start"; sleep 60; done

k() { tr ',' ' ' < "$1"; }
export BASE_KNOBS="V2_PRESET=shipped V2_SHIP_GEN=6"
export CAND_KNOBS="$(k /mnt/e/chess_data/q166/knobs_v3s.txt);$(k /mnt/e/chess_data/q173/knobs_cr1_s.txt);$(k /mnt/e/chess_data/q173/knobs_cr2_s.txt);$(k /mnt/e/chess_data/q173/knobs_cr3_s.txt);$(k /mnt/e/chess_data/q174/knobs_all0.txt);$(k /mnt/e/chess_data/q174/knobs_all2.txt);ASPIRATION_DELTA=300;ASPIRATION_DELTA=800;EVAL_NOISE_SIGMA=30"
export CAND_NAME="S;cr1;cr2;cr3;all0;all2;null_asp300;null_asp800;null_noise30"

for S in game_regret_set game_regret_set_v2; do
  stopped "$S"
  echo "[q176] $(date) $S: gen 6.7 base + 9 arms at d7"
  DUMP=$LOGD/${S}_dump.csv nice -n 5 bash "$R" pyrun diagnostics/_ks_footprint_regret.py SET=ks_sets/$S.csv DEPTH=7 JOBS=3 > $LOGD/$S.txt 2>&1
  echo "[q176] $(date) $S done: $(grep -m1 '^SET=' $LOGD/$S.txt)"
  grep -E "^  (candidate|S |cr|all|null)" $LOGD/$S.txt | sed 's/^/[q176]   /'
done

echo "[q176] $(date) paired against the three nulls (arm% - null% on the positions both changed):"
for S in game_regret_set game_regret_set_v2; do
  N=$LOGD/${S}_dump_null_asp300.csv,$LOGD/${S}_dump_null_asp800.csv,$LOGD/${S}_dump_null_noise30.csv
  for a in S cr1 cr2 cr3 all0 all2; do
    [ -f $LOGD/${S}_dump_$a.csv ] || { echo "[q176]   $S $a: no dump"; continue; }
    echo "[q176]   $S $a: $(bash "$R" pyrun diagnostics/_paired_null.py ARM=$LOGD/${S}_dump_$a.csv NULLS=$N 2>&1 | grep -E 'mean arm-null')"
  done
done
echo "[q176] $(date) QUEUE 176 DONE"
