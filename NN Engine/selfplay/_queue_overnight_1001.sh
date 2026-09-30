#!/bin/bash
# 2026-10-01 overnight queue #2 (owner: order doesn't matter overnight). Waits for queue #1 (WSF SPRT + replication,
# `_queue_wsf_sprt.sh`) to finish — it runs from the working tree, so NO build before then — then:
#   1 build + fingerprints (ABORT ALL on mismatch)   2 Kaufman FORM 3 closure + symmetry, both arms (skip Kaufman on fail)
#   3 depth-residual pass (4 shards)                 4 Kaufman gauntlets vs SF18 @250k, fresh seeds 47 + 48
#   5 v2 vs v1, 2,000 games at EQUAL NODES (conservative: v2 gets ~40% more nodes at equal time)
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
Q1LOG="/mnt/c/Users/Kumodth/AppData/Local/Temp/claude/c--Users-Kumodth-OneDrive-Desktop-Programming-Chess-Engine-Chess-Engine/97a258a9-0c6e-4a2c-8ae5-fa3a9a6ce239/scratchpad/wsf_sprt.log"
TX=/mnt/e/chess_data/texel
cd "$ND" || exit 1
echo "[q2] $(date) waiting for queue #1"
until grep -q "WSF SPRT QUEUE DONE" "$Q1LOG" 2>/dev/null; do sleep 300; done
echo "[q2] $(date) queue #1 done"

# 1 — build + fingerprints
bash "$R" build > /tmp/q2_build.txt 2>&1
FP2=$(bash "$R" wac fp_v2 V2_PRESET=shipped 2>&1 | grep -E "SOLVED|NODES" | tr '\n' ' ')
FP1=$(bash "$R" wac fp_v1 2>&1 | grep -E "SOLVED|NODES" | tr '\n' ' ')
echo "[q2] fp_v2: $FP2"; echo "[q2] fp_v1: $FP1"
if ! echo "$FP2" | grep -q "252/300.*49094807" || ! echo "$FP1" | grep -q "250/300.*35310778"; then
  echo "[q2] ☠️ FINGERPRINT MISMATCH — aborting the whole queue"; exit 1
fi

# 2 — Kaufman checks (both arms)
KOK=1
for arm in full queen; do
  K="V2_PRESET=shipped KAUF_V2_MAG=1000 KAUF_V2_FORM=3 KAUF_V2_FILE=$TX/kauf_$arm.txt"
  env $K bash "$R" pyrun diagnostics/_texel_kauf_fit.py MODE=closure 2>&1 | grep "KAUF CLOSURE" | sed "s/^/[q2] $arm /"
  [ "${PIPESTATUS[0]}" = "0" ] || KOK=0
  SYM=$(env $K bash "$R" pyrun diagnostics/_eval_symmetry.py N=4000 2>&1 | grep -E "violations")
  echo "[q2] $arm symmetry: $(echo "$SYM" | tr '\n' ' ')"
  [ "$(echo "$SYM" | grep -c 'violations 0 ')" = "2" ] || KOK=0
done
echo "[q2] Kaufman checks OK=$KOK"

# 3 — depth-residual pass, 4 shards
for s in 0 1 2 3; do
  env V2_PRESET=shipped PRESET=LONG_FORMAT MAX_DEPTH=10 USE_OPENING_BOOK=0 bash "$R" pyrun diagnostics/_depth_residual_pass.py \
    IN=ks_sets/fitC_mg_sf18.csv OUT=ks_sets/fitC_mg_ours_d10.csv SHARD=$s/4 > /tmp/q2_depth_$s.txt 2>&1 &
done
wait
echo "[q2] $(date) depth pass done: $(tail -1 /tmp/q2_depth_*.txt | grep -c 'DEPTH PASS') shards"

# 4 — Kaufman gauntlets
if [ "$KOK" = "1" ]; then
  for s in 47 48; do
    bash "$R" gauntlet 500 400 4 gauntlet_ship5_s$s $s V2_PRESET=shipped
    for arm in full queen; do
      bash "$R" gauntlet 500 400 4 gauntlet_kauf${arm}_s$s $s V2_PRESET=shipped KAUF_V2_MAG=1000 KAUF_V2_FORM=3 \
        KAUF_V2_FILE=$TX/kauf_$arm.txt
    done
  done
else
  echo "[q2] ☠️ Kaufman checks failed — gauntlets skipped"
fi

# 5 — v2 vs v1 at equal nodes
bash "$R" pyrun selfplay/tournament.py --p1-label v2 --p1-config "V2_PRESET=shipped MAX_DEPTH=64 NODE_LIMIT=50000" \
  --p2-label v1 --p2-config "MAX_DEPTH=64 NODE_LIMIT=50000" --games 2000 --concurrency 4 --preset LONG_FORMAT \
  --max-plies 400 --openings selfplay/openings_uho.txt --seed 49 --tag v2_vs_v1_1001 --quiet
echo "[q2] $(date) OVERNIGHT QUEUE 2 DONE"
