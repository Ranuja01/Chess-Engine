#!/bin/bash
# 2026-10-07 night, queue #36 (owner OK). THREATS-LED CANDIDATE GATE. Real d10 re-search (queues #32-#35, val 4,952 rows):
# threats −7.8% vs ship (null −0.1%); on top of threats: mobility EG LEG ONLY −3.2% (mg cost removed), space −0.9% (mg −3.5%),
# rook files −1.0%. Steps: fingerprint → symmetry of the combo → combined d10 re-search (do the parts stack?) → games on BOTH
# instruments, fresh seeds: SF18 @1000 (the 10-07 anchor) ship / threats / combo on seeds 101-102, then self-play @50k combo vs
# ship and threats vs ship (seed 103). Node-limited only (safe in the owner's evening window). ≤ 4 engines. No build.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
O=/mnt/e/chess_data/texel/revival
TH="THREAT_V2_PCT=100 THREAT_V2_HANGING=1 THREAT_V2_KING=1 THREAT_V2_PAWN_TARGETS=1"
COMBO="$TH C1_V2_FIT=1 C1_V2_FILE=$O/mob_depth_c1_egonly.txt SPACE_V2_MAG=1320 ROOKFILE_V2_OPEN=200 ROOKFILE_V2_SEMI=90"
cd "$ND" || exit 1
echo "[q36] $(date) start"
FP=$(env V2_PRESET=shipped bash "$R" wac fp_v2 2>&1 | grep -E "SOLVED|NODES" | tr '\n' ' ')
echo "[q36] v2 shipped: $FP"
echo "$FP" | grep -q "254/300.*50622239" || { echo "[q36] ☠️ FINGERPRINT MISMATCH — aborting"; exit 1; }
for arm in TH COMBO; do
  SYM=$(env V2_PRESET=shipped ${!arm} bash "$R" pyrun diagnostics/_eval_symmetry.py N=4000 2>&1 | grep -E "violations")
  echo "[q36] $arm symmetry: $(echo "$SYM" | tr '\n' ' ')"
  [ "$(echo "$SYM" | grep -c 'violations 0 ')" = "2" ] || { echo "[q36] ☠️ $arm symmetry FAILED — aborting"; exit 1; }
done
for s in 0 1 2 3; do
  ( until env V2_PRESET=shipped $COMBO PRESET=LONG_FORMAT MAX_DEPTH=10 USE_OPENING_BOOK=0 bash "$R" pyrun \
        diagnostics/_depth_residual_pass.py IN=ks_sets/dual_val_sf18.csv OUT=ks_sets/dual_val_combo_d10.csv \
        SHARD=$s/4 CHUNK=800 > /tmp/q36_combo_$s.txt 2>&1; [ $? -ne 3 ]; do :; done ) &
done
wait
echo "[q36] $(date) combo d10 rows: $(cat diagnostics/ks_sets/dual_val_combo_d10_s*of4.csv 2>/dev/null | grep -vc '^fen,')"
for s in 101 102; do
  bash "$R" gauntlet 500 1000 4 g1000_ship_s$s $s V2_PRESET=shipped
  bash "$R" gauntlet 500 1000 4 g1000_threats_s$s $s V2_PRESET=shipped $TH
  bash "$R" gauntlet 500 1000 4 g1000_thcombo_s$s $s V2_PRESET=shipped $COMBO
  echo "[q36] $(date) SF18 seed $s done"
done
for arm in COMBO TH; do
  tag=$( [ "$arm" = COMBO ] && echo thcombo_selfplay || echo threats_selfplay )
  bash "$R" pyrun selfplay/tournament.py --p1-label $tag --p1-config "V2_PRESET=shipped ${!arm} MAX_DEPTH=64 NODE_LIMIT=50000" \
    --p2-label shipped --p2-config "V2_PRESET=shipped MAX_DEPTH=64 NODE_LIMIT=50000" --games 2000 --concurrency 4 \
    --preset LONG_FORMAT --max-plies 400 --openings selfplay/openings_uho.txt --seed 103 --tag $tag --quiet
  echo "[q36] $(date) $tag: $(grep -o '"elo": [-0-9.]*' selfplay/games/$tag/tournament.json)"
done
echo "[q36] $(date) QUEUE 36 DONE"
