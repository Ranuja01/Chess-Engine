#!/bin/bash
# 2026-10-08 night, queue #42 (owner OK). STRUCTURAL GATE vs the SHIP. Real d10 re-search on the ship base (vs ship control,
# queues #40/#41): PX (passer cells + jointly fitted passer re-price) −5.56% (eg −5.15, mg −6.76) · KFL −2.78% (mg −9.60) ·
# KFL+PX only −3.94% (KFL costs PX's endgame gain) · path ladder −1.25% (☠️ DEAD under any C1 table: passer_value_mp's C1
# branch `continue`s before the ladder — the "stack" arm was byte-identical to KFL+PX). ⇒ two separate arms.
# Games, BOTH instruments, fresh seeds: SF18 @1000 ship / PX / KFL on 108-109, then self-play @50k PX vs ship and KFL vs ship
# (seed 110). Node-limited only (safe in the owner's evening window). ≤ 4 engines. No build.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
O=/mnt/e/chess_data/texel/revival
TX=/mnt/e/chess_data/texel
PX="C1_V2_FIT=1 C1_V2_FILE=$TX/px_depth_c1.txt PX_V2=1 PX_V2_FILE=$TX/px_depth_px.txt"
KFL="KFL_V2=1 KFL_V2_FILE=$O/kfl_depth.txt"
cd "$ND" || exit 1
echo "[q42] $(date) start"
FP=$(env V2_PRESET=shipped bash "$R" wac fp_v2 2>&1 | grep -E "SOLVED|NODES" | tr '\n' ' ')
echo "[q42] v2 shipped: $FP"
echo "$FP" | grep -q "254/300.*50622239" || { echo "[q42] ☠️ FINGERPRINT MISMATCH — aborting"; exit 1; }
for arm in PX KFL; do
  SYM=$(env V2_PRESET=shipped ${!arm} bash "$R" pyrun diagnostics/_eval_symmetry.py N=4000 2>&1 | grep -E "violations")
  echo "[q42] $arm symmetry: $(echo "$SYM" | tr '\n' ' ')"
  [ "$(echo "$SYM" | grep -c 'violations 0 ')" = "2" ] || { echo "[q42] ☠️ $arm symmetry FAILED — aborting"; exit 1; }
done
for s in 108 109; do
  bash "$R" gauntlet 500 1000 4 g1000_ship_s$s $s V2_PRESET=shipped
  bash "$R" gauntlet 500 1000 4 g1000_px_s$s $s V2_PRESET=shipped $PX
  bash "$R" gauntlet 500 1000 4 g1000_kfl_s$s $s V2_PRESET=shipped $KFL
  echo "[q42] $(date) SF18 seed $s done"
done
for arm in PX KFL; do
  tag=$(echo "$arm" | tr 'A-Z' 'a-z')_selfplay
  bash "$R" pyrun selfplay/tournament.py --p1-label $tag --p1-config "V2_PRESET=shipped ${!arm} MAX_DEPTH=64 NODE_LIMIT=50000" \
    --p2-label shipped --p2-config "V2_PRESET=shipped MAX_DEPTH=64 NODE_LIMIT=50000" --games 2000 --concurrency 4 \
    --preset LONG_FORMAT --max-plies 400 --openings selfplay/openings_uho.txt --seed 110 --tag $tag --quiet
  echo "[q42] $(date) $tag: $(grep -o '"elo": [-0-9.]*' selfplay/games/$tag/tournament.json)"
done
echo "[q42] $(date) QUEUE 42 DONE"
