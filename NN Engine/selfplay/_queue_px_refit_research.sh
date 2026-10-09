#!/bin/bash
# 2026-10-08 night, queue #43 (owner: "maximize the night"). The passer subsystem (rank tables + king terms + 51 PX cells)
# REFITTED jointly on the CURRENT ship base (`ours1004`, with connected pawns) with the 10-05 controls (by-game split, SCALE
# nuisance): `_px_depth_fit.py OURS_*=…ours1004… TAG=px_depth1008` → px_depth1008_{c1,px}.txt. Real d10 re-search of the 4,952
# val rows on the SHIP base, same layout as #40/#41; compare with the 10-03-base PX (dual_val_sh_pxall, −5.56%) and the ship
# control. WAITS for queue #42 (games hold the 4 engines). No build.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
TX=/mnt/e/chess_data/texel
PX8="C1_V2_FIT=1 C1_V2_FILE=$TX/px_depth1008_c1.txt PX_V2=1 PX_V2_FILE=$TX/px_depth1008_px.txt"
until grep -q "QUEUE 42 DONE" /mnt/e/chess_data/q42_structural_gate.log 2>/dev/null; do
  grep -qE "☠️|aborting" /mnt/e/chess_data/q42_structural_gate.log 2>/dev/null && { echo "[q43] queue #42 aborted — not starting"; exit 1; }
  sleep 120
done
cd "$ND" || exit 1
echo "[q43] $(date) start"
[ -s $TX/px_depth1008_c1.txt ] && [ -s $TX/px_depth1008_px.txt ] || { echo "[q43] ☠️ refit tables missing — aborting"; exit 1; }
for s in 0 1 2 3; do
  ( until env V2_PRESET=shipped $PX8 PRESET=LONG_FORMAT MAX_DEPTH=10 USE_OPENING_BOOK=0 bash "$R" pyrun \
        diagnostics/_depth_residual_pass.py IN=ks_sets/dual_val_sf18.csv OUT=ks_sets/dual_val_sh_px1008_d10.csv \
        SHARD=$s/4 CHUNK=800 > /tmp/q43_$s.txt 2>&1; [ $? -ne 3 ]; do :; done ) &
done
wait
echo "[q43] $(date) ship+px1008 rows: $(cat diagnostics/ks_sets/dual_val_sh_px1008_d10_s*of4.csv 2>/dev/null | grep -vc '^fen,')"
echo "[q43] $(date) QUEUE 43 DONE"
