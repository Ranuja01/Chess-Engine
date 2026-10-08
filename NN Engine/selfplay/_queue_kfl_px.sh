#!/bin/bash
# 2026-10-08 evening, queue #41 (owner OK). Queue #40 on the SHIP base (vs ship control): PX + passer re-price −5.56% (eg −5.15,
# mg −6.76) · KFL −2.78% (mg −9.60) · path −1.25% · stack KFL+PX+path only −3.94% (path overlaps PX in the eg). The untested
# pair KFL + PX (no path ladder): one real d10 re-search, same rows/layout. ≤ 4 engines. No build. Gate decided after reading.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
O=/mnt/e/chess_data/texel/revival
TX=/mnt/e/chess_data/texel
PAIR="KFL_V2=1 KFL_V2_FILE=$O/kfl_depth.txt C1_V2_FIT=1 C1_V2_FILE=$TX/px_depth_c1.txt PX_V2=1 PX_V2_FILE=$TX/px_depth_px.txt"
cd "$ND" || exit 1
echo "[q41] $(date) start"
for s in 0 1 2 3; do
  ( until env V2_PRESET=shipped $PAIR PRESET=LONG_FORMAT MAX_DEPTH=10 USE_OPENING_BOOK=0 bash "$R" pyrun \
        diagnostics/_depth_residual_pass.py IN=ks_sets/dual_val_sf18.csv OUT=ks_sets/dual_val_sh_kflpx_d10.csv \
        SHARD=$s/4 CHUNK=800 > /tmp/q41_$s.txt 2>&1; [ $? -ne 3 ]; do :; done ) &
done
wait
echo "[q41] $(date) ship+kfl+px rows: $(cat diagnostics/ks_sets/dual_val_sh_kflpx_d10_s*of4.csv 2>/dev/null | grep -vc '^fen,')"
echo "[q41] $(date) QUEUE 41 DONE"
