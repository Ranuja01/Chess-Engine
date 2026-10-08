#!/bin/bash
# 2026-10-08 evening, queue #40 (owner OK). Queue #39 (on top of threats): KFL −2.0%, PX + passer re-price −2.3%, free-path
# ladder −2.6% (re-price alone +4.1%). Threats is NOT shipping (SF18 −5 ± 9) ⇒ re-measure on the SHIP base, the base these
# structural terms would actually join, and check they STACK (the space/rook-files lesson). Real d10 re-search of the 4,952 val
# rows, same SHARD=4 / CHUNK=800 layout; reference = the identical-conditions ship control `dual_val_shipctl_d10` (#33).
# ≤ 4 engines. No build. The game gate (#41) is launched after reading these.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
O=/mnt/e/chess_data/texel/revival
TX=/mnt/e/chess_data/texel
declare -A ARM
ARM[kfl]="KFL_V2=1 KFL_V2_FILE=$O/kfl_depth.txt"
ARM[pxall]="C1_V2_FIT=1 C1_V2_FILE=$TX/px_depth_c1.txt PX_V2=1 PX_V2_FILE=$TX/px_depth_px.txt"
ARM[path]="PASSER_V2_PATH_PCT=100"
ARM[stack]="${ARM[kfl]} ${ARM[pxall]} ${ARM[path]}"
cd "$ND" || exit 1
echo "[q40] $(date) start"
FP=$(env V2_PRESET=shipped bash "$R" wac fp_v2 2>&1 | grep -E "SOLVED|NODES" | tr '\n' ' ')
echo "[q40] v2 shipped: $FP"
echo "$FP" | grep -q "254/300.*50622239" || { echo "[q40] ☠️ FINGERPRINT MISMATCH — aborting"; exit 1; }
for a in stack kfl pxall path; do
  for s in 0 1 2 3; do
    ( until env V2_PRESET=shipped ${ARM[$a]} PRESET=LONG_FORMAT MAX_DEPTH=10 USE_OPENING_BOOK=0 bash "$R" pyrun \
          diagnostics/_depth_residual_pass.py IN=ks_sets/dual_val_sf18.csv OUT=ks_sets/dual_val_sh_${a}_d10.csv \
          SHARD=$s/4 CHUNK=800 > /tmp/q40_${a}_$s.txt 2>&1; [ $? -ne 3 ]; do :; done ) &
  done
  wait
  echo "[q40] $(date) ship+$a rows: $(cat diagnostics/ks_sets/dual_val_sh_${a}_d10_s*of4.csv 2>/dev/null | grep -vc '^fen,')"
done
echo "[q40] $(date) QUEUE 40 DONE"
