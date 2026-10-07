#!/bin/bash
# 2026-10-07 evening, queue #35 (owner: "split mg and eg to optimise both?"). Queue #34's threats+mobility read −2.7% overall but
# eg −4.3 / non-eg +1.6 vs threats alone. v2 is already per-term mg/eg (tapered, no v1-style cliff), so the clean test is the
# fitted mobility table's ENDGAME leg only (mg = the shipped start values; built from mob_depth_c1.txt; total differences scale
# with (256 − phase), checked on 399 rows). Same rows / layout as #32-#34. WAITS for queue #34 to finish (≤ 4 engines).
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
O=/mnt/e/chess_data/texel/revival
TH="THREAT_V2_PCT=100 THREAT_V2_HANGING=1 THREAT_V2_KING=1 THREAT_V2_PAWN_TARGETS=1"
until grep -q "QUEUE 34 DONE" /mnt/e/chess_data/q34_dynamic_research.log 2>/dev/null; do sleep 60; done
cd "$ND" || exit 1
echo "[q35] $(date) start"
for s in 0 1 2 3; do
  ( until env V2_PRESET=shipped $TH C1_V2_FIT=1 C1_V2_FILE=$O/mob_depth_c1_egonly.txt PRESET=LONG_FORMAT MAX_DEPTH=10 \
        USE_OPENING_BOOK=0 bash "$R" pyrun diagnostics/_depth_residual_pass.py IN=ks_sets/dual_val_sf18.csv \
        OUT=ks_sets/dual_val_th_mobeg_d10.csv SHARD=$s/4 CHUNK=800 > /tmp/q35_$s.txt 2>&1; [ $? -ne 3 ]; do :; done ) &
done
wait
N=$(cat diagnostics/ks_sets/dual_val_th_mobeg_d10_s*of4.csv 2>/dev/null | grep -vc "^fen,")
echo "[q35] $(date) th+mob(eg leg only) rows: $N"
echo "[q35] $(date) QUEUE 35 DONE"
