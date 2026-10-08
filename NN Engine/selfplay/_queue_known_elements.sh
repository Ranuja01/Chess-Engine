#!/bin/bash
# 2026-10-08, queue #39 (owner: "prepare the next items — everything in place before POT"). The KNOWN ELEMENTS left after
# threats, per the SF11 endgame decomposition (bench §3a: king −18% · placement −15% · passed −12%): every table here was
# fitted on the ROOT-Δ depth proxy, and king/passer terms are partly DYNAMIC (races, path safety, king moves) ⇒ a REAL d10
# re-search of the 4,952 val rows, each ON TOP OF threats (the candidate base), same SHARD=4 / CHUNK=800 layout as #32-#35
# (threats-only = dual_val_threats_d10). WAITS for queue #38 (SF18 games) so engines never exceed 4. No build.
#   kfl        KFL C3-b (king-to-own/enemy-pawn distance + pawnless flank), depth-fit cells
#   kfleg      the same, ENDGAME LEG ONLY (mg = 0 = as shipped)
#   passer     queue #14 passer re-price (rank tables both legs + king coefficients, cells 78-96 via C1)
#   pxall      PX passer cells + passer re-price fitted jointly (the 10-04 "ALL" arm)
#   path       SF11's passer free-path ladder at 100% (rejected 09-25 on 7 arms at the OLD passer magnitude)
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
O=/mnt/e/chess_data/texel/revival
TX=/mnt/e/chess_data/texel
TH="THREAT_V2_PCT=100 THREAT_V2_HANGING=1 THREAT_V2_KING=1 THREAT_V2_PAWN_TARGETS=1"
declare -A ARM
ARM[kfl]="KFL_V2=1 KFL_V2_FILE=$O/kfl_depth.txt"
ARM[kfleg]="KFL_V2=1 KFL_V2_FILE=$O/kfl_depth_egonly.txt"
ARM[passer]="C1_V2_FIT=1 C1_V2_FILE=$TX/pawn_depth_passer.txt"
ARM[pxall]="C1_V2_FIT=1 C1_V2_FILE=$TX/px_depth_c1.txt PX_V2=1 PX_V2_FILE=$TX/px_depth_px.txt"
ARM[path]="PASSER_V2_PATH_PCT=100"
until grep -q "QUEUE 38 DONE" /mnt/e/chess_data/q38_threats_sf18.log 2>/dev/null; do
  grep -qE "☠️|aborting" /mnt/e/chess_data/q38_threats_sf18.log 2>/dev/null && { echo "[q39] queue #38 aborted — not starting"; exit 1; }
  sleep 120
done
cd "$ND" || exit 1
echo "[q39] $(date) start"
FP=$(env V2_PRESET=shipped bash "$R" wac fp_v2 2>&1 | grep -E "SOLVED|NODES" | tr '\n' ' ')
echo "[q39] v2 shipped: $FP"
echo "$FP" | grep -q "254/300.*50622239" || { echo "[q39] ☠️ FINGERPRINT MISMATCH — aborting"; exit 1; }
for a in kfl kfleg passer pxall path; do
  for s in 0 1 2 3; do
    ( until env V2_PRESET=shipped $TH ${ARM[$a]} PRESET=LONG_FORMAT MAX_DEPTH=10 USE_OPENING_BOOK=0 bash "$R" pyrun \
          diagnostics/_depth_residual_pass.py IN=ks_sets/dual_val_sf18.csv OUT=ks_sets/dual_val_th_${a}_d10.csv \
          SHARD=$s/4 CHUNK=800 > /tmp/q39_${a}_$s.txt 2>&1; [ $? -ne 3 ]; do :; done ) &
  done
  wait
  echo "[q39] $(date) th+$a rows: $(cat diagnostics/ks_sets/dual_val_th_${a}_d10_s*of4.csv 2>/dev/null | grep -vc '^fen,')"
done
echo "[q39] $(date) QUEUE 39 DONE"
