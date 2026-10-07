#!/bin/bash
# 2026-10-07 evening, queue #34 (owner: "understand if they even can fit in" — option 2). Every DYNAMIC term closed or parked on
# the ROOT-Δ depth proxy (memory root-delta-depth-proxy-is-biased-against-dynamic-terms) gets a REAL d10 re-search of the 4,952
# val rows, each ON TOP OF the threats arm (does it add to the candidate?), same SHARD=4 / CHUNK=800 layout as queues #32/#33
# (threats-only = dual_val_threats_d10, ship control = dual_val_shipctl_d10). Arm values = the ones each was screened at.
# Fixed-depth only (safe in the owner's evening window). ≤ 4 engines. No build.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
O=/mnt/e/chess_data/texel/revival
TH="THREAT_V2_PCT=100 THREAT_V2_HANGING=1 THREAT_V2_KING=1 THREAT_V2_PAWN_TARGETS=1"
declare -A ARM
ARM[mob]="C1_V2_FIT=1 C1_V2_FILE=$O/mob_depth_c1.txt"
ARM[kprot]="KPROT_V2=1 KPROT_V2_FILE=$O/kprot_depth.txt"
ARM[space]="SPACE_V2_MAG=1320"
ARM[longdiag]="LONGDIAG_V2_PCT=100"
ARM[reach]="REACH_V2_PCT=100"
ARM[latent]="LATENT_V2_PCT=100"
ARM[rookfile]="ROOKFILE_V2_OPEN=200 ROOKFILE_V2_SEMI=90"
cd "$ND" || exit 1
echo "[q34] $(date) start"
FP2=$(env V2_PRESET=shipped bash "$R" wac fp_v2 2>&1 | grep -E "SOLVED|NODES" | tr '\n' ' ')
echo "[q34] v2 shipped: $FP2"
echo "$FP2" | grep -q "254/300.*50622239" || { echo "[q34] ☠️ FINGERPRINT MISMATCH — aborting"; exit 1; }
for a in mob kprot space longdiag reach latent rookfile; do
  for s in 0 1 2 3; do
    ( until env V2_PRESET=shipped $TH ${ARM[$a]} PRESET=LONG_FORMAT MAX_DEPTH=10 USE_OPENING_BOOK=0 bash "$R" pyrun \
          diagnostics/_depth_residual_pass.py IN=ks_sets/dual_val_sf18.csv OUT=ks_sets/dual_val_th_${a}_d10.csv \
          SHARD=$s/4 CHUNK=800 > /tmp/q34_${a}_$s.txt 2>&1; [ $? -ne 3 ]; do :; done ) &
  done
  wait
  N=$(cat diagnostics/ks_sets/dual_val_th_${a}_d10_s*of4.csv 2>/dev/null | grep -vc "^fen,")
  echo "[q34] $(date) th+$a rows: $N"
done
echo "[q34] $(date) QUEUE 34 DONE"
