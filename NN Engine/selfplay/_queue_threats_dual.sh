#!/bin/bash
# 2026-10-07 evening, queue #32 — THREATS: "improve actual play AND what pruning sees" (owner). The dual fit (MODE=dual) gives
# the engine-realisable arm (PCT 100 + hanging + king + pawn targets; restrict/push off) STATIC −4.1% (eg −4.9%), while the
# root-Δ depth PROXY reads +12.6% — a proxy that adds the ROOT threat delta to the d10 score and so cannot see search
# resolving it. This queue measures what the proxy cannot: (1) SPEED (pinned wac_speed, ship vs arm, 5 reps each, alone);
# (2) a REAL d10 re-search of the 4,961 held-out rows with the arm on, vs the ship's d10 on the same rows (`ours1004`);
# a 300-row ship re-run first checks that the ship's d10 still reproduces exactly. ≤ 4 engines. No build.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
ARM="THREAT_V2_PCT=100 THREAT_V2_HANGING=1 THREAT_V2_KING=1 THREAT_V2_PAWN_TARGETS=1"
cd "$ND" || exit 1
echo "[q32] $(date) start"
FP1=$(bash "$R" wac fp_v1 2>&1 | grep -E "SOLVED|NODES" | tr '\n' ' ')
FP2=$(env V2_PRESET=shipped bash "$R" wac fp_v2 2>&1 | grep -E "SOLVED|NODES" | tr '\n' ' ')
echo "[q32] v1: $FP1"; echo "[q32] v2 shipped: $FP2"
if ! echo "$FP1" | grep -q "250/300.*35310778" || ! echo "$FP2" | grep -q "254/300.*50622239"; then
  echo "[q32] ☠️ FINGERPRINT MISMATCH — aborting"; exit 1
fi
echo "[q32] $(date) speed: ship"
env V2_PRESET=shipped bash "$R" wac_speed th_ship 5 V2_PRESET=shipped
echo "[q32] $(date) speed: arm"
env V2_PRESET=shipped $ARM bash "$R" wac_speed th_arm 5 V2_PRESET=shipped $ARM
echo "[q32] $(date) ship d10 reproduction check (300 rows)"
env V2_PRESET=shipped PRESET=LONG_FORMAT MAX_DEPTH=10 USE_OPENING_BOOK=0 bash "$R" pyrun diagnostics/_depth_residual_pass.py \
    IN=ks_sets/dual_val_sf18.csv OUT=ks_sets/dual_val_ship_d10.csv LIMIT=300 > /tmp/q32_ship.txt 2>&1
tail -1 /tmp/q32_ship.txt
for s in 0 1 2 3; do
  ( until env V2_PRESET=shipped $ARM PRESET=LONG_FORMAT MAX_DEPTH=10 USE_OPENING_BOOK=0 bash "$R" pyrun \
        diagnostics/_depth_residual_pass.py IN=ks_sets/dual_val_sf18.csv OUT=ks_sets/dual_val_threats_d10.csv \
        SHARD=$s/4 CHUNK=800 > /tmp/q32_arm_$s.txt 2>&1; [ $? -ne 3 ]; do :; done ) &
done
wait
N=$(cat diagnostics/ks_sets/dual_val_threats_d10_s*of4.csv 2>/dev/null | grep -vc "^fen,")
echo "[q32] $(date) arm depth rows: $N"
echo "[q32] $(date) QUEUE 32 DONE"
