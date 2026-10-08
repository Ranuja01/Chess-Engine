#!/bin/bash
# 2026-10-08, queue #38 (owner OK). Queue #37: threats +32.1 ± 9 self-play (2,000 g) but +2 ± 12 vs SF18 @1000 (seeds 101-102)
# — the instruments disagree; self-play also inflates same-family differences. Four more FRESH SF18 seeds (104-107), ship vs
# threats, each 500 paired games, so the pooled SF18 read reaches ~±7 Elo over 6 seeds. Node-limited. ≤ 4 engines. No build.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
TH="THREAT_V2_PCT=100 THREAT_V2_HANGING=1 THREAT_V2_KING=1 THREAT_V2_PAWN_TARGETS=1"
cd "$ND" || exit 1
echo "[q38] $(date) start"
FP=$(env V2_PRESET=shipped bash "$R" wac fp_v2 2>&1 | grep -E "SOLVED|NODES" | tr '\n' ' ')
echo "[q38] v2 shipped: $FP"
echo "$FP" | grep -q "254/300.*50622239" || { echo "[q38] ☠️ FINGERPRINT MISMATCH — aborting"; exit 1; }
for s in 104 105 106 107; do
  bash "$R" gauntlet 500 1000 4 g1000_ship_s$s $s V2_PRESET=shipped
  bash "$R" gauntlet 500 1000 4 g1000_threats_s$s $s V2_PRESET=shipped $TH
  echo "[q38] $(date) SF18 seed $s done"
done
echo "[q38] $(date) QUEUE 38 DONE"
