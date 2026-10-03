#!/bin/bash
# 2026-10-03, queue #13: CONFIRMATION of the 10-03 ship (Kaufman fitted cells + joint KS/KS-B) — NEW shipped vs OLD shipped
# (the 10-01 config, reproduced via env overrides: 251 / 49,211,859 exactly). Calibrated SF18 @800, fresh seeds 66/67,
# both arms per seed (paired) → self-play 2,000 @50k new vs old (seed 68). Per-game engine processes (bounded memory).
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
cd "$ND" || exit 1
OLD="KS_V2_W_N=31 KS_V2_W_B=31 KS_V2_W_R=47 KS_V2_W_Q=78 KS_V2_COORD=256 KS_V2_WEAK=64 KS_V2_ADJ=50 KS_V2_NO_QUEEN=402 KS_V2_CHK_Q=260 KS_V2_CHK_R=193 KS_V2_CHK_B=141 KS_V2_CHK_N=189 KS_V2_MAX=4000 KS_V2_HALF=646 KS_V2_EG_PCT=100 KS_V2_ADJ_INST=-12 KS_V2_UNSAFE=19 KS_V2_FLANK_ATT=11 KS_V2_FLANK_ATT2=-1 KS_V2_KNIGHT_DEF=15 KS_V2_CONTEST_EXCESS=14 KS_V2_CONTEST_SQ=30 KS_V2_CONTEST_SQ_Q=19 KS_V2_BLOCKERS=0 KSB_V2=0 KAUF_V2_MAG=0"
for s in 66 67; do
  bash "$R" gauntlet 500 800 4 g800_old1001_s$s $s V2_PRESET=shipped $OLD
  bash "$R" gauntlet 500 800 4 g800_new1003_s$s $s V2_PRESET=shipped
done
bash "$R" pyrun selfplay/tournament.py --p1-label new1003 --p1-config "V2_PRESET=shipped MAX_DEPTH=64 NODE_LIMIT=50000" \
  --p2-label old1001 --p2-config "V2_PRESET=shipped $OLD MAX_DEPTH=64 NODE_LIMIT=50000" --games 2000 --concurrency 4 \
  --preset LONG_FORMAT --max-plies 400 --openings selfplay/openings_uho.txt --seed 68 --tag confirm_1003_selfplay --quiet
echo "[q13] $(date) QUEUE 13 DONE"
