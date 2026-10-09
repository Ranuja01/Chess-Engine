#!/bin/bash
# 2026-10-09, queue #44 (owner OK). INSTRUMENT VALIDATION: real d10 MSE vs SF18 improved for threats/PX/KFL yet all three were
# null-to-negative vs SF18 in games (memory real-d10-accuracy-gains-did-not-become-elo). Does the MOVE-based footprint regret
# (changed-move win%, null band 49.8-50.4 on game_regret_set, 50.7 on _v2; real bar ~2-2.5 pp, cross-set mandatory) rank KNOWN
# game outcomes correctly? Base = today's ship; arms (each inherits the base):
#   KNOWN NULL (added):     th · px · kfl                         → expect ≈ the null band
#   KNOWN GAIN (REMOVED):   noconn (−connected, +14.5) · no1003 (−10-03 −connected, ≈ −30) · nofitA (pre-PST-fit, +111/+38)
#                           → expect clearly WORSE than the band (win% < 50)
#   NEUTRAL:                asp300 (ASPIRATION_DELTA=300, the documented neutral)
# Fixed depth 7 (the instrument's validated setting). Both cross-sets. ≤ 4 engines (JOBS=4). No build.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
O=/mnt/e/chess_data/texel/revival
TX=/mnt/e/chess_data/texel
TH="THREAT_V2_PCT=100 THREAT_V2_HANGING=1 THREAT_V2_KING=1 THREAT_V2_PAWN_TARGETS=1"
PX="C1_V2_FIT=1 C1_V2_FILE=$TX/px_depth_c1.txt PX_V2=1 PX_V2_FILE=$TX/px_depth_px.txt"
KFL="KFL_V2=1 KFL_V2_FILE=$O/kfl_depth.txt"
OLD="KS_V2_W_N=31 KS_V2_W_B=31 KS_V2_W_R=47 KS_V2_W_Q=78 KS_V2_COORD=256 KS_V2_WEAK=64 KS_V2_ADJ=50 KS_V2_NO_QUEEN=402 KS_V2_CHK_Q=260 KS_V2_CHK_R=193 KS_V2_CHK_B=141 KS_V2_CHK_N=189 KS_V2_MAX=4000 KS_V2_HALF=646 KS_V2_EG_PCT=100 KS_V2_ADJ_INST=-12 KS_V2_UNSAFE=19 KS_V2_FLANK_ATT=11 KS_V2_FLANK_ATT2=-1 KS_V2_KNIGHT_DEF=15 KS_V2_CONTEST_EXCESS=14 KS_V2_CONTEST_SQ=30 KS_V2_CONTEST_SQ_Q=19 KS_V2_BLOCKERS=0 KSB_V2=0 KAUF_V2_MAG=0 PS_V2_CONN_MAG=0"
CAND="$TH;$PX;$KFL;PS_V2_CONN_MAG=0;$OLD;PST_V2_TAPERED=1;ASPIRATION_DELTA=300"
cd "$ND" || exit 1
echo "[q44] $(date) start"
FP=$(env V2_PRESET=shipped bash "$R" wac fp_v2 2>&1 | grep -E "SOLVED|NODES" | tr '\n' ' ')
echo "[q44] v2 shipped: $FP"
echo "$FP" | grep -q "254/300.*50622239" || { echo "[q44] ☠️ FINGERPRINT MISMATCH — aborting"; exit 1; }
for SET in game_regret_set game_regret_set_v2; do
  echo "[q44] $(date) set $SET"
  # MAXN=6000 per set (smoke test: 100 rows × 4 configs = 42 s ⇒ full sets ≈ 6 h; 6k ≈ 1.5 h each, ~2k changed moves/arm)
  env V2_PRESET=shipped bash "$R" pyrun diagnostics/_ks_footprint_regret.py SET=ks_sets/$SET.csv DEPTH=7 JOBS=4 MAXN=6000 \
      BASE_KNOBS="V2_PRESET=shipped" CAND_KNOBS="$CAND" > /mnt/e/chess_data/q44_$SET.txt 2>&1
  echo "[q44] $(date) $SET done (exit $?)"
done
echo "[q44] $(date) QUEUE 44 DONE"
