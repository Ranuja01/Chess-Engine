#!/bin/bash
# 2026-10-07 night, queue #37 (owner OK; replaces #36, stopped 44 s into its first gauntlet). THREATS-LED CANDIDATE GATE.
# Real d10 re-search (val 4,952 rows): threats −7.8% vs ship (null −0.1%); on top of threats: mobility EG LEG ONLY −3.2%
# (mg −0.3%), space −0.9%, rook files −1.0% — but #36's COMBINED re-search (threats + mob-eg + space + rook files) read only
# −0.7% vs threats with middlegames +4.4% WORSE (space/rook files interact) ⇒ the gated pair is THREATS + MOBILITY-EG only.
# Games on BOTH instruments, fresh seeds: SF18 @1000 ship / threats / threats+mob-eg on 101-102, then self-play @50k each
# vs ship (seed 103). Node-limited only (safe in the owner's evening window). ≤ 4 engines. No build.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
O=/mnt/e/chess_data/texel/revival
TH="THREAT_V2_PCT=100 THREAT_V2_HANGING=1 THREAT_V2_KING=1 THREAT_V2_PAWN_TARGETS=1"
PAIR="$TH C1_V2_FIT=1 C1_V2_FILE=$O/mob_depth_c1_egonly.txt"
cd "$ND" || exit 1
echo "[q37] $(date) start"
FP=$(env V2_PRESET=shipped bash "$R" wac fp_v2 2>&1 | grep -E "SOLVED|NODES" | tr '\n' ' ')
echo "[q37] v2 shipped: $FP"
echo "$FP" | grep -q "254/300.*50622239" || { echo "[q37] ☠️ FINGERPRINT MISMATCH — aborting"; exit 1; }
SYM=$(env V2_PRESET=shipped $PAIR bash "$R" pyrun diagnostics/_eval_symmetry.py N=4000 2>&1 | grep -E "violations")
echo "[q37] PAIR symmetry: $(echo "$SYM" | tr '\n' ' ')"
[ "$(echo "$SYM" | grep -c 'violations 0 ')" = "2" ] || { echo "[q37] ☠️ PAIR symmetry FAILED — aborting"; exit 1; }
for s in 101 102; do
  bash "$R" gauntlet 500 1000 4 g1000_ship_s$s $s V2_PRESET=shipped
  bash "$R" gauntlet 500 1000 4 g1000_threats_s$s $s V2_PRESET=shipped $TH
  bash "$R" gauntlet 500 1000 4 g1000_thmobeg_s$s $s V2_PRESET=shipped $PAIR
  echo "[q37] $(date) SF18 seed $s done"
done
for arm in PAIR TH; do
  tag=$( [ "$arm" = PAIR ] && echo thmobeg_selfplay || echo threats_selfplay )
  bash "$R" pyrun selfplay/tournament.py --p1-label $tag --p1-config "V2_PRESET=shipped ${!arm} MAX_DEPTH=64 NODE_LIMIT=50000" \
    --p2-label shipped --p2-config "V2_PRESET=shipped MAX_DEPTH=64 NODE_LIMIT=50000" --games 2000 --concurrency 4 \
    --preset LONG_FORMAT --max-plies 400 --openings selfplay/openings_uho.txt --seed 103 --tag $tag --quiet
  echo "[q37] $(date) $tag: $(grep -o '"elo": [-0-9.]*' selfplay/games/$tag/tournament.json)"
done
echo "[q37] $(date) QUEUE 37 DONE"
