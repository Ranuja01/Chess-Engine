#!/bin/bash
# 2026-10-16 night, queue #75 (v3) — TARGETED checks of POT set A on BALANCED endgame starts, OPENING BOOK OFF.
# ☠️ v2's node-limited match (+8.3 ± 23.4) had the OPENING BOOK on: games from FEN starts consult the book for the first 30
# plies and M11.2.bin contains endgame positions ⇒ 5.3% of moves (23% of games, mostly plies 1-12) were BOOK moves, not either
# eval's — diluting the A/B toward 0. The book path also drops the promotion piece (illegal `h7h8`, 29 voids; ChessAI.pyx fix
# pending a rebuild). ⇒ USE_OPENING_BOOK=0 for BOTH players here (and for every FEN-start venue from now on).
#  1. NODE-LIMITED @50k, 1,200 games (600 balanced starts × both colours), seed 267, book off — starts now.
#     PREDICTION (registered): slightly above v2's +8.3 (book dilution removed), 0 void games.
#  2. TIMED (LIGHTNING), 600 games, seed 268, book off — only 00:00-08:59 (owner plays 9pm-midnight).
#     PREDICTION: positive, 10-30% below step 1 (race costs −22% NPS inside pure pawn endings).
# ≤ 3 processes. ⚠️ Machine suspend corrupts timed games.
A="POT_V2_RACE=1 POT_V2_WIN_PAWNEND=64 POT_V2_WIN_ONEMINOR=16 POT_V2_WIN_RACE=-12 POT_V2_RACE_W_BEST=137 POT_V2_RACE_W_PROMO=217 POT_V2_RACE_W_TEMPI=202 POT_V2_RACE_W_DECISIVE=51"
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
OPEN=selfplay/eg_balanced_fens.txt
cd "$ND" || exit 1
echo "[q75] $(date) start (v3, book off)"
FP=$(env V2_PRESET=shipped bash "$R" wac fp_v2 2>&1 | grep -E "SOLVED|NODES" | tr '\n' ' ')
echo "$FP" | grep -q "259/300.*47869841" || { echo "[q75] ☠️ FINGERPRINT MISMATCH (expected gen 3) — aborting"; exit 1; }
match() {   # $1 tag, $2 games, $3 seed, $4 preset, $5 node-limit config suffix
  bash "$R" pyrun selfplay/tournament.py --p1-label $1 --p1-config "V2_PRESET=shipped $A USE_OPENING_BOOK=0 MAX_DEPTH=64 $5" \
    --p2-label shipped --p2-config "V2_PRESET=shipped USE_OPENING_BOOK=0 MAX_DEPTH=64 $5" --games $2 --concurrency 3 \
    --preset $4 --max-plies 400 --openings $OPEN --seed $3 --tag $1 --quiet
  echo "[q75] $(date) $1: $(grep -oE '"elo(_margin)?": [-0-9.]+' selfplay/games/$1/tournament.json | tr '\n' ' ')"
}
match potA_egbal_nb267 1200 267 LONG_FORMAT NODE_LIMIT=50000
until [ "$(date +%H)" -lt 9 ]; do sleep 300; done
match potA_egbal_timed_nb268 600 268 LIGHTNING ""
echo "[q75] $(date) QUEUE 75 DONE"
