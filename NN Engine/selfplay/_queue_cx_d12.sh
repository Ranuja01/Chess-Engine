#!/bin/bash
# 2026-10-18, queue #80 — COMPLEXITY SCALER test data (the last untested piece of POT's time box; memory
# pot-may-not-work-use-stopping-points). Candidate base (gen 3 + C64), d12 search on 6,000 EGX endgame rows sampled by game
# (≤ 2 per game, 3,218 games; E:\chess_data\cx\cx_in.csv). Resign disabled (amplified evals once dropped 113 rows).
# ONE process, fixed depth, single-threaded math ⇒ safe beside the owner's games and server.
# KEEP criterion (registered before the data): the SF-bounded complexity edge leg, fitted on the d12 residual and held out by
# game, cuts endgame d12 win% MSE by ≥ 2% beyond the cells. PREDICTION: miss (≈ −0.5%): the 10-13 probe's depth gain was
# −1.5% at d10 and its biggest input (+580 pure pawn ending) is what the class cells now own.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
C=/mnt/e/chess_data/cx
cd "$ND" || exit 1
echo "[q80] $(date) start"
until timeout 5400 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PRESET=LONG_FORMAT MAX_DEPTH=12 \
      USE_OPENING_BOOK=0 V2_PRESET=candidate RESIGN_THRESHOLD=-99999999 bash "$R" pyrun \
      diagnostics/_depth_residual_pass.py IN=$C/cx_in.csv OUT=$C/cx_cand_d12.csv CHUNK=500 > /dev/null 2>&1; [ $? -ne 3 ]; do
  echo "[q80] $(date) chunk done: $(grep -vc '^fen' $C/cx_cand_d12.csv) rows"
done
echo "[q80] $(date) QUEUE 80 DONE: $(grep -vc '^fen' $C/cx_cand_d12.csv) rows"
