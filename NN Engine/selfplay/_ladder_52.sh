#!/bin/bash
# Owner UI game 11-12 vs SF1.1 (loss from a won position): which pruning hides 52...Bc4's refutation from depth 11 on (one
# technique off at a time, depth 16), and SF18's view of 67...Kc3, the move our search prefers there.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
G=/mnt/e/chess_data/owner_games/2026-11-12-sf11-equal-60p06-loss
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
bash "$R" pyrun diagnostics/_depth_ladder.py PGN=$G.pgn SIDE=black MOVES=52 DEPTH=16 MAXSEC=90 VERIFY_DEPTH=16 \
	"VARIANTS=normal:|lmr_off:ENABLE_LMR=0|futility_off:ENABLE_FUTILITY=0|razor_off:ENABLE_RAZORING=0,ENABLE_ROOT_RAZOR=0|lmp_off:ENABLE_LMP=0|rfp_off:ENABLE_RFP=0|seeprune_off:SEE_PRUNE_CAPTURES=0|null_off:ENABLE_NULLMOVE=0" \
	> $G.ladder52.txt 2>&1
bash "$R" pyrun diagnostics/_depth_ladder.py PGN=$G.pgn SIDE=black MOVES=67 DEPTH=12 MAXSEC=30 VERIFY_DEPTH=16 EXTRA=Kc3 \
	"VARIANTS=normal:" > $G.ladder67.txt 2>&1
echo "LADDER DONE" >> $G.ladder67.txt
