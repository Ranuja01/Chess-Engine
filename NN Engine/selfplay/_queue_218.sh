#!/bin/bash
# 2026-11-12 ~15:40, queue #218: SF18 grades our moves in the owner's two UI games of 11-12 (60+0.6 each), one SF18 thread,
# 4M nodes per position (diagnostics/_grade_game.py). The equal-time loss to SF1.1 (ours = Black, "a winning position that
# was not seen through"), then the knight-odds win over SF18 (ours = Black). Runs beside #217's build; #217's pinned bench
# waits for it. Output E:\chess_data\owner_games\*.grade.txt.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
G=/mnt/e/chess_data/owner_games
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
echo "[q218] $(date '+%Y-%m-%d %H:%M:%S') start"
for f in 2026-11-12-sf11-equal-60p06-loss 2026-11-12-sf18-knight-odds-60p06-win; do
	bash "$R" pyrun diagnostics/_grade_game.py PGN=$G/$f.pgn SIDE=black NODES=4000000 > $G/$f.grade.txt 2>&1
	echo "[q218] $f:"; sed 's/^/[q218]   /' $G/$f.grade.txt
done
echo "[q218] $(date) QUEUE 218 DONE"
