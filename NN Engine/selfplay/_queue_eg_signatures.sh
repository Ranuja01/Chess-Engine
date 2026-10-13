#!/bin/bash
# 2026-10-13, queue #62 — POT / endgame-knowledge groundwork: the reference ladder (SF11 · SF15.1 classical · SF15.1 NNUE · SF18
# static · v2 gen 3) on ~6,000 held-out positions grouped by Stockfish endgame.cpp signature (_eg_signatures.py MODE=build),
# then the per-signature table (MODE=read). Static, 1 engine. Waits for queue #61.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
B=/mnt/e/chess_data/bench1013
until grep -qE "QUEUE 61 DONE|☠️" /mnt/e/chess_data/q61_round3.log 2>/dev/null; do sleep 120; done
cd "$ND" || exit 1
echo "[q62] $(date) start"
rm -f $B/eg_sig_dump.csv
env PRESET=LONG_FORMAT USE_OPENING_BOOK=0 CORPUS=$B/eg_signatures.csv N=99999 DUMP=$B/eg_sig_dump.csv REFS=1 DUMPLABEL="v2 gen3" \
    V2_PRESET=shipped bash "$R" pyrun diagnostics/_reference_ceiling.py 2>&1 | grep -E "^  [A-Za-z].* [0-9n].* [0-9]+$|rror" | sed 's/^/[q62] /'
bash "$R" pyrun diagnostics/_eg_signatures.py MODE=read DUMP=$B/eg_sig_dump.csv SIG=$B/eg_signatures.csv 2>&1 | sed 's/^/[q62] /'
echo "[q62] $(date) QUEUE 62 DONE"
