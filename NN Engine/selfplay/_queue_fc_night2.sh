#!/bin/bash
# 2026-10-20, queue #84 (owner-approved): on the FIXED binary (PRINT_CACHE_STATS=0), concurrency 3, book off, UHO openings.
#   A  v2 gen 4 vs v1, st=1, 300 games — completes the 10-19 run that stopped at 155 ("v1 not responsive"); fastchess trace log
#      kept to catch that abort if it recurs
#   B  v2 gen 4 @ 1 s/move vs Stockfish 18 @ 5 ms/move (the owner's UI matchup), 200 games
#   C  v2 gen 4 @ 1 s/move vs Stockfish 18 @ 20 ms/move, 200 games
# Bench fingerprint check first; hard stop EXIT_AT.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
B="$ND/build_uci/chess_engine_uci"
SF="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/stockfish_18_linux/stockfish-ubuntu-x86-64-avx2"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
EXIT_AT=${EXIT_AT:-09:00}
cd "$ND" || exit 1
echo "[q84] $(date) start"
FP=$(printf 'bench\nquit\n' | "$B" 2>/dev/null | grep "^bench")
echo "[q84] $FP"
echo "$FP" | grep -q "solved 258/300 nodes 47689413" || { echo "[q84] ☠️ BENCH MISMATCH — aborting"; exit 1; }
[ -x "$SF" ] || { echo "[q84] ☠️ Stockfish 18 not found at $SF — aborting"; exit 1; }
COMMON="-openings file=$O/openings_uho.epd format=epd order=random -srand 84 -repeat -concurrency 3 -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
run() {   # $1 tag, $2 rounds, $3... engine args
  local tag=$1 rounds=$2; shift 2
  local secs=$(( $(date -d "$EXIT_AT" +%s) - $(date +%s) )); [ $secs -lt 0 ] && secs=$(( secs + 86400 ))
  echo "[q84] $(date) $tag ($((rounds * 2)) games), up to ${secs}s"
  timeout $secs nice -n 5 $FC "$@" $COMMON -rounds $rounds -games 2 -pgnout file=$O/$tag.pgn notation=san nodes=true append=false > $O/$tag.log 2>&1
  echo "[q84] $(date) $tag done (rc $?): $(grep -E '^Games:|^Elo:' $O/$tag.log | tail -2 | tr '\n' ' ') $(grep -c 'not responsive' $O/$tag.log) unresponsive"
}
run v2_v1_st1_full 150 -engine cmd="$B" name=v2_gen4 option.OwnBook=false -engine cmd="$B" name=v1 option.EvalArm=0 option.OwnBook=false \
    -each st=1 timemargin=250 -log file=$O/v2_v1_st1_full.trace level=trace
run v2_vs_sf18_5ms 100 -engine cmd="$B" name=v2_gen4 option.OwnBook=false st=1 timemargin=250 \
    -engine cmd="$SF" name=sf18_5ms option.Threads=1 option.Hash=16 st=0.005 timemargin=250
run v2_vs_sf18_20ms 100 -engine cmd="$B" name=v2_gen4 option.OwnBook=false st=1 timemargin=250 \
    -engine cmd="$SF" name=sf18_20ms option.Threads=1 option.Hash=16 st=0.02 timemargin=250
echo "[q84] $(date) QUEUE 84 DONE"
