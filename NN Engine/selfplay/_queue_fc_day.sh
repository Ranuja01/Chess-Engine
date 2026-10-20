#!/bin/bash
# 2026-10-20, queue #86 (owner: start now) — the runs the overnight queue lost, on restart=on (a fresh engine per game; the
# memory growth that stalled fastchess). Concurrency 3, book off, UHO openings, adjudication. Bench check first; hard stop EXIT_AT.
#   SF5   v2 gen 4 @ 1 s vs Stockfish 18 @ 5 ms, 200 g        SF20  … vs SF18 @ 20 ms, 200 g
#   A2    FIX_MIN_SCOUT on vs off (same binary), st=1, 200 g    A1    FIX_RESEARCH on vs off, st=1, 200 g
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
B="$ND/build_uci/chess_engine_uci"
SF="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/stockfish_18_linux/stockfish-ubuntu-x86-64-avx2"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
EXIT_AT=${EXIT_AT:-18:00}
cd "$ND" || exit 1
echo "[q86] $(date) start"
FP=$(printf 'bench\nquit\n' | "$B" 2>/dev/null | grep "^bench")
echo "[q86] $FP"
echo "$FP" | grep -q "solved 258/300 nodes 47689413" || { echo "[q86] ☠️ BENCH MISMATCH — aborting"; exit 1; }
COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
OURS="option.OwnBook=false st=1 timemargin=250 restart=on"
run() {   # $1 tag, $2 seed, $3... engine args
  local tag=$1 seed=$2; shift 2
  local secs=$(( $(date -d "$EXIT_AT" +%s) - $(date +%s) ))
  [ $secs -lt 900 ] && { echo "[q86] $(date) skip $tag: past EXIT_AT"; return; }
  echo "[q86] $(date) $tag (200 games), up to ${secs}s"
  timeout $secs nice -n 5 $FC "$@" $COMMON -srand $seed -rounds 100 -games 2 \
    -pgnout file=$O/$tag.pgn notation=san nodes=true append=false > $O/$tag.log 2>&1
  echo "[q86] $(date) $tag done (rc $?): $(grep -E '^Games:|^Elo:' $O/$tag.log | tail -2 | tr '\n' ' ') $(grep -c 'not responsive' $O/$tag.log) unresponsive"
}
run v2_vs_sf18_5ms 861 -engine cmd="$B" name=v2_gen4 $OURS \
    -engine cmd="$SF" name=sf18_5ms option.Threads=1 option.Hash=16 st=0.005 timemargin=250 restart=on
run v2_vs_sf18_20ms 862 -engine cmd="$B" name=v2_gen4 $OURS \
    -engine cmd="$SF" name=sf18_20ms option.Threads=1 option.Hash=16 st=0.02 timemargin=250 restart=on
run fix_A2 863 -engine cmd="$B" name=fix_A2 $OURS option.Knob_FIX_MIN_SCOUT=1 -engine cmd="$B" name=gen4 $OURS
run fix_A1 864 -engine cmd="$B" name=fix_A1 $OURS option.Knob_FIX_RESEARCH=1 -engine cmd="$B" name=gen4 $OURS
echo "[q86] $(date) QUEUE 86 DONE"
