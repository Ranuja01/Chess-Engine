#!/bin/bash
# 2026-10-24, queue #99 (owner: Mediocre at mixed time controls "for fun" + the depthcap probe alongside).
#  A  gen 6 (production binary) vs Mediocre 0.5 (Java): st=1 100 g · 60+0.6 100 g · 10+0.1 100 g, 3 at a time, -recover
#  B  in parallel, one job: _fix_firecount.py SIDES=2 MOVETIME=1 OFF_ONLY=1 — how often does iterative deepening reach the
#     MAX_PLY - 4 cap in real play (FIXCNT depthcap), next to drawexit (FIX_DRAW_EXIT firing)
# Registered: A st=1 +100 … +200, 60+0.6 +100 … +200, 10+0.1 +50 … +250 (10-07: gen ~1 scored 45% at 1 s over 50 g).
#             B depthcap > 0 in some endgame-set games, and those games also show drawexit > 0.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
ENG="$ND/build_uci/chess_engine_uci"
MED="$ND/selfplay/mediocre_uci.sh"
RUNNER="$ND/selfplay/overnight_runner.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
cd "$ND" || exit 1
echo "[q99] $(date) start; engine bench: $(printf 'bench\nquit\n' | "$ENG" 2>/dev/null | grep '^bench')  (want 257 / 46196950)"

( OMP_NUM_THREADS=1 bash "$RUNNER" pyrun diagnostics/_fix_firecount.py BIN="$ENG" MOVETIME=1 OFF_ONLY=1 SIDES=2 JOBS=1 N=15 \
    PLIES=200 SEED=99 > /mnt/e/chess_data/q99_depthcap.log 2>&1; echo "[q99] $(date) depthcap probe done" ) &

COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
med() {   # $1 tag, $2 seed, $3 time control args, $4 rounds
  echo "[q99] $(date) $1 ($(( $4 * 2 )) games)"
  nice -n 5 "$FC" -engine cmd="$ENG" name=gen6 option.OwnBook=false $3 timemargin=250 restart=on \
    -engine cmd="$MED" name=mediocre $3 timemargin=250 restart=on $COMMON -srand $2 -rounds $4 -games 2 \
    -pgnout file=$O/$1.pgn notation=san nodes=true append=false > $O/$1.log 2>&1
  echo "[q99] $(date) $1 done (rc $?): $(grep -E '^Games:|^Elo:' $O/$1.log | tail -2 | tr '\n' ' ')"
  echo "[q99]   disconnects/stalls: $(grep -ciE 'disconnect|stall|not responsive' $O/$1.log)  time losses: $(grep -ciE 'loses on time' $O/$1.log)"
}
med med_st1   991 st=1        50
med med_60p6  992 tc=60+0.6   50
med med_10p01 993 tc=10+0.1   50
wait
echo "[q99] depthcap probe summary:"
grep -E '^===|diverged|drawexit|depthcap|CRASH' /mnt/e/chess_data/q99_depthcap.log | sed 's/^/[q99]   /'
echo "[q99] $(date) QUEUE 99 DONE"
