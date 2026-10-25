#!/bin/bash
# 2026-10-24 night, queue #100 (owner: long runs only, no builds). Engine = production (v2 gen 6 + PGO).
#  A concurrency stability: self-play at st=0.5, concurrency 1 / 3 / 4, gen-6 tables vs the old 1.23 GB tables (Knobs); the PGNs
#    keep per-move nodes and time, so per-engine nps and its spread can be compared (owner: "resources not fighting").
#  C Mediocre 0.5 at 180+1.8, 40 g (does the clock gap of 60+0.6 / 10+0.1 hold with more time?)
#  D warm depthcap probe: _fix_firecount.py SIDES=2 MOVETIME=1 OFF_ONLY=1, 150 games (does real play reach the depth cap?)
# Registered: A nps spread grows with concurrency and more so with the old tables; C within ±60 of level; D depthcap rare (0-3).
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
ENG="$ND/build_uci/chess_engine_uci"
MED="$ND/selfplay/mediocre_uci.sh"
RUNNER="$ND/selfplay/overnight_runner.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
OLD="TT_LOG2=24,EVAL_LOG2=23,QCACHE_LOG2=23,MGCACHE_LOG2=23"
cd "$ND" || exit 1
echo "[q100] $(date) start; bench $(printf 'bench\nquit\n' | "$ENG" 2>/dev/null | grep '^bench')  (want 257 / 46196950)"
stab() {   # $1 tag, $2 concurrency, $3 rounds, $4 knobs ('' = gen-6 tables)
  local ko=""; [ -n "$4" ] && ko="option.Knobs=$4"
  echo "[q100] $(date) $1 (concurrency $2, $(( $3 * 2 )) games)"
  "$FC" -engine cmd="$ENG" name=a option.OwnBook=false $ko st=0.5 timemargin=250 restart=on \
    -engine cmd="$ENG" name=b option.OwnBook=false $ko st=0.5 timemargin=250 restart=on \
    -openings file=$O/openings_uho.epd format=epd order=random -srand 100 -repeat -rounds $3 -games 2 -concurrency $2 -recover \
    -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10 \
    -pgnout file=$O/$1.pgn notation=san nodes=true append=false > $O/$1.log 2>&1
  echo "[q100]   $1 done: $(grep -E '^Games:' $O/$1.log | tail -1)  disconnects $(grep -ciE 'disconnect|stall|not responsive' $O/$1.log)"
}
stab stab_new_c1 1 4  ''
stab stab_new_c3 3 12 ''
stab stab_new_c4 4 16 ''
stab stab_old_c1 1 4  "$OLD"
stab stab_old_c3 3 12 "$OLD"
stab stab_old_c4 4 16 "$OLD"
echo "[q100] $(date) med_180 (40 games)"
nice -n 5 "$FC" -engine cmd="$ENG" name=gen6 option.OwnBook=false tc=180+1.8 timemargin=250 restart=on \
  -engine cmd="$MED" name=mediocre tc=180+1.8 timemargin=250 restart=on \
  -openings file=$O/openings_uho.epd format=epd order=random -srand 1001 -repeat -rounds 20 -games 2 -concurrency 3 -recover \
  -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10 \
  -pgnout file=$O/med_180.pgn notation=san nodes=true append=false > $O/med_180.log 2>&1
echo "[q100]   med_180 done: $(grep -E '^Games:|^Elo:' $O/med_180.log | tail -2 | tr '\n' ' ')  disconnects $(grep -ciE 'disconnect|stall|not responsive' $O/med_180.log)"
echo "[q100] $(date) depthcap probe (150 games)"
OMP_NUM_THREADS=1 bash "$RUNNER" pyrun diagnostics/_fix_firecount.py BIN="$ENG" MOVETIME=1 OFF_ONLY=1 SIDES=2 JOBS=2 N=50 \
  PLIES=200 SEED=100 > /mnt/e/chess_data/q100_depthcap.log 2>&1
grep -E '^===|diverged|drawexit|depthcap' /mnt/e/chess_data/q100_depthcap.log | sed 's/^/[q100]   /'
echo "[q100] $(date) QUEUE 100 DONE"
