#!/bin/bash
# 2026-10-19, queue #83 — FIRST TIMED BASELINES through the native UCI binary + fastchess (UCI design M5; owner 10-19).
#   (a) v2 gen 4 vs v1 at a FIXED 1 s per move (st=1) — the clean "equal time" comparison; actual time per move logged in PGN
#   (b) v2 gen 4 vs v1 on a 60+0.6 CLOCK — first ever time-pressure test (placeholder clock rule, design §4)
# 300 games each (150 UHO openings, colours swapped), concurrency 3, book off, adjudication (resign 3 moves ≥ 10 pawns, draw
# after move 40 with 8 moves within 0.1). Waits for queue #82 to finish (its log line) unless NO_WAIT=1. Hard stop EXIT_AT.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
B="$ND/build_uci/chess_engine_uci"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
Q82=/mnt/e/chess_data/q82_wdl_full.log
EXIT_AT=${EXIT_AT:-09:00}
echo "[q83] $(date) armed"
if [ "${NO_WAIT:-0}" != "1" ]; then
  until grep -q "QUEUE 82 DONE" "$Q82" 2>/dev/null; do sleep 60; done
fi
echo "[q83] $(date) start; fingerprint check via bench"
FP=$(printf 'bench\nquit\n' | "$B" 2>/dev/null | grep "^bench")
echo "[q83] $FP"
echo "$FP" | grep -q "solved 258/300 nodes 47689413" || { echo "[q83] ☠️ BENCH MISMATCH (expected 258 / 47689413) — aborting"; exit 1; }
run() {   # $1 tag, $2 time control args
  local secs=$(( $(date -d "$EXIT_AT" +%s) - $(date +%s) )); [ $secs -lt 0 ] && secs=$(( secs + 86400 ))
  echo "[q83] $(date) $1 ($2), up to ${secs}s"
  timeout $secs nice -n 5 $FC -engine cmd="$B" name=v2_gen4 option.OwnBook=false -engine cmd="$B" name=v1 option.EvalArm=0 option.OwnBook=false \
    -each $2 timemargin=250 -openings file=$O/openings_uho.epd format=epd order=random -srand 83 \
    -rounds 150 -games 2 -repeat -concurrency 3 -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10 \
    -pgnout file=$O/$1.pgn notation=san nodes=true append=false > $O/$1.log 2>&1
  echo "[q83] $(date) $1 done (rc $?): $(grep -E '^Games:|^Elo:|^Ptnml' $O/$1.log | tail -3 | tr '\n' ' ')"
}
run v2_v1_st1 "st=1"
run v2_v1_60p06 "tc=60+0.6"
echo "[q83] $(date) QUEUE 83 DONE"
