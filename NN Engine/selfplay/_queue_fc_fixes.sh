#!/bin/bash
# 2026-10-20, queue #85 — first EQUAL-TIME screens of the search correctness fixes (SEARCH-CORRECTNESS-PLAN-2026-10-19.md).
# Same binary both sides (build_uci/chess_engine_uci_fix), fix knob ON vs OFF, st=1, 200 games each, concurrency 3, book off.
#   A2  FIX_MIN_SCOUT (min PVS/LMR scout at the min bound) — WAC d10: 250 (−8), nodes −12.6%
#   A1  FIX_RESEARCH  (reduced fail-high re-searched; min VERIFY mirrored) — WAC d10: 259 (+1), nodes +33.6%
# Waits for queue #84 to finish unless NO_WAIT=1. Hard stop EXIT_AT.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
B="$ND/build_uci/chess_engine_uci_fix"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
EXIT_AT=${EXIT_AT:-09:00}
echo "[q85] $(date) armed"
[ "${NO_WAIT:-0}" = "1" ] || until grep -q "QUEUE 84 DONE" /mnt/e/chess_data/q84_fc_night2.log 2>/dev/null; do sleep 60; done
cd "$ND" || exit 1
echo "[q85] $(date) start"
FP=$(printf 'bench\nquit\n' | "$B" 2>/dev/null | grep "^bench")
echo "[q85] off-identity: $FP"
echo "$FP" | grep -q "solved 258/300 nodes 47689413" || { echo "[q85] ☠️ BENCH MISMATCH — aborting"; exit 1; }
COMMON="-openings file=$O/openings_uho.epd format=epd order=random -srand 85 -repeat -concurrency 3 -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10 -each st=1 timemargin=250 restart=on"
run() {   # $1 tag, $2 knob
  local secs=$(( $(date -d "$EXIT_AT" +%s) - $(date +%s) )); [ $secs -lt 0 ] && secs=$(( secs + 86400 ))
  [ $secs -lt 600 ] && { echo "[q85] $(date) skip $1: past EXIT_AT"; return; }
  echo "[q85] $(date) $1 ($2 on vs off, 200 games), up to ${secs}s"
  timeout $secs nice -n 5 $FC -engine cmd="$B" name=fix_$1 option.OwnBook=false option.Knob_$2=1 \
    -engine cmd="$B" name=gen4 option.OwnBook=false $COMMON -rounds 100 -games 2 \
    -pgnout file=$O/fix_$1.pgn notation=san nodes=true append=false > $O/fix_$1.log 2>&1
  echo "[q85] $(date) $1 done (rc $?): $(grep -E '^Games:|^Elo:' $O/fix_$1.log | tail -2 | tr '\n' ' ')"
}
run A2 FIX_MIN_SCOUT
run A1 FIX_RESEARCH
echo "[q85] $(date) QUEUE 85 DONE"
