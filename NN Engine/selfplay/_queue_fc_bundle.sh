#!/bin/bash
# 2026-10-21, queue #89 (owner: go) — the cheap correctness bundle at EQUAL TIME, same binary both sides (build_uci/chess_engine_uci_c1),
# fresh seeds, st=1, concurrency 3, book off, UHO, restart=on. A2 / A1 stay OFF (A2 not confirmed in #88).
#   BUNDLE   B1 FIX_HALFMOVE + A4 FIX_MAX_NULL + B3 FIX_DRAW_EXIT + B2 bit 1 FIX_REP_KEY=1 + C1 FIX_QS_CHECK  vs gen 4, 400 g
#   REPB2    bundle with FIX_REP_KEY=3 (adds SF's pre-root 3-fold rule) vs the bundle, 200 g  (bit 2 cost 5 WAC solves)
# Registered: bundle ≈ 0 … +15 (bookkeeping + rare-mate fixes; "ship on not harmful"); REPB2 ≈ 0 ± noise.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
B="$ND/build_uci/chess_engine_uci_c1"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
EXIT_AT=${EXIT_AT:-20:30}
cd "$ND" || exit 1
echo "[q89] $(date) start"
FP=$(printf 'bench\nquit\n' | "$B" 2>/dev/null | grep "^bench")
echo "[q89] $FP"
echo "$FP" | grep -q "solved 258/300 nodes 47689413" || { echo "[q89] ☠️ BENCH MISMATCH — aborting"; exit 1; }
( while true; do sleep 1800; echo "[q89] $(date +%H:%M) mem: $(free -g | awk '/Mem:/{print "used " $3 "G avail " $7 "G"}') swap $(free -g | awk '/Swap:/{print $3 "G"}')"; done ) &
MEMPID=$!
COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
OURS="option.OwnBook=false st=1 timemargin=250 restart=on"
FIXES="option.Knob_FIX_HALFMOVE=1 option.Knob_FIX_MAX_NULL=1 option.Knob_FIX_DRAW_EXIT=1 option.Knob_FIX_QS_CHECK=1"
run() {   # $1 tag, $2 seed, $3 rounds (2 games each), $4... engine args
  local tag=$1 seed=$2 rounds=$3; shift 3
  local secs=$(( $(date -d "$EXIT_AT" +%s) - $(date +%s) )); [ $secs -lt 0 ] && secs=$(( secs + 86400 ))
  [ $secs -lt 900 ] && { echo "[q89] $(date) skip $tag: past EXIT_AT"; return; }
  echo "[q89] $(date) $tag ($(( rounds * 2 )) games), up to ${secs}s"
  timeout $secs nice -n 5 $FC "$@" $COMMON -srand $seed -rounds $rounds -games 2 \
    -pgnout file=$O/$tag.pgn notation=san nodes=true append=false > $O/$tag.log 2>&1
  echo "[q89] $(date) $tag done (rc $?): $(grep -E '^Games:|^Elo:' $O/$tag.log | tail -2 | tr '\n' ' ') $(grep -c 'not responsive' $O/$tag.log) unresponsive"
}
run fix_bundle_c1 891 200 -engine cmd="$B" name=bundle $OURS $FIXES option.Knob_FIX_REP_KEY=1 -engine cmd="$B" name=gen4 $OURS
run fix_repb2 892 100 -engine cmd="$B" name=bundle_rep3 $OURS $FIXES option.Knob_FIX_REP_KEY=3 \
    -engine cmd="$B" name=bundle $OURS $FIXES option.Knob_FIX_REP_KEY=1
kill $MEMPID 2>/dev/null
echo "[q89] $(date) QUEUE 89 DONE"
