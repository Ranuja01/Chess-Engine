#!/bin/bash
# 2026-10-20, queue #88 (owner: go, not gaming tonight) — fresh seeds, restart=on, concurrency 3, book off, UHO, adjudication.
#   A2C    FIX_MIN_SCOUT on vs off, st=1, 400 g (confirmation of q86's +49 ±37 on fresh seeds)
#   A2A1   A2+A1 vs A2, st=1, 200 g (the missing 2x2 cell: A1's value given A2; base-vs-A2 / base-vs-A1 = q86)
#   SF11L  v2 gen 4 @ 1 s vs SF11 @ 10 / 30 / 100 ms, 200 g each (the "playable with SF11" ladder)
#   AUT    loss autopsy of v2_vs_sf18_5ms.pgn (SF18 d14 WDL; one process, untimed — runs even past EXIT_AT)
# Bench check first; hard stop EXIT_AT for the timed games; RAM sampled every 30 min.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
B="$ND/build_uci/chess_engine_uci"
SF11="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/stockfish_11_linux/stockfish-11-linux/Linux/stockfish_20011801_x64_bmi2"
SF18="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/stockfish_18_linux/stockfish-ubuntu-x86-64-avx2"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
EXIT_AT=${EXIT_AT:-10:00}
cd "$ND" || exit 1
echo "[q88] $(date) start"
FP=$(printf 'bench\nquit\n' | "$B" 2>/dev/null | grep "^bench")
echo "[q88] $FP"
echo "$FP" | grep -q "solved 258/300 nodes 47689413" || { echo "[q88] ☠️ BENCH MISMATCH — aborting"; exit 1; }
[ -x "$SF11" ] || { echo "[q88] ☠️ SF11 binary missing — aborting"; exit 1; }
( while true; do sleep 1800; echo "[q88] $(date +%H:%M) mem: $(free -g | awk '/Mem:/{print "used " $3 "G avail " $7 "G"}') swap $(free -g | awk '/Swap:/{print $3 "G"}')"; done ) &
MEMPID=$!
COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
OURS="option.OwnBook=false st=1 timemargin=250 restart=on"
run() {   # $1 tag, $2 seed, $3 rounds (2 games each), $4... engine args
  local tag=$1 seed=$2 rounds=$3; shift 3
  local secs=$(( $(date -d "$EXIT_AT" +%s) - $(date +%s) )); [ $secs -lt 0 ] && secs=$(( secs + 86400 ))
  [ $secs -lt 900 ] && { echo "[q88] $(date) skip $tag: past EXIT_AT"; return; }
  echo "[q88] $(date) $tag ($(( rounds * 2 )) games), up to ${secs}s"
  timeout $secs nice -n 5 $FC "$@" $COMMON -srand $seed -rounds $rounds -games 2 \
    -pgnout file=$O/$tag.pgn notation=san nodes=true append=false > $O/$tag.log 2>&1
  echo "[q88] $(date) $tag done (rc $?): $(grep -E '^Games:|^Elo:' $O/$tag.log | tail -2 | tr '\n' ' ') $(grep -c 'not responsive' $O/$tag.log) unresponsive"
}
run fix_A2_confirm 881 200 -engine cmd="$B" name=fix_A2 $OURS option.Knob_FIX_MIN_SCOUT=1 -engine cmd="$B" name=gen4 $OURS
run fix_A2A1_vs_A2 882 100 -engine cmd="$B" name=fix_A2A1 $OURS option.Knob_FIX_MIN_SCOUT=1 option.Knob_FIX_RESEARCH=1 \
    -engine cmd="$B" name=fix_A2 $OURS option.Knob_FIX_MIN_SCOUT=1
for MS in 10 30 100; do
  run v2_vs_sf11_${MS}ms 88$(( MS == 10 ? 3 : MS == 30 ? 4 : 5 )) 100 -engine cmd="$B" name=v2_gen4 $OURS \
    -engine cmd="$SF11" name=sf11_${MS}ms option.Threads=1 option.Hash=16 st=0.$(printf '%03d' $MS) timemargin=250 restart=on
done
echo "[q88] $(date) autopsy start"
bash selfplay/overnight_runner.sh pyrun diagnostics/_loss_autopsy.py PGN=$O/v2_vs_sf18_5ms.pgn OURS=v2_gen4 BIN="$B" \
  SF="$SF18" DEPTH=14 LONG=10 OUT=/mnt/e/chess_data/autopsy > /mnt/e/chess_data/autopsy_sf18_5ms.log 2>&1
echo "[q88] $(date) autopsy done (rc $?): $(grep -c '^\[autopsy\] game' /mnt/e/chess_data/autopsy_sf18_5ms.log) games"
kill $MEMPID 2>/dev/null
echo "[q88] $(date) QUEUE 88 DONE"
