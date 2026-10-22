#!/bin/bash
# 2026-10-22, queue #92 (owner: ok) — the REAL gen 5 harm check (the 10-21 bundle match that stood in for it was void: undeclared
# Knob_ options never reached the engine). gen 5 (default) vs gen 4 (option.Knobs=V2_SHIP_GEN=4), same binary, st=1, 400 g, UHO,
# fresh seed, restart=on, concurrency 3, book off. Guards: both benches, then 0 "doesn't have option" warnings + the knob echo.
# Registered: gen 5 − gen 4 ≈ 0 (B3 / C1 fire in ~1/6 / ~1/3 of general games, small per-move effect; STS −47 / −156 does not
# predict Elo); harm would be < −20.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
B="$ND/build_uci/chess_engine_uci"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
EXIT_AT=${EXIT_AT:-20:30}
cd "$ND" || exit 1
echo "[q92] $(date) start"
G5=$(printf 'bench\nquit\n' | "$B" 2>/dev/null | grep "^bench")
G4=$(printf 'setoption name Knobs value V2_SHIP_GEN=4\nisready\nbench\nquit\n' | "$B" 2>/dev/null | grep -E "^bench")
echo "[q92] gen5 $G5"; echo "[q92] gen4 $G4"
echo "$G5" | grep -q "solved 258/300 nodes 47689586" || { echo "[q92] ☠️ gen 5 BENCH MISMATCH — aborting"; exit 1; }
echo "$G4" | grep -q "solved 258/300 nodes 47689413" || { echo "[q92] ☠️ gen 4 (Knobs) BENCH MISMATCH — aborting"; exit 1; }
( while true; do sleep 1800; echo "[q92] $(date +%H:%M) mem: $(free -g | awk '/Mem:/{print "used " $3 "G avail " $7 "G"}')"; done ) &
MEMPID=$!
OURS="option.OwnBook=false st=1 timemargin=250 restart=on"
# Pre-flight: 2 fast games with the engine I/O log — the knob must reach the engine (echo) and fastchess must not drop it.
timeout 120 $FC -engine cmd="$B" name=gen5 option.OwnBook=false st=0.05 -engine cmd="$B" name=gen4 option.OwnBook=false st=0.05 \
  option.Knobs=V2_SHIP_GEN=4 -openings file=$O/openings_uho.epd format=epd order=random -rounds 1 -games 2 -concurrency 1 \
  -log file=/tmp/q92_preflight.engine.log level=info engine=true > /tmp/q92_preflight.log 2>&1
W=$(grep -c "have option" /tmp/q92_preflight.log); E=$(grep -c 'knobs V2_SHIP_GEN=4' /tmp/q92_preflight.engine.log)
echo "[q92] preflight: option warnings $W, gen4 knob echoes $E"
[ "$W" = "0" ] && [ "$E" -ge 1 ] || { echo "[q92] ☠️ KNOB CHANNEL CHECK FAILED — aborting"; kill $MEMPID; exit 1; }
secs=$(( $(date -d "$EXIT_AT" +%s) - $(date +%s) )); [ $secs -lt 0 ] && secs=$(( secs + 86400 ))
echo "[q92] $(date) gen5_vs_gen4 (400 games), up to ${secs}s"
timeout $secs nice -n 5 $FC -engine cmd="$B" name=gen5 $OURS -engine cmd="$B" name=gen4 $OURS option.Knobs=V2_SHIP_GEN=4 \
  -openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -resign movecount=3 score=1000 \
  -draw movenumber=40 movecount=8 score=10 -srand 921 -rounds 200 -games 2 \
  -pgnout file=$O/gen5_vs_gen4.pgn notation=san nodes=true append=false > $O/gen5_vs_gen4.log 2>&1
echo "[q92] $(date) done (rc $?): $(grep -E '^Games:|^Elo:' $O/gen5_vs_gen4.log | tail -2 | tr '\n' ' ') $(grep -c 'not responsive' $O/gen5_vs_gen4.log) unresponsive · option warnings: $(grep -c "have option" $O/gen5_vs_gen4.log)"
kill $MEMPID 2>/dev/null
echo "[q92] $(date) QUEUE 92 DONE"
