#!/bin/bash
# 2026-10-25 evening, queue #103. Never edit this file while it runs.
#  1 fingerprints: production gen 6.1 (PGO) and the plain gen 6.1 A/B binary
#  2 gen 6.1 production vs Stockfish 1.1 (2008; native build of the owner's copy, -fno-aggressive-loop-optimizations), 60+0.6,
#    100 g — a first reading of a new external opponent
#  3 (dropped before launch: step 3 failed its judge)
# -recover; disconnects and time forfeits counted; new WSL crash dumps listed.
# Registered: SF 1.1 -100 ... -350 (no prior; wide); step 3 st=1 -10 ... +25 (ship needs > -1 sigma and a positive judge).
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
PROD="$ND/build_uci/chess_engine_uci"
AB="$ND/build_uci/chess_engine_uci_g61"
SF1="/mnt/e/chess_data/engines/sf1.1/stockfish-1.1"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
cd "$ND" || exit 1
START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[q103] $START start"
nodes() { printf "$2bench\nquit\n" | "$1" 2>/dev/null | grep '^bench' | sed -n 's/.* nodes \([0-9]*\).*/\1/p'; }
p2=$(nodes "$PROD" ''); p1=$(nodes "$PROD" 'setoption name EvalArm value 0\n'); p5=$(nodes "$PROD" 'setoption name Knobs value V2_SHIP_GEN=5\n')
a2=$(nodes "$AB" ''); ap=$(nodes "$AB" 'setoption name Knobs value TM_PARTIAL=1\n')
echo "[q103] production: v2 $p2 (want 46196950) v1 $p1 (want 35310778) gen5 $p5 (want 47689586)"
echo "[q103] plain g61:  v2 $a2 (want 46196950) TM_PARTIAL=1 $ap (want 46196950)"
[ "$p2" = 46196950 ] && [ "$p1" = 35310778 ] && [ "$p5" = 47689586 ] && [ "$a2" = 46196950 ] && [ "$ap" = 46196950 ] \
  || { echo "[q103] ☠️ fingerprints failed — QUEUE 103 ABORTED"; exit 1; }

COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
SF1OPT="option.Threads=1 option.Ponder=false option.OwnBook=false option.Hash=128"
report() {   # $1 tag
  echo "[q103] $(date) $1 done: $(grep -E '^Games:|^Elo:' $O/$1.log | tail -2 | tr '\n' ' ')"
  echo "[q103]   disconnects/stalls: $(grep -ciE 'disconnect|stall|not responsive' $O/$1.log)  time forfeits: $(grep -c 'loses on time' $O/$1.pgn)"
}

# 2: Stockfish 1.1. Preflight: 2 short games, 0 "doesn't have option" warnings, both engines produce moves.
timeout 120 "$FC" -engine cmd="$PROD" name=g61 option.OwnBook=false st=0.1 -engine cmd="$SF1" name=sf1.1 $SF1OPT st=0.1 \
  -openings file=$O/openings_uho.epd format=epd order=random -rounds 1 -games 2 -concurrency 1 -pgnout file=/tmp/q103_pre.pgn > /tmp/q103_pre.log 2>&1
W=$(grep -c "have option" /tmp/q103_pre.log); G=$(grep -c '^\[Result ' /tmp/q103_pre.pgn 2>/dev/null)
echo "[q103] sf1.1 preflight: warnings $W, games $G"
if [ "$W" = "0" ] && [ "$G" = "2" ]; then
  echo "[q103] $(date) g61_sf11_60p6 (100 games)"
  nice -n 5 "$FC" -engine cmd="$PROD" name=g61 option.OwnBook=false tc=60+0.6 timemargin=250 restart=on \
    -engine cmd="$SF1" name=sf1.1 $SF1OPT tc=60+0.6 timemargin=250 restart=on $COMMON -srand 1031 -rounds 50 -games 2 \
    -pgnout file=$O/g61_sf11_60p6.pgn notation=san nodes=true append=false > $O/g61_sf11_60p6.log 2>&1
  report g61_sf11_60p6
else echo "[q103] ☠️ sf1.1 PREFLIGHT FAILED — skipped (see /tmp/q103_pre.log)"; fi

# 3 (dropped before launch): step 3 failed its judge (-0.61 pp, adopted better in 42% of non-tied cases), so its no-harm
# match was not run.

echo "[q103] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | sed 's/^/[q103]   /'
echo "[q103] $(date) QUEUE 103 DONE"
