#!/bin/bash
# 2026-10-26 morning, queue #105: chained after #104 (waits for "QUEUE 104 DONE"). Never edit this file while it runs.
#  1 step 4 all together: TM_SCALE=7,TM_NORM=95,TM_OPT_PLY=1 vs the no-knob binary (= gen 6.1), 60+0.6, 200 g
#    (TM_NORM 95 = the probe's median product of all three factors in the 60+0.6 decision window)
#  2 the movestogo path in real play: gen 6.1 production vs Mediocre 0.5 at 40/60 (40 moves per minute, repeating), 100 g
#  3 Stockfish 1.1 at time odds: gen 6.1 production at 60+0.6 vs SF 1.1 at 20+0.2, 100 g (a tighter reading than 6.5%)
# -recover; disconnects / forfeits (by side) counted; new WSL crash dumps listed.
# Registered: 1 +0 ... +35 (and mean time within 15% of the control); 2 +0 ... +150 with 0 forfeits by gen 6.1;
# 3 -150 ... -350.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
BIN="$ND/build_uci/chess_engine_uci_tm4"
PROD="$ND/build_uci/chess_engine_uci"
MED="$ND/selfplay/mediocre_uci.sh"
SF1="/mnt/e/chess_data/engines/sf1.1/stockfish-1.1"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
cd "$ND" || exit 1
echo "[q105] $(date) waiting for queue 104"
until grep -q 'QUEUE 104 DONE' /mnt/e/chess_data/q104.log 2>/dev/null; do sleep 60; done
START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[q105] $START start"
nodes() { printf "$2bench\nquit\n" | "$1" 2>/dev/null | grep '^bench' | sed -n 's/.* nodes \([0-9]*\).*/\1/p'; }
p2=$(nodes "$PROD" ''); nk=$(nodes "$BIN" 'setoption name Knobs value TM_SCALE=7,TM_NORM=95,TM_OPT_PLY=1\n')
echo "[q105] production v2 $p2, tm4 all knobs $nk (both want 46196950)"
[ "$p2" = 46196950 ] && [ "$nk" = 46196950 ] || { echo "[q105] ☠️ fingerprints failed — QUEUE 105 ABORTED"; exit 1; }

COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
report() {   # $1 tag
  echo "[q105] $(date) $1 done: $(grep -E '^Games:|^Elo:' $O/$1.log | tail -2 | tr '\n' ' ')"
  echo "[q105]   disconnects/stalls: $(grep -ciE 'disconnect|stall|not responsive' $O/$1.log)  time forfeits: $(grep -c 'loses on time' $O/$1.pgn)"
}

# 1: all of step 4 together. Preflight: the knob string reaches the arm, the control runs without knobs.
K=TM_SCALE=7,TM_NORM=95,TM_OPT_PLY=1
timeout 120 "$FC" -engine cmd="$BIN" name=arm option.OwnBook=false option.Knobs=$K st=0.05 -engine cmd="$BIN" name=g61 option.OwnBook=false st=0.05 \
  -openings file=$O/openings_uho.epd format=epd order=random -rounds 1 -games 2 -concurrency 1 \
  -log file=/tmp/q105_pre.engine.log level=info engine=true > /tmp/q105_pre.log 2>&1
W=$(grep -c "have option" /tmp/q105_pre.log); E=$(grep -cF "knobs $K" /tmp/q105_pre.engine.log); N=$(grep -cF "knobs (none)" /tmp/q105_pre.engine.log)
echo "[q105] tm4all preflight: warnings $W, arm echoes $E, control echoes $N"
if [ "$W" = "0" ] && [ "$E" -ge 1 ] && [ "$N" -ge 1 ]; then
  echo "[q105] $(date) tm4all_60p6 (200 games)"
  nice -n 5 "$FC" -engine cmd="$BIN" name=arm option.OwnBook=false option.Knobs=$K tc=60+0.6 timemargin=250 restart=on \
    -engine cmd="$BIN" name=g61 option.OwnBook=false tc=60+0.6 timemargin=250 restart=on $COMMON -srand 1051 -rounds 100 -games 2 \
    -pgnout file=$O/tm4all_60p6.pgn notation=san nodes=true append=false > $O/tm4all_60p6.log 2>&1
  report tm4all_60p6
else echo "[q105] ☠️ tm4all PREFLIGHT FAILED — skipped"; fi

# 2: 40 moves per minute, repeating. Preflight at 40/10 (2 games must finish).
timeout 180 "$FC" -engine cmd="$PROD" name=g61 option.OwnBook=false tc=40/10 -engine cmd="$MED" name=mediocre tc=40/10 \
  -openings file=$O/openings_uho.epd format=epd order=random -rounds 1 -games 2 -concurrency 1 -pgnout file=/tmp/q105_pre2.pgn > /tmp/q105_pre2.log 2>&1
G=$(grep -c '^\[Result ' /tmp/q105_pre2.pgn 2>/dev/null); W=$(grep -c "have option" /tmp/q105_pre2.log)
echo "[q105] 40/60 preflight (at 40/10): games $G, warnings $W, forfeits $(grep -c 'loses on time' /tmp/q105_pre2.pgn 2>/dev/null)"
if [ "$G" = "2" ] && [ "$W" = "0" ]; then
  echo "[q105] $(date) g61_med_40m60 (100 games)"
  nice -n 5 "$FC" -engine cmd="$PROD" name=g61 option.OwnBook=false tc=40/60 timemargin=250 restart=on \
    -engine cmd="$MED" name=mediocre tc=40/60 timemargin=250 restart=on $COMMON -srand 1052 -rounds 50 -games 2 \
    -pgnout file=$O/g61_med_40m60.pgn notation=san nodes=true append=false > $O/g61_med_40m60.log 2>&1
  report g61_med_40m60
else echo "[q105] ☠️ 40/60 PREFLIGHT FAILED — skipped"; fi

# 3: Stockfish 1.1 at time odds (it gets a third of our clock).
SF1OPT="option.Threads=1 option.Ponder=false option.OwnBook=false option.Hash=128"
echo "[q105] $(date) g61_sf11_odds (100 games: g61 60+0.6, sf1.1 20+0.2)"
nice -n 5 "$FC" -engine cmd="$PROD" name=g61 option.OwnBook=false tc=60+0.6 timemargin=250 restart=on \
  -engine cmd="$SF1" name=sf1.1 $SF1OPT tc=20+0.2 timemargin=250 restart=on $COMMON -srand 1053 -rounds 50 -games 2 \
  -pgnout file=$O/g61_sf11_odds.pgn notation=san nodes=true append=false > $O/g61_sf11_odds.log 2>&1
report g61_sf11_odds

echo "[q105] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | sed 's/^/[q105]   /'
echo "[q105] $(date) QUEUE 105 DONE"
