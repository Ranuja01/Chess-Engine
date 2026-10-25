#!/bin/bash
# 2026-10-25, queue #102: time manager step 1 (TM_MODE=1: optimum / maximum allocation, soft stop between iterations once the
# total elapsed time passes the optimum; dev_notes/TIME-MANAGER-RESEARCH-2026-10-25.md). Never edit this file while it runs.
#  1 fingerprints of the plain step-1 binary (TM_MODE=1 must leave the bench identical)
#  2 self-play 60+0.6, TM_MODE=1 vs the same binary with no knobs (= gen 6's clock rule), 200 g
#  3 TM_MODE=1 vs Mediocre 0.5 at 60+0.6, 200 g (baseline: gen 6 +7.4, 95% -39 ... +54, 200 g pooled)
# -recover keeps a match going if an engine dies; disconnects and time forfeits are counted, new WSL crash dumps listed.
# Registered: self-play +25 ... +70; Mediocre +35 ... +75; TM_MODE=1 time per move in moves 1-40 2.0-3.0 s; 0 forfeits.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
BIN="$ND/build_uci/chess_engine_uci_tm1"
MED="$ND/selfplay/mediocre_uci.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
cd "$ND" || exit 1
START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[q102] $START start"
nodes() { printf "$2bench\nquit\n" | "$1" 2>/dev/null | grep '^bench' | sed -n 's/.* nodes \([0-9]*\).*/\1/p'; }
n2=$(nodes "$BIN" ''); n1=$(nodes "$BIN" 'setoption name EvalArm value 0\n'); n5=$(nodes "$BIN" 'setoption name Knobs value V2_SHIP_GEN=5\n')
nt=$(nodes "$BIN" 'setoption name Knobs value TM_MODE=1\n')
echo "[q102] tm1: v2 $n2 (want 46196950) v1 $n1 (want 35310778) gen5 $n5 (want 47689586) TM_MODE=1 $nt (want 46196950)"
[ "$n2" = 46196950 ] && [ "$n1" = 35310778 ] && [ "$n5" = 47689586 ] && [ "$nt" = 46196950 ] \
  || { echo "[q102] ☠️ fingerprints failed — QUEUE 102 ABORTED"; exit 1; }

COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
report() {   # $1 tag
  echo "[q102] $(date) $1 done: $(grep -E '^Games:|^Elo:' $O/$1.log | tail -2 | tr '\n' ' ')"
  echo "[q102]   disconnects/stalls: $(grep -ciE 'disconnect|stall|not responsive' $O/$1.log)  time forfeits: $(grep -ciE 'loses on time|timeout' $O/$1.pgn)"
}
# The arm's knob must reach the engine: 0 "doesn't have option" warnings, the arm echoes "knobs TM_MODE=1", the control "knobs (none)".
preflight() {   # $1 tag, $2 second engine command, $3 second engine's extra options
  timeout 120 "$FC" -engine cmd="$BIN" name=tm1 option.OwnBook=false option.Knobs=TM_MODE=1 st=0.05 \
    -engine cmd="$2" name=other $3 st=0.05 \
    -openings file=$O/openings_uho.epd format=epd order=random -rounds 1 -games 2 -concurrency 1 \
    -log file=/tmp/q102_pre.engine.log level=info engine=true > /tmp/q102_pre.log 2>&1
  local W K; W=$(grep -c "have option" /tmp/q102_pre.log); K=$(grep -cF "knobs TM_MODE=1" /tmp/q102_pre.engine.log)
  echo "[q102] $1 preflight: warnings $W, 'knobs TM_MODE=1' echoes $K"
  [ "$W" = "0" ] && [ "$K" -ge 1 ]
}

if preflight tm1_60p6 "$BIN" "option.OwnBook=false"; then
  N=$(grep -cF "knobs (none)" /tmp/q102_pre.engine.log); echo "[q102] control 'knobs (none)' echoes $N"
  echo "[q102] $(date) tm1_60p6 (200 games)"
  nice -n 5 "$FC" -engine cmd="$BIN" name=tm1 option.OwnBook=false option.Knobs=TM_MODE=1 tc=60+0.6 timemargin=250 restart=on \
    -engine cmd="$BIN" name=gen6 option.OwnBook=false tc=60+0.6 timemargin=250 restart=on $COMMON -srand 1021 -rounds 100 -games 2 \
    -pgnout file=$O/tm1_60p6.pgn notation=san nodes=true append=false > $O/tm1_60p6.log 2>&1
  report tm1_60p6
else echo "[q102] ☠️ tm1_60p6 PREFLIGHT FAILED — skipped"; fi

if preflight tm1_med "$MED" ""; then
  echo "[q102] $(date) tm1_med_60p6 (200 games)"
  nice -n 5 "$FC" -engine cmd="$BIN" name=tm1 option.OwnBook=false option.Knobs=TM_MODE=1 tc=60+0.6 timemargin=250 restart=on \
    -engine cmd="$MED" name=mediocre tc=60+0.6 timemargin=250 restart=on $COMMON -srand 1022 -rounds 100 -games 2 \
    -pgnout file=$O/tm1_med_60p6.pgn notation=san nodes=true append=false > $O/tm1_med_60p6.log 2>&1
  report tm1_med_60p6
else echo "[q102] ☠️ tm1_med PREFLIGHT FAILED — skipped"; fi

echo "[q102] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | sed 's/^/[q102]   /'
echo "[q102] $(date) QUEUE 102 DONE"
