#!/bin/bash
# 2026-10-24, queue #98b (owner: "full confirmation of everything"). Never edit this file while it runs.
#  1 fingerprints: production copy (gen 5 + S1-S5, 10-23) and gen 6 + PGO (speed batches 2-6, the depth-cap crash fix,
#    EVAL_CACHE_FORM 2, gen-6 table sizes, UCI Hash)
#  2 gen 6 + PGO vs production, st=1, 400 g          3 the same at 60+0.6, 200 g
# -recover keeps a match going if an engine dies; every disconnect is counted, and new WSL crash dumps are listed.
# Registered: st=1 +40 ... +70, 60+0.6 +30 ... +60, 0 crashes and 0 disconnects.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
CUR="$ND/build_uci/chess_engine_uci_q96cur"
NEW="$ND/build_uci/chess_engine_uci_g6pgo"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
cd "$ND" || exit 1
START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[q98b] $START start"
nodes() { printf "$2bench\nquit\n" | "$1" 2>/dev/null | grep '^bench' | sed -n 's/.* nodes \([0-9]*\).*/\1/p'; }
c2=$(nodes "$CUR" ''); c1=$(nodes "$CUR" 'setoption name EvalArm value 0\n')
n2=$(nodes "$NEW" ''); n1=$(nodes "$NEW" 'setoption name EvalArm value 0\n'); n5=$(nodes "$NEW" 'setoption name Knobs value V2_SHIP_GEN=5\n')
echo "[q98b] production: v2 $c2 (want 47689586) v1 $c1 (want 35310778)"
echo "[q98b] gen6pgo:    v2 $n2 (want 46196950) v1 $n1 (want 35310778) gen5 $n5 (want 47689586)"
[ "$c2" = 47689586 ] && [ "$c1" = 35310778 ] && [ "$n2" = 46196950 ] && [ "$n1" = 35310778 ] && [ "$n5" = 47689586 ] \
  || { echo "[q98b] ☠️ fingerprints failed — QUEUE 98b ABORTED"; exit 1; }

COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
match() {   # $1 tag, $2 seed, $3 time control args, $4 rounds
  timeout 120 "$FC" -engine cmd="$NEW" name=arm option.OwnBook=false st=0.05 -engine cmd="$CUR" name=cur option.OwnBook=false st=0.05 \
    -openings file=$O/openings_uho.epd format=epd order=random -rounds 1 -games 2 -concurrency 1 \
    -log file=/tmp/q98b_pre.engine.log level=info engine=true > /tmp/q98b_pre.log 2>&1
  local W E; W=$(grep -c "have option" /tmp/q98b_pre.log); E=$(grep -cF "knobs (none)" /tmp/q98b_pre.engine.log)
  echo "[q98b] $1 preflight: warnings $W, 'knobs (none)' echoes $E"
  [ "$W" = "0" ] && [ "$E" -ge 1 ] || { echo "[q98b] ☠️ $1 PREFLIGHT FAILED — skipped"; return; }
  echo "[q98b] $(date) $1 ($(( $4 * 2 )) games)"
  nice -n 5 "$FC" -engine cmd="$NEW" name=arm option.OwnBook=false $3 timemargin=250 restart=on \
    -engine cmd="$CUR" name=cur option.OwnBook=false $3 timemargin=250 restart=on $COMMON -srand $2 -rounds $4 -games 2 \
    -pgnout file=$O/$1.pgn notation=san nodes=true append=false > $O/$1.log 2>&1
  echo "[q98b] $(date) $1 done (rc $?): $(grep -E '^Games:|^Elo:' $O/$1.log | tail -2 | tr '\n' ' ')"
  echo "[q98b]   disconnects/stalls: $(grep -ciE 'disconnect|stall|not responsive' $O/$1.log)"
}
match g6_st1  982 st=1 200
match g6_60p6 983 tc=60+0.6 100
echo "[q98b] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | sed 's/^/[q98b]   /'
echo "[q98b] $(date) QUEUE 98b DONE"
