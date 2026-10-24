#!/bin/bash
# 2026-10-23, queue #97 (owner: option B). Starts when queue #96 logs QUEUE 96 DONE. Never edit this file while it runs.
#  1 fingerprints of the current production copy and of batch 6 + PGO (both must reproduce v2 and v1)
#  2 build identity at fixed nodes: current -> batch 6 + PGO (every set 0 divergences, 0 crashes)
#  3 batch 6 + PGO vs current production, st=1, 400 g (everything from 10-23 in one row: speed batch 2, batch 3, the
#    isNearGameEnd fix, F6 king-safety mask reuse, F1 four-table prefetch, PGO)
# Registered: identity 0 divergences · real-play nps +15 ... +25% · Elo +30 ... +60.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
CUR="$ND/build_uci/chess_engine_uci_q96cur"
NEW="$ND/build_uci/chess_engine_uci_sb6pgo"
RUNNER="$ND/selfplay/overnight_runner.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
Q96LOG=/mnt/e/chess_data/q96_night.log
EXIT_AT=${EXIT_AT:-21:00}
V2_NODES=47689586
V1_NODES=35310778
cd "$ND" || exit 1
echo "[q97] $(date) waiting for queue 96"
until grep -qE 'QUEUE 96 DONE|QUEUE ABORTED' "$Q96LOG"; do sleep 60; done
echo "[q97] $(date) start ($(grep -E 'QUEUE 96 DONE|QUEUE ABORTED' "$Q96LOG" | tail -1))"

benchnodes() {
  printf "$2bench\nquit\n" | "$1" 2>/dev/null | grep '^bench' | sed -n 's/.* nodes \([0-9]*\).*/\1/p'
}
fingerprints_ok() {
  [ -x "$1" ] || { echo "[q97] ☠️ $2: binary missing"; return 1; }
  local v2 v1
  v2=$(benchnodes "$1" '')
  v1=$(benchnodes "$1" 'setoption name EvalArm value 0\n')
  echo "[q97] $2 bench: v2 nodes '$v2' (want $V2_NODES), v1 nodes '$v1' (want $V1_NODES)"
  [ "$v2" = "$V2_NODES" ] && [ "$v1" = "$V1_NODES" ]
}
secs_left() {
  local s=$(( $(date -d "$EXIT_AT" +%s) - $(date +%s) )); [ $s -lt 0 ] && s=$(( s + 86400 )); echo $s
}

fingerprints_ok "$CUR" cur || { echo "[q97] ☠️ QUEUE 97 ABORTED"; exit 1; }
[ -f "$ND/build_uci/sb6pgo.ready" ] && fingerprints_ok "$NEW" sb6pgo || { echo "[q97] ☠️ sb6pgo not ready or failed — QUEUE 97 ABORTED"; exit 1; }

log="/mnt/e/chess_data/q97_sb6pgo_identity.log"
OMP_NUM_THREADS=1 bash "$RUNNER" pyrun diagnostics/_fix_firecount.py BIN="$CUR" BIN_ON="$NEW" ON= N=40 NODES=100000 PLIES=200 \
  JOBS=3 SEED=97 > "$log" 2>&1
sets=$(grep -c '^  diverged:' "$log"); zero=$(grep -c '^  diverged: 0 /' "$log"); crash=$(grep '^  diverged:' "$log" | grep -vc 'engine crashes: 0$')
echo "[q97] $(date) identity: sets $sets, sets with 0 divergences $zero, sets with crashes $crash"
grep '^  diverged:' "$log" | sed 's/^/[q97]   /'
[ "$sets" = "3" ] && [ "$zero" = "3" ] && [ "$crash" = "0" ] || { echo "[q97] ☠️ identity failed — QUEUE 97 STOPPED"; exit 1; }

secs=$(secs_left)
[ $secs -lt 900 ] && { echo "[q97] $(date) skip match: past EXIT_AT"; exit 0; }
COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
timeout 120 "$FC" -engine cmd="$NEW" name=arm option.OwnBook=false st=0.05 -engine cmd="$CUR" name=cur option.OwnBook=false st=0.05 \
  -openings file=$O/openings_uho.epd format=epd order=random -rounds 1 -games 2 -concurrency 1 \
  -log file=/tmp/q97_pre.engine.log level=info engine=true > /tmp/q97_pre.log 2>&1
W=$(grep -c "have option" /tmp/q97_pre.log); E=$(grep -cF "knobs (none)" /tmp/q97_pre.engine.log)
echo "[q97] preflight: warnings $W, 'knobs (none)' echoes $E"
[ "$W" = "0" ] && [ "$E" -ge 1 ] || { echo "[q97] ☠️ PREFLIGHT FAILED — QUEUE 97 STOPPED"; exit 1; }
echo "[q97] $(date) speed_sb6pgo (400 games), up to ${secs}s"
timeout $secs nice -n 5 "$FC" -engine cmd="$NEW" name=arm option.OwnBook=false st=1 timemargin=250 restart=on \
  -engine cmd="$CUR" name=cur option.OwnBook=false st=1 timemargin=250 restart=on $COMMON -srand 971 -rounds 200 -games 2 \
  -pgnout file=$O/speed_sb6pgo.pgn notation=san nodes=true append=false > $O/speed_sb6pgo.log 2>&1
echo "[q97] $(date) speed_sb6pgo done (rc $?): $(grep -E '^Games:|^Elo:' $O/speed_sb6pgo.log | tail -2 | tr '\n' ' ') $(grep -c 'not responsive' $O/speed_sb6pgo.log) unresponsive"
echo "[q97] $(date) QUEUE 97 DONE"
