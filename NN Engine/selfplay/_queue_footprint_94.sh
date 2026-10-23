#!/bin/bash
# 2026-10-22, queue #94 (owner: "mostly reads and long runs overnight") — waits for queue #93, then:
#  1 build S5 + runtime table sizing to chess_engine_uci_s6; require identical benches (v2 47689586, v1 35310778); install as production
#    + Cython rebuild + fingerprints. Any mismatch => abort, production untouched.
#  2 eval-cache identity: v2 bench at EVAL_LOG2=20 (identical nodes => v2's eval cache has no side-effect coupling).
#  3 TT size speed sweep (fixed-depth bench, pinned core 4, x2): TT_LOG2 20..24 + a "regular engine" set (all caches small).
#  4 equal-time strength, small footprint vs current (same binary, option.Knobs): st=1 400 g, then 60+0.6 200 g. Measurement only.
# Registered: (2) identical; (3) small TT faster per node at equal depth, nodes up slightly; (4) small vs current within ±20 Elo.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
SMALL="TT_LOG2=21,EVAL_LOG2=20,QCACHE_LOG2=20,MGCACHE_LOG2=20"
EXIT_AT=${EXIT_AT:-10:00}
cd "$ND" || exit 1
echo "[q94] $(date) armed"
until grep -q "QUEUE 93 DONE" /mnt/e/chess_data/q93_speed_batch.log 2>/dev/null; do sleep 60; done
echo "[q94] $(date) start"
b() { printf "$1bench\nquit\n" | taskset -c 4 "$2" 2>/dev/null | grep '^bench'; }

# 1 build + identity + install
OUT=chess_engine_uci_s6 bash selfplay/build_uci.sh || { echo "[q94] ☠️ BUILD FAILED"; exit 1; }
N=build_uci/chess_engine_uci_s6
V2=$(b '' $N); V1=$(b 'setoption name EvalArm value 0\n' $N)
echo "[q94] s6 v2 $V2"; echo "[q94] s6 v1 $V1"
echo "$V2" | grep -q "solved 258/300 nodes 47689586" && echo "$V1" | grep -q "solved 250/300 nodes 35310778" \
  || { echo "[q94] ☠️ IDENTITY FAILED — production untouched, aborting"; exit 1; }
cp build_uci/chess_engine_uci build_uci/chess_engine_uci_pre_s6 && cp $N build_uci/chess_engine_uci
echo "[q94] installed s6 as production (previous kept as chess_engine_uci_pre_s6)"
bash "$R" build 2>&1 | grep -iE 'error' | head -3
echo "[q94] cython fp_v1: $(bash "$R" wac fp_v1 2>&1 | grep -E 'NODES' | tr '\n' ' ')"
echo "[q94] cython fp_v2: $(V2_PRESET=shipped bash "$R" wac fp_v2 V2_PRESET=shipped 2>&1 | grep -E 'NODES' | tr '\n' ' ')"
B="$ND/build_uci/chess_engine_uci"

# 2 eval-cache identity
echo "[q94] EVAL_LOG2=20 v2: $(b 'setoption name Knobs value EVAL_LOG2=20\n' $B)   (identical = 47689586)"

# 3 TT size speed sweep
for cfg in TT_LOG2=24 TT_LOG2=23 TT_LOG2=22 TT_LOG2=21 TT_LOG2=20 $SMALL; do
  for r in 1 2; do echo "[q94] sweep $cfg #$r: $(b "setoption name Knobs value $cfg\n" $B)"; done
done

# 4 equal-time strength (Knobs channel pre-flight first)
timeout 120 $FC -engine cmd="$B" name=small option.OwnBook=false option.Knobs=$SMALL st=0.05 -engine cmd="$B" name=cur option.OwnBook=false st=0.05 \
  -openings file=$O/openings_uho.epd format=epd order=random -rounds 1 -games 2 -concurrency 1 \
  -log file=/tmp/q94_pre.engine.log level=info engine=true > /tmp/q94_pre.log 2>&1
W=$(grep -c "have option" /tmp/q94_pre.log); E=$(grep -c "knobs $SMALL" /tmp/q94_pre.engine.log)
echo "[q94] preflight: option warnings $W, small-footprint echoes $E"
[ "$W" = "0" ] && [ "$E" -ge 1 ] || { echo "[q94] ☠️ KNOB CHANNEL CHECK FAILED — no matches"; exit 1; }
COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
match() {   # $1 tag, $2 seed, $3 rounds, $4 time control args
  local secs=$(( $(date -d "$EXIT_AT" +%s) - $(date +%s) )); [ $secs -lt 0 ] && secs=$(( secs + 86400 ))
  [ $secs -lt 900 ] && { echo "[q94] $(date) skip $1: past EXIT_AT"; return; }
  echo "[q94] $(date) $1 ($(( $3 * 2 )) games), up to ${secs}s"
  timeout $secs nice -n 5 $FC -engine cmd="$B" name=small option.OwnBook=false option.Knobs=$SMALL $4 timemargin=250 restart=on \
    -engine cmd="$B" name=cur option.OwnBook=false $4 timemargin=250 restart=on $COMMON -srand $2 -rounds $3 -games 2 \
    -pgnout file=$O/$1.pgn notation=san nodes=true append=false > $O/$1.log 2>&1
  echo "[q94] $(date) $1 done (rc $?): $(grep -E '^Games:|^Elo:' $O/$1.log | tail -2 | tr '\n' ' ') $(grep -c 'not responsive' $O/$1.log) unresponsive · warnings $(grep -c 'have option' $O/$1.log)"
}
match footprint_st1 941 200 "st=1"
match footprint_60p06 942 100 "tc=60+0.6"
echo "[q94] $(date) QUEUE 94 DONE"
