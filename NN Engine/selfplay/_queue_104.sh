#!/bin/bash
# 2026-10-25 night, queue #104: time manager step 4, one factor family per arm, each vs the same binary with no knobs (= gen 6.1).
# Never edit this file while it runs.
#  A TM_SCALE=1 (best-move node share), TM_NORM=55   B TM_SCALE=6 (stability x score drop), TM_NORM=165
#  C TM_OPT_PLY=1 (the optimum's share grows with the ply, SF 15.1 curve)
# TM_NORM keeps the median factor product at ~1 in the 60+0.6 decision window (diagnostics/_tm_factor_probe.py, 600 positions:
# node share median 0.40 -> f_nodes 1.81; stab*drop median 0.606), so A and B redistribute time rather than add it.
# Plain build_uci/chess_engine_uci_tm4. 60+0.6, 200 g each; -recover; disconnects / forfeits counted; new WSL dumps listed.
# Registered: A +0 ... +25, B -10 ... +15, C -10 ... +20; each arm's mean time per move within 15% of the control's (A, B).
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
BIN="$ND/build_uci/chess_engine_uci_tm4"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
cd "$ND" || exit 1
START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[q104] $START start"
nodes() { printf "$2bench\nquit\n" | "$1" 2>/dev/null | grep '^bench' | sed -n 's/.* nodes \([0-9]*\).*/\1/p'; }
n2=$(nodes "$BIN" ''); n1=$(nodes "$BIN" 'setoption name EvalArm value 0\n'); n5=$(nodes "$BIN" 'setoption name Knobs value V2_SHIP_GEN=5\n')
nk=$(nodes "$BIN" 'setoption name Knobs value TM_SCALE=7,TM_NORM=150,TM_OPT_PLY=1\n')
echo "[q104] tm4: v2 $n2 (want 46196950) v1 $n1 (want 35310778) gen5 $n5 (want 47689586) all TM knobs $nk (want 46196950)"
[ "$n2" = 46196950 ] && [ "$n1" = 35310778 ] && [ "$n5" = 47689586 ] && [ "$nk" = 46196950 ] \
  || { echo "[q104] ☠️ fingerprints failed — QUEUE 104 ABORTED"; exit 1; }

COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
arm() {   # $1 tag, $2 knobs, $3 seed
  timeout 120 "$FC" -engine cmd="$BIN" name=arm option.OwnBook=false option.Knobs=$2 st=0.05 \
    -engine cmd="$BIN" name=g61 option.OwnBook=false st=0.05 \
    -openings file=$O/openings_uho.epd format=epd order=random -rounds 1 -games 2 -concurrency 1 \
    -log file=/tmp/q104_pre.engine.log level=info engine=true > /tmp/q104_pre.log 2>&1
  local W K N; W=$(grep -c "have option" /tmp/q104_pre.log); K=$(grep -cF "knobs $2" /tmp/q104_pre.engine.log); N=$(grep -cF "knobs (none)" /tmp/q104_pre.engine.log)
  echo "[q104] $1 preflight: warnings $W, 'knobs $2' echoes $K, 'knobs (none)' echoes $N"
  [ "$W" = "0" ] && [ "$K" -ge 1 ] && [ "$N" -ge 1 ] || { echo "[q104] ☠️ $1 PREFLIGHT FAILED — skipped"; return; }
  echo "[q104] $(date) $1 (200 games)"
  nice -n 5 "$FC" -engine cmd="$BIN" name=arm option.OwnBook=false option.Knobs=$2 tc=60+0.6 timemargin=250 restart=on \
    -engine cmd="$BIN" name=g61 option.OwnBook=false tc=60+0.6 timemargin=250 restart=on $COMMON -srand $3 -rounds 100 -games 2 \
    -pgnout file=$O/$1.pgn notation=san nodes=true append=false > $O/$1.log 2>&1
  echo "[q104] $(date) $1 done: $(grep -E '^Games:|^Elo:' $O/$1.log | tail -2 | tr '\n' ' ')"
  echo "[q104]   disconnects/stalls: $(grep -ciE 'disconnect|stall|not responsive' $O/$1.log)  time forfeits: $(grep -c 'loses on time' $O/$1.pgn)"
}
arm tm4a_nodes TM_SCALE=1,TM_NORM=55 1041
arm tm4b_stab  TM_SCALE=6,TM_NORM=165 1042
arm tm4c_ply   TM_OPT_PLY=1 1043
echo "[q104] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | sed 's/^/[q104]   /'
echo "[q104] $(date) QUEUE 104 DONE"
