#!/bin/bash
# 2026-10-26, queue #106: time manager step 4b, the iteration-start rule. Never edit this file while it runs.
# Why: in #104/#105 both sides spent their whole clock (60 s + 40 increments) by move 40, because the soft stop only acts
# after the iteration that crosses the optimum (cumulative growth ~1.93 per iteration ⇒ spend ≈ 1.3–1.5× the optimum). The
# references guard against it (SF: no deeper iteration past 50% of the target; Weiss "don't start an iteration you can't use",
# +57; Velvet: start only if 1.75× the last iteration fits). TM_START=175 = Velvet's predictor (ours measured ~1.8×).
#  D TM_START=175,TM_SINGLE=1 vs the no-knob binary (= gen 6.1)
#  E D + TM_SCALE=7,TM_NORM=95,TM_OPT_PLY=1 (everything) vs the no-knob binary
# Plain build_uci/chess_engine_uci_tm4b; 60+0.6, 200 g each; -recover; disconnects / forfeits counted; new WSL dumps listed.
# Registered: D -10 ... +30, E +5 ... +45; D/E spend less than the control in moves 1-20 and more after move 40; 0 forfeits.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
BIN="$ND/build_uci/chess_engine_uci_tm4b"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
cd "$ND" || exit 1
START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[q106] $START start"
nodes() { printf "$2bench\nquit\n" | "$1" 2>/dev/null | grep '^bench' | sed -n 's/.* nodes \([0-9]*\).*/\1/p'; }
n2=$(nodes "$BIN" ''); n1=$(nodes "$BIN" 'setoption name EvalArm value 0\n'); n5=$(nodes "$BIN" 'setoption name Knobs value V2_SHIP_GEN=5\n')
nk=$(nodes "$BIN" 'setoption name Knobs value TM_START=175,TM_SINGLE=1,TM_SCALE=7,TM_NORM=95,TM_OPT_PLY=1\n')
echo "[q106] tm4b: v2 $n2 (want 46196950) v1 $n1 (want 35310778) gen5 $n5 (want 47689586) all TM knobs $nk (want 46196950)"
[ "$n2" = 46196950 ] && [ "$n1" = 35310778 ] && [ "$n5" = 47689586 ] && [ "$nk" = 46196950 ] \
  || { echo "[q106] ☠️ fingerprints failed — QUEUE 106 ABORTED"; exit 1; }

COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
arm() {   # $1 tag, $2 knobs, $3 seed
  timeout 120 "$FC" -engine cmd="$BIN" name=arm option.OwnBook=false option.Knobs=$2 st=0.05 \
    -engine cmd="$BIN" name=g61 option.OwnBook=false st=0.05 \
    -openings file=$O/openings_uho.epd format=epd order=random -rounds 1 -games 2 -concurrency 1 \
    -log file=/tmp/q106_pre.engine.log level=info engine=true > /tmp/q106_pre.log 2>&1
  local W K N; W=$(grep -c "have option" /tmp/q106_pre.log); K=$(grep -cF "knobs $2" /tmp/q106_pre.engine.log); N=$(grep -cF "knobs (none)" /tmp/q106_pre.engine.log)
  echo "[q106] $1 preflight: warnings $W, 'knobs $2' echoes $K, 'knobs (none)' echoes $N"
  [ "$W" = "0" ] && [ "$K" -ge 1 ] && [ "$N" -ge 1 ] || { echo "[q106] ☠️ $1 PREFLIGHT FAILED — skipped"; return; }
  echo "[q106] $(date) $1 (200 games)"
  nice -n 5 "$FC" -engine cmd="$BIN" name=arm option.OwnBook=false option.Knobs=$2 tc=60+0.6 timemargin=250 restart=on \
    -engine cmd="$BIN" name=g61 option.OwnBook=false tc=60+0.6 timemargin=250 restart=on $COMMON -srand $3 -rounds 100 -games 2 \
    -pgnout file=$O/$1.pgn notation=san nodes=true append=false > $O/$1.log 2>&1
  echo "[q106] $(date) $1 done: $(grep -E '^Games:|^Elo:' $O/$1.log | tail -2 | tr '\n' ' ')"
  echo "[q106]   disconnects/stalls: $(grep -ciE 'disconnect|stall|not responsive' $O/$1.log)  time forfeits: $(grep -c 'loses on time' $O/$1.pgn)"
}
arm tm4d_start TM_START=175,TM_SINGLE=1 1061
arm tm4e_all   TM_START=175,TM_SINGLE=1,TM_SCALE=7,TM_NORM=95,TM_OPT_PLY=1 1062
echo "[q106] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | sed 's/^/[q106]   /'
echo "[q106] $(date) QUEUE 106 DONE"
