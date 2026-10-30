#!/bin/bash
# 2026-10-30, queue #119: the endgame material-class caps (built 10-09, never played) on gen 6.4, at SF15.1's values
# (scale factors are unit-free 1/64ths in both engines): OCBX −42 + 3/piece, ROOKE −28, QNOQ −27 + 3/minor, LONEMINOR −64.
# Their only verdict so far was a static fit on the material-scaled cp label, which drove them to ~0 (round 3, 10-13); on the
# SF18 WDL label the endgame gap is SF11's scaling (OPTIMIZATION_LOG 10-30, "WHERE THE EVAL GAP IS").
#  0 symmetry gate (caps on) · bench liveness (caps nodes must differ from the base) · static WDL read by stage, caps on
#  1 caps_uho  caps vs gen 6.4, 10+0.1, 1,000 g, UHO, fresh seed 1191
# Registered: −5 … +15 (QNOQ fires ~11% and OCBX ~8% of depth rows, ROOKE / LONEMINOR ~0.2%; endgame accuracy has paid less
# Elo than middlegame). Static: eg-stage MSE down, egl not up by more than it gains in eg.
# Never edit this file while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
BIN="$ND/build_uci/chess_engine_uci_g64pgo"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
B=/mnt/e/chess_data/bench1030
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
BASE=THREAT_ORDER_W=100
CAPS=$BASE,POT_V2_WIN_OCBX=-42,POT_V2_WIN_OCBX_PC=3,POT_V2_WIN_ROOKE=-28,POT_V2_WIN_QNOQ=-27,POT_V2_WIN_QNOQ_MINOR=3,POT_V2_WIN_LONEMINOR=-64
CAPS_ENV="POT_V2_WIN_OCBX=-42 POT_V2_WIN_OCBX_PC=3 POT_V2_WIN_ROOKE=-28 POT_V2_WIN_QNOQ=-27 POT_V2_WIN_QNOQ_MINOR=3 POT_V2_WIN_LONEMINOR=-64"
cd "$ND" || exit 1
START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[q119] $START start"
while pgrep -f "fastchess|vs_sf.py" >/dev/null; do sleep 60; done

SYM=$(env V2_PRESET=shipped $CAPS_ENV bash "$R" pyrun diagnostics/_eval_symmetry.py N=4000 2>&1 | grep -E "violations")
echo "[q119] caps symmetry: $(echo "$SYM" | tr '\n' ' ')"
[ "$(echo "$SYM" | grep -c 'violations 0 ')" = "2" ] || { echo "[q119] ☠️ symmetry FAILED — QUEUE 119 ABORTED"; exit 1; }

nodes() { printf "$2bench\nquit\n" | "$1" 2>/dev/null | grep '^bench' | sed -n 's/.* nodes \([0-9]*\).*/\1/p'; }
nb=$(nodes "$BIN" "setoption name Knobs value $BASE\n"); nc=$(nodes "$BIN" "setoption name Knobs value $CAPS\n")
echo "[q119] g64pgo: base $nb (want 36991901) caps $nc (must differ)"
[ "$nb" = 36991901 ] && [ "$nc" != "$nb" ] && [ -n "$nc" ] || { echo "[q119] ☠️ fingerprints failed — QUEUE 119 ABORTED"; exit 1; }

env OMP_NUM_THREADS=1 V2_PRESET=shipped $CAPS_ENV bash "$R" pyrun diagnostics/_triangulate_sf11.py MODE=egshare \
  WDL=/mnt/e/chess_data/wdlcal STAGES=1 SCALE_US=2.76 > $B/stages_caps.txt 2>&1
echo "[q119] static by stage, caps (base: eg 290.8 · egl 275.9 · mgl 390.4 · mg 576.7):"
grep -E '^  (eg|egl|mgl|mg) ' $B/stages_caps.txt | awk '{print "[q119]   " $1 " MSE " $5}'

COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
preflight() {
  timeout 180 "$FC" -engine cmd="$BIN" name=arm option.OwnBook=false option.Knobs=$CAPS st=0.05 \
    -engine cmd="$BIN" name=ctl option.OwnBook=false option.Knobs=$BASE st=0.05 \
    -openings file=$O/openings_uho.epd format=epd order=random -rounds 1 -games 2 -concurrency 1 \
    -log file=/tmp/q119_pre.engine.log level=info engine=true > /tmp/q119_pre.log 2>&1
  local W K N; W=$(grep -c "have option" /tmp/q119_pre.log); K=$(grep -cF "knobs $CAPS" /tmp/q119_pre.engine.log)
  N=$(grep -oE "knobs [A-Z0-9_=,-]+" /tmp/q119_pre.engine.log | grep -cxF "knobs $BASE")
  echo "[q119] preflight: warnings $W, caps echoes $K, base echoes $N"
  [ "$W" = "0" ] && [ "$K" -ge 1 ] && [ "$N" -ge 1 ] || { echo "[q119] ☠️ PREFLIGHT FAILED — QUEUE 119 ABORTED"; exit 1; }
}
preflight
echo "[q119] $(date) caps_uho (1000 games, 10+0.1)"
nice -n 5 "$FC" -engine cmd="$BIN" name=caps option.OwnBook=false option.Knobs=$CAPS tc=10+0.1 timemargin=250 \
  -engine cmd="$BIN" name=g64 option.OwnBook=false option.Knobs=$BASE tc=10+0.1 timemargin=250 $COMMON -srand 1191 -rounds 500 -games 2 \
  -pgnout file=$O/caps_uho.pgn notation=san nodes=true append=false > $O/caps_uho.log 2>&1
echo "[q119] $(date) caps_uho done: $(grep -E '^Games:|^Elo:' $O/caps_uho.log | tail -2 | tr '\n' ' ')"
echo "[q119]   $(grep -E '^Ptnml' $O/caps_uho.log | tail -1)  disconnects/stalls: $(grep -ciE 'disconnect|stall|not responsive' $O/caps_uho.log)  time forfeits: $(grep -c 'loses on time' $O/caps_uho.pgn)"
echo "[q119] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | sed 's/^/[q119]   /'
echo "[q119] $(date) QUEUE 119 DONE"
