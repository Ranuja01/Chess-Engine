#!/bin/bash
# 2026-10-29, queue #116: ship checks for threats in eval at mg 50 / eg 100 on top of gen 6.3. Chained after #115.
# Never edit this file while it runs.
# #114 vs gen 6.3 (THREAT_ORDER_W=100 both sides): 10+0.1 mg 50 +38.4 +- 19.7 (seed 1141), mg 0 +31.0 +- 19.1 (seed 1142);
# 60+0.6 mg 50 +61.4 +- 42.8 (200 g, seed 1143). Both arms were above their registered bands.
# Binary build_uci/chess_engine_uci_thr2 (as #114); the SF18 leg runs on the Cython .so, rebuilt 10-28 with the gen 6.3 default.
#  1 thr_m50_10b  fresh-seed 10+0.1 re-run vs gen 6.3, 1,000 g, seed 1161
#  2 SF18 gauntlet @1100 nodes, ours 250k nodes, 500 g, seed 1122 = the openings of #112's gen 6.3 run (53.8%)
#  3 med_m50      vs Mediocre 0.5, 60+0.6, 200 g, seed 1071 = the openings of #112's gen 6.3 run (+113.3 +- 45.9)
# Registered: 1 +10 ... +35 (one-seed results above their band shrink) and > 0 by 2 sigma pooled with #114;
# 2 arm - 53.8% > -1 sigma (about -3 points), expected -2 ... +4 points; 3 not below gen 6.3 by more than 1 sigma paired,
# expected +100 ... +170.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
BIN="$ND/build_uci/chess_engine_uci_thr2"
MED="$ND/selfplay/mediocre_uci.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
BASE=TM_START=175,TM_SINGLE=1,TM_SCALE=7,TM_NORM=95,TM_OPT_PLY=1,THREAT_ORDER_W=100
M50=THREAT_V2_PCT=100,THREAT_V2_HANGING=1,THREAT_V2_KING=1,THREAT_V2_PAWN_TARGETS=1,THREAT_V2_MG_PCT=50,THREAT_V2_EG_PCT=100
cd "$ND" || exit 1
echo "[q116] $(date) waiting for queue 115"
until grep -q 'QUEUE 115 DONE' /mnt/e/chess_data/q115.log 2>/dev/null; do sleep 60; done
START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[q116] $START start"
nodes() { printf "$2bench\nquit\n" | "$1" 2>/dev/null | grep '^bench' | sed -n 's/.* nodes \([0-9]*\).*/\1/p'; }
nb=$(nodes "$BIN" "setoption name Knobs value $BASE\n"); nm=$(nodes "$BIN" "setoption name Knobs value $BASE,$M50\n")
echo "[q116] thr2: base $nb (want 42263170) m50 $nm (want 43040865)"
[ "$nb" = 42263170 ] && [ "$nm" = 43040865 ] || { echo "[q116] ☠️ fingerprints failed — QUEUE 116 ABORTED"; exit 1; }

COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
report() {   # $1 tag
  echo "[q116] $(date) $1 done: $(grep -E '^Games:|^Elo:' $O/$1.log | tail -2 | tr '\n' ' ')"
  echo "[q116]   $(grep -E '^Ptnml' $O/$1.log | tail -1)  disconnects/stalls: $(grep -ciE 'disconnect|stall|not responsive' $O/$1.log)  time forfeits: $(grep -c 'loses on time' $O/$1.pgn)"
}
# $1 tag, $2 arm knobs, $3 control cmd, $4 control knobs ("" = none)
preflight() {
  local copt=""; [ -n "$4" ] && copt="option.Knobs=$4"
  rm -f /tmp/q116_pre.engine.log
  timeout 180 "$FC" -engine cmd="$BIN" name=arm option.OwnBook=false option.Knobs=$2 st=0.05 \
    -engine cmd="$3" name=ctl $copt st=0.05 \
    -openings file=$O/openings_uho.epd format=epd order=random -rounds 1 -games 2 -concurrency 1 \
    -log file=/tmp/q116_pre.engine.log level=info engine=true > /tmp/q116_pre.log 2>&1
  local W K N; W=$(grep -c "have option" /tmp/q116_pre.log)
  K=$(grep -cxF "knobs $2" <(grep -oE "knobs [A-Z0-9_=,]+" /tmp/q116_pre.engine.log))
  if [ -n "$4" ]; then N=$(grep -cxF "knobs $4" <(grep -oE "knobs [A-Z0-9_=,]+" /tmp/q116_pre.engine.log)); else N=1; fi
  echo "[q116] $1 preflight: warnings $W, arm echoes $K, control echoes $N"
  [ "$W" = "0" ] && [ "$K" -ge 1 ] && [ "$N" -ge 1 ] || { echo "[q116] ☠️ $1 PREFLIGHT FAILED — skipped"; return 1; }
}

# 1: fresh-seed self-play at 10+0.1
if preflight thr_m50_10b "$BASE,$M50" "$BIN" "$BASE"; then
  echo "[q116] $(date) thr_m50_10b (1000 games, 10+0.1)"
  nice -n 5 "$FC" -engine cmd="$BIN" name=arm option.OwnBook=false option.Knobs=$BASE,$M50 tc=10+0.1 timemargin=250 restart=on \
    -engine cmd="$BIN" name=ctl option.OwnBook=false option.Knobs=$BASE tc=10+0.1 timemargin=250 restart=on $COMMON -srand 1161 -rounds 500 -games 2 \
    -pgnout file=$O/thr_m50_10b.pgn notation=san nodes=true append=false > $O/thr_m50_10b.log 2>&1
  report thr_m50_10b
fi

# 2: SF18 gauntlet (node-limited, Cython path). The .so must be gen 6.3 and match the UCI binary with the term on.
TH_ENV="THREAT_V2_PCT=100 THREAT_V2_HANGING=1 THREAT_V2_KING=1 THREAT_V2_PAWN_TARGETS=1 THREAT_V2_MG_PCT=50 THREAT_V2_EG_PCT=100"
c=$(env V2_PRESET=shipped bash "$R" wac q116_ctl); a=$(env V2_PRESET=shipped $TH_ENV bash "$R" wac q116_m50)
echo "[q116] cython: ctl $(echo "$c" | grep -E '^SOLVED|^NODES' | tr '\n' ' ') · m50 $(echo "$a" | grep -E '^SOLVED|^NODES' | tr '\n' ' ')"
if echo "$c" | grep -q "NODES: 42263170" && echo "$a" | grep -q "NODES: 43040865"; then
  echo "[q116] $(date) gauntlet g1100_m50_s1122"
  bash "$R" gauntlet 500 1100 3 g1100_m50_s1122 1122 V2_PRESET=shipped $TH_ENV 2>&1 | grep -E "our score|games in"
else
  echo "[q116] ☠️ Cython fingerprints failed — gauntlet skipped"
fi

# 3: Mediocre at 60+0.6
if preflight med_m50 "$BASE,$M50" "$MED" ""; then
  echo "[q116] $(date) med_m50 (200 games, 60+0.6)"
  nice -n 5 "$FC" -engine cmd="$BIN" name=m50 option.OwnBook=false option.Knobs=$BASE,$M50 tc=60+0.6 timemargin=250 restart=on \
    -engine cmd="$MED" name=mediocre tc=60+0.6 timemargin=250 restart=on $COMMON -srand 1071 -rounds 100 -games 2 \
    -pgnout file=$O/med_m50.pgn notation=san nodes=true append=false > $O/med_m50.log 2>&1
  report med_m50
fi

echo "[q116] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | sed 's/^/[q116]   /'
echo "[q116] $(date) QUEUE 116 DONE"
