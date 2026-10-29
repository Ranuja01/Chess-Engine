#!/bin/bash
# 2026-10-29, queue #116: ship checks for threats in eval at mg 50 / eg 100 together with RFP_MARGIN 600 (candidate C), on top
# of gen 6.3. Never edit this file while it runs.
# Evidence, all 1,000 g at 10+0.1 vs gen 6.3 (RFP 1000, no threats term):
#   #114 mg 50 @1000 +38.4 +- 19.7 · mg 0 @1000 +31.0 +- 19.1 · mg 50 @1000 at 60+0.6 +61.4 +- 42.8 (200 g)
#   #115 RFP 600 alone -27.5 +- 19.1 · mg 0 @600 +39.1 +- 19.1 · mg 50 @600 +73.0 +- 19.3
# The term is what makes the tighter margin pay (the crank channel). Binary build_uci/chess_engine_uci_thr2 (as #114/#115);
# the SF18 and WAC legs run on the Cython .so (gen 6.3 default since 10-28).
#  1 c_10b        C vs gen 6.3, fresh seed, 10+0.1, 1,000 g, seed 1161
#  2 WAC at 250k nodes per position, gen 6.3 vs C (the tactical veto; at d10 RFP 600 cost the base 11 solves)
#  3 SF18 @1100 nodes, ours 250k nodes, 500 g each: C on seed 1122 (paired with #112's gen 6.3 run, 53.8%), then gen 6.3
#    and C on a fresh seed 1162
#  4 c_60         C vs gen 6.3, 60+0.6, 200 g, seed 1163
#  5 med_c        C vs Mediocre 0.5, 60+0.6, 200 g, seed 1071 (= #112's gen 6.3 run, +113.3 +- 45.9)
#  6 plateau      C at RFP 800 and at RFP 400, each vs C at 600, 10+0.1, 1,000 g, seeds 1164 / 1165
# Registered: 1 +35 ... +75; 2 C not more than 5 solves below gen 6.3; 3 C - gen 6.3 pooled over both seeds > -1 sigma,
# expected -2 ... +5 points; 4 +20 ... +80; 5 not below gen 6.3 by more than 1 sigma, expected +100 ... +180;
# 6 RFP 800 -30 ... +10, RFP 400 -40 ... +10 (either one above +10 => the optimum is past 600 that way).
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
BIN="$ND/build_uci/chess_engine_uci_thr2"
MED="$ND/selfplay/mediocre_uci.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
BASE=TM_START=175,TM_SINGLE=1,TM_SCALE=7,TM_NORM=95,TM_OPT_PLY=1,THREAT_ORDER_W=100
M50=THREAT_V2_PCT=100,THREAT_V2_HANGING=1,THREAT_V2_KING=1,THREAT_V2_PAWN_TARGETS=1,THREAT_V2_MG_PCT=50,THREAT_V2_EG_PCT=100
C=$M50,RFP_MARGIN=600
TH_ENV="THREAT_V2_PCT=100 THREAT_V2_HANGING=1 THREAT_V2_KING=1 THREAT_V2_PAWN_TARGETS=1 THREAT_V2_MG_PCT=50 THREAT_V2_EG_PCT=100"
cd "$ND" || exit 1
START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[q116] $START start"
nodes() { printf "$2bench\nquit\n" | "$1" 2>/dev/null | grep '^bench' | sed -n 's/.* nodes \([0-9]*\).*/\1/p'; }
nb=$(nodes "$BIN" "setoption name Knobs value $BASE\n"); nc=$(nodes "$BIN" "setoption name Knobs value $BASE,$C\n")
n8=$(nodes "$BIN" "setoption name Knobs value $BASE,$M50,RFP_MARGIN=800\n"); n4=$(nodes "$BIN" "setoption name Knobs value $BASE,$M50,RFP_MARGIN=400\n")
echo "[q116] thr2: base $nb (want 42263170) C $nc (want 36991901) C@800 $n8 C@400 $n4 (each must differ from C)"
[ "$nb" = 42263170 ] && [ "$nc" = 36991901 ] && [ -n "$n8" ] && [ "$n8" != "$nc" ] && [ -n "$n4" ] && [ "$n4" != "$nc" ] \
  || { echo "[q116] ☠️ fingerprints failed — QUEUE 116 ABORTED"; exit 1; }

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
selfplay() {   # $1 tag, $2 arm knobs, $3 control knobs, $4 tc, $5 rounds, $6 seed
  preflight $1 "$2" "$BIN" "$3" || return
  echo "[q116] $(date) $1 ($((2 * $5)) games, $4)"
  nice -n 5 "$FC" -engine cmd="$BIN" name=arm option.OwnBook=false option.Knobs=$2 tc=$4 timemargin=250 restart=on \
    -engine cmd="$BIN" name=ctl option.OwnBook=false option.Knobs=$3 tc=$4 timemargin=250 restart=on $COMMON -srand $6 -rounds $5 -games 2 \
    -pgnout file=$O/$1.pgn notation=san nodes=true append=false > $O/$1.log 2>&1
  report $1
}

# 1: fresh-seed self-play at 10+0.1
selfplay c_10b "$BASE,$C" "$BASE" 10+0.1 500 1161

# 2 + 3: Cython legs. The .so must be gen 6.3 and give C the same nodes as the UCI binary.
c=$(env V2_PRESET=shipped bash "$R" wac q116_ctl); a=$(env V2_PRESET=shipped $TH_ENV RFP_MARGIN=600 bash "$R" wac q116_c)
echo "[q116] cython d10: gen 6.3 $(echo "$c" | grep -E '^SOLVED|^NODES' | tr '\n' ' ') · C $(echo "$a" | grep -E '^SOLVED|^NODES' | tr '\n' ' ')"
if echo "$c" | grep -q "NODES: 42263170" && echo "$a" | grep -q "NODES: 36991901"; then
  for arm in "g63|" "c|$TH_ENV RFP_MARGIN=600"; do
    t=${arm%%|*}; k=${arm#*|}
    w=$(env V2_PRESET=shipped $k bash "$R" wac q116_n_$t MAX_DEPTH=64 NODE_LIMIT=250000)
    echo "[q116] WAC @250k nodes $t: $(echo "$w" | grep -E '^SOLVED|^NODES' | tr '\n' ' ')"
  done
  echo "[q116] $(date) gauntlet g1100_c_s1122"
  bash "$R" gauntlet 500 1100 3 g1100_c_s1122 1122 V2_PRESET=shipped $TH_ENV RFP_MARGIN=600 2>&1 | grep -E "our score|games in"
  echo "[q116] $(date) gauntlet g1100_g63_s1162"
  bash "$R" gauntlet 500 1100 3 g1100_g63_s1162 1162 V2_PRESET=shipped 2>&1 | grep -E "our score|games in"
  echo "[q116] $(date) gauntlet g1100_c_s1162"
  bash "$R" gauntlet 500 1100 3 g1100_c_s1162 1162 V2_PRESET=shipped $TH_ENV RFP_MARGIN=600 2>&1 | grep -E "our score|games in"
else
  echo "[q116] ☠️ Cython fingerprints failed — WAC and SF18 legs skipped"
fi

# 4: self-play at 60+0.6
selfplay c_60 "$BASE,$C" "$BASE" 60+0.6 100 1163

# 5: Mediocre at 60+0.6
if preflight med_c "$BASE,$C" "$MED" ""; then
  echo "[q116] $(date) med_c (200 games, 60+0.6)"
  nice -n 5 "$FC" -engine cmd="$BIN" name=c option.OwnBook=false option.Knobs=$BASE,$C tc=60+0.6 timemargin=250 restart=on \
    -engine cmd="$MED" name=mediocre tc=60+0.6 timemargin=250 restart=on $COMMON -srand 1071 -rounds 100 -games 2 \
    -pgnout file=$O/med_c.pgn notation=san nodes=true append=false > $O/med_c.log 2>&1
  report med_c
fi

# 6: margin plateau around 600
selfplay c_r800_10 "$BASE,$M50,RFP_MARGIN=800" "$BASE,$C" 10+0.1 500 1164
selfplay c_r400_10 "$BASE,$M50,RFP_MARGIN=400" "$BASE,$C" 10+0.1 500 1165

echo "[q116] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | sed 's/^/[q116]   /'
echo "[q116] $(date) QUEUE 116 DONE"
