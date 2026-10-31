#!/bin/bash
# 2026-10-31 evening, queue #127: chained after #126 (waits for it), for the night the owner is away. It branches on RFP 500's
# pooled 10+0.1 result vs gen 6.4 (#125 rfp500 + #126 rfp500_r):
#  A  pooled ≥ 2σ ⇒ the ship checks for RFP 500:
#     WAC at 250k nodes (veto: not more than 5 solves below gen 6.4's 247) · SF18 @1100 nodes, ours 250k, 500 g on seeds 1122 and
#     1162 (paired with gen 6.4's 54.0% / 55.4% in #116; pooled RFP 500 − gen 6.4 > −1σ, expected −2 … +4 points) · Mediocre 0.5
#     at 60+0.6, 200 g, seed 1071 (gen 6.4 76.9% over the first 80 in #116; not below it by > 1σ) · plateau RFP 450 and 550,
#     each vs RFP 500, 1,000 g at 10+0.1, seeds 1272 / 1273 (registered −20 … +10 each; either above +10 ⇒ optimum past 500)
#  B  0.5σ < pooled < 2σ ⇒ the margin curve near 500, all vs gen 6.4 at 10+0.1, 1,000 g: RFP 500 again (seed 1274, −5 … +20),
#     RFP 450 (1275, −10 … +20), RFP 550 (1276, −10 … +20); then, if RFP 500 pooled over three runs reaches 2σ, WAC + SF18 as in A
#  C  pooled ≤ 0.5σ ⇒ nothing (the margin lead is noise; the box stays idle)
# The SF18 and WAC legs run on the live Cython module (gen 6.4 default; RFP_MARGIN by env); they run only if its fingerprints
# match the UCI binary. Timed legs start only if they can finish by Sunday 13:00. Damp is not in any arm: if #126's damp_on500
# comes out ≥ 2σ, that is decided by hand.
# Stop between legs: touch /mnt/e/chess_data/q127.stop. Never edit this file while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
DMP="$ND/build_uci/chess_engine_uci_damp"
MED="$ND/selfplay/mediocre_uci.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
STOP=/mnt/e/chess_data/q127.stop
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
BASE=THREAT_ORDER_W=100
R500=$BASE,RFP_MARGIN=500
cd "$ND" || exit 1
rm -f "$STOP"
echo "[q127] $(date) waiting for queue 126 to finish"
while pgrep -f "fastchess|vs_sf.py|_queue_126.sh" >/dev/null; do sleep 30; done
START=$(date '+%Y-%m-%d %H:%M:%S')
DEADLINE=$(date -d 'tomorrow 13:00' +%s); [ "$(date +%H)" -lt 13 ] && DEADLINE=$(date -d 'today 13:00' +%s)
echo "[q127] $START start (timed legs must finish by $(date -d @$DEADLINE); stop file $STOP)"

# pooled Elo over runs: each input line "elo ci95"; prints "mean ci95 z"
elo_of() { grep -E '^Elo:' $O/$1.log 2>/dev/null | tail -1 | awk '{gsub(",","",$4); print $2, $4}'; }
pool() { awk '{n++; s+=$1; v+=$2*$2} END{if(n==0){print "0 0 0"; exit} m=s/n; c=sqrt(v)/n; printf "%.2f %.2f %.2f\n", m, c, 1.96*m/c}'; }
P=$( { elo_of rfp500; elo_of rfp500_r; } | pool ); Z=$(echo $P | awk '{print $3}')
echo "[q127] RFP 500 pooled over #125 + #126: $(echo $P | awk '{printf "%+.1f ± %.1f (z %.2f)", $1, $2, $3}')"
echo "[q127] #126 damp_on500: $(elo_of damp_on500)  ·  rfp500_60: $(elo_of rfp500_60)"
BR=C; awk -v z=$Z 'BEGIN{exit !(z >= 2)}' && BR=A || { awk -v z=$Z 'BEGIN{exit !(z > 0.5)}' && BR=B; }
echo "[q127] branch $BR"
[ "$BR" = C ] && { echo "[q127] $(date) QUEUE 127 DONE (nothing to run)"; exit 0; }

nodes() { printf "setoption name Knobs value $2\nbench\nquit\n" | "$1" 2>/dev/null | grep '^bench' | sed -n 's/.* nodes \([0-9]*\).*/\1/p'; }
db=$(nodes "$DMP" $BASE); dr=$(nodes "$DMP" $R500); d45=$(nodes "$DMP" $BASE,RFP_MARGIN=450); d55=$(nodes "$DMP" $BASE,RFP_MARGIN=550)
echo "[q127] damp binary: base $db (want 36991901) rfp500 $dr (want 36670022) rfp450 $d45 rfp550 $d55 (must differ)"
[ "$db" = 36991901 ] && [ "$dr" = 36670022 ] && [ -n "$d45" ] && [ "$d45" != "$dr" ] && [ "$d45" != "$db" ] \
  && [ -n "$d55" ] && [ "$d55" != "$dr" ] && [ "$d55" != "$db" ] \
  || { echo "[q127] ☠️ fingerprints failed — QUEUE 127 ABORTED"; exit 1; }

COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
report() {
  echo "[q127] $(date) $1 done: $(grep -E '^Games:|^Elo:' $O/$1.log | tail -2 | tr '\n' ' ')"
  echo "[q127]   $(grep -E '^Ptnml' $O/$1.log | tail -1)  disconnects/stalls: $(grep -ciE 'disconnect|stall|not responsive' $O/$1.log)  time forfeits: $(grep -c 'loses on time' $O/$1.pgn)"
}
gate() {   # $1 tag, $2 seconds needed
  [ -f "$STOP" ] && { echo "[q127] $(date) skip $1: stop file present"; return 1; }
  local left=$(( DEADLINE - $(date +%s) ))
  [ $left -ge $2 ] || { echo "[q127] $(date) skip $1: only ${left}s before the deadline (needs $2)"; return 1; }
}
preflight() {   # $1 tag, $2 arm knobs, $3 control cmd, $4 control knobs ("" = none)
  local copt=""; [ -n "$4" ] && copt="option.Knobs=$4"
  rm -f /tmp/q127_pre.engine.log
  timeout 180 "$FC" -engine cmd="$DMP" name=arm option.OwnBook=false option.Knobs=$2 st=0.05 \
    -engine cmd="$3" name=ctl $copt st=0.05 \
    -openings file=$O/openings_uho.epd format=epd order=random -rounds 1 -games 2 -concurrency 1 \
    -log file=/tmp/q127_pre.engine.log level=info engine=true > /tmp/q127_pre.log 2>&1
  local W K N; W=$(grep -c "have option" /tmp/q127_pre.log)
  K=$(grep -oE "knobs [A-Z0-9_=,-]+" /tmp/q127_pre.engine.log | grep -cxF "knobs $2")
  if [ -n "$4" ]; then N=$(grep -oE "knobs [A-Z0-9_=,-]+" /tmp/q127_pre.engine.log | grep -cxF "knobs $4"); else N=1; fi
  echo "[q127] $1 preflight: warnings $W, arm echoes $K, control echoes $N"
  [ "$W" = "0" ] && [ "$K" -ge 1 ] && [ "$N" -ge 1 ] || { echo "[q127] ☠️ $1 PREFLIGHT FAILED — skipped"; return 1; }
}
selfplay() {   # $1 tag, $2 arm knobs, $3 control knobs, $4 tc, $5 rounds, $6 seed, $7 seconds needed
  gate $1 $7 && preflight $1 "$2" "$DMP" "$3" || return
  echo "[q127] $(date) $1 ($((2 * $5)) games, $4)"
  nice -n 5 "$FC" -engine cmd="$DMP" name=arm option.OwnBook=false option.Knobs=$2 tc=$4 timemargin=250 \
    -engine cmd="$DMP" name=ctl option.OwnBook=false option.Knobs=$3 tc=$4 timemargin=250 $COMMON -srand $6 -rounds $5 -games 2 \
    -pgnout file=$O/$1.pgn notation=san nodes=true append=false > $O/$1.log 2>&1
  report $1
}
mediocre() {   # $1 tag, $2 arm knobs, $3 rounds, $4 seed, $5 seconds needed
  gate $1 $5 && preflight $1 "$2" "$MED" "" || return
  echo "[q127] $(date) $1 ($((2 * $3)) games, 60+0.6)"
  nice -n 5 "$FC" -engine cmd="$DMP" name=arm option.OwnBook=false option.Knobs=$2 tc=60+0.6 timemargin=250 restart=on \
    -engine cmd="$MED" name=mediocre tc=60+0.6 timemargin=250 restart=on $COMMON -srand $4 -rounds $3 -games 2 \
    -pgnout file=$O/$1.pgn notation=san nodes=true append=false > $O/$1.log 2>&1
  report $1
}
cython_checks() {   # WAC at 250k nodes and SF18 on two seeds for RFP 500, on the live module after a fingerprint match
  [ -f "$STOP" ] && { echo "[q127] $(date) skip cython checks: stop file present"; return; }
  local c a w
  c=$(env V2_PRESET=shipped bash "$R" wac q127_ctl); a=$(env V2_PRESET=shipped RFP_MARGIN=500 bash "$R" wac q127_r500)
  echo "[q127] cython d10: gen 6.4 $(echo "$c" | grep -E '^SOLVED|^NODES' | tr '\n' ' ') · RFP 500 $(echo "$a" | grep -E '^SOLVED|^NODES' | tr '\n' ' ')"
  if echo "$c" | grep -q "NODES: 36991901" && echo "$a" | grep -q "NODES: 36670022"; then
    w=$(env V2_PRESET=shipped RFP_MARGIN=500 bash "$R" wac q127_n_r500 MAX_DEPTH=64 NODE_LIMIT=250000)
    echo "[q127] WAC @250k nodes RFP 500: $(echo "$w" | grep -E '^SOLVED|^NODES' | tr '\n' ' ') (gen 6.4: 247/300)"
    for s in 1122 1162; do
      [ -f "$STOP" ] && { echo "[q127] $(date) skip gauntlet $s: stop file present"; continue; }
      echo "[q127] $(date) gauntlet g1100_r500_s$s"
      bash "$R" gauntlet 500 1100 3 g1100_r500_s$s $s V2_PRESET=shipped RFP_MARGIN=500 2>&1 | grep -E "our score|games in" | sed 's/^/[q127]   /'
    done
    echo "[q127]   gen 6.4 on the same seeds (#116): 1122 54.0% · 1162 55.4%"
  else
    echo "[q127] ☠️ Cython fingerprints failed — WAC and SF18 legs skipped"
  fi
}

if [ "$BR" = A ]; then
  cython_checks
  mediocre  med_r500    "$R500" 100 1071 12600
  selfplay  r450_vs500  "$BASE,RFP_MARGIN=450" "$R500" 10+0.1 500 1272 10200
  selfplay  r550_vs500  "$BASE,RFP_MARGIN=550" "$R500" 10+0.1 500 1273 10200
else
  selfplay  rfp500_r2   "$R500"                "$BASE" 10+0.1 500 1274 10200
  selfplay  rfp450      "$BASE,RFP_MARGIN=450" "$BASE" 10+0.1 500 1275 10200
  selfplay  rfp550      "$BASE,RFP_MARGIN=550" "$BASE" 10+0.1 500 1276 10200
  P3=$( { elo_of rfp500; elo_of rfp500_r; elo_of rfp500_r2; } | pool ); Z3=$(echo $P3 | awk '{print $3}')
  echo "[q127] RFP 500 pooled over three runs: $(echo $P3 | awk '{printf "%+.1f ± %.1f (z %.2f)", $1, $2, $3}')"
  if awk -v z=$Z3 'BEGIN{exit !(z >= 2)}'; then cython_checks; else echo "[q127] below 2σ: WAC / SF18 not run"; fi
fi
echo "[q127] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | sed 's/^/[q127]   /'
echo "[q127] $(date) QUEUE 127 DONE"
