#!/bin/bash
# 2026-11-01, queue #133: ship checks for the endgame caps, chained after #132, run ONLY if caps pooled over the two transition
# runs (#131 caps_tr + #132 caps_tr_r) is ≥ 2σ (z ≥ 1.96). Caps are a narrow item: their venue (transition starts) carries the
# ≥ 2σ, UHO already reads not harmful (pooled ≈ +6 ± 14, 60+0.6 −8.7 ± 42.9), so these legs are the external vetoes.
#  1 WAC at 250k nodes per position, gen 6.4 vs caps (tactical veto)
#  2 SF18 @1100 nodes, ours 250k nodes, 500 g each, gen 6.4 and caps on FRESH seeds 1331 and 1332 (seeds 1122 / 1162 selected
#    gen 6.4, so its saved games there are not an unbiased baseline)
#  3 caps vs Mediocre 0.5, 60+0.6, 200 g, seed 1071 (gen 6.4 there: +208.7 ± 86.9 on 80 g, #116; gen 6.3 +113.3 ± 45.9 on 200)
# Registered: 1 caps not more than 5 solves below gen 6.4; 2 caps − gen 6.4 pooled over both seeds > −1σ, expected −1 … +3
# points (caps fire in few SF18 games); 3 not below gen 6.4 by more than 1σ, expected +120 … +220.
# All pass ⇒ caps go to the owner as a ship candidate. Timed legs must finish by Monday 18:00. Stop between legs: touch
# /mnt/e/chess_data/q133.stop. Never edit this file while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
KS4="$ND/build_uci/chess_engine_uci_ks4"
MED="$ND/selfplay/mediocre_uci.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
STOP=/mnt/e/chess_data/q133.stop
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
BASE=THREAT_ORDER_W=100
CAPS=$BASE,POT_V2_WIN_OCBX=-42,POT_V2_WIN_OCBX_PC=3,POT_V2_WIN_ROOKE=-28,POT_V2_WIN_QNOQ=-27,POT_V2_WIN_QNOQ_MINOR=3,POT_V2_WIN_LONEMINOR=-64
# Cython legs: gen 6.4 stated in full, so the result does not depend on which generation the .so's preset holds.
G64_ENV="V2_PRESET=shipped THREAT_ORDER_W=100 THREAT_V2_PCT=100 THREAT_V2_HANGING=1 THREAT_V2_KING=1 THREAT_V2_PAWN_TARGETS=1 THREAT_V2_MG_PCT=50 THREAT_V2_EG_PCT=100 RFP_MARGIN=600"
CAPS_ENV="$G64_ENV POT_V2_WIN_OCBX=-42 POT_V2_WIN_OCBX_PC=3 POT_V2_WIN_ROOKE=-28 POT_V2_WIN_QNOQ=-27 POT_V2_WIN_QNOQ_MINOR=3 POT_V2_WIN_LONEMINOR=-64"
cd "$ND" || exit 1
rm -f "$STOP"
echo "[q133] $(date) waiting for queue 132 to finish"
while pgrep -f "fastchess|vs_sf.py|_queue_132\.sh" >/dev/null; do sleep 30; done
START=$(date '+%Y-%m-%d %H:%M:%S')
DEADLINE=$(date -d '2026-11-02 18:00' +%s)
echo "[q133] $START start (timed legs must finish by $(date -d @$DEADLINE); stop file $STOP)"

elo_of() { grep -E '^Elo:' $O/$1.log 2>/dev/null | tail -1 | awk '{gsub(",","",$4); print $2, $4}'; }
pool() { awk '{n++; s+=$1; v+=$2*$2} END{if(n==0){print "0 0 0"; exit} m=s/n; c=sqrt(v)/n; printf "%.2f %.2f %.2f\n", m, c, 1.96*m/c}'; }
[ -n "$(elo_of caps_tr)" ] && [ -n "$(elo_of caps_tr_r)" ] || { echo "[q133] caps runs incomplete (caps_tr $(elo_of caps_tr) · caps_tr_r $(elo_of caps_tr_r)): no ship checks. $(date) QUEUE 133 DONE"; exit 0; }
P=$( { elo_of caps_tr; elo_of caps_tr_r; } | pool )
echo "[q133] caps over two transition runs: $(echo $P | awk '{printf "%+.1f ± %.1f (z %.2f)", $1, $2, $3}')"
echo $P | awk '{exit !($3 >= 1.96)}' || { echo "[q133] caps below 2σ: no ship checks. $(date) QUEUE 133 DONE"; exit 0; }

gate() {   # $1 tag, $2 seconds needed
  [ -f "$STOP" ] && { echo "[q133] $(date) skip $1: stop file present"; return 1; }
  local left=$(( DEADLINE - $(date +%s) ))
  [ $left -ge $2 ] || { echo "[q133] $(date) skip $1: only ${left}s before the deadline (needs $2)"; return 1; }
}

# 1 + 2: Cython legs. The .so must give gen 6.4 and caps the same d10 nodes as the UCI binary.
c=$(env $G64_ENV bash "$R" wac q133_ctl); a=$(env $CAPS_ENV bash "$R" wac q133_caps)
echo "[q133] cython d10: gen 6.4 $(echo "$c" | grep -E '^SOLVED|^NODES' | tr '\n' ' ') · caps $(echo "$a" | grep -E '^SOLVED|^NODES' | tr '\n' ' ')"
if echo "$c" | grep -q "NODES: 36991901" && echo "$a" | grep -q "NODES: 38998902"; then
  for arm in "g64|$G64_ENV" "caps|$CAPS_ENV"; do
    t=${arm%%|*}; k=${arm#*|}
    w=$(env $k bash "$R" wac q133_n_$t MAX_DEPTH=64 NODE_LIMIT=250000)
    echo "[q133] WAC @250k nodes $t: $(echo "$w" | grep -E '^SOLVED|^NODES' | tr '\n' ' ')"
  done
  for s in 1331 1332; do
    gate g1100_g64_s$s 3600 && { echo "[q133] $(date) gauntlet g1100_g64_s$s"
      bash "$R" gauntlet 500 1100 3 g1100_g64_s$s $s $G64_ENV 2>&1 | grep -E "our score|games in"; }
    gate g1100_caps_s$s 3600 && { echo "[q133] $(date) gauntlet g1100_caps_s$s"
      bash "$R" gauntlet 500 1100 3 g1100_caps_s$s $s $CAPS_ENV 2>&1 | grep -E "our score|games in"; }
  done
else
  echo "[q133] ☠️ Cython fingerprints failed (want 36991901 / 38998902) — WAC and SF18 legs skipped"
fi

# 3: Mediocre at 60+0.6 on the UCI binary.
nodes() { printf "setoption name Knobs value $2\nbench\nquit\n" | "$1" 2>/dev/null | grep '^bench' | sed -n 's/.* nodes \([0-9]*\).*/\1/p'; }
dc=$(nodes "$KS4" $CAPS)
echo "[q133] ks4 binary: caps $dc (want 38998902)"
if [ "$dc" = 38998902 ] && gate med_caps 13000; then
  rm -f /tmp/q133_pre.engine.log
  timeout 180 "$FC" -engine cmd="$KS4" name=caps option.OwnBook=false option.Knobs=$CAPS st=0.05 \
    -engine cmd="$MED" name=mediocre st=0.05 \
    -openings file=$O/openings_uho.epd format=epd order=random -rounds 1 -games 2 -concurrency 1 \
    -log file=/tmp/q133_pre.engine.log level=info engine=true > /tmp/q133_pre.log 2>&1
  W=$(grep -c "have option" /tmp/q133_pre.log)
  A=$(grep -oE " caps ---> info string knobs [A-Z0-9_=,-]+" /tmp/q133_pre.engine.log | grep -cxF " caps ---> info string knobs $CAPS")
  echo "[q133] med_caps preflight: warnings $W, caps echoes $A"
  if [ "$W" = "0" ] && [ "$A" -ge 1 ]; then
    echo "[q133] $(date) med_caps (200 games, 60+0.6)"
    nice -n 5 "$FC" -engine cmd="$KS4" name=caps option.OwnBook=false option.Knobs=$CAPS tc=60+0.6 timemargin=250 restart=on \
      -engine cmd="$MED" name=mediocre tc=60+0.6 timemargin=250 restart=on \
      -openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 \
      -draw movenumber=40 movecount=8 score=10 -srand 1071 -rounds 100 -games 2 \
      -pgnout file=$O/med_caps.pgn notation=san nodes=true append=false > $O/med_caps.log 2>&1
    echo "[q133] $(date) med_caps done: $(grep -E '^Games:|^Elo:' $O/med_caps.log | tail -2 | tr '\n' ' ')"
    echo "[q133]   $(grep -E '^Ptnml' $O/med_caps.log | tail -1)  disconnects/stalls: $(grep -ciE 'disconnect|stall|not responsive' $O/med_caps.log)  time forfeits: $(grep -c 'loses on time' $O/med_caps.pgn)"
  else
    echo "[q133] ☠️ med_caps PREFLIGHT FAILED — skipped"
  fi
else
  [ "$dc" = 38998902 ] || echo "[q133] ☠️ ks4 caps fingerprint failed — Mediocre leg skipped"
fi

echo "[q133] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | sed 's/^/[q133]   /'
echo "[q133] $(date) QUEUE 133 DONE"
