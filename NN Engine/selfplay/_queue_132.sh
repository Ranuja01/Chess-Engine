#!/bin/bash
# 2026-11-01, queue #132: chained after #131, fills the night and Monday daytime (owner 11-01: the box may run until Monday
# 18:00). Legs in priority order, each only if its condition holds and it can finish by the deadline:
#   1. caps_tr_r  caps replicate on the transition starts (fresh seed), if caps_tr read +8 or more.
#   2. damp_tr_r  damp replicate on the transition starts (fresh seed), if damp_tr read +8 or more.
#   3. shelter    the shelter -> danger coupling KS_V2_SHELTER_K=12 (SF11's 6/8 in our units) vs gen 6.4 on UHO: the one
#                 king-safety item with no game read, and its own mechanism, so pinned defenders parking does not settle it.
#   4. shelter_r  its replicate (fresh seed), if shelter read +8 or more.
# A single 1,000-game run is a screen; +8 is the bar for spending a replicate (owner 11-01). Registered: a replicate lands within
# its first run's ± (regression toward 0 expected); shelter −10 … +15. A pair pooled ≥ 2σ ⇒ ship checks (caps and damp by the
# narrow-item rule: their venue ≥ 2σ, UHO not harmful); pooled ≤ 0.5σ ⇒ parked.
# Binary ks4 both sides, 10+0.1, 1,000 games per leg. Fingerprints (ks4): default 249 / 36,991,901 · caps 38,998,902 ·
# damp 35,917,541 · shelter K 12 38,767,998. Stop between legs: touch /mnt/e/chess_data/q132.stop. Never edit this file while
# it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
KS4="$ND/build_uci/chess_engine_uci_ks4"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
STOP=/mnt/e/chess_data/q132.stop
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
BASE=THREAT_ORDER_W=100
CAPS=$BASE,POT_V2_WIN_OCBX=-42,POT_V2_WIN_OCBX_PC=3,POT_V2_WIN_ROOKE=-28,POT_V2_WIN_QNOQ=-27,POT_V2_WIN_QNOQ_MINOR=3,POT_V2_WIN_LONEMINOR=-64
DAMP=$BASE,WIN_V2=1,WIN_V2_DAMP=1,WIN_V2_PHASE_MAX=0,WIN_V2_SKIP_PAWNEND=1,WIN_V2_PASSED=42,WIN_V2_PAWNS=52,WIN_V2_OUTFLANK=42,WIN_V2_INFILT=56,WIN_V2_FLANKS=98,WIN_V2_UNWIN=-202,WIN_V2_BASE=-469
SHELTER=$BASE,KS_V2_SHELTER_K=12
TR=$O/transition.epd
UHO=$O/openings_uho.epd
cd "$ND" || exit 1
rm -f "$STOP"
echo "[q132] $(date) waiting for queue 131 to finish"
while pgrep -f "fastchess|vs_sf.py|_queue_131\.sh" >/dev/null; do sleep 30; done
START=$(date '+%Y-%m-%d %H:%M:%S')
DEADLINE=$(date -d '2026-11-02 18:00' +%s)
echo "[q132] $START start (timed legs must finish by $(date -d @$DEADLINE); stop file $STOP)"

nodes() { printf "setoption name Knobs value $2\nbench\nquit\n" | "$1" 2>/dev/null | grep '^bench' | sed -n 's/.* nodes \([0-9]*\).*/\1/p'; }
db=$(nodes "$KS4" $BASE); dc=$(nodes "$KS4" $CAPS); dd=$(nodes "$KS4" $DAMP); ds=$(nodes "$KS4" $SHELTER)
echo "[q132] ks4 binary: base $db (want 36991901) caps $dc (want 38998902) damp $dd (want 35917541) shelter $ds (want 38767998)"
[ "$db" = 36991901 ] && [ "$dc" = 38998902 ] && [ "$dd" = 35917541 ] && [ "$ds" = 38767998 ] || { echo "[q132] ☠️ fingerprints failed — QUEUE 132 ABORTED"; exit 1; }
[ "$(grep -c . $TR)" = 500 ] || { echo "[q132] ☠️ transition.epd is not 500 starts — QUEUE 132 ABORTED"; exit 1; }

elo_of() { grep -E '^Elo:' $O/$1.log 2>/dev/null | tail -1 | awk '{gsub(",","",$4); print $2, $4}'; }
pool() { awk '{n++; s+=$1; v+=$2*$2} END{if(n==0){print "0 0 0"; exit} m=s/n; c=sqrt(v)/n; printf "%.2f %.2f %.2f\n", m, c, 1.96*m/c}'; }
ge8() { awk -v e="$(elo_of $1 | awk '{print $1}')" 'BEGIN{exit !(e != "" && e + 0 >= 8)}'; }
pooled() {   # $1 tag of the first run; prints the two-run pool only when the replicate finished
  [ -n "$(elo_of ${1}_r)" ] || return
  echo "[q132] $1 pooled over two runs: $( { elo_of $1; elo_of ${1}_r; } | pool | awk '{printf "%+.1f ± %.1f (z %.2f)", $1, $2, $3}')"
}
echo "[q132] #131: caps_tr $(elo_of caps_tr) · damp_tr $(elo_of damp_tr)"

COMMON="-repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
report() {
  echo "[q132] $(date) $1 done: $(grep -E '^Games:|^Elo:' $O/$1.log | tail -2 | tr '\n' ' ')"
  echo "[q132]   $(grep -E '^Ptnml' $O/$1.log | tail -1)  disconnects/stalls: $(grep -ciE 'disconnect|stall|not responsive' $O/$1.log)  time forfeits: $(grep -c 'loses on time' $O/$1.pgn)"
}
gate() {   # $1 tag, $2 seconds needed
  [ -f "$STOP" ] && { echo "[q132] $(date) skip $1: stop file present"; return 1; }
  local left=$(( DEADLINE - $(date +%s) ))
  [ $left -ge $2 ] || { echo "[q132] $(date) skip $1: only ${left}s before the deadline (needs $2)"; return 1; }
}
preflight() {   # $1 tag, $2 arm knobs, $3 control knobs, $4 openings
  rm -f /tmp/q132_pre.engine.log
  timeout 180 "$FC" -engine cmd="$KS4" name=arm option.OwnBook=false option.Knobs=$2 st=0.05 \
    -engine cmd="$KS4" name=ctl option.OwnBook=false option.Knobs=$3 st=0.05 \
    -openings file=$4 format=epd order=random -rounds 1 -games 2 -concurrency 1 \
    -log file=/tmp/q132_pre.engine.log level=info engine=true > /tmp/q132_pre.log 2>&1
  local W A C; W=$(grep -c "have option" /tmp/q132_pre.log)
  A=$(grep -oE " arm ---> info string knobs [A-Z0-9_=,-]+" /tmp/q132_pre.engine.log | grep -cxF " arm ---> info string knobs $2")
  C=$(grep -oE " ctl ---> info string knobs [A-Z0-9_=,-]+" /tmp/q132_pre.engine.log | grep -cxF " ctl ---> info string knobs $3")
  echo "[q132] $1 preflight: warnings $W, arm echoes $A, control echoes $C"
  [ "$W" = "0" ] && [ "$A" -ge 1 ] && [ "$C" -ge 1 ] || { echo "[q132] ☠️ $1 PREFLIGHT FAILED — skipped"; return 1; }
}
selfplay() {   # $1 tag, $2 arm knobs, $3 control knobs, $4 openings, $5 tc, $6 rounds, $7 seed, $8 seconds needed
  gate $1 $8 && preflight $1 "$2" "$3" $4 || return
  echo "[q132] $(date) $1 ($((2 * $6)) games, $5, $(basename $4))"
  nice -n 5 "$FC" -engine cmd="$KS4" name=arm option.OwnBook=false option.Knobs=$2 tc=$5 timemargin=250 \
    -engine cmd="$KS4" name=ctl option.OwnBook=false option.Knobs=$3 tc=$5 timemargin=250 \
    -openings file=$4 format=epd order=random $COMMON -srand $7 -rounds $6 -games 2 \
    -pgnout file=$O/$1.pgn notation=san nodes=true append=false > $O/$1.log 2>&1
  report $1
}

if ge8 caps_tr; then selfplay caps_tr_r "$CAPS" "$BASE" $TR 10+0.1 500 1321 10200; pooled caps_tr
else echo "[q132] no caps replicate: caps_tr below +8"; fi
if ge8 damp_tr; then selfplay damp_tr_r "$DAMP" "$BASE" $TR 10+0.1 500 1322 10200; pooled damp_tr
else echo "[q132] no damp replicate: damp_tr below +8"; fi
selfplay shelter "$SHELTER" "$BASE" $UHO 10+0.1 500 1323 10200
if ge8 shelter; then selfplay shelter_r "$SHELTER" "$BASE" $UHO 10+0.1 500 1324 10200; pooled shelter
else echo "[q132] no shelter replicate: shelter below +8 or not run"; fi
echo "[q132] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | sed 's/^/[q132]   /'
echo "[q132] $(date) QUEUE 132 DONE"
