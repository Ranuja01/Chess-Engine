#!/bin/bash
# 2026-11-01, queue #131: the endgame caps and the damp-only drawishness threshold, re-judged WHERE THEY FIRE. Both were parked
# on UHO self-play alone (caps pooled ≈ +6 ± 14, damp −5.9 ± 19.9), which rarely reaches the endings they act on; the
# narrow-item rule judges a term in its own venue (≥ +2σ there, not harmful elsewhere). A drawishness term pays mostly in the
# trade-down decision, so the venue is transition.epd: 500 late-middlegame / early-endgame starts with a modest edge (phase256
# 64-160, |cp| 40-200 by the mover's own search, one per source opening; diagnostics/_make_transition_starts.py).
# Each arm vs gen 6.4 at 10+0.1, 1,000 games, on the ks4 binary for BOTH sides. Registered: caps −5 … +20 · damp −10 … +15.
# Caps ≥ 2σ here ⇒ caps meet the narrow-item ship rule (UHO already not harmful) and go to replication + ship checks; both
# within ±1σ ⇒ the parked verdict stands, now measured where the terms fire.
# Fingerprints (ks4 binary): default 249 / 36,991,901 · caps 248 / 38,998,902 · damp 249 / 35,917,541.
# Waits for queue 130. Timed legs start only if they can finish by Monday 08:00. Stop between legs: touch
# /mnt/e/chess_data/q131.stop. Never edit this file while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
KS4="$ND/build_uci/chess_engine_uci_ks4"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
STOP=/mnt/e/chess_data/q131.stop
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
BASE=THREAT_ORDER_W=100
CAPS=$BASE,POT_V2_WIN_OCBX=-42,POT_V2_WIN_OCBX_PC=3,POT_V2_WIN_ROOKE=-28,POT_V2_WIN_QNOQ=-27,POT_V2_WIN_QNOQ_MINOR=3,POT_V2_WIN_LONEMINOR=-64
DAMP=$BASE,WIN_V2=1,WIN_V2_DAMP=1,WIN_V2_PHASE_MAX=0,WIN_V2_SKIP_PAWNEND=1,WIN_V2_PASSED=42,WIN_V2_PAWNS=52,WIN_V2_OUTFLANK=42,WIN_V2_INFILT=56,WIN_V2_FLANKS=98,WIN_V2_UNWIN=-202,WIN_V2_BASE=-469
cd "$ND" || exit 1
rm -f "$STOP"
echo "[q131] $(date) waiting for queue 130 to finish"
while pgrep -f "fastchess|vs_sf.py|_queue_130\.sh" >/dev/null; do sleep 30; done
START=$(date '+%Y-%m-%d %H:%M:%S')
DEADLINE=$(date -d 'tomorrow 08:00' +%s); [ "$(date +%H)" -lt 8 ] && DEADLINE=$(date -d 'today 08:00' +%s)
echo "[q131] $START start (timed legs must finish by $(date -d @$DEADLINE); stop file $STOP)"

nodes() { printf "setoption name Knobs value $2\nbench\nquit\n" | "$1" 2>/dev/null | grep '^bench' | sed -n 's/.* nodes \([0-9]*\).*/\1/p'; }
db=$(nodes "$KS4" $BASE); dc=$(nodes "$KS4" $CAPS); dd=$(nodes "$KS4" $DAMP)
echo "[q131] ks4 binary: base $db (want 36991901) caps $dc (want 38998902) damp $dd (want 35917541)"
[ "$db" = 36991901 ] && [ "$dc" = 38998902 ] && [ "$dd" = 35917541 ] || { echo "[q131] ☠️ fingerprints failed — QUEUE 131 ABORTED"; exit 1; }
[ "$(grep -c . $O/transition.epd)" = 500 ] || { echo "[q131] ☠️ transition.epd is not 500 starts — QUEUE 131 ABORTED"; exit 1; }

COMMON="-openings file=$O/transition.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
report() {
  echo "[q131] $(date) $1 done: $(grep -E '^Games:|^Elo:' $O/$1.log | tail -2 | tr '\n' ' ')"
  echo "[q131]   $(grep -E '^Ptnml' $O/$1.log | tail -1)  disconnects/stalls: $(grep -ciE 'disconnect|stall|not responsive' $O/$1.log)  time forfeits: $(grep -c 'loses on time' $O/$1.pgn)"
}
gate() {   # $1 tag, $2 seconds needed
  [ -f "$STOP" ] && { echo "[q131] $(date) skip $1: stop file present"; return 1; }
  local left=$(( DEADLINE - $(date +%s) ))
  [ $left -ge $2 ] || { echo "[q131] $(date) skip $1: only ${left}s before the deadline (needs $2)"; return 1; }
}
preflight() {   # $1 tag, $2 arm knobs, $3 control knobs
  rm -f /tmp/q131_pre.engine.log
  timeout 180 "$FC" -engine cmd="$KS4" name=arm option.OwnBook=false option.Knobs=$2 st=0.05 \
    -engine cmd="$KS4" name=ctl option.OwnBook=false option.Knobs=$3 st=0.05 \
    -openings file=$O/transition.epd format=epd order=random -rounds 1 -games 2 -concurrency 1 \
    -log file=/tmp/q131_pre.engine.log level=info engine=true > /tmp/q131_pre.log 2>&1
  local W A C; W=$(grep -c "have option" /tmp/q131_pre.log)
  A=$(grep -oE " arm ---> info string knobs [A-Z0-9_=,-]+" /tmp/q131_pre.engine.log | grep -cxF " arm ---> info string knobs $2")
  C=$(grep -oE " ctl ---> info string knobs [A-Z0-9_=,-]+" /tmp/q131_pre.engine.log | grep -cxF " ctl ---> info string knobs $3")
  echo "[q131] $1 preflight: warnings $W, arm echoes $A, control echoes $C"
  [ "$W" = "0" ] && [ "$A" -ge 1 ] && [ "$C" -ge 1 ] || { echo "[q131] ☠️ $1 PREFLIGHT FAILED — skipped"; return 1; }
}
selfplay() {   # $1 tag, $2 arm knobs, $3 control knobs, $4 tc, $5 rounds, $6 seed, $7 seconds needed
  gate $1 $7 && preflight $1 "$2" "$3" || return
  echo "[q131] $(date) $1 ($((2 * $5)) games, $4)"
  nice -n 5 "$FC" -engine cmd="$KS4" name=arm option.OwnBook=false option.Knobs=$2 tc=$4 timemargin=250 \
    -engine cmd="$KS4" name=ctl option.OwnBook=false option.Knobs=$3 tc=$4 timemargin=250 $COMMON -srand $6 -rounds $5 -games 2 \
    -pgnout file=$O/$1.pgn notation=san nodes=true append=false > $O/$1.log 2>&1
  report $1
}

selfplay  caps_tr  "$CAPS"  "$BASE" 10+0.1 500 1311 10200
selfplay  damp_tr  "$DAMP"  "$BASE" 10+0.1 500 1312 10200
echo "[q131] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | sed 's/^/[q131]   /'
echo "[q131] $(date) QUEUE 131 DONE"
