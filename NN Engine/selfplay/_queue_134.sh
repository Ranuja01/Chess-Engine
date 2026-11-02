#!/bin/bash
# 2026-11-02, queue #134: the endgame caps judged on a TARGETED venue, chained after #133. Caps pooled over the two transition
# runs (#131 caps_tr +12.9 ± 14.5, #132 caps_tr_r +5.2 ± 13.9) = +9.0 ± 10.0 (z 1.77): positive where they fire, under 2σ. Per
# the 11-01 plan (1-2σ ⇒ targeted venue), the next read is on starts where a capped ending is on the board or one trade away.
#  0 roles read of the two transition runs (descriptive, no engines): does caps' gain come from converting more when ahead or
#    holding more when behind? Registered: both deltas ≥ 0, neither resolved alone (≈ ±4-5 pp per role).
#  1 generator check: the default mode must still rebuild transition.epd byte for byte (the caps mode was added 11-02).
#  2 capped.epd: _make_transition_starts.py NEAR=caps from the ten gen 6.4-era UHO runs, excluding the transition starts: per game
#    the onset of a cap class (opposite-coloured bishops with other pieces, queen vs no queen, the one-flank rook ending, pawnless
#    leader with at most a lone minor) held 6 plies, onset phase ≤ 192; start = a quiet ply from 10 before to 6 after the onset,
#    mover's |score| 30-250; one per opening, up to 500. Needs ≥ 300 starts.
#  3 caps_ca vs gen 6.4 on capped.epd, 10+0.1, ks4 binary both sides, seed 1341; replicate caps_ca_r (seed 1342) if ≥ +8.
#    Registered: caps_ca −5 … +30 (expected +10 … +20, larger than the transition read since the caps fire in more of the games);
#    replicate within its first run's ±; venue Ptnml middle (1-point pairs) ≤ 65% of pairs.
#    Decision: pooled ≥ 2σ ⇒ caps meet the narrow-item rule (targeted ≥ 2σ, UHO not harmful) ⇒ ship checks proposed to the owner;
#    positive under 2σ ⇒ the narrow-item pool (owner decides); ≤ 0 ⇒ caps stay parked.
#  4 filler if time remains: gen 6.4 vs SF18 @1100 nodes (ours 250k), 500 g on fresh seeds 1331 and 1332 — the fresh-seed
#    baseline every next candidate's ship check needs (node-limited, deterministic).
# Timed legs must finish by Monday 18:00. Stop between legs: touch /mnt/e/chess_data/q134.stop. Never edit this file while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
KS4="$ND/build_uci/chess_engine_uci_ks4"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
STOP=/mnt/e/chess_data/q134.stop
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
BASE=THREAT_ORDER_W=100
CAPS=$BASE,POT_V2_WIN_OCBX=-42,POT_V2_WIN_OCBX_PC=3,POT_V2_WIN_ROOKE=-28,POT_V2_WIN_QNOQ=-27,POT_V2_WIN_QNOQ_MINOR=3,POT_V2_WIN_LONEMINOR=-64
G64_ENV="V2_PRESET=shipped THREAT_ORDER_W=100 THREAT_V2_PCT=100 THREAT_V2_HANGING=1 THREAT_V2_KING=1 THREAT_V2_PAWN_TARGETS=1 THREAT_V2_MG_PCT=50 THREAT_V2_EG_PCT=100 RFP_MARGIN=600"
TR=$O/transition.epd
CA=$O/capped.epd
TR_SRC=$O/rfp450.pgn,$O/rfp500.pgn,$O/rfp500_r.pgn,$O/rfp500_r2.pgn,$O/rfp550.pgn,$O/damp_uho.pgn
CA_SRC=$TR_SRC,$O/aa1.pgn,$O/aa2.pgn,$O/pin.pgn,$O/pin_b0.pgn
PY1="env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1"
cd "$ND" || exit 1
rm -f "$STOP"
echo "[q134] $(date) waiting for queues 132 and 133 to finish"
while pgrep -f "fastchess|vs_sf.py|_queue_13[23]\.sh" >/dev/null; do sleep 30; done
START=$(date '+%Y-%m-%d %H:%M:%S')
DEADLINE=$(date -d '2026-11-02 18:00' +%s)
echo "[q134] $START start (timed legs must finish by $(date -d @$DEADLINE); stop file $STOP)"

nodes() { printf "setoption name Knobs value $2\nbench\nquit\n" | "$1" 2>/dev/null | grep '^bench' | sed -n 's/.* nodes \([0-9]*\).*/\1/p'; }
db=$(nodes "$KS4" $BASE); dc=$(nodes "$KS4" $CAPS)
echo "[q134] ks4 binary: base $db (want 36991901) caps $dc (want 38998902)"
[ "$db" = 36991901 ] && [ "$dc" = 38998902 ] || { echo "[q134] ☠️ fingerprints failed — QUEUE 134 ABORTED"; exit 1; }

elo_of() { grep -E '^Elo:' $O/$1.log 2>/dev/null | tail -1 | awk '{gsub(",","",$4); print $2, $4}'; }
pool() { awk '{n++; s+=$1; v+=$2*$2} END{if(n==0){print "0 0 0"; exit} m=s/n; c=sqrt(v)/n; printf "%.2f %.2f %.2f\n", m, c, 1.96*m/c}'; }
ge8() { awk -v e="$(elo_of $1 | awk '{print $1}')" 'BEGIN{exit !(e != "" && e + 0 >= 8)}'; }
roles() { $PY1 timeout 900 bash "$R" pyrun selfplay/fastchess_report.py "$@" --p1 arm --roles 30 2>&1 \
  | grep -E "vs the field|Elo |roles by|^    (ahead|behind|level)" | sed 's/^/[q134]   /'; }

# 0: roles read of the transition runs.
echo "[q134] $(date) roles, caps_tr + caps_tr_r:"
roles $O/caps_tr.pgn $O/caps_tr_r.pgn

# 1 + 2: venue.
$PY1 timeout 1800 bash "$R" pyrun diagnostics/_make_transition_starts.py PGNS=$TR_SRC N=500 OUT=/tmp/q134_tr_check.epd > /tmp/q134_gen.log 2>&1
if cmp -s /tmp/q134_tr_check.epd $TR; then echo "[q134] generator check: default mode rebuilds transition.epd IDENTICALLY"
else echo "[q134] ☠️ generator check: default mode DIFFERS from transition.epd ($(grep -c . /tmp/q134_tr_check.epd) lines; $(tail -1 /tmp/q134_gen.log))"; fi
echo "[q134] $(date) building capped.epd"
rm -f $CA
$PY1 timeout 2400 bash "$R" pyrun diagnostics/_make_transition_starts.py PGNS=$CA_SRC N=500 OUT=$CA NEAR=caps WIN=-10,6 PERSIST=6 \
  PH=0,192 CP=30,250 MIN_PLY=16 SEED=1341 ID=ca EXCLUDE=$TR 2>&1 | sed 's/^/[q134]   /'
L=$(grep -c . $CA 2>/dev/null); L=${L:-0}
echo "[q134] capped.epd: $L starts"

COMMON="-repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
report() {
  echo "[q134] $(date) $1 done: $(grep -E '^Games:|^Elo:' $O/$1.log | tail -2 | tr '\n' ' ')"
  echo "[q134]   $(grep -E '^Ptnml' $O/$1.log | tail -1)  disconnects/stalls: $(grep -ciE 'disconnect|stall|not responsive' $O/$1.log)  time forfeits: $(grep -c 'loses on time' $O/$1.pgn)"
}
gate() {   # $1 tag, $2 seconds needed
  [ -f "$STOP" ] && { echo "[q134] $(date) skip $1: stop file present"; return 1; }
  local left=$(( DEADLINE - $(date +%s) ))
  [ $left -ge $2 ] || { echo "[q134] $(date) skip $1: only ${left}s before the deadline (needs $2)"; return 1; }
}
preflight() {   # $1 tag, $2 arm knobs, $3 control knobs, $4 openings
  rm -f /tmp/q134_pre.engine.log
  timeout 180 "$FC" -engine cmd="$KS4" name=arm option.OwnBook=false option.Knobs=$2 st=0.05 \
    -engine cmd="$KS4" name=ctl option.OwnBook=false option.Knobs=$3 st=0.05 \
    -openings file=$4 format=epd order=random -rounds 1 -games 2 -concurrency 1 \
    -log file=/tmp/q134_pre.engine.log level=info engine=true > /tmp/q134_pre.log 2>&1
  local W A C; W=$(grep -c "have option" /tmp/q134_pre.log)
  A=$(grep -oE " arm ---> info string knobs [A-Z0-9_=,-]+" /tmp/q134_pre.engine.log | grep -cxF " arm ---> info string knobs $2")
  C=$(grep -oE " ctl ---> info string knobs [A-Z0-9_=,-]+" /tmp/q134_pre.engine.log | grep -cxF " ctl ---> info string knobs $3")
  echo "[q134] $1 preflight: warnings $W, arm echoes $A, control echoes $C"
  [ "$W" = "0" ] && [ "$A" -ge 1 ] && [ "$C" -ge 1 ] || { echo "[q134] ☠️ $1 PREFLIGHT FAILED — skipped"; return 1; }
}
selfplay() {   # $1 tag, $2 arm knobs, $3 control knobs, $4 openings, $5 tc, $6 rounds, $7 seed, $8 seconds needed
  gate $1 $8 && preflight $1 "$2" "$3" $4 || return
  echo "[q134] $(date) $1 ($((2 * $6)) games, $5, $(basename $4))"
  nice -n 5 "$FC" -engine cmd="$KS4" name=arm option.OwnBook=false option.Knobs=$2 tc=$5 timemargin=250 \
    -engine cmd="$KS4" name=ctl option.OwnBook=false option.Knobs=$3 tc=$5 timemargin=250 \
    -openings file=$4 format=epd order=random $COMMON -srand $7 -rounds $6 -games 2 \
    -pgnout file=$O/$1.pgn notation=san nodes=true append=false > $O/$1.log 2>&1
  report $1
}

# 3: caps on the capped starts (each start once per run when there are fewer than 500).
if [ "$L" -ge 300 ]; then
  RN=$(( L < 500 ? L : 500 ))
  selfplay caps_ca "$CAPS" "$BASE" $CA 10+0.1 $RN 1341 10200
  [ -n "$(elo_of caps_ca)" ] && { echo "[q134] roles, caps_ca:"; roles $O/caps_ca.pgn; }
  if ge8 caps_ca; then
    selfplay caps_ca_r "$CAPS" "$BASE" $CA 10+0.1 $RN 1342 10200
    if [ -n "$(elo_of caps_ca_r)" ]; then
      echo "[q134] caps_ca pooled over two runs: $( { elo_of caps_ca; elo_of caps_ca_r; } | pool | awk '{printf "%+.1f ± %.1f (z %.2f)", $1, $2, $3}')"
      echo "[q134] roles, caps_ca + caps_ca_r:"; roles $O/caps_ca.pgn $O/caps_ca_r.pgn
    fi
  else echo "[q134] no caps_ca replicate: caps_ca below +8 or not run"; fi
else
  echo "[q134] ☠️ capped.epd has $L starts (< 300) — caps legs skipped"
fi

# 4: fresh-seed gen 6.4 SF18 baseline (Cython path; the .so must give gen 6.4's d10 nodes).
c=$(env $G64_ENV bash "$R" wac q134_ctl)
echo "[q134] cython d10 gen 6.4: $(echo "$c" | grep -E '^SOLVED|^NODES' | tr '\n' ' ')"
if echo "$c" | grep -q "NODES: 36991901"; then
  for s in 1331 1332; do
    gate g1100_g64_s$s 3600 && { echo "[q134] $(date) gauntlet g1100_g64_s$s"
      bash "$R" gauntlet 500 1100 3 g1100_g64_s$s $s $G64_ENV 2>&1 | grep -E "our score|games in" | sed 's/^/[q134]   /'; }
  done
else
  echo "[q134] ☠️ Cython fingerprint failed (want 36991901) — SF18 baseline skipped"
fi

echo "[q134] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | sed 's/^/[q134]   /'
echo "[q134] $(date) QUEUE 134 DONE"
