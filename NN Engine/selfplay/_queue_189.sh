#!/bin/bash
# 2026-11-09 ~14:20, queue #189 (owner 11-09 "ok" to the lean margins re-sweep): correction history on gen 7.1, with lazy eval
# deciding on the corrected value (ac7ba67). Binary build_uci/chess_engine_uci_lazy4 (plain build of the current sources; its
# default == gen 7.1, 247 / 39,852,210). Arms are Knobs on lazy4; the control is the default on the same binary.
#   c2q = CORR2=1,CORR2_WP=88,CORR2_WN=100: quarter weights (on gen 6.2 at RFP 1000: +7.6 and +3.5 over 1,000 g each)
#   c2r = c2q + RFP_MARGIN=500: does the corrected static eval carry a tighter reverse-futility margin? (600 is gen 7.1's)
#  1 timed self-play c2q vs gen 7.1, 2,000 g at 10+0.1 (8 chunks of 250, fresh seeds 189201-189208)
#  2 only if 1 reads >= 0 Elo: c2r vs gen 7.1, 2,000 g (seeds 189221-189228)
#  3 node-limited, in the 21:00-00:00 window (the owner's games, when timed games do not run): SF18 at equal nodes (ours 250k,
#    SF18 1,100), 250 g per arm per seed, seeds 189111-189113, arms ctl / c2q / c2r (if 2 runs) / off (LAZY_V2=0, watch W1)
#  4 Mediocre at 10+0.1, 300 g per arm: ctl / c2q / c2r (if 2 ran)
# Timed chunks start only before 20:15 today or from 00:00 to 10:00 on 11-10.
# REGISTERED before it runs (ship rule: self-play >= +2 sigma, SF18 > -1 sigma, Mediocre not negative):
#   c2q gains if its Elo vs gen 7.1 >= +2 sigma; not harmful at >= -1 sigma. Prediction 0 to +10.
#   c2r gains if >= +2 sigma; the tighter margin pays if c2r - c2q >= +10 Elo (separate runs, a screen, not a verdict).
#     Prediction: c2r within +-10 of c2q.
#   SF18: arm - ctl > -1 sigma on the seeds both played. W1: off - ctl reported (#187: +1.0 pp over 8 sets).
#   Mediocre: arm - ctl not negative.
#   Every corrhist arm's bench must print "failed to cut 0".
# Stop: touch /mnt/e/chess_data/q189.stop (checked between chunks). Never edit this file or lazy4 while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
SF18="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/stockfish_18_linux/stockfish-ubuntu-x86-64-avx2"
MED="$ND/selfplay/mediocre_uci.sh"
O=/mnt/e/chess_data/fastchess
BIN="$ND/build_uci/chess_engine_uci_lazy4"
C2Q="CORR2=1,CORR2_WP=88,CORR2_WN=100"
C2R="$C2Q,RFP_MARGIN=500"
STOP=/mnt/e/chess_data/q189.stop
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
EVE=$(date -d '2026-11-09 20:15' +%s); MID=$(date -d '2026-11-10 00:00' +%s); END=$(date -d '2026-11-10 10:00' +%s)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"
show() { sed "s/^/[q189]   /"; }
busy() { pgrep -f "fastchess-linux-x86-64/fastchess|vs_sf.py|_depth_residual_pass|_build_regret_set|_retune_fit|sts_test|tactical_test|_theme_movetime|chess_engine_uci_lzprobe2" >/dev/null; }
kn() { case $1 in ctl) echo "" ;; c2q) echo "$C2Q" ;; c2r) echo "$C2R" ;; off) echo "LAZY_V2=0" ;; esac; }
bn() { local k=""; [ -n "$1" ] && k="setoption name Knobs value $1"; printf '%s\nbench\nquit\n' "$k" | "$BIN" 2>/dev/null | grep -E '^bench|lazy eval' | sed -E 's/ time [0-9]+ ms nps [0-9]+//'; }
START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[q189] $START start"
busy && { echo "[q189] ☠️ a games or bench job is running — QUEUE 189 ABORTED"; exit 1; }

# ── 0 fingerprints ────────────────────────────────────────────────────────────────────────────────────────────────────
F0=$(bn ""); FQ=$(bn "$C2Q"); FR=$(bn "$C2R")
echo "[q189] default (want 247 / 39852210):"; echo "$F0" | show
echo "[q189] c2q:"; echo "$FQ" | show
echo "[q189] c2r:"; echo "$FR" | show
echo "$F0" | grep -q 'solved 247/300 nodes 39852210$' && echo "$FQ" | grep -q 'failed to cut 0$' && echo "$FR" | grep -q 'failed to cut 0$' \
  && [ "$(echo "$FQ" | grep '^bench')" != "$(echo "$FR" | grep '^bench')" ] && [ "$(echo "$FQ" | grep '^bench')" != "$(echo "$F0" | grep '^bench')" ] \
  || { echo "[q189] ☠️ fingerprints or the failed-to-cut check — QUEUE 189 ABORTED"; exit 1; }
rm -f /tmp/q189_pre.engine.log
timeout 180 "$FC" -engine "cmd=$BIN" name=arm option.OwnBook=false "option.Knobs=$C2R" st=0.05 \
  -engine "cmd=$BIN" name=ctl option.OwnBook=false st=0.05 \
  -openings file=$O/openings_uho.epd format=epd order=random -rounds 1 -games 2 -concurrency 1 \
  -log file=/tmp/q189_pre.engine.log level=info engine=true > /tmp/q189_pre.log 2>&1
W=$(grep -c 'have option' /tmp/q189_pre.log)
echo "[q189] preflight: $(grep -E '^Games:' /tmp/q189_pre.log | tail -1) · warnings $W · arm echoes $(grep -cF "knobs $C2R" /tmp/q189_pre.engine.log)"
[ "$W" = 0 ] || { echo "[q189] ☠️ preflight warnings — QUEUE 189 ABORTED"; exit 1; }

# ── helpers ───────────────────────────────────────────────────────────────────────────────────────────────────────────
COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
S2=0
sf_chunk() {  # seed, arm; skipped when that chunk is already complete
  local seed=$1 a=$2 k f=$O/g18_q189_s$1_$2.log
  [ -e "$STOP" ] && return
  grep -qE '^Games: 250,' "$f" 2>/dev/null && return
  k=$(kn $a); [ -n "$k" ] && k="option.Knobs=$k"
  nice -n 5 "$FC" -engine "cmd=$BIN" name=ours option.OwnBook=false $k nodes=250000 \
    -engine "cmd=$SF18" name=sf18 option.Threads=1 option.Hash=16 nodes=1100 \
    -each tc=inf $COMMON -srand $seed -rounds 125 -games 2 -pgnout file=$O/g18_q189_s${seed}_$a.pgn notation=san nodes=true append=false > "$f" 2>&1
  echo "[q189] $(date) SF18 seed $seed $a: $(grep -E '^Games:' "$f" | tail -1) · warnings $(grep -c 'have option' "$f") · disconnects/stalls $(grep -ciE 'disconnect|stall|not responsive' "$f")"
}
sf_block() {  # every arm on every seed, the order rotating by seed
  local arms="ctl c2q off"; [ $S2 = 1 ] && arms="ctl c2q c2r off"
  echo "[q189] $(date) SF18 at equal nodes (node-limited), arms: $arms"
  set -- $arms
  for a in "$@"; do sf_chunk 189111 $a; done
  for (( i = $#; i >= 1; i-- )); do sf_chunk 189112 ${!i}; done
  for a in "${@:2}" "$1"; do sf_chunk 189113 $a; done
}
timed_ok() { local n=$(date +%s); [ $n -lt $EVE ] || { [ $n -ge $MID ] && [ $n -lt $END ]; }; }
SF_RAN=0
wait_timed() {  # 0 when a timed chunk may start; 1 when timed games are over for this queue
  while ! timed_ok; do
    [ -e "$STOP" ] && return 1
    [ $(date +%s) -ge $END ] && return 1
    [ $SF_RAN = 0 ] && { SF_RAN=1; while busy; do sleep 30; done; sf_block; }
    sleep 60
  done
  while busy; do sleep 30; done
  return 0
}
tchunk() {  # tag, arm, seed
  local T=$1 a=$2 seed=$3
  [ -e "$STOP" ] && return
  wait_timed || { echo "[q189] $(date) skip $T seed $seed (stop, or past 10:00)"; return; }
  nice -n 5 "$FC" -engine "cmd=$BIN" name=arm option.OwnBook=false "option.Knobs=$(kn $a)" tc=10+0.1 timemargin=250 \
    -engine "cmd=$BIN" name=ctl option.OwnBook=false tc=10+0.1 timemargin=250 \
    $COMMON -srand $seed -rounds 125 -games 2 -pgnout file=$O/${T}_s$seed.pgn notation=san nodes=true append=false > $O/${T}_s$seed.log 2>&1
  echo "[q189] $(date) $T seed $seed: $(grep -E '^Elo:' $O/${T}_s$seed.log | tail -1) · $(grep -E '^Games:' $O/${T}_s$seed.log | tail -1 | cut -d, -f1) · warnings $(grep -c 'have option' $O/${T}_s$seed.log) · disconnects/stalls $(grep -ciE 'disconnect|stall|not responsive' $O/${T}_s$seed.log) · time forfeits $(grep -c 'loses on time' $O/${T}_s$seed.pgn)"
}
ELO=""
tread() {  # tag, label, rule
  local f=$(ls $O/$1_s*.pgn 2>/dev/null) out
  [ -n "$f" ] || { echo "[q189] $2: no games"; ELO=""; return; }
  out=$(timeout 900 bash "$R" pyrun selfplay/fastchess_report.py $f --p1 arm 2>&1 | sed -n '/vs the field/,$p')
  echo "[q189] $(date) REGISTERED READ, $2 ($3):"
  echo "$out" | show
  ELO=$(echo "$out" | grep -m1 -E 'Elo [-+]?[0-9.]+' | awk '{gsub(/[()]/,""); gsub(/\+/,""); print $2}')
  echo "$out" | grep -m1 -E 'Elo [-+]?[0-9.]+' | awk '{gsub(/[()]/,""); gsub(/\+/,""); printf "[q189]   z = %.2f\n", $2 / (($6 - $4) / 3.92)}'
}

# ── 1 c2q vs gen 7.1 ──────────────────────────────────────────────────────────────────────────────────────────────────
rm -f $O/c2q_10p1_s* $O/c2r_10p1_s* $O/g18_q189_s* $O/med_q189_*
for c in 1 2 3 4 5 6 7 8; do tchunk c2q_10p1 c2q $(( 189200 + c )); done
tread c2q_10p1 "c2q vs gen 7.1 at 10+0.1" "gain at +2 sigma, not harmful at >= -1 sigma"
if [ -n "$ELO" ] && awk -v e="$ELO" 'BEGIN { exit !(e >= 0) }'; then S2=1; echo "[q189] c2q reads >= 0: stage 2 (c2r) runs"
else echo "[q189] c2q reads below 0 (or did not finish): no stage 2"; fi

# ── 2 c2r vs gen 7.1 ──────────────────────────────────────────────────────────────────────────────────────────────────
if [ $S2 = 1 ]; then
  for c in 1 2 3 4 5 6 7 8; do tchunk c2r_10p1 c2r $(( 189220 + c )); done
  tread c2r_10p1 "c2r vs gen 7.1 at 10+0.1" "gain at +2 sigma; c2r - c2q >= +10 = the tighter margin pays (screen)"
fi

# ── 4 Mediocre ────────────────────────────────────────────────────────────────────────────────────────────────────────
med_chunk() {  # seed, arm
  local seed=$1 a=$2 k f=$O/med_q189_s$1_$2.log
  [ -e "$STOP" ] && return
  wait_timed || { echo "[q189] $(date) skip Mediocre $seed $a (stop, or past 10:00)"; return; }
  k=$(kn $a); [ -n "$k" ] && k="option.Knobs=$k"
  nice -n 5 "$FC" -engine "cmd=$BIN" name=ours option.OwnBook=false $k tc=10+0.1 timemargin=250 \
    -engine "cmd=$MED" name=mediocre tc=10+0.1 timemargin=1000 \
    $COMMON -srand $seed -rounds 75 -games 2 -pgnout file=$O/med_q189_s${seed}_$a.pgn notation=san append=false > "$f" 2>&1
  echo "[q189] $(date) Mediocre seed $seed $a: $(grep -E '^Games:' "$f" | tail -1) · warnings $(grep -c 'have option' "$f") · disconnects/stalls $(grep -ciE 'disconnect|stall|not responsive' "$f") · time losses $(grep -c 'loses on time' "$f")"
}
MARMS="ctl c2q"; [ $S2 = 1 ] && MARMS="ctl c2q c2r"
set -- $MARMS
for a in "$@"; do med_chunk 189311 $a; done
for (( i = $#; i >= 1; i-- )); do med_chunk 189312 ${!i}; done

# ── 3 SF18 (if the evening window did not already run it all) ─────────────────────────────────────────────────────────
while busy; do sleep 30; done
sf_block

# ── reads ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
row() {  # seed, arm, log: complete chunks only
  grep -E '^Games:' "$3" 2>/dev/null | tail -1 | awk -v s=$1 -v a=$2 '{ gsub(/,/, ""); for (i = 1; i <= NF; i++) { if ($i == "Games:") n = $(i+1); if ($i == "Wins:") w = $(i+1); if ($i == "Losses:") l = $(i+1); if ($i == "Draws:") d = $(i+1) } if (n > 0) print s, a, n, w, l, d }'
}
cmp() {  # A, B, rows file, label: A - B over the seeds both played
  awk -v A=$1 -v B=$2 -v lab="$4" '
    { if ($2 == A) { hA[$1] = 1; nA[$1] = $3; wA[$1] = $4; dA[$1] = $6 } if ($2 == B) { hB[$1] = 1; nB[$1] = $3; wB[$1] = $4; dB[$1] = $6 } }
    END {
      for (k in hA) if (hB[k]) { na += nA[k]; wa += wA[k]; da += dA[k]; nb += nB[k]; wb += wB[k]; db += dB[k]; seeds++ }
      if (na == 0 || nb == 0) { printf "  %s %-4s - %-4s: no shared seeds\n", lab, A, B; exit }
      sa = (wa + da / 2) / na; va = ((wa + da / 4) / na - sa * sa) / na
      sb = (wb + db / 2) / nb; vb = ((wb + db / 4) / nb - sb * sb) / nb
      d = sa - sb; se = sqrt(va + vb)
      printf "  %s %-4s - %-4s (%d seeds, %d / %d games): %.1f%% vs %.1f%%  %+.2f pp  ± %.2f (95%%)  z %+.2f\n", lab, A, B, seeds, na, nb, 100 * sa, 100 * sb, 100 * d, 196 * se, d / se
    }' $3
}
for f in $O/g18_q189_s*_*.log; do [ -e "$f" ] || continue; b=$(basename $f .log); b=${b#g18_q189_s}; row ${b%%_*} ${b#*_} $f; done > /tmp/q189_sf.txt
for f in $O/med_q189_s*_*.log; do [ -e "$f" ] || continue; b=$(basename $f .log); b=${b#med_q189_s}; row ${b%%_*} ${b#*_} $f; done > /tmp/q189_med.txt
echo "[q189] $(date) REGISTERED READS, SF18 at equal nodes (arm - ctl > -1 sigma; W1 = off - ctl, reported):"
{ cmp c2q ctl /tmp/q189_sf.txt SF18; cmp c2r ctl /tmp/q189_sf.txt SF18; cmp off ctl /tmp/q189_sf.txt "SF18 W1"; } | show
echo "[q189] REGISTERED READS, Mediocre (arm - ctl not negative):"
{ cmp c2q ctl /tmp/q189_med.txt Mediocre; cmp c2r ctl /tmp/q189_med.txt Mediocre; } | show
echo "[q189] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | show
echo "[q189] $(date) QUEUE 189 DONE"
