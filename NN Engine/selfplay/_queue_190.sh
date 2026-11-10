#!/bin/bash
# 2026-11-09 ~18:50, queue #190 (owner 11-09 "sure"): the overnight follow-on to #189, using the box until the owner is back
# (~10:30 on 11-10). It waits for #189 to finish, then:
#  A only if #189 skipped its stage 2 (c2q read below 0): c2r vs gen 7.1 anyway, as #189 registered it (2,000 g at 10+0.1,
#    seeds 189221-189228), then its Mediocre (seeds 189311 / 189312, 150 g each) and SF18 (seeds 189111-189113), on the seeds
#    #189 played ctl on. A DEVIATION from #189's stage-2 gate, decided before #189's read: the gate only saved box time, and
#    an eval correction is judged together with its margin (memory eval-accuracy-payoff-is-pruning), so c2q alone reading
#    about 0 is what that predicts and should not decide whether the margin is tested. c2r's bars are #189's.
#  B always: watch item W1 (LAZY-EVAL.md section 7), SF18 at equal nodes on six fresh seeds 190111-190116, gen 7.1 (ctl) vs
#    LAZY_V2=0 (off), 250 g per arm per seed, both arms of a seed before the next.
# Timed chunks start only from 00:00 to 09:20 on 11-10 (the last ends by ~10:00); SF18 chunks (node-limited) only before 10:15.
# REGISTERED before it runs:
#   c2r (if A runs): gain at >= +2 sigma; not harmful at >= -1 sigma; the tighter margin pays if c2r - c2q >= +10 Elo (a
#     screen). Prediction: c2r within +-10 of c2q. SF18 c2r - ctl > -1 sigma; Mediocre c2r - ctl not negative.
#   W1, lazy on - off, over the fresh sets of #189 and #190 (9 sets, none used for a decision): level if > -1 sigma (W1
#     closes); lazy eval costs strength at equal nodes if <= -2 sigma (raise with the owner); in between, W1 stays open.
#     Prediction: about -1 pp (#187: -1.02 pp over 8 sets, -2.08 on its 5 fresh ones).
# Stop: touch /mnt/e/chess_data/q190.stop (checked between chunks). It does not run if #189 was stopped (q189.stop present).
# Never edit this file or lazy4 while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
SF18="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/stockfish_18_linux/stockfish-ubuntu-x86-64-avx2"
MED="$ND/selfplay/mediocre_uci.sh"
O=/mnt/e/chess_data/fastchess
BIN="$ND/build_uci/chess_engine_uci_lazy4"
C2Q="CORR2=1,CORR2_WP=88,CORR2_WN=100"
C2R="$C2Q,RFP_MARGIN=500"
STOP=/mnt/e/chess_data/q190.stop
STOP189=/mnt/e/chess_data/q189.stop
L189=/mnt/e/chess_data/q189.log
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
MID=$(date -d '2026-11-10 00:00' +%s); TEND=$(date -d '2026-11-10 09:20' +%s); SEND=$(date -d '2026-11-10 10:15' +%s)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"
show() { sed "s/^/[q190]   /"; }
busy() { pgrep -f "fastchess-linux-x86-64/fastchess|vs_sf.py|_depth_residual_pass|_build_regret_set|_retune_fit|sts_test|tactical_test|_theme_movetime|chess_engine_uci_lzprobe2" >/dev/null; }
kn() { case $1 in ctl) echo "" ;; c2q) echo "$C2Q" ;; c2r) echo "$C2R" ;; off) echo "LAZY_V2=0" ;; esac; }
bn() { local k=""; [ -n "$1" ] && k="setoption name Knobs value $1"; printf '%s\nbench\nquit\n' "$k" | "$BIN" 2>/dev/null | grep -E '^bench|lazy eval' | sed -E 's/ time [0-9]+ ms nps [0-9]+//'; }

# ── wait for #189 ─────────────────────────────────────────────────────────────────────────────────────────────────────
echo "[q190] $(date '+%Y-%m-%d %H:%M:%S') queued; waiting for #189"
until grep -q 'QUEUE 189 DONE' "$L189" 2>/dev/null || ! pgrep -f "_queue_189.sh" >/dev/null; do
  [ -e "$STOP" ] && { echo "[q190] $(date) stopped while waiting: QUEUE 190 NOT RUN"; exit 0; }
  sleep 60
done
while busy; do sleep 30; done
START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[q190] $START start (#189's last line: $(tail -1 "$L189"))"
[ -e "$STOP189" ] && { echo "[q190] #189 was stopped (q189.stop present): QUEUE 190 NOT RUN"; exit 0; }
[ -e "$STOP" ] && { echo "[q190] stopped: QUEUE 190 NOT RUN"; exit 0; }

# ── 0 fingerprints ────────────────────────────────────────────────────────────────────────────────────────────────────
F0=$(bn ""); FR=$(bn "$C2R")
echo "[q190] default (want 247 / 39852210):"; echo "$F0" | show
echo "[q190] c2r (want 245 / 36768650, failed to cut 0):"; echo "$FR" | show
echo "$F0" | grep -q 'solved 247/300 nodes 39852210$' && echo "$FR" | grep -q 'solved 245/300 nodes 36768650$' \
  && echo "$FR" | grep -q 'failed to cut 0$' || { echo "[q190] ☠️ fingerprints or the failed-to-cut check — QUEUE 190 ABORTED"; exit 1; }

# ── helpers (as #189's; #190's own files are g18_q190_* / med_q190_*) ─────────────────────────────────────────────────
COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
rm -f $O/g18_q190_s* $O/med_q190_s*
wait_timed() {  # 0 when a timed chunk may start; 1 when timed games are over for this queue
  local n
  while :; do
    [ -e "$STOP" ] && return 1
    n=$(date +%s); [ $n -ge $TEND ] && return 1; [ $n -ge $MID ] && break
    sleep 60
  done
  while busy; do sleep 30; done
  return 0
}
tchunk() {  # tag, arm, seed
  local T=$1 a=$2 seed=$3
  [ -e "$STOP" ] && return
  wait_timed || { echo "[q190] $(date) skip $T seed $seed (stop, or past 09:20)"; return; }
  nice -n 5 "$FC" -engine "cmd=$BIN" name=arm option.OwnBook=false "option.Knobs=$(kn $a)" tc=10+0.1 timemargin=250 \
    -engine "cmd=$BIN" name=ctl option.OwnBook=false tc=10+0.1 timemargin=250 \
    $COMMON -srand $seed -rounds 125 -games 2 -pgnout file=$O/${T}_s$seed.pgn notation=san nodes=true append=false > $O/${T}_s$seed.log 2>&1
  echo "[q190] $(date) $T seed $seed: $(grep -E '^Elo:' $O/${T}_s$seed.log | tail -1) · $(grep -E '^Games:' $O/${T}_s$seed.log | tail -1 | cut -d, -f1) · warnings $(grep -c 'have option' $O/${T}_s$seed.log) · disconnects/stalls $(grep -ciE 'disconnect|stall|not responsive' $O/${T}_s$seed.log) · time forfeits $(grep -c 'loses on time' $O/${T}_s$seed.pgn)"
}
med_chunk() {  # seed, arm
  local seed=$1 a=$2 k f=$O/med_q190_s$1_$2.log
  [ -e "$STOP" ] && return
  wait_timed || { echo "[q190] $(date) skip Mediocre $seed $a (stop, or past 09:20)"; return; }
  k=$(kn $a); [ -n "$k" ] && k="option.Knobs=$k"
  nice -n 5 "$FC" -engine "cmd=$BIN" name=ours option.OwnBook=false $k tc=10+0.1 timemargin=250 \
    -engine "cmd=$MED" name=mediocre tc=10+0.1 timemargin=1000 \
    $COMMON -srand $seed -rounds 75 -games 2 -pgnout file=$O/med_q190_s${seed}_$a.pgn notation=san append=false > "$f" 2>&1
  echo "[q190] $(date) Mediocre seed $seed $a: $(grep -E '^Games:' "$f" | tail -1) · warnings $(grep -c 'have option' "$f") · disconnects/stalls $(grep -ciE 'disconnect|stall|not responsive' "$f") · time losses $(grep -c 'loses on time' "$f")"
}
sf_chunk() {  # seed, arm
  local seed=$1 a=$2 k f=$O/g18_q190_s$1_$2.log
  [ -e "$STOP" ] && return
  [ $(date +%s) -ge $SEND ] && { echo "[q190] $(date) skip SF18 seed $seed $a (past 10:15)"; return; }
  while busy; do sleep 30; done
  k=$(kn $a); [ -n "$k" ] && k="option.Knobs=$k"
  nice -n 5 "$FC" -engine "cmd=$BIN" name=ours option.OwnBook=false $k nodes=250000 \
    -engine "cmd=$SF18" name=sf18 option.Threads=1 option.Hash=16 nodes=1100 \
    -each tc=inf $COMMON -srand $seed -rounds 125 -games 2 -pgnout file=$O/g18_q190_s${seed}_$a.pgn notation=san nodes=true append=false > "$f" 2>&1
  echo "[q190] $(date) SF18 seed $seed $a: $(grep -E '^Games:' "$f" | tail -1) · warnings $(grep -c 'have option' "$f") · disconnects/stalls $(grep -ciE 'disconnect|stall|not responsive' "$f")"
}

# ── A c2r, only if #189 skipped its stage 2 ───────────────────────────────────────────────────────────────────────────
A=0; grep -q 'no stage 2' "$L189" && A=1
if [ $A = 1 ]; then
  echo "[q190] #189 skipped its stage 2: c2r runs here (a deviation from #189's gate, registered above)"
  for c in 1 2 3 4 5 6 7 8; do tchunk c2r_10p1 c2r $(( 189220 + c )); done
  med_chunk 189311 c2r; med_chunk 189312 c2r
  for s in 189111 189112 189113; do sf_chunk $s c2r; done
else
  echo "[q190] #189 ran its stage 2 (or did not reach the decision): no c2r here"
fi

# ── B W1: lazy eval on vs off at equal nodes, fresh seeds ─────────────────────────────────────────────────────────────
echo "[q190] $(date) W1, SF18 at equal nodes (node-limited), ctl vs off, seeds 190111-190116"
for s in 190111 190112 190113 190114 190115 190116; do
  if [ $(( s % 2 )) = 1 ]; then sf_chunk $s ctl; sf_chunk $s off; else sf_chunk $s off; sf_chunk $s ctl; fi
done

# ── reads ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
ELO=""
tread() {  # tag, label, rule
  local f=$(ls $O/$1_s*.pgn 2>/dev/null) out
  [ -n "$f" ] || { echo "[q190] $2: no games"; ELO=""; return; }
  out=$(timeout 900 bash "$R" pyrun selfplay/fastchess_report.py $f --p1 arm 2>&1 | sed -n '/vs the field/,$p')
  echo "[q190] $(date) REGISTERED READ, $2 ($3):"
  echo "$out" | show
  ELO=$(echo "$out" | grep -m1 -E 'Elo [-+]?[0-9.]+' | awk '{gsub(/[()]/,""); gsub(/\+/,""); print $2}')
  echo "$out" | grep -m1 -E 'Elo [-+]?[0-9.]+' | awk '{gsub(/[()]/,""); gsub(/\+/,""); printf "[q190]   z = %.2f\n", $2 / (($6 - $4) / 3.92)}'
}
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
if [ $A = 1 ]; then
  tread c2r_10p1 "c2r vs gen 7.1 at 10+0.1" "gain at +2 sigma, not harmful at >= -1 sigma"; ER=$ELO
  tread c2q_10p1 "c2q vs gen 7.1 at 10+0.1, #189's games read again" "for the screen"; EQ=$ELO
  [ -n "$ER" ] && [ -n "$EQ" ] && awk -v r="$ER" -v q="$EQ" 'BEGIN { printf "[q190]   c2r - c2q = %+.1f Elo (the tighter margin pays if >= +10; a screen)\n", r - q }'
fi
for f in $O/g18_q189_s*_*.log $O/g18_q190_s*_*.log; do [ -e "$f" ] || continue; b=$(basename $f .log); b=${b#g18_q*_s}; row ${b%%_*} ${b#*_} $f; done > /tmp/q190_sf.txt
for f in $O/g18_q190_s*_*.log; do [ -e "$f" ] || continue; b=$(basename $f .log); b=${b#g18_q*_s}; row ${b%%_*} ${b#*_} $f; done > /tmp/q190_sf_new.txt
for f in $O/med_q189_s*_*.log $O/med_q190_s*_*.log; do [ -e "$f" ] || continue; b=$(basename $f .log); b=${b#med_q*_s}; row ${b%%_*} ${b#*_} $f; done > /tmp/q190_med.txt
echo "[q190] $(date) REGISTERED READS, SF18 at equal nodes (c2r - ctl > -1 sigma; W1 lazy on - off: level > -1 sigma, costs <= -2 sigma):"
{ [ $A = 1 ] && cmp c2r ctl /tmp/q190_sf.txt SF18
  cmp ctl off /tmp/q190_sf.txt "SF18 W1 (#189 + #190 fresh sets)"
  cmp ctl off /tmp/q190_sf_new.txt "SF18 W1 (#190 sets only)"; } | show
if [ $A = 1 ]; then
  echo "[q190] REGISTERED READ, Mediocre (c2r - ctl not negative):"
  cmp c2r ctl /tmp/q190_med.txt Mediocre | show
fi
echo "[q190] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | show
echo "[q190] $(date) QUEUE 190 DONE"
