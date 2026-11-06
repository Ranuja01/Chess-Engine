#!/bin/bash
# 2026-11-06, queue #168 (owner 11-06: run it after arm S's check): does MORE DATA buy Elo, or is it a ghost? The accuracy learning
# curve says more rows help, but accuracy has not reliably become Elo, so this plays the curve. Nothing here needs new data.
# Starts when #167's timed legs end, and runs beside its A2 pass (node-limited, safe at any hour).
#  1 recipe S again (same rows, stress slices at 0.5), fitted on 4 quarter-size and 2 half-size subsets of the own-play training
#    games that do not overlap within a size, plus the full set, which must reproduce fit_hybrid_v3s.npz (the check on the new path).
#  2 per arm: tables, closure (>= 70% realised), symmetry, UCI == Cython, as in #166.
#  3 50k nodes vs gen 6.7, 500 g chunks, interleaved, every chunk a fresh seed: quarters 1,000 g each, halves 2,000 g each, S 4,000 g.
#    12,000 g, about 6 h. S's 2,000 g from #166 are left out: S was picked on them.
# Registered: 4,000 g per size is about +-10 at 95%. Read as level(100%) - level(25%), about 1.7 sigma at the +12 bar:
#    >= +12 (>= +6 per doubling, so about +10 by 1M positions): the data seems to matter. Next: refit with A2's rows, screen it directly,
#           and only if that holds up spend a day and night on more data.
#    +6 ... +12: no dedicated time; A2 finishes in idle time and its fit gets a normal screen.
#    < +6: more rows are a ghost. Stop adding rows and go on to the static ladder.
#    Expected: flat. A go/no-go for a cheap step, not a proof: 100% is one table, and 50k nodes is far below play depth (~360k).
#  The spread among the 4 quarter tables also sizes how much near-equal tables differ, which bears on S - N (+23 +- 10).
#  4 from #167's end, the A2 d10 pass goes on to the end of the file (it stops at 20:00 there), at most 3 processes, cut-off 11-07 08:00.
# Stop: touch /mnt/e/chess_data/q168.stop (checked between steps and chunks). Never edit this file while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
D=/mnt/e/chess_data/retune
BIN="$ND/build_uci/chess_engine_uci_g67pgo"
LOGD=/mnt/e/chess_data/q168
STOP=/mnt/e/chess_data/q168.stop
PY=/home/ranuja/anaconda3/bin/python
CUTOFF=$(date -d '2026-11-07 08:00' +%s)
TAG=ours1105
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"; mkdir -p "$LOGD"
show() { sed "s/^/[q168]   /"; }
stopped() { [ -e "$STOP" ] && { echo "[q168] $(date) stop file present before $1 — QUEUE 168 DONE"; exit 0; }; }
echo "[q168] $(date) waiting for #167's timed legs to end (stop: $STOP)"
until grep -qE 'A2 d10 again: |QUEUE 167 (DONE|ABORTED)' /mnt/e/chess_data/q167.log 2>/dev/null; do
  [ -e "$STOP" ] && { echo "[q168] stop file present — QUEUE 168 DONE (nothing run)"; exit 0; }
  sleep 60
done
START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[q168] $START start"

# ── 4 (in the background) the A2 pass after #167 has stopped at its cut-off ─────────────────────────────────────────────
late() { [ -e "$STOP" ] || [ $(date +%s) -ge $CUTOFF ]; }
job() {
  local rc
  while :; do
    timeout 3600 env V2_PRESET=shipped PRESET=LONG_FORMAT MAX_DEPTH=10 USE_OPENING_BOOK=0 RESIGN_THRESHOLD=-99999999 \
      bash "$R" pyrun diagnostics/_depth_residual_pass.py IN="$2" OUT="$3" SHARD=$4 CHUNK=800 MOVE=1 >> "$LOGD/$1.log" 2>&1
    rc=$?
    [ $rc -ne 3 ] && break
    late && { echo "[q168] $(date) $1 paused at a chunk boundary (stop or cut-off)"; return; }
  done
  echo "[q168] $(date) $1 done (rc $rc)"
}
(
  until grep -qE 'QUEUE 167 (DONE|ABORTED)' /mnt/e/chess_data/q167.log 2>/dev/null; do
    late && exit 0
    sleep 60
  done
  sleep 20
  JOBS=()
  for p in $(ls $D/A2_fens_sf18_s*.csv 2>/dev/null); do
    i=$(basename $p .csv | sed 's/.*_s//')
    for k in 0 1 2; do JOBS+=("A2_f${i}_k$k|$p|$D/A2_${TAG}_d10_f$i.csv|$k/3"); done
  done
  echo "[q168] $(date) A2 d10 goes on: ${#JOBS[@]} jobs · rows so far $(cat $D/A2_${TAG}_d10_f*.csv 2>/dev/null | grep -vc '^fen,')"
  for j in "${JOBS[@]}"; do
    IFS='|' read -r name in out shard <<< "$j"
    late && { echo "[q168] $(date) not starting $name: stop or cut-off"; continue; }
    while [ $(jobs -rp | wc -l) -ge 3 ]; do wait -n; done
    job "$name" "$in" "$out" "$shard" &
  done
  wait
  echo "[q168] $(date) A2 d10 rows: $(cat $D/A2_${TAG}_d10_f*.csv 2>/dev/null | grep -vc '^fen,') / ~399k — A2 PASS OVER"
) &
A2PID=$!

# ── 1 the fits ───────────────────────────────────────────────────────────────────────────────────────────────────────
stopped "the fits"
TIME=""; [ -x /usr/bin/time ] && TIME="/usr/bin/time -v"
LCS="0.25@0,0.25@0.25,0.25@0.5,0.25@0.75,0.5@0,0.5@0.5,1"
echo "[q168] $(date) fits: $LCS"
[ -s $D/fit_hybrid_v3lc.npz ] || $TIME bash "$R" pyrun diagnostics/_retune_fit.py EXPERIMENT=hybrid MODES=hybrid VAL=1 WIN_AWARE=1 LAMBDAS=0.03 \
  CORPUS=corpus_v3.csv FEAT=feat_g67 EXTRA_SCOPES=1 TRAIN_N=500000 VAL_N=120000 DEPTH_SLICES=A,GAUNT,EGX,CUR,VARG,ODDS,STRIP \
  SLICE_W=VARG:0.5,ODDS:0.5,STRIP:0.5 LC=$LCS TAG=_v3lc > $LOGD/fit.log 2> $LOGD/fit.err
if [ ! -s $D/fit_hybrid_v3lc.npz ]; then
  echo "[q168] ☠️ the fit failed — QUEUE 168 ABORTED"
  grep -vE '^\s+(Command|User|System|Percent|Elapsed|Average|Maximum|Major|Minor|Voluntary|Involuntary|Swaps|File|Socket|Signals|Page|Exit)' $LOGD/fit.err | tail -6 | show
  wait $A2PID; exit 1
fi
echo "[q168] $(date) fit ok · peak RSS $(grep -E 'Maximum resident' $LOGD/fit.err | grep -oE '[0-9]+$') kB"
grep -E "rows:|design:|LC [0-9.@]+:|hybrid  λ|^  scope|static all|depth all|depth (mg|mgl|egl|eg) |val CUR|val GAUNT|val EGX|hold |rror|Trace" $LOGD/fit.log | show
$PY - <<EOF 2>&1 | show
import numpy as np
a = np.load("$D/fit_hybrid_v3lc.npz"); b = np.load("$D/fit_hybrid_v3s.npz")
print("labels:", list(a["labels"]))
ib = {n: i for i, n in enumerate(b["names"])}
da, db = a["deltas"][-1], b["deltas"][0]
idx = [ib[n] for n in a["names"]]
print("full-set arm vs fit_hybrid_v3s (want about 0): max |d| %.3f mp over %d cells" % (np.max(np.abs(da[:len(idx)] - db[idx])), len(idx)))
d = a["deltas"][:, :len(idx)]
for i in range(6):
    print("arm %d: rms move against the full fit %.1f mp" % (i, np.sqrt(np.mean((d[i] - d[-1]) ** 2))))
EOF
stopped "the gates"

# ── 2 gates ──────────────────────────────────────────────────────────────────────────────────────────────────────────
PASS=""
for i in 0 1 2 3 4 5; do
  t=lc$i
  rm -rf $D/arm_$t
  env V2_PRESET=shipped bash "$R" pyrun diagnostics/_retune_export.py MODE=tables NPZ=fit_hybrid_v3lc.npz ARM_I=$i OUT=$D/arm_$t > $LOGD/tables_$t.log 2>&1
  grep -E "values:|⚠️|rror" $LOGD/tables_$t.log | sed "s/^/[q168] $t /"
  [ -s $D/arm_$t/env.txt ] || { echo "[q168] $t ✗ no tables — dropped"; continue; }
  ENVL=$(cat $D/arm_$t/env.txt)
  env V2_PRESET=shipped $ENVL bash "$R" pyrun diagnostics/_retune_export.py MODE=verify NPZ=fit_hybrid_v3lc.npz ARM_I=$i ARMDIR=$t N=3000 \
    FEAT=feat_g67 WIN_AWARE=1 > $LOGD/verify_$t.log 2>&1
  grep -E "skipped|REALISED|rror" $LOGD/verify_$t.log | sed "s/^/[q168] $t /"
  set -- $(grep REALISED $LOGD/verify_$t.log | grep -oE '\(-?[0-9.]+%\)' | tr -d '()%')
  awk -v a="$1" -v b="$2" 'BEGIN{exit !(a < 0 && b/a >= 0.7)}' || { echo "[q168] $t ✗ closure gate (predicted $1% / engine $2%) — dropped"; continue; }
  SYM=$(env V2_PRESET=shipped $ENVL bash "$R" pyrun diagnostics/_eval_symmetry.py N=4000 2>&1 | grep -E "violations")
  echo "[q168] $t symmetry: $(echo "$SYM" | tr '\n' ' ')"
  [ "$(echo "$SYM" | grep -c 'violations 0 ')" = "2" ] || { echo "[q168] $t ✗ symmetry — dropped"; continue; }
  KN=$(tr -s ' \n' ',' < $D/arm_$t/env.txt | sed 's/,$//')
  U=$(printf 'setoption name Knobs value %s\nbench\nquit\n' "$KN" | "$BIN" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+')
  W=$(env V2_PRESET=shipped $ENVL bash "$R" wac fp_$t 2>&1 | grep -E "^SOLVED|^NODES" | tr '\n' ' ' | grep -oE 'Solved [0-9]+/300 NODES: [0-9]+')
  un=$(echo "$U" | grep -oE '[0-9]+$'); wn=$(echo "$W" | grep -oE '[0-9]+$')
  echo "[q168] $t UCI bench: $U · Cython WAC: $W"
  [ -n "$un" ] && [ "$un" = "$wn" ] && [ "$un" != 36859850 ] || { echo "[q168] $t ✗ UCI arm ≠ Cython arm (or knobs dead) — dropped"; continue; }
  echo "$KN" > $LOGD/knobs_$t.txt
  PASS="$PASS $t"
done
cp /mnt/e/chess_data/q166/knobs_v3s.txt $LOGD/knobs_s.txt
echo "[q168] $(date) arms for the games:$PASS s"

# ── 3 games ──────────────────────────────────────────────────────────────────────────────────────────────────────────
GC="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
has() { [ -s $LOGD/knobs_$1.txt ]; }
n=0
play() {     # $1 arm, $2 chunk
  has $1 || return 0
  n=$(( n + 1 ))
  stopped "$1 chunk $2"
  local T=lcg_$1
  nice -n 5 "$FC" -engine "cmd=$BIN" name=arm option.OwnBook=false "option.Knobs=$(cat $LOGD/knobs_$1.txt)" nodes=50000 \
    -engine "cmd=$BIN" name=ctl option.OwnBook=false nodes=50000 $GC -srand $(( 168000 + n )) -rounds 250 -games 2 \
    -pgnout file=$O/${T}_c$2.pgn notation=san nodes=true append=false > $O/${T}_c$2.log 2>&1
  echo "[q168] $(date) $T chunk $2: $(grep -E '^Elo:' $O/${T}_c$2.log | tail -1) · warnings $(grep -c 'have option' $O/${T}_c$2.log) · stalls $(grep -ciE 'disconnect|stall|not responsive' $O/${T}_c$2.log)"
}
rm -f $O/lcg_*_c*.pgn $O/lcg_*_c*.log
for r in 1 2; do
  for a in lc0 lc1 lc2 lc3 lc4 lc5; do play $a $r; done
  play s $(( 2 * r - 1 )); play s $(( 2 * r ))
done
for r in 3 4; do
  play lc4 $r; play lc5 $r
  play s $(( 2 * r - 1 )); play s $(( 2 * r ))
done
rep() { timeout 900 bash "$R" pyrun selfplay/fastchess_report.py "$@" --p1 arm 2>&1 | sed -n '/vs the field/,/pentanomial/p' | show; }
for a in lc0 lc1 lc2 lc3 lc4 lc5 s; do
  f=$(ls $O/lcg_${a}_c*.pgn 2>/dev/null); [ -n "$f" ] || continue
  echo "[q168] arm $a pooled:"; rep $f
done
for lvl in "25%|lc0 lc1 lc2 lc3" "50%|lc4 lc5" "100%|s"; do
  f=""; for a in ${lvl#*|}; do f="$f $(ls $O/lcg_${a}_c*.pgn 2>/dev/null | tr '\n' ' ')"; done
  [ -n "$(echo $f)" ] || continue
  echo "[q168] LEVEL ${lvl%%|*}:"; rep $f
done
echo "[q168] $(date) GAMES OVER (registered: 100% - 25% >= +12 go · +6 to +12 no dedicated time · < +6 ghost)"
wait $A2PID
echo "[q168] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | show
echo "[q168] $(date) QUEUE 168 DONE"
