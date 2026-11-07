#!/bin/bash
# 2026-11-06, queue #173 (owner 11-06: keep working from about 07:00 on 11-07; timed games are allowed that afternoon, that day only): the first
# pass of stage 2 of the retune, criticality. Critical rows join the same hybrid fit as extra weight. Nothing is installed.
#  1 data, after #172: corpus_v4 = corpus_v3 + the critical rows (slices CRP puzzles, CRG our own games), their features, weight files, held-out rows.
#  2 fits, 3 at a time. The base recipe is S, or S x 2 / S x 4 if #171 picked one (then the fits and gates run again on that share).
#      cr0 the base on corpus_v4, new rows at 0, as a check: it must reproduce the base fit exactly
#      cr1 base + own-play critical rows x 3        cr2 cr1 + new rows at 0.5        cr3 cr1 + new rows at 1        cr4 cr1 + new rows at 3
#      crlc25, crlc50: cr3 with 25% and 50% of the new rows, for the accuracy curve only
#  3 gates per arm, as in #166. Steps 1-3 run at night beside #170's node-limited games if they can finish before #171's timed games; otherwise
#    they wait for #171.
#  4 after #171: 2,000 g each at 50k nodes vs gen 6.7, fresh seeds, interleaved. Beside the games: d10 with the move on the held-out critical rows
#    for the base and each arm, then full STS d10 and STS300 at 249k per arm. STS is never trained on (owner: "we shouldn't train on a bench").
#  5 the afternoon: if the best arm screens well, timed 10+0.1 self-play of it; otherwise 200k-node self-play of S and the best arm.
# Registered before any stage 2 fit is read.
#  Fit check: cr0 within 0.01 mp of the base fit, or the queue aborts (the new rows would be disturbing the old ones).
#  Expected in the fits: held-out critical scopes (static / depth hold CRP, hold CRG, val critical) improve as the dose rises, ordinary own-play
#  val gives a little back. Accuracy curve on the critical rows, cr3 at 25 / 50 / 100%: a last doubling worth >= 10% on depth hold CRP says more
#  critical rows would still help.
#  Moves: an arm finds more critical moves if its miss rate on the held-out rows is lower than the base arm's by 2 sigma of the two paired
#  differences. Expected: under 1 pp at 0.5, perhaps 1-2 pp at 3.
#  Games: an arm screens well if it beats the base arm's level at 50k (S: +15.4 on 4,000 g) by >= +10. That is about 1.2 sigma, a screen and
#  not a verdict. Expected: within +-15 of S, so most likely no arm clears it.
#  STS: read against gen 6.7 (full 9,830, STS300 @ 249k 1,928) and S (10,006, 1,890). No bar, it is a judge of the type of position, not of strength.
#  A ship still needs +2 sigma timed, SF18, Mediocre and the owner.
# Stop: touch /mnt/e/chess_data/q173.stop (checked between steps and chunks). Never edit this file while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
D=/mnt/e/chess_data/retune
BIN="$ND/build_uci/chess_engine_uci_g67pgo"
LOGD=/mnt/e/chess_data/q173
STOP=/mnt/e/chess_data/q173.stop
PY=/home/ranuja/anaconda3/bin/python
NIGHT_END=$(date -d '2026-11-07 01:30' +%s)
TIMED_LATEST=$(date -d '2026-11-07 16:00' +%s)
NODES_LATEST=$(date -d '2026-11-07 17:00' +%s)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"; mkdir -p "$LOGD"
show() { sed "s/^/[q173]   /"; }
stopped() { [ -e "$STOP" ] && { echo "[q173] $(date) stop file present before $1 — QUEUE 173 DONE"; exit 0; }; }
over() { grep -qE "QUEUE $1 (DONE|ABORTED)" /mnt/e/chess_data/q$1.log 2>/dev/null; }
busy() { pgrep -f "fastchess-linux-x86-64/fastchess|vs_sf.py|_depth_residual_pass|_build_regret_set|_px_export|_retune_fit|_reference_ceiling|sts_test|tactical_test" >/dev/null; }
# Work that is not node-limited-safe to overlap with #171 goes ahead only while #170 still plays and well before its expected end; else it waits for #171.
pace() {
  over 171 && return
  if over 170 || [ $(date +%s) -ge $NIGHT_END ]; then
    echo "[q173] $(date) waiting for #171 before $1"
    until over 171; do stopped "$1"; sleep 60; done
    echo "[q173] $(date) #171 is over"
  fi
}
echo "[q173] $(date) waiting for #172 (stop: $STOP)"
until over 172; do stopped "the data"; sleep 60; done

# ── 1 data ───────────────────────────────────────────────────────────────────────────────────────────────────────────
pace "the data"
stopped "the data"
if [ ! -s $D/CR_hold.csv ]; then
  bash "$R" pyrun diagnostics/_critical_fit_data.py MODE=assemble 2>&1 | tee $LOGD/assemble.log | show
fi
[ -s $D/CR_hold.csv ] && [ -s $D/corpus_v4.csv ] || { echo "[q173] ☠️ assembly failed — QUEUE 173 ABORTED"; exit 1; }
env V2_PRESET=shipped bash "$R" pyrun diagnostics/_px_export.py IN=$D/CR_rows.csv OUT=$D/feat_cr.npz > $LOGD/export.log 2>&1
grep -E "DONE|rror|Trace" $LOGD/export.log | show
[ -s $D/feat_cr_c00.npz ] || { echo "[q173] ☠️ feature export failed — QUEUE 173 ABORTED"; exit 1; }
for f in $D/feat_g67_c??.npz; do
  t=$D/feat_v4_$(basename "$f" | sed 's/^feat_g67_//')
  [ -e "$t" ] || ln -s "$(basename "$f")" "$t" 2>/dev/null || cp "$f" "$t"
done
n=$(ls $D/feat_g67_c??.npz | wc -l)
for f in $D/feat_cr_c??.npz; do
  t=$D/feat_v4_c$(printf %02d $n)
  [ -e "$t.npz" ] || ln -s "$(basename "$f")" "$t.npz" 2>/dev/null || cp "$f" "$t.npz"
  n=$(( n + 1 ))
done
echo "[q173] $(date) corpus_v4 $(( $(wc -l < $D/corpus_v4.csv) - 1 )) rows · feature chunks $(ls $D/feat_v4_c??.npz | wc -l)"

# ── 2-3 fits and gates, for one base ─────────────────────────────────────────────────────────────────────────────────
TIME=""; [ -x /usr/bin/time ] && TIME="/usr/bin/time -v"
COMMON="EXPERIMENT=hybrid MODES=hybrid VAL=1 WIN_AWARE=1 LAMBDAS=0.03 CORPUS=corpus_v4.csv FEAT=feat_v4 EXTRA_SCOPES=1 SCOPE_KEYS=CR_scope.csv TRAIN_N=500000 VAL_N=120000 DEPTH_SLICES=A,GAUNT,EGX,CUR,VARG,ODDS,STRIP,CRP,CRG"
fit() {   # $1 tag, $2 stress weight, $3 new rows' weight, $4 ROW_W file (optional)
  [ -s $D/fit_hybrid_$1.npz ] || $TIME bash "$R" pyrun diagnostics/_retune_fit.py $COMMON STRESS_W=$2 SLICE_W=VARG:$2,ODDS:$2,STRIP:$2,CRP:$3,CRG:$3 ${4:+ROW_W=$4} TAG=_$1 > $LOGD/fit_$1.log 2> $LOGD/fit_$1.err
}
report_fit() {
  echo "[q173] $(date) fit $1: $( [ -s $D/fit_hybrid_$1.npz ] && echo ok || echo '☠️ FAILED') · peak RSS $(grep -E 'Maximum resident' $LOGD/fit_$1.err | grep -oE '[0-9]+$') kB"
  grep -E "weight by slice|hybrid  λ|static all|depth all|critical|hold CR|val CUR|rror|Trace" $LOGD/fit_$1.log | show
}
gate() {  # $1 tag; adds it to PASS
  local t=$1 ENVL KN U W un wn SYM
  if [ -s $LOGD/knobs_$t.txt ]; then PASS="$PASS $t"; return; fi
  [ -s $D/fit_hybrid_$t.npz ] || return
  rm -rf $D/arm_$t
  env V2_PRESET=shipped bash "$R" pyrun diagnostics/_retune_export.py MODE=tables NPZ=fit_hybrid_$t.npz ARM_I=0 OUT=$D/arm_$t > $LOGD/tables_$t.log 2>&1
  grep -E "values:|⚠️|rror" $LOGD/tables_$t.log | sed "s/^/[q173] $t /"
  [ -s $D/arm_$t/env.txt ] || { echo "[q173] $t ✗ no tables — dropped"; return; }
  ENVL=$(cat $D/arm_$t/env.txt)
  env V2_PRESET=shipped $ENVL bash "$R" pyrun diagnostics/_retune_export.py MODE=verify NPZ=fit_hybrid_$t.npz ARM_I=0 ARMDIR=$t N=3000 \
    FEAT=feat_g67 WIN_AWARE=1 > $LOGD/verify_$t.log 2>&1
  grep -E "skipped|REALISED|rror" $LOGD/verify_$t.log | sed "s/^/[q173] $t /"
  set -- $(grep REALISED $LOGD/verify_$t.log | grep -oE '\(-?[0-9.]+%\)' | tr -d '()%')
  awk -v a="$1" -v b="$2" 'BEGIN{exit !(a < 0 && b/a >= 0.7)}' || { echo "[q173] $t ✗ closure gate (predicted $1% / engine $2%) — dropped"; return; }
  SYM=$(env V2_PRESET=shipped $ENVL bash "$R" pyrun diagnostics/_eval_symmetry.py N=4000 2>&1 | grep -E "violations")
  echo "[q173] $t symmetry: $(echo "$SYM" | tr '\n' ' ')"
  [ "$(echo "$SYM" | grep -c 'violations 0 ')" = "2" ] || { echo "[q173] $t ✗ symmetry — dropped"; return; }
  KN=$(tr -s ' \n' ',' < $D/arm_$t/env.txt | sed 's/,$//')
  pace "the benches of $t"   # #171 skips its chunks if it finds a bench running, so a bench must not start once #170 has ended
  U=$(printf 'setoption name Knobs value %s\nbench\nquit\n' "$KN" | "$BIN" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+')
  W=$(env V2_PRESET=shipped $ENVL bash "$R" wac fp_$t 2>&1 | grep -E "^SOLVED|^NODES" | tr '\n' ' ' | grep -oE 'Solved [0-9]+/300 NODES: [0-9]+')
  un=$(echo "$U" | grep -oE '[0-9]+$'); wn=$(echo "$W" | grep -oE '[0-9]+$')
  echo "[q173] $t UCI bench: $U · Cython WAC: $W"
  [ -n "$un" ] && [ "$un" = "$wn" ] && [ "$un" != 36859850 ] || { echo "[q173] $t ✗ UCI arm ≠ Cython arm (or knobs dead) — dropped"; return; }
  echo "$KN" > $LOGD/knobs_$t.txt
  PASS="$PASS $t"
}
fits_and_gates() {   # $1 base tag (s, x2, x4), $2 stress weight, $3 the base's fit tag
  local B=$1 SW=$2 BF=$3
  pace "the fits"; stopped "the fits"
  echo "[q173] $(date) fits on base $B (stress $SW): cr0, cr1, cr2"
  fit cr0_$B $SW 0 & fit cr1_$B $SW 0 CR_w_all.csv & fit cr2_$B $SW 0.5 CR_w_all.csv & wait
  for t in cr0 cr1 cr2; do report_fit ${t}_$B; done
  $PY - > $LOGD/check_cr0_$B.txt 2>&1 <<EOF
import numpy as np
a = np.load("$D/fit_hybrid_cr0_$B.npz"); b = np.load("$D/fit_hybrid_$BF.npz")
da = dict(zip(a["names"].tolist(), a["deltas"][0].tolist())); db = dict(zip(b["names"].tolist(), b["deltas"][0].tolist()))
c = sorted(set(da) & set(db))
print("cells %d / %d, common %d, max |d| %.4f mp" % (len(da), len(db), len(c), max(abs(da[k] - db[k]) for k in c)))
EOF
  echo "[q173] cr0_$B vs $BF: $(tail -1 $LOGD/check_cr0_$B.txt)"
  awk '/max \|d\|/ {ok = ($(NF-1) + 0 < 0.01)} END {exit !ok}' $LOGD/check_cr0_$B.txt || { echo "[q173] ☠️ cr0 does not reproduce the base fit — QUEUE 173 ABORTED"; exit 1; }
  pace "the next fits"; stopped "the next fits"
  echo "[q173] $(date) fits: cr3, cr4, 25% of the new rows"
  fit cr3_$B $SW 1 CR_w_all.csv & fit cr4_$B $SW 3 CR_w_all.csv & fit crlc25_$B $SW 1 CR_w_lc25.csv & wait
  for t in cr3 cr4 crlc25; do report_fit ${t}_$B; done
  pace "the last fit"; stopped "the last fit"
  fit crlc50_$B $SW 1 CR_w_lc50.csv
  report_fit crlc50_$B
  PASS=""
  for t in cr1 cr2 cr3 cr4; do
    pace "the gates of ${t}_$B"; stopped "the gates of ${t}_$B"
    gate ${t}_$B
  done
  echo "[q173] $(date) arms for the games on base $B:${PASS:- none}"
}
fits_and_gates s 0.5 v3s

# ── 4 after #171: the base, the games and the judges ─────────────────────────────────────────────────────────────────
pace "the games"
until over 171; do stopped "the games"; sleep 60; done
A=$(grep -oE 'arm for the timed games: [a-z0-9]+' /mnt/e/chess_data/q171.log | tail -1 | awk '{print $NF}')
case "$A" in
  v3x2) BASE=v3x2; fits_and_gates x2 1 v3x2 ;;
  v3x4) BASE=v3x4; fits_and_gates x4 2 v3x4 ;;
  *)    BASE=v3s ;;
esac
echo "[q173] $(date) base arm $BASE · stage 2 arms:${PASS:- none}"
[ -n "$PASS" ] || { echo "[q173] no arm passed the gates — QUEUE 173 DONE"; exit 0; }
until ! pgrep -f "fastchess-linux-x86-64/fastchess|vs_sf.py" >/dev/null; do stopped "the games"; sleep 60; done

judges() {
  local t a arms="" ENVL k
  for t in $BASE $PASS; do
    [ -e "$STOP" ] && return
    ENVL=$(cat $D/arm_$t/env.txt)
    for k in 0 1 2; do
      ( until timeout 3600 env V2_PRESET=shipped $ENVL PRESET=LONG_FORMAT MAX_DEPTH=10 USE_OPENING_BOOK=0 RESIGN_THRESHOLD=-99999999 \
            bash "$R" pyrun diagnostics/_depth_residual_pass.py IN=$D/CR_hold.csv OUT=$D/CRH_${t}_d10.csv SHARD=$k/3 CHUNK=800 MOVE=1 >> $LOGD/hold_$t.log 2>&1; \
            [ $? -ne 3 ]; do [ -e "$STOP" ] && break; done ) &
    done
    wait
    echo "[q173] $(date) held-out critical moves, $t done"
    arms="$arms,$t:CRH_${t}_d10_s*of3.csv"
  done
  bash "$R" pyrun diagnostics/_critical_fit_data.py MODE=moves "ARMS=${arms#,}" > $LOGD/moves.txt 2>&1
  bash "$R" pyrun diagnostics/_critical_fit_data.py MODE=moves "ARMS=${arms#,}" REF=$BASE > $LOGD/moves_vs_base.txt 2>&1
  echo "[q173] $(date) moves on the held-out critical rows, against gen 6.7 ($LOGD/moves.txt) and against the base arm:"
  cut -c1-120 $LOGD/moves.txt | show
  cut -c1-120 $LOGD/moves_vs_base.txt | show
  for t in $PASS; do
    [ -e "$STOP" ] && return
    ENVL=$(cat $D/arm_$t/env.txt)
    echo "[q173] $(date) $t STS full d10: $(bash "$R" sts_full recf_$t V2_PRESET=shipped $ENVL | grep 'STS score' | tail -1)"
    echo "[q173] $(date) $t STS300 @249k: $(bash "$R" sts sts_eqn_$t V2_PRESET=shipped $ENVL MAX_DEPTH=64 NODE_LIMIT=249014 | tail -1)"
  done
  echo "[q173] $(date) JUDGES OVER"
}
judges &
JP=$!

GC="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
n=0
rm -f $O/crg_*_c*.pgn $O/crg_*_c*.log
for c in 1 2 3 4; do
  for t in $PASS; do
    n=$(( n + 1 ))
    [ -e "$STOP" ] && break 2
    T=crg_$t
    nice -n 5 "$FC" -engine "cmd=$BIN" name=arm option.OwnBook=false "option.Knobs=$(cat $LOGD/knobs_$t.txt)" nodes=50000 \
      -engine "cmd=$BIN" name=ctl option.OwnBook=false nodes=50000 $GC -srand $(( 173000 + n )) -rounds 250 -games 2 \
      -pgnout file=$O/${T}_c$c.pgn notation=san nodes=true append=false > $O/${T}_c$c.log 2>&1
    echo "[q173] $(date) $T chunk $c: $(grep -E '^Elo:' $O/${T}_c$c.log | tail -1) · warnings $(grep -c 'have option' $O/${T}_c$c.log) · stalls $(grep -ciE 'disconnect|stall|not responsive' $O/${T}_c$c.log)"
  done
done
elo() { timeout 900 bash "$R" pyrun selfplay/fastchess_report.py "$@" --p1 arm 2>&1 | grep -m1 -oE 'Elo [-+]?[0-9.]+' | awk '{print $2}'; }
BEST=""; BE=-99999
for t in $PASS; do
  echo "[q173] arm $t pooled:"
  timeout 900 bash "$R" pyrun selfplay/fastchess_report.py $O/crg_${t}_c*.pgn --p1 arm 2>&1 | sed -n '/vs the field/,/pentanomial/p' | show
  e=$(elo $O/crg_${t}_c*.pgn)
  if [ -n "$e" ] && awk -v a="$e" -v b="$BE" 'BEGIN{exit !(a + 0 > b + 0)}'; then BE=$e; BEST=$t; fi
done
case "$BASE" in v3s) TB=s ;; *) TB=$BASE ;; esac
BL=$(elo $O/lcg_${TB}_c*.pgn)
echo "[q173] $(date) GAMES OVER · best arm $BEST $BE · base $BASE $BL · screen: best - base >= +10"
echo "[q173] $(date) waiting for the judges"
wait $JP
stopped "the afternoon"

# ── 5 the afternoon ──────────────────────────────────────────────────────────────────────────────────────────────────
if [ -n "$BEST" ] && [ -n "$BL" ] && awk -v a="$BE" -v b="$BL" 'BEGIN{exit !(a - b >= 10)}'; then
  echo "[q173] $(date) $BEST screens well: timed 10+0.1 self-play, no chunk after 16:00"
  KN=$(cat $LOGD/knobs_$BEST.txt)
  rm -f /tmp/q173_pre.engine.log
  timeout 180 "$FC" -engine "cmd=$BIN" name=arm option.OwnBook=false "option.Knobs=$KN" st=0.05 \
    -engine "cmd=$BIN" name=ctl option.OwnBook=false st=0.05 \
    -openings file=$O/openings_uho.epd format=epd order=random -rounds 1 -games 2 -concurrency 1 \
    -log file=/tmp/q173_pre.engine.log level=info engine=true > /tmp/q173_pre.log 2>&1
  W=$(grep -c 'have option' /tmp/q173_pre.log)
  echo "[q173] preflight: $(grep -E '^Games:' /tmp/q173_pre.log | tail -1) · warnings $W"
  [ "$W" = 0 ] || { echo "[q173] ☠️ preflight warnings — QUEUE 173 ABORTED"; exit 1; }
  sleep 20
  T=rt_${BEST}_10p1
  rm -f $O/${T}_c[1-4].pgn $O/${T}_c[1-4].log
  for c in 1 2 3 4; do
    # Wait for a quiet box here: skipping on busy would lose every remaining chunk at once.
    while busy && [ ! -e "$STOP" ] && [ $(date +%s) -le $TIMED_LATEST ]; do sleep 60; done
    { [ -e "$STOP" ] || [ $(date +%s) -gt $TIMED_LATEST ] || busy; } && { echo "[q173] $(date) skip $T chunk $c (stop, late or busy)"; continue; }
    nice -n 5 "$FC" -engine "cmd=$BIN" name=arm option.OwnBook=false "option.Knobs=$KN" tc=10+0.1 timemargin=250 \
      -engine "cmd=$BIN" name=ctl option.OwnBook=false tc=10+0.1 timemargin=250 \
      -openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10 \
      -srand $(( 173110 + c )) -rounds 250 -games 2 -pgnout file=$O/${T}_c$c.pgn notation=san nodes=true append=false > $O/${T}_c$c.log 2>&1
    echo "[q173] $(date) $T chunk $c: $(grep -E '^Elo:' $O/${T}_c$c.log | tail -1) · warnings $(grep -c 'have option' $O/${T}_c$c.log) · stalls $(grep -ciE 'disconnect|stall|not responsive' $O/${T}_c$c.log) · time forfeits $(grep -c 'loses on time' $O/${T}_c$c.pgn)"
  done
  f=$(ls $O/${T}_c[1-4].pgn 2>/dev/null)
  [ -n "$f" ] && { echo "[q173] $T pooled:"; timeout 900 bash "$R" pyrun selfplay/fastchess_report.py $f --p1 arm 2>&1 | sed -n '/vs the field/,$p' | show; }
else
  echo "[q173] $(date) no arm screens well: 200k-node self-play of the base and the best arm, no chunk after 17:00"
  n=0
  rm -f $O/n200k_*_c*.pgn $O/n200k_*_c*.log
  for c in 1 2 3 4; do
    for t in $BASE $BEST; do
      n=$(( n + 1 ))
      { [ -e "$STOP" ] || [ $(date +%s) -gt $NODES_LATEST ]; } && break 2
      if [ "$t" = v3s ]; then KF=/mnt/e/chess_data/q166/knobs_v3s.txt; elif [ "$t" = "$BASE" ]; then KF=/mnt/e/chess_data/q170/knobs_$t.txt; else KF=$LOGD/knobs_$t.txt; fi
      T=n200k_$t
      nice -n 5 "$FC" -engine "cmd=$BIN" name=arm option.OwnBook=false "option.Knobs=$(cat $KF)" nodes=200000 \
        -engine "cmd=$BIN" name=ctl option.OwnBook=false nodes=200000 $GC -srand $(( 173200 + n )) -rounds 250 -games 2 \
        -pgnout file=$O/${T}_c$c.pgn notation=san nodes=true append=false > $O/${T}_c$c.log 2>&1
      echo "[q173] $(date) $T chunk $c: $(grep -E '^Elo:' $O/${T}_c$c.log | tail -1) · warnings $(grep -c 'have option' $O/${T}_c$c.log) · stalls $(grep -ciE 'disconnect|stall|not responsive' $O/${T}_c$c.log)"
    done
  done
  for t in $BASE $BEST; do
    f=$(ls $O/n200k_${t}_c*.pgn 2>/dev/null)
    [ -n "$f" ] && { echo "[q173] $t at 200k pooled:"; timeout 900 bash "$R" pyrun selfplay/fastchess_report.py $f --p1 arm 2>&1 | sed -n '/vs the field/,/pentanomial/p' | show; }
  done
fi
echo "[q173] $(date) QUEUE 173 DONE"
