#!/bin/bash
# 2026-11-06, queue #174 (owner 11-06: "keep pushing the training to see if we can reduce error even more", and a held-out error reduction that
# stays at least neutral in games is worth taking). Two jobs, both beside #173. Nothing is installed.
#  1 a fit on everything already labelled, no new data: all own-play training rows (S used 500,000 of about 872,000) and A2's d10 rows (the depth
#    target goes from about 274,000 own-play rows to about 600,000). corpus_v5 = corpus_v4 with A2's scores filled in, same rows, same features.
#      all0 recipe S on all rows        all2 all0 + own-play critical rows x 3 + the new critical rows at 0.5
#    Registered as conditional on #170's read before the owner asked; run now because it costs about half an hour of box and no data. A day and
#    night of NEW data stays conditional on #170.
#  2 the error scoreboard: win% accuracy on the SF18 WDL label (#169's 5,988 held-out quiet rows), static and d10, for every arm of #173 and this
#    queue, paired against gen 6.7. Gen 6.7's 384.2 / 201.3, S's 372.3 / 197.6, SF11 static 334.9 are the references.
#  3 after #171: 50k nodes vs gen 6.7, 2,000 g each, fresh seeds.
# Steps 1-2 run at night beside #170's node-limited games while that is safe for #171, otherwise after it.
# Registered: the accuracy curve expects a small gain, depth val from -2.86% to about -3.1%, and WDL static a few tenths past S. The fit's own val
# scopes change composition with A2's rows, so compare arms on the fixed WDL rows. Candidate by the owner's rule (not confirmed as a bar): lower
# held-out WDL error than S, and games not worse than S by 1 sigma (about -8 on 2,000 g against S's 4,000). Expected: within +-15 of S.
# Stop: touch /mnt/e/chess_data/q174.stop (checked between steps and chunks). Never edit this file while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
D=/mnt/e/chess_data/retune
W=/mnt/e/chess_data/wdlcal
BIN="$ND/build_uci/chess_engine_uci_g67pgo"
LOGD=/mnt/e/chess_data/q174
Q173=/mnt/e/chess_data/q173
STOP=/mnt/e/chess_data/q174.stop
PY=/home/ranuja/anaconda3/bin/python
NIGHT_END=$(date -d '2026-11-07 01:10' +%s)
GAMES_LATEST=$(date -d '2026-11-07 11:30' +%s)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"; mkdir -p "$LOGD"
show() { sed "s/^/[q174]   /"; }
stopped() { [ -e "$STOP" ] && { echo "[q174] $(date) stop file present before $1 — QUEUE 174 DONE"; exit 0; }; }
over() { grep -qE "QUEUE $1 (DONE|ABORTED)" /mnt/e/chess_data/q$1.log 2>/dev/null; }
# Work goes ahead at night only while #170 still plays and well before its expected end; else it waits for #171's timed games to finish.
pace() {
  over 171 && return
  if over 170 || [ $(date +%s) -ge $NIGHT_END ]; then
    echo "[q174] $(date) waiting for #171 before $1"
    until over 171; do stopped "$1"; sleep 60; done
    echo "[q174] $(date) #171 is over"
  fi
}
echo "[q174] $(date) waiting for #173's gates and the end of the A2 pass (stop: $STOP)"
until { grep -qE 'arms for the games on base s|QUEUE 173 (DONE|ABORTED)' /mnt/e/chess_data/q173.log 2>/dev/null; } && { grep -q 'A2 PASS OVER' /mnt/e/chess_data/q168.log 2>/dev/null; }; do
  stopped "the fits"; sleep 60
done
[ -s $D/corpus_v4.csv ] && [ -s $D/feat_v4_c12.npz ] || { echo "[q174] ☠️ corpus_v4 or its features are missing — QUEUE 174 ABORTED"; exit 1; }

# ── 1 the corpus and the fits ────────────────────────────────────────────────────────────────────────────────────────
pace "the corpus"; stopped "the corpus"
if [ ! -s $D/corpus_v5.csv ]; then
  $PY - <<EOF 2>&1 | show
import csv, glob, os
csv.field_size_limit(10 ** 7)
d10 = {}
for p in glob.glob("$D/A2_ours1105_d10_f*_s*of3.csv"):
    for r in csv.DictReader(open(p, newline="")):
        if r.get("ours_cp_white") not in (None, ""):
            d10[" ".join(r["fen"].split()[:4])] = (r["ours_cp_white"], r.get("move", ""))
n = t = 0
with open("$D/corpus_v4.csv", newline="") as fi, open("$D/corpus_v5.tmp", "w", newline="") as fo:
    rd, w = csv.reader(fi), csv.writer(fo)
    h = next(rd); w.writerow(h)
    ik, iz, im, isl = h.index("key"), h.index("ours_d10"), h.index("ours_move"), h.index("slice")
    for r in rd:
        t += 1
        if r[isl] == "A2" and r[iz] == "" and r[ik] in d10:
            r[iz], r[im] = d10[r[ik]]
            n += 1
        w.writerow(r)
os.replace("$D/corpus_v5.tmp", "$D/corpus_v5.csv")
print("A2 d10 rows read %d, filled %d, corpus rows %d" % (len(d10), n, t))
EOF
fi
[ -s $D/corpus_v5.csv ] || { echo "[q174] ☠️ corpus_v5 failed — QUEUE 174 ABORTED"; exit 1; }
TIME=""; [ -x /usr/bin/time ] && TIME="/usr/bin/time -v"
COMMON="EXPERIMENT=hybrid MODES=hybrid VAL=1 WIN_AWARE=1 LAMBDAS=0.03 CORPUS=corpus_v5.csv FEAT=feat_v4 EXTRA_SCOPES=1 SCOPE_KEYS=CR_scope.csv TRAIN_N=900000 VAL_N=200000 DEPTH_SLICES=A,A2,GAUNT,EGX,CUR,VARG,ODDS,STRIP,CRP,CRG STRESS_W=0.5"
fit() {   # $1 tag, $2 new rows' weight, $3 ROW_W file (optional)
  [ -s $D/fit_hybrid_$1.npz ] || $TIME bash "$R" pyrun diagnostics/_retune_fit.py $COMMON SLICE_W=VARG:0.5,ODDS:0.5,STRIP:0.5,CRP:$2,CRG:$2 ${3:+ROW_W=$3} TAG=_$1 > $LOGD/fit_$1.log 2> $LOGD/fit_$1.err
}
echo "[q174] $(date) fits on all rows, side by side"
fit all0 0 & fit all2 0.5 CR_w_all.csv & wait
for t in all0 all2; do
  echo "[q174] $(date) fit $t: $( [ -s $D/fit_hybrid_$t.npz ] && echo ok || echo '☠️ FAILED') · peak RSS $(grep -E 'Maximum resident' $LOGD/fit_$t.err | grep -oE '[0-9]+$') kB"
  grep -E "rows: train|weight by slice|hybrid  λ|static all|depth all|critical|hold CR|val CUR|rror|Trace" $LOGD/fit_$t.log | cut -c1-200 | show
done

# ── gates ────────────────────────────────────────────────────────────────────────────────────────────────────────────
gate() {  # $1 tag; adds it to PASS
  local t=$1 ENVL KN U Wc un wn SYM
  if [ -s $LOGD/knobs_$t.txt ]; then PASS="$PASS $t"; return; fi
  [ -s $D/fit_hybrid_$t.npz ] || return
  rm -rf $D/arm_$t
  env V2_PRESET=shipped bash "$R" pyrun diagnostics/_retune_export.py MODE=tables NPZ=fit_hybrid_$t.npz ARM_I=0 OUT=$D/arm_$t > $LOGD/tables_$t.log 2>&1
  grep -E "values:|⚠️|rror" $LOGD/tables_$t.log | sed "s/^/[q174] $t /"
  [ -s $D/arm_$t/env.txt ] || { echo "[q174] $t ✗ no tables — dropped"; return; }
  ENVL=$(cat $D/arm_$t/env.txt)
  env V2_PRESET=shipped $ENVL bash "$R" pyrun diagnostics/_retune_export.py MODE=verify NPZ=fit_hybrid_$t.npz ARM_I=0 ARMDIR=$t N=3000 \
    FEAT=feat_g67 WIN_AWARE=1 > $LOGD/verify_$t.log 2>&1
  grep -E "skipped|REALISED|rror" $LOGD/verify_$t.log | sed "s/^/[q174] $t /"
  set -- $(grep REALISED $LOGD/verify_$t.log | grep -oE '\(-?[0-9.]+%\)' | tr -d '()%')
  awk -v a="$1" -v b="$2" 'BEGIN{exit !(a < 0 && b/a >= 0.7)}' || { echo "[q174] $t ✗ closure gate (predicted $1% / engine $2%) — dropped"; return; }
  SYM=$(env V2_PRESET=shipped $ENVL bash "$R" pyrun diagnostics/_eval_symmetry.py N=4000 2>&1 | grep -E "violations")
  echo "[q174] $t symmetry: $(echo "$SYM" | tr '\n' ' ')"
  [ "$(echo "$SYM" | grep -c 'violations 0 ')" = "2" ] || { echo "[q174] $t ✗ symmetry — dropped"; return; }
  KN=$(tr -s ' \n' ',' < $D/arm_$t/env.txt | sed 's/,$//')
  pace "the benches of $t"   # #171 skips its chunks if it finds a bench running, so a bench must not start once #170 has ended
  U=$(printf 'setoption name Knobs value %s\nbench\nquit\n' "$KN" | "$BIN" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+')
  Wc=$(env V2_PRESET=shipped $ENVL bash "$R" wac fp_$t 2>&1 | grep -E "^SOLVED|^NODES" | tr '\n' ' ' | grep -oE 'Solved [0-9]+/300 NODES: [0-9]+')
  un=$(echo "$U" | grep -oE '[0-9]+$'); wn=$(echo "$Wc" | grep -oE '[0-9]+$')
  echo "[q174] $t UCI bench: $U · Cython WAC: $Wc"
  [ -n "$un" ] && [ "$un" = "$wn" ] && [ "$un" != 36859850 ] || { echo "[q174] $t ✗ UCI arm ≠ Cython arm (or knobs dead) — dropped"; return; }
  echo "$KN" > $LOGD/knobs_$t.txt
  PASS="$PASS $t"
}
PASS=""
for t in all0 all2; do
  pace "the gates of $t"; stopped "the gates of $t"
  gate $t
done
echo "[q174] $(date) arms of this queue:${PASS:- none}"

# ── 2 the error scoreboard ───────────────────────────────────────────────────────────────────────────────────────────
ARMS="$(ls $Q173/knobs_cr*.txt 2>/dev/null | sed 's#.*/knobs_##; s/\.txt$//' | tr '\n' ' ')$PASS"
echo "[q174] $(date) scoreboard arms: $ARMS"
board() {  # $1 reference arm
  env OMP_NUM_THREADS=1 bash "$R" pyrun diagnostics/_v1_v2_accuracy.py MODE=read REF=$1 > $LOGD/accuracy_$1.txt 2>&1
  grep -E "label|arm  |^  (static|d10)_(g67|v3s|sf11|cr[0-9]_[a-z0-9]+|all[0-9]) |calibrated|paired" $LOGD/accuracy_$1.txt | cut -c1-130 | show
}
for t in $ARMS; do
  pace "the static row of $t"; stopped "the static row of $t"
  [ -s $W/static_$t.csv ] || env V2_PRESET=shipped $(cat $D/arm_$t/env.txt) bash "$R" pyrun diagnostics/_v1_v2_accuracy.py MODE=static TAG=$t > /dev/null 2>&1
done
echo "[q174] $(date) STATIC SCOREBOARD, paired vs gen 6.7:"
board static_g67
d10() {   # $1 tag, the rest the engine env
  local tag=$1 k; shift
  [ -s $W/d10_$tag.csv ] && return
  for k in 0 1; do
    ( until env OMP_NUM_THREADS=1 PRESET=LONG_FORMAT MAX_DEPTH=10 USE_OPENING_BOOK=0 "$@" nice -n 10 bash "$R" pyrun \
          diagnostics/_depth_residual_pass.py IN=$W/cal_d10_in.csv OUT=$W/d10_$tag.csv SHARD=$k/2 CHUNK=1000 > /dev/null 2>&1; \
          [ $? -ne 3 ]; do [ -e "$STOP" ] && break; done ) &
  done
  wait
  [ -e "$STOP" ] && return
  { head -1 $W/d10_${tag}_s0of2.csv; tail -q -n +2 $W/d10_${tag}_s0of2.csv $W/d10_${tag}_s1of2.csv; } > $W/d10_$tag.csv
  echo "[q174] $(date) d10 $tag rows $(($(wc -l < $W/d10_$tag.csv) - 1))"
}
for t in $ARMS; do
  pace "the d10 row of $t"; stopped "the d10 row of $t"
  d10 $t V2_PRESET=shipped $(cat $D/arm_$t/env.txt)
done
stopped "the d10 table"
echo "[q174] $(date) D10 SCOREBOARD, paired vs gen 6.7:"
board d10_g67

# ── 3 games ──────────────────────────────────────────────────────────────────────────────────────────────────────────
[ -n "$PASS" ] || { echo "[q174] no arm of this queue passed the gates — QUEUE 174 DONE"; exit 0; }
until over 171; do stopped "the games"; sleep 60; done
# #173 checks that no games run before it starts its own, so this queue starts after it.
until grep -qE 'stage 2 arms:|QUEUE 173 (DONE|ABORTED)' /mnt/e/chess_data/q173.log 2>/dev/null; do stopped "the games"; sleep 60; done
sleep 180
GC="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
n=0
rm -f $O/crall_*_c*.pgn $O/crall_*_c*.log
for c in 1 2 3 4; do
  for t in $PASS; do
    n=$(( n + 1 ))
    # #173's afternoon leg wants a quiet box, so no new chunk once its games are over.
    { [ -e "$STOP" ] || [ $(date +%s) -gt $GAMES_LATEST ] || grep -q 'GAMES OVER' /mnt/e/chess_data/q173.log; } && { echo "[q174] $(date) no more chunks (stop, late, or #173's games are over)"; break 2; }
    T=crall_$t
    nice -n 5 "$FC" -engine "cmd=$BIN" name=arm option.OwnBook=false "option.Knobs=$(cat $LOGD/knobs_$t.txt)" nodes=50000 \
      -engine "cmd=$BIN" name=ctl option.OwnBook=false nodes=50000 $GC -srand $(( 174000 + n )) -rounds 250 -games 2 \
      -pgnout file=$O/${T}_c$c.pgn notation=san nodes=true append=false > $O/${T}_c$c.log 2>&1
    echo "[q174] $(date) $T chunk $c: $(grep -E '^Elo:' $O/${T}_c$c.log | tail -1) · warnings $(grep -c 'have option' $O/${T}_c$c.log) · stalls $(grep -ciE 'disconnect|stall|not responsive' $O/${T}_c$c.log)"
  done
done
for t in $PASS; do
  f=$(ls $O/crall_${t}_c*.pgn 2>/dev/null)
  [ -n "$f" ] && { echo "[q174] arm $t pooled:"; timeout 900 bash "$R" pyrun selfplay/fastchess_report.py $f --p1 arm 2>&1 | sed -n '/vs the field/,/pentanomial/p' | show; }
done
echo "[q174] $(date) QUEUE 174 DONE"
