#!/bin/bash
# 2026-11-05 evening, queue #165: the box time from the end of #164 (~03:00) to the owner's return (~10:00).
#  1 corpus_v3 for the retune: first the assembler identity check (the edited _corpus_assemble.py with default env must
#    write byte-for-byte what the pre-edit copy writes), then SLICES_EXTRA=EGX,CUR,VARG,ODDS,STRIP D10_TAG=ours1105
#    CRIT_GLOB=CRIT*_sf18k4_s*.csv → corpus_v3.csv, then gen 6.7's feature / static export → feat_g67 (_cNN.npz chunks).
#  2 TM_LOWCLOCK replicate (TIMED, box otherwise quiet): 10+0 self-play on chess_engine_uci_tmlc, arm Knobs=TM_LOWCLOCK=1 vs
#    ctl, 4 x 200 g, fresh seeds 165111-165114. Registered: pooled with #159 (+20.4 ± 9.9) the item stays ≥ +2σ; a replicate
#    can only revoke. Then Mediocre 0.5 at 10+0, arm and ctl 300 g each on one seed (1651): not negative.
#  3 the rest of the night: gen 6.7's d10 pass on slice A2 (399k, older own-play; resumable), for the case the first fit's
#    learning curve says more depth rows help. No new shard after 09:30.
# Timed legs start only before 07:00 and never beside a data job. Stop: touch /mnt/e/chess_data/q165.stop (between steps).
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
D=/mnt/e/chess_data/retune
MED="$ND/selfplay/mediocre_uci.sh"
BIN="$ND/build_uci/chess_engine_uci_tmlc"
LOGD=/mnt/e/chess_data/q165
STOP=/mnt/e/chess_data/q165.stop
TIMED_LATEST=$(date -d '2026-11-06 07:00' +%s)
CUTOFF=$(date -d '2026-11-06 09:30' +%s)
TAG=ours1105
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"; mkdir -p "$LOGD"
show() { sed "s/^/[q165]   /"; }
echo "[q165] $(date) waiting for queue #164 (stop: $STOP)"
until grep -qE 'QUEUE 164 (DONE|ABORTED)' /mnt/e/chess_data/q164.log 2>/dev/null; do
  [ -e "$STOP" ] && { echo "[q165] stop file present — QUEUE 165 DONE (nothing run)"; exit 0; }
  sleep 60
done
sleep 20
busy() { pgrep -f "fastchess-linux-x86-64/fastchess|vs_sf.py|_depth_residual_pass|_build_regret_set|_px_export" >/dev/null; }
busy && { echo "[q165] another job is running — QUEUE 165 ABORTED"; exit 1; }
START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[q165] $START start"

# ── 1 corpus_v3 ───────────────────────────────────────────────────────────────────────────────────────────────────────
if [ ! -s $D/corpus_v3.csv ]; then
  bash "$R" pyrun diagnostics/_corpus_assemble_prev.py MODE=assemble OUT=corpus_chk_prev.csv > $LOGD/chk_prev.log 2>&1
  bash "$R" pyrun diagnostics/_corpus_assemble.py MODE=assemble OUT=corpus_chk_new.csv > $LOGD/chk_new.log 2>&1
  if cmp -s $D/corpus_chk_prev.csv $D/corpus_chk_new.csv && [ -s $D/corpus_chk_new.csv ]; then
    echo "[q165] assembler identity: default output byte-identical to the pre-edit copy ($(($(wc -l < $D/corpus_chk_new.csv) - 1)) rows)"
    rm -f $D/corpus_chk_prev.csv $D/corpus_chk_new.csv diagnostics/_corpus_assemble_prev.py
    bash "$R" pyrun diagnostics/_corpus_assemble.py MODE=assemble SLICES_EXTRA=EGX,CUR,VARG,ODDS,STRIP D10_TAG=$TAG \
      'CRIT_GLOB=CRIT*_sf18k4_s*.csv' OUT=corpus_v3.csv > $LOGD/assemble_v3.log 2>&1
    sed -n '/game map/,/d10 depth rows/p;/multi-PV gap/p;/SUMMARY/,$p' $LOGD/assemble_v3.log | show
  else
    echo "[q165] ☠️ assembler identity FAILED (prev $(wc -l < $D/corpus_chk_prev.csv 2>/dev/null) vs new $(wc -l < $D/corpus_chk_new.csv 2>/dev/null) lines) — corpus_v3 not built"
    tail -3 $LOGD/chk_new.log | show
  fi
fi
if [ -s $D/corpus_v3.csv ] && [ ! -e "$STOP" ]; then
  env V2_PRESET=shipped bash "$R" pyrun diagnostics/_px_export.py IN=$D/corpus_v3.csv OUT=$D/feat_g67.npz > $LOGD/export.log 2>&1
  echo "[q165] $(date) export: $(grep -E 'CORPUS' $LOGD/export.log | tail -1) · chunks $(ls $D/feat_g67_c*.npz 2>/dev/null | wc -l)"
fi

# ── 2 TM_LOWCLOCK replicate + Mediocre (timed) ────────────────────────────────────────────────────────────────────────
timed_ok() { [ ! -e "$STOP" ] && [ $(date +%s) -le $TIMED_LATEST ] && ! busy; }
if timed_ok; then
  d=$(printf 'bench\nquit\n' | "$BIN" 2>/dev/null | grep '^bench')
  echo "[q165] tmlc bench (want 251 / 36859850): $d"
  if echo "$d" | grep -q "solved 251/300 nodes 36859850 "; then
    COMMON="-repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
    T=tmlc10r; rm -f $O/${T}_c[1-5].pgn $O/${T}_c[1-5].log
    for c in 1 2 3 4; do
      timed_ok || { echo "[q165] $(date) skip $T chunk $c (stop, late or busy)"; break; }
      nice -n 5 "$FC" -engine cmd="$BIN" name=arm option.OwnBook=false option.Knobs=TM_LOWCLOCK=1 tc=10+0 timemargin=250 \
        -engine cmd="$BIN" name=ctl option.OwnBook=false tc=10+0 timemargin=250 \
        -openings file=$O/openings_uho.epd format=epd order=random $COMMON -srand $(( 165110 + c )) -rounds 100 -games 2 \
        -pgnout file=$O/${T}_c$c.pgn notation=san nodes=true append=false > $O/${T}_c$c.log 2>&1
      echo "[q165] $(date) $T chunk $c: $(grep -E '^Elo:' $O/${T}_c$c.log | tail -1)  disconnects/stalls: $(grep -ciE 'disconnect|stall|not responsive' $O/${T}_c$c.log)  time forfeits: $(grep -c 'loses on time' $O/${T}_c$c.pgn)"
    done
    f=$(ls $O/${T}_c[1-4].pgn 2>/dev/null)
    if [ -n "$f" ]; then
      echo "[q165] $T pooled:"
      timeout 600 bash "$R" pyrun selfplay/fastchess_report.py $f --p1 arm 2>&1 | sed -n '/vs the field/,/terminations/p' | show
      echo "[q165] with #159 (all 8 chunks):"
      timeout 600 bash "$R" pyrun selfplay/fastchess_report.py $O/tmlc10_c[1-4].pgn $f --p1 arm 2>&1 | sed -n '/vs the field/,/pentanomial/p' | show
      for s in arm ctl; do
        echo "[q165]   time forfeits $s: $(cat $f | awk -v s=$s '/^\[White /{w=$2; gsub(/[]"[]/,"",w)} /^\[Black /{b=$2; gsub(/[]"[]/,"",b)} /loses on time/{ if ((index($0,"White loses")>0 && w==s) || (index($0,"Black loses")>0 && b==s)) n++ } END {print n+0}')"
      done
    fi
    MC="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
    for arm in on off; do
      timed_ok || { echo "[q165] $(date) skip Mediocre $arm (stop, late or busy)"; continue; }
      K=""; [ $arm = on ] && K="option.Knobs=TM_LOWCLOCK=1"
      T=med10z_$arm
      nice -n 5 "$FC" -engine cmd="$BIN" name=ours option.OwnBook=false $K tc=10+0 timemargin=250 restart=on \
        -engine cmd="$MED" name=mediocre tc=10+0 timemargin=250 restart=on $MC -srand 1651 -rounds 150 -games 2 \
        -pgnout file=$O/$T.pgn notation=san nodes=true append=false > $O/$T.log 2>&1
      echo "[q165] $(date) $T: $(grep -E '^Games:|^Elo:' $O/$T.log | tail -2 | tr '\n' ' ')"
      echo "[q165]   time forfeits ours: $(cat $O/$T.pgn | awk '/^\[White /{w=$2; gsub(/[]"[]/,"",w)} /^\[Black /{b=$2; gsub(/[]"[]/,"",b)} /loses on time/{ if ((index($0,"White loses")>0 && w=="ours") || (index($0,"Black loses")>0 && b=="ours")) n++ } END {print n+0}') · all: $(grep -c 'loses on time' $O/$T.pgn)"
    done
  else
    echo "[q165] ☠️ tmlc fingerprint failed — timed legs skipped"
  fi
else
  echo "[q165] $(date) timed legs skipped (stop, past 07:00 or a job running)"
fi

# ── 3 A2 d10 pass until the cut-off ───────────────────────────────────────────────────────────────────────────────────
late() { [ -e "$STOP" ] || [ $(date +%s) -ge $CUTOFF ]; }
job() {
  local rc
  while :; do
    timeout 3600 env V2_PRESET=shipped PRESET=LONG_FORMAT MAX_DEPTH=10 USE_OPENING_BOOK=0 RESIGN_THRESHOLD=-99999999 \
      bash "$R" pyrun diagnostics/_depth_residual_pass.py IN="$2" OUT="$3" SHARD=$4 CHUNK=800 MOVE=1 >> "$LOGD/$1.log" 2>&1
    rc=$?
    [ $rc -ne 3 ] && break
    late && { echo "[q165] $(date) $1 paused at a chunk boundary (stop or cut-off)"; return; }
  done
  echo "[q165] $(date) $1 done (rc $rc)"
}
JOBS=()
for p in $(ls $D/A2_fens_sf18_s*.csv 2>/dev/null); do
  i=$(basename $p .csv | sed 's/.*_s//')
  for k in 0 1 2; do JOBS+=("A2_f${i}_k$k|$p|$D/A2_${TAG}_d10_f$i.csv|$k/3"); done
done
echo "[q165] $(date) A2 d10: ${#JOBS[@]} jobs"
for j in "${JOBS[@]}"; do
  IFS='|' read -r name in out shard <<< "$j"
  late && { echo "[q165] $(date) not starting $name: stop or cut-off"; continue; }
  while [ $(jobs -rp | wc -l) -ge 3 ]; do wait -n; done
  job "$name" "$in" "$out" "$shard" &
done
wait
echo "[q165] $(date) A2 d10 rows: $(cat $D/A2_${TAG}_d10_f*.csv 2>/dev/null | grep -vc '^fen,') / ~399k"
echo "[q165] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | show
echo "[q165] $(date) QUEUE 165 DONE"
