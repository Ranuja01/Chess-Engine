#!/bin/bash
# 2026-11-06 ~06:20, queue #167: the ship check of retune arm S (#166: tables in retune/arm_v3s, knobs in q166/knobs_v3s.txt), the
# only arm that screened well: S +17.6 (+4.0 ... +31.2) and N -5.6 (-19.2 ... +8.1) over 2,000 g each at 50k nodes vs gen 6.7.
# Nothing is installed; a ship needs the owner.
#  1 SF18 @ 1,100 nodes (our side 250k nodes, so safe beside the A2 d10 pass): gen 6.7 and arm S on a fresh seed, 1671, 500 g each.
#    The saved baselines on seed 1521 helped choose gen 6.7, so pairing against them would read low. Registered: > -1 sigma,
#    expected -2 ... +3 pp.
#  2 stop #165's A2 pass (it pauses at a chunk boundary, within ~5 min, and resumes in step 6). Timed games need the quiet box.
#  3 knob preflight, then pinned bench alternating x3, arm vs ctl. Same code, only tables: nps expected within 1%.
#  4 TIMED self-play, 10+0.1 UHO, arm vs ctl on chess_engine_uci_g67pgo, seeds 167111-167114, 4 x 500 g, with an interim pool
#    after 2 chunks. Registered: +0 ... +20; the ship rule asks >= +2 sigma on the 2,000 g.
#  5 TIMED Mediocre 0.5 at 10+0.1, seed 1671, arm and ctl 300 g each, between chunks 2 and 3. Registered: arm not below ctl by more
#    than its noise (about 34 Elo for 1 sigma). A sanity leg, not a measurement of the gain.
#  6 the rest of the day: the A2 d10 pass again (at most 3 processes, no new shard after 20:00), because the fit's learning curve
#    says more depth rows still help.
# Timed legs start only before 19:30 (owner games from ~21:00). Stop: touch /mnt/e/chess_data/q167.stop (between steps and chunks).
# Never edit this file while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
D=/mnt/e/chess_data/retune
MED="$ND/selfplay/mediocre_uci.sh"
BIN="$ND/build_uci/chess_engine_uci_g67pgo"
LOGD=/mnt/e/chess_data/q167
STOP=/mnt/e/chess_data/q167.stop
TIMED_LATEST=$(date -d '2026-11-06 19:30' +%s)
CUTOFF=$(date -d '2026-11-06 20:00' +%s)
TAG=ours1105
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"; mkdir -p "$LOGD"
show() { sed "s/^/[q167]   /"; }
stopped() { [ -e "$STOP" ] && { echo "[q167] $(date) stop file present before $1 — QUEUE 167 DONE"; exit 0; }; }
busy() { pgrep -f "fastchess-linux-x86-64/fastchess|vs_sf.py|_depth_residual_pass|_build_regret_set|_px_export" >/dev/null; }
pgrep -f "fastchess-linux-x86-64/fastchess|vs_sf.py" >/dev/null && { echo "[q167] a match is running — QUEUE 167 ABORTED"; exit 1; }
START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[q167] $START start"
ENVL=$(cat $D/arm_v3s/env.txt)
KN=$(cat /mnt/e/chess_data/q166/knobs_v3s.txt)
[ -n "$ENVL" ] && [ -n "$KN" ] || { echo "[q167] ☠️ arm S's tables or knobs are missing — QUEUE 167 ABORTED"; exit 1; }
UB=$(printf 'bench\nquit\n' | "$BIN" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+')
UA=$(printf 'setoption name Knobs value %s\nbench\nquit\n' "$KN" | "$BIN" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+')
echo "[q167] gen 6.7 bench (want 251 / 36859850): $UB · arm S (want 246 / 36509830): $UA"
[ "$UB" = "solved 251/300 nodes 36859850" ] && [ "$UA" = "solved 246/300 nodes 36509830" ] || { echo "[q167] ☠️ fingerprints differ — QUEUE 167 ABORTED"; exit 1; }

# ── 1 SF18 @ 1,100 nodes, fresh seed ─────────────────────────────────────────────────────────────────────────────────
stopped "SF18"
echo "[q167] $(date) gauntlet g1100_g67_s1671 (500 games)"
bash "$R" gauntlet 500 1100 3 g1100_g67_s1671 1671 V2_PRESET=shipped 2>&1 | grep -E 'decided|games in' | show
stopped "SF18 arm"
echo "[q167] $(date) gauntlet g1100_v3s_s1671 (500 games)"
bash "$R" gauntlet 500 1100 3 g1100_v3s_s1671 1671 V2_PRESET=shipped $ENVL 2>&1 | grep -E 'decided|games in' | show
echo "[q167] SF18, arm S vs gen 6.7 (registered > -1 sigma):"
bash "$R" pyrun diagnostics/_gauntlet_pair.py g1100_g67_s1671:g1100_v3s_s1671 2>&1 | tail -2 | show

# ── 2 the box goes quiet for the timed legs ──────────────────────────────────────────────────────────────────────────
stopped "stopping A2"
touch /mnt/e/chess_data/q165.stop
echo "[q167] $(date) asked #165's A2 pass to pause"
for i in $(seq 1 40); do grep -qE 'QUEUE 165 (DONE|ABORTED)' /mnt/e/chess_data/q165.log 2>/dev/null && break; sleep 30; done
grep -qE 'QUEUE 165 (DONE|ABORTED)' /mnt/e/chess_data/q165.log 2>/dev/null \
  || { echo "[q167] ☠️ #165 did not stop within 20 min — timed legs not started, QUEUE 167 ABORTED"; exit 1; }
sleep 20
timed_ok() { [ ! -e "$STOP" ] && [ $(date +%s) -le $TIMED_LATEST ] && ! busy; }

# ── 3 preflight + pinned bench ───────────────────────────────────────────────────────────────────────────────────────
PF=0
if timed_ok; then
  rm -f /tmp/q167_pre.engine.log
  timeout 180 "$FC" -engine "cmd=$BIN" name=arm option.OwnBook=false "option.Knobs=$KN" st=0.05 \
    -engine "cmd=$BIN" name=ctl option.OwnBook=false st=0.05 \
    -openings file=$O/openings_uho.epd format=epd order=random -rounds 1 -games 2 -concurrency 1 \
    -log file=/tmp/q167_pre.engine.log level=info engine=true > /tmp/q167_pre.log 2>&1
  W=$(grep -c 'have option' /tmp/q167_pre.log)
  echo "[q167] preflight: $(grep -E '^Games:' /tmp/q167_pre.log | tail -1) · warnings $W · arm echoes $(grep -c 'knobs C1_V2_FIT=1' /tmp/q167_pre.engine.log)"
  [ "$W" = 0 ] && PF=1 || echo "[q167] ☠️ preflight warnings — timed legs skipped"
  if [ $PF = 1 ]; then
    echo "[q167] $(date) pinned bench, alternating x3:"
    for i in 1 2 3; do
      echo "[q167]   ctl $(printf 'bench\nquit\n' | taskset -c 5 "$BIN" 2>/dev/null | grep '^bench')"
      echo "[q167]   arm $(printf 'setoption name Knobs value %s\nbench\nquit\n' "$KN" | taskset -c 5 "$BIN" 2>/dev/null | grep '^bench')"
    done
  fi
else
  echo "[q167] $(date) timed legs skipped (stop, late or a job running)"
fi

# ── 4 + 5 timed self-play with Mediocre in the middle ────────────────────────────────────────────────────────────────
COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
T=rt_v3s_10p1
chunk() {
  local c=$1
  timed_ok || { echo "[q167] $(date) skip $T chunk $c (stop, late or busy)"; return; }
  nice -n 5 "$FC" -engine "cmd=$BIN" name=arm option.OwnBook=false "option.Knobs=$KN" tc=10+0.1 timemargin=250 \
    -engine "cmd=$BIN" name=ctl option.OwnBook=false tc=10+0.1 timemargin=250 \
    $COMMON -srand $(( 167110 + c )) -rounds 250 -games 2 \
    -pgnout file=$O/${T}_c$c.pgn notation=san nodes=true append=false > $O/${T}_c$c.log 2>&1
  echo "[q167] $(date) $T chunk $c: $(grep -E '^Elo:' $O/${T}_c$c.log | tail -1) · warnings $(grep -c 'have option' $O/${T}_c$c.log) · disconnects/stalls $(grep -ciE 'disconnect|stall|not responsive' $O/${T}_c$c.log) · time forfeits $(grep -c 'loses on time' $O/${T}_c$c.pgn)"
}
pool() {
  local f; f=$(ls $O/${T}_c[1-4].pgn 2>/dev/null)
  [ -n "$f" ] || return
  echo "[q167] $(date) $T pooled ($1):"
  timeout 900 bash "$R" pyrun selfplay/fastchess_report.py $f --p1 arm 2>&1 | sed -n '/vs the field/,$p' | show
}
if [ $PF = 1 ]; then
  rm -f $O/${T}_c[1-4].pgn $O/${T}_c[1-4].log
  chunk 1; chunk 2; pool "2 chunks"
  for arm in v3s g67; do
    timed_ok || { echo "[q167] $(date) skip Mediocre $arm (stop, late or busy)"; continue; }
    M=med10p1_$arm
    if [ $arm = v3s ]; then
      nice -n 5 "$FC" -engine "cmd=$BIN" name=ours option.OwnBook=false "option.Knobs=$KN" tc=10+0.1 timemargin=250 restart=on \
        -engine "cmd=$MED" name=mediocre tc=10+0.1 timemargin=250 restart=on $COMMON -srand 1671 -rounds 150 -games 2 \
        -pgnout file=$O/$M.pgn notation=san nodes=true append=false > $O/$M.log 2>&1
    else
      nice -n 5 "$FC" -engine "cmd=$BIN" name=ours option.OwnBook=false tc=10+0.1 timemargin=250 restart=on \
        -engine "cmd=$MED" name=mediocre tc=10+0.1 timemargin=250 restart=on $COMMON -srand 1671 -rounds 150 -games 2 \
        -pgnout file=$O/$M.pgn notation=san nodes=true append=false > $O/$M.log 2>&1
    fi
    echo "[q167] $(date) $M: $(grep -E '^Games:|^Elo:' $O/$M.log | tail -2 | tr '\n' ' ') · time forfeits, all $(grep -c 'loses on time' $O/$M.pgn)"
  done
  chunk 3; chunk 4; pool "all chunks"
fi

# ── 6 A2 d10 pass again until the cut-off ────────────────────────────────────────────────────────────────────────────
late() { [ -e "$STOP" ] || [ $(date +%s) -ge $CUTOFF ]; }
job() {
  local rc
  while :; do
    timeout 3600 env V2_PRESET=shipped PRESET=LONG_FORMAT MAX_DEPTH=10 USE_OPENING_BOOK=0 RESIGN_THRESHOLD=-99999999 \
      bash "$R" pyrun diagnostics/_depth_residual_pass.py IN="$2" OUT="$3" SHARD=$4 CHUNK=800 MOVE=1 >> "$LOGD/$1.log" 2>&1
    rc=$?
    [ $rc -ne 3 ] && break
    late && { echo "[q167] $(date) $1 paused at a chunk boundary (stop or cut-off)"; return; }
  done
  echo "[q167] $(date) $1 done (rc $rc)"
}
JOBS=()
for p in $(ls $D/A2_fens_sf18_s*.csv 2>/dev/null); do
  i=$(basename $p .csv | sed 's/.*_s//')
  for k in 0 1 2; do JOBS+=("A2_f${i}_k$k|$p|$D/A2_${TAG}_d10_f$i.csv|$k/3"); done
done
echo "[q167] $(date) A2 d10 again: ${#JOBS[@]} jobs · rows so far $(cat $D/A2_${TAG}_d10_f*.csv 2>/dev/null | grep -vc '^fen,')"
for j in "${JOBS[@]}"; do
  IFS='|' read -r name in out shard <<< "$j"
  late && { echo "[q167] $(date) not starting $name: stop or cut-off"; continue; }
  while [ $(jobs -rp | wc -l) -ge 3 ]; do wait -n; done
  job "$name" "$in" "$out" "$shard" &
done
wait
echo "[q167] $(date) A2 d10 rows: $(cat $D/A2_${TAG}_d10_f*.csv 2>/dev/null | grep -vc '^fen,') / ~399k"
echo "[q167] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | show
echo "[q167] $(date) QUEUE 167 DONE"
