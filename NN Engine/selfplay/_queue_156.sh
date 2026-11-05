#!/bin/bash
# 2026-11-05 ~09:15, queue #156 — the idle 45 minutes before the owner is back (~10:00). #155 left ENABLE_TT_FLAG_FIX with
# self-play and SF18 both passed (SF18 flag fix on gen 6.6 +0.50 pp, z +0.20; on gen 6.5 +0.30 pp, z +0.12). The one leg of
# the gain-ship rule still missing is Mediocre (not negative). One leg:
#  Mediocre 0.5 at 10+0.1, 300 games, seed 1432 (the same opening order as #143's med10_prod, gen 6.5 + speed batches 7-8:
#  79.6%, +236.8 [+198.8 … +280.4] on 400 g): gen 6.5 + the flag fix, knobs on the production binary (UCI == Cython was
#  checked in #155). Registered: not below #143's gen 6.5 read by more than its noise (point estimate ≥ about +200).
#  Weak instrument (±40 Elo per run); it is the ship rule's sanity leg, not a measurement of the gain.
# Nothing is installed. Starts only before 09:30. Stop: touch /mnt/e/chess_data/q156.stop (checked before the leg).
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
MED="$ND/selfplay/mediocre_uci.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
PROD="$ND/build_uci/chess_engine_uci_g66pgo"
STOP=/mnt/e/chess_data/q156.stop
LATEST=$(date -d '2026-11-05 09:30' +%s)
G65K=FIX_TT_MATE=0,FIX_MDP=0,ENABLE_NULL_MATE_CLAMP=0
AK=$G65K,ENABLE_TT_FLAG_FIX=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"
pgrep -f "fastchess-linux-x86-64/fastchess|vs_sf.py" >/dev/null && { echo "[q156] a match is running — QUEUE 156 ABORTED"; exit 1; }
START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[q156] $START start"
cmp -s build_uci/chess_engine_uci "$PROD" && echo "[q156] production == chess_engine_uci_g66pgo" \
  || { echo "[q156] ☠️ production differs from chess_engine_uci_g66pgo — QUEUE 156 ABORTED"; exit 1; }
b=$(printf "setoption name Knobs value $AK\nbench\nquit\n" | "$PROD" 2>/dev/null | grep '^bench')
echo "[q156] gen 6.5 + flag fix bench (want 251 / 36859850): $b"
echo "$b" | grep -q "solved 251/300 nodes 36859850 " || { echo "[q156] ☠️ fingerprint failed — QUEUE 156 ABORTED"; exit 1; }

T=med10_g65f
[ -e "$STOP" ] && { echo "[q156] $(date) skip $T: stop file present — QUEUE 156 DONE"; exit 0; }
[ $(date +%s) -le $LATEST ] || { echo "[q156] $(date) skip $T: past 09:30 — QUEUE 156 DONE"; exit 0; }
rm -f /tmp/q156_pre.engine.log
timeout 180 "$FC" -engine cmd="$PROD" name=ours option.OwnBook=false option.Knobs=$AK st=0.05 \
  -engine cmd="$MED" name=mediocre st=0.05 \
  -openings file=$O/openings_uho.epd format=epd order=random -rounds 1 -games 2 -concurrency 1 \
  -log file=/tmp/q156_pre.engine.log level=info engine=true > /tmp/q156_pre.log 2>&1
W=$(grep -c 'have option' /tmp/q156_pre.log)
E=$(grep -c "knobs $AK" /tmp/q156_pre.engine.log)
echo "[q156] $T preflight: $(grep -E '^Games:' /tmp/q156_pre.log | tail -1) · warnings $W · knob echoes $E"
[ "$W" = 0 ] && [ "$E" -ge 1 ] || { echo "[q156] ☠️ $T PREFLIGHT FAILED — QUEUE 156 ABORTED"; exit 1; }
COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
echo "[q156] $(date) $T (300 games, 10+0.1)"
nice -n 5 "$FC" -engine cmd="$PROD" name=ours option.OwnBook=false option.Knobs=$AK tc=10+0.1 timemargin=250 restart=on \
  -engine cmd="$MED" name=mediocre tc=10+0.1 timemargin=250 restart=on $COMMON -srand 1432 -rounds 150 -games 2 \
  -pgnout file=$O/$T.pgn notation=san nodes=true append=false > $O/$T.log 2>&1
echo "[q156] $(date) $T done: $(grep -E '^Games:|^Elo:' $O/$T.log | tail -2 | tr '\n' ' ')"
echo "[q156]   $(grep -E '^Ptnml' $O/$T.log | tail -1)  disconnects/stalls: $(grep -ciE 'disconnect|stall|not responsive' $O/$T.log)  time forfeits: $(grep -c 'loses on time' $O/$T.pgn)"
timeout 600 bash "$R" pyrun selfplay/fastchess_report.py $O/$T.pgn --p1 ours 2>&1 | sed -n '/vs the field/,$p' | sed 's/^/[q156]   /'
echo "[q156] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | sed 's/^/[q156]   /'
echo "[q156] $(date) QUEUE 156 DONE"
