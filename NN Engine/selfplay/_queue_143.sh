#!/bin/bash
# 2026-11-04 overnight, queue #143 (owner 11-03: chain it after #142): the outside-opponent read for today's two ships.
# Waits for queue #142 to finish, then:
#  1 SF18 at 1100 nodes vs ours at 250k nodes (Cython .so, node-limited, so speed plays no part), 500 games each on one seed
#    (1431): gen 6.5 (V2_PRESET=shipped) and gen 6.4 (the seven correctness fixes at 0). This is the correctness ship's
#    outside check: registered gen 6.5 - gen 6.4 > -1 sigma, expected -3 ... +5 points.
#  2 Mediocre 0.5 at 10+0.1, 400 games each on one seed (1432): today's production binary (gen 6.5 + speed batches 7 + 8) and
#    gen 6.4's production binary. The absolute "where do we stand today" read, speed included: registered production
#    ahead of gen 6.4 by 0 ... +40 Elo.
# Timed legs must finish by 12:00. Stop: touch /mnt/e/chess_data/q143.stop (checked between legs and while waiting).
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
MED="$ND/selfplay/mediocre_uci.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
STOP=/mnt/e/chess_data/q143.stop
Q142=/mnt/e/chess_data/q142.log
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
PROD="$ND/build_uci/chess_engine_uci_g65b8pgo"
G64BIN="$ND/build_uci/chess_engine_uci_g64pgo"
G64ENV="FIX_MAX_NULL=0 NULL_NPM_GUARD=0 FIX_MATE_50=0 FIX_HALFMOVE=0 FIX_REP_KEY=0 FIX_KILLER_SPLICE=0 FIX_MATE_GATE=0"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"
DEADLINE=$(date -d '2026-11-04 12:00' +%s)
echo "[q143] $(date) waiting for queue #142 to finish (stop: $STOP)"
until grep -qE 'QUEUE 142 DONE|QUEUE 142 ABORTED' "$Q142" 2>/dev/null; do
  [ -e "$STOP" ] && { echo "[q143] $(date) stop file present — QUEUE 143 DONE (nothing run)"; exit 0; }
  sleep 60
done
sleep 30
pgrep -f "fastchess" >/dev/null && { echo "[q143] matches are still running after #142 — QUEUE 143 ABORTED"; exit 1; }
START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[q143] $START start"

# ── fingerprints ────────────────────────────────────────────────────────────────────────────────────────────────────
b() { printf 'bench\nquit\n' | "$1" 2>/dev/null | grep '^bench'; }
bp=$(b "$PROD"); bg=$(b "$G64BIN")
echo "[q143] production (want 252 / 36960934): $bp"
echo "[q143] gen 6.4 binary (want 249 / 36991901): $bg"
c65=$(env V2_PRESET=shipped bash "$R" wac q143_g65 V2_PRESET=shipped 2>&1 | grep -E 'SOLVED|NODES' | tr '\n' ' ')
c64=$(env V2_PRESET=shipped $G64ENV bash "$R" wac q143_g64 V2_PRESET=shipped $G64ENV 2>&1 | grep -E 'SOLVED|NODES' | tr '\n' ' ')
echo "[q143] Cython gen 6.5 (want 252 / 36960934): $c65"
echo "[q143] Cython gen 6.4 (want 249 / 36991901): $c64"

# ── 1 SF18 legs ─────────────────────────────────────────────────────────────────────────────────────────────────────
if echo "$c65" | grep -q "NODES: 36960934" && echo "$c64" | grep -q "NODES: 36991901"; then
  for arm in "g65|" "g64|$G64ENV"; do
    t=${arm%%|*}; k=${arm#*|}
    [ -e "$STOP" ] && { echo "[q143] $(date) skip SF18 $t: stop file present"; continue; }
    echo "[q143] $(date) gauntlet g1100_${t}_s1431 (500 games)"
    bash "$R" gauntlet 500 1100 3 g1100_${t}_s1431 1431 V2_PRESET=shipped $k 2>&1 | grep -E "our score|games in" | sed 's/^/[q143]   /'
  done
else
  echo "[q143] ☠️ Cython fingerprints failed — SF18 legs skipped"
fi

# ── 2 Mediocre legs ─────────────────────────────────────────────────────────────────────────────────────────────────
COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
med() {   # $1 tag, $2 binary
  [ -e "$STOP" ] && { echo "[q143] $(date) skip $1: stop file present"; return; }
  local left=$(( DEADLINE - $(date +%s) ))
  [ $left -ge 5400 ] || { echo "[q143] $(date) skip $1: only ${left}s before the deadline"; return; }
  timeout 180 "$FC" -engine cmd="$2" name=ours option.OwnBook=false st=0.05 -engine cmd="$MED" name=mediocre st=0.05 \
    -openings file=$O/openings_uho.epd format=epd order=random -rounds 1 -games 2 -concurrency 1 > /tmp/q143_pre.log 2>&1
  local W; W=$(grep -c 'have option' /tmp/q143_pre.log)
  echo "[q143] $1 preflight: $(grep -E '^Games:' /tmp/q143_pre.log | tail -1) · warnings $W"
  [ "$W" = "0" ] || { echo "[q143] ☠️ $1 PREFLIGHT FAILED — skipped"; return; }
  echo "[q143] $(date) $1 (400 games, 10+0.1)"
  nice -n 5 "$FC" -engine cmd="$2" name=ours option.OwnBook=false tc=10+0.1 timemargin=250 restart=on \
    -engine cmd="$MED" name=mediocre tc=10+0.1 timemargin=250 restart=on $COMMON -srand 1432 -rounds 200 -games 2 \
    -pgnout file=$O/$1.pgn notation=san nodes=true append=false > $O/$1.log 2>&1
  echo "[q143] $(date) $1 done: $(grep -E '^Games:|^Elo:' $O/$1.log | tail -2 | tr '\n' ' ')"
  echo "[q143]   $(grep -E '^Ptnml' $O/$1.log | tail -1)  disconnects/stalls: $(grep -ciE 'disconnect|stall|not responsive' $O/$1.log)  time forfeits: $(grep -c 'loses on time' $O/$1.pgn)"
  timeout 600 bash "$R" pyrun selfplay/fastchess_report.py $O/$1.pgn --p1 ours 2>&1 | sed -n '/vs the field/,$p' | sed 's/^/[q143]   /'
}
if echo "$bp" | grep -q "nodes 36960934 "; then med med10_prod "$PROD"; else echo "[q143] ☠️ production fingerprint failed — med10_prod skipped"; fi
if echo "$bg" | grep -q "nodes 36991901 "; then med med10_g64 "$G64BIN"; else echo "[q143] ☠️ gen 6.4 binary fingerprint failed — med10_g64 skipped"; fi

echo "[q143] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | sed 's/^/[q143]   /'
echo "[q143] $(date) QUEUE 143 DONE"
