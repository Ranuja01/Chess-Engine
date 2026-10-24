#!/bin/bash
# 2026-10-23, queue #96 (owner: "ok lets go"; runs may go into Saturday afternoon). Never edit this file while it runs.
#  1 footprint arm A: q-cache 2^20 alone vs current, st=1, 400 g                          (Knobs QCACHE_LOG2=20)
#  2 footprint arm B: TT 2^22 + move-gen 2^20 + eval 2^20 vs current, st=1, 400 g          (Knobs, one binary)
#  3 build identity at fixed nodes (_fix_firecount.py BIN_ON): current -> speed batch 2, speed batch 2 -> its PGO build
#  4 speed batch 2 vs current, st=1, 400 g          (needs build_uci/sb2.ready AND stage 3 identity for sb2)
#  5 PGO(speed batch 2) vs speed batch 2, st=1, 400 g (needs build_uci/sb2pgo.ready AND stage 3 identity for the PGO build)
# A .ready marker is written only after that binary reproduced BOTH fingerprints (v2 bench + EvalArm 0 bench); each stage
# re-checks the bench itself before playing. A missing marker or a failed check skips the stage with a log line.
# Registered: A >= -20 (no harm expected; q-cache is a search cache) · B -10 ... +30 (TT small +29 earlier, MG -3, eval is
# decision-neutral) · identity 0 divergences, 0 crashes · stage 4 +5 ... +25 (real-play nps +3-10%) · stage 5 +10 ... +30
# (PGO nps +5-15%).
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
CUR="$ND/build_uci/chess_engine_uci_q96cur"
SB2="$ND/build_uci/chess_engine_uci_sb2"
PGO="$ND/build_uci/chess_engine_uci_sb2pgo"
RUNNER="$ND/selfplay/overnight_runner.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
EXIT_AT=${EXIT_AT:-19:00}
V2_NODES=47689586
V1_NODES=35310778
cd "$ND" || exit 1
echo "[q96] $(date) start"

benchnodes() {   # $1 binary, $2 extra setoption line(s); prints the node count, or nothing
  printf "$2bench\nquit\n" | "$1" 2>/dev/null | grep '^bench' | sed -n 's/.* nodes \([0-9]*\).*/\1/p'
}
fingerprints_ok() {   # $1 binary, $2 label
  [ -x "$1" ] || { echo "[q96] ☠️ $2: binary missing"; return 1; }
  local v2 v1
  v2=$(benchnodes "$1" '')
  v1=$(benchnodes "$1" 'setoption name EvalArm value 0\n')
  echo "[q96] $2 bench: v2 nodes '$v2' (want $V2_NODES), v1 nodes '$v1' (want $V1_NODES)"
  [ "$v2" = "$V2_NODES" ] && [ "$v1" = "$V1_NODES" ]
}
secs_left() {
  local s=$(( $(date -d "$EXIT_AT" +%s) - $(date +%s) )); [ $s -lt 0 ] && s=$(( s + 86400 )); echo $s
}

COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
match() {   # $1 tag, $2 seed, $3 knobs for the arm ('' = none), $4 arm binary, $5 control binary
  local secs; secs=$(secs_left)
  [ $secs -lt 900 ] && { echo "[q96] $(date) skip $1: past EXIT_AT"; return; }
  local ko=""; [ -n "$3" ] && ko="option.Knobs=$3"
  timeout 120 "$FC" -engine cmd="$4" name=arm option.OwnBook=false $ko st=0.05 -engine cmd="$5" name=cur option.OwnBook=false st=0.05 \
    -openings file=$O/openings_uho.epd format=epd order=random -rounds 1 -games 2 -concurrency 1 \
    -log file=/tmp/q96_pre.engine.log level=info engine=true > /tmp/q96_pre.log 2>&1
  local W E want
  W=$(grep -c "have option" /tmp/q96_pre.log)
  want="knobs ${3:-(none)}"
  E=$(grep -cF "$want" /tmp/q96_pre.engine.log)
  echo "[q96] $1 preflight: warnings $W, '$want' echoes $E"
  [ "$W" = "0" ] && [ "$E" -ge 1 ] || { echo "[q96] ☠️ $1 PREFLIGHT FAILED — skipped"; return; }
  echo "[q96] $(date) $1 (400 games), up to ${secs}s"
  timeout $secs nice -n 5 "$FC" -engine cmd="$4" name=arm option.OwnBook=false $ko st=1 timemargin=250 restart=on \
    -engine cmd="$5" name=cur option.OwnBook=false st=1 timemargin=250 restart=on $COMMON -srand $2 -rounds 200 -games 2 \
    -pgnout file=$O/$1.pgn notation=san nodes=true append=false > $O/$1.log 2>&1
  echo "[q96] $(date) $1 done (rc $?): $(grep -E '^Games:|^Elo:' $O/$1.log | tail -2 | tr '\n' ' ') $(grep -c 'not responsive' $O/$1.log) unresponsive"
}

identity() {   # $1 tag, $2 old binary, $3 new binary; 0 = every set diverged 0 with 0 crashes
  local log="/mnt/e/chess_data/q96_${1}_identity.log"
  OMP_NUM_THREADS=1 bash "$RUNNER" pyrun diagnostics/_fix_firecount.py BIN="$2" BIN_ON="$3" ON= N=40 NODES=100000 PLIES=200 \
    JOBS=3 SEED=96 > "$log" 2>&1
  local sets zero crash
  sets=$(grep -c '^  diverged:' "$log")
  zero=$(grep -c '^  diverged: 0 /' "$log")
  crash=$(grep '^  diverged:' "$log" | grep -vc 'engine crashes: 0$')
  echo "[q96] $(date) identity $1: sets $sets, sets with 0 divergences $zero, sets with crashes $crash"
  grep '^  diverged:' "$log" | sed 's/^/[q96]   /'
  [ "$sets" = "3" ] && [ "$zero" = "3" ] && [ "$crash" = "0" ]
}

# ── stages 1-2: footprint arms on a frozen copy of today's production binary ─────────────────────────────────────
fingerprints_ok "$CUR" cur || { echo "[q96] ☠️ current binary failed its fingerprints — QUEUE ABORTED"; exit 1; }
match fp_q20  961 QCACHE_LOG2=20                          "$CUR" "$CUR"
match fp_cand 962 TT_LOG2=22,MGCACHE_LOG2=20,EVAL_LOG2=20 "$CUR" "$CUR"

# ── stage 3: build identity at fixed nodes ───────────────────────────────────────────────────────────────────────
SB2_OK=0; PGO_OK=0
if [ -f "$ND/build_uci/sb2.ready" ] && fingerprints_ok "$SB2" sb2; then
  identity sb2 "$CUR" "$SB2" && SB2_OK=1
else
  echo "[q96] $(date) sb2 not ready or failed its fingerprints — stages 3-5 skipped"
fi
if [ $SB2_OK = 1 ] && [ -f "$ND/build_uci/sb2pgo.ready" ] && fingerprints_ok "$PGO" sb2pgo; then
  identity sb2pgo "$SB2" "$PGO" && PGO_OK=1
else
  [ $SB2_OK = 1 ] && echo "[q96] $(date) PGO build not ready, failed its fingerprints, or sb2 not identical — stage 5 skipped"
fi
echo "[q96] identity verdicts: sb2 $SB2_OK, sb2pgo $PGO_OK"

# ── stages 4-5: equal-time strength ──────────────────────────────────────────────────────────────────────────────
[ $SB2_OK = 1 ] && match speed_sb2 963 '' "$SB2" "$CUR"
[ $PGO_OK = 1 ] && match speed_pgo 964 '' "$PGO" "$SB2"
echo "[q96] $(date) QUEUE 96 DONE"
