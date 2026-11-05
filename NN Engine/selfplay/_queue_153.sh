#!/bin/bash
# 2026-11-04 night, queue #153 (owner 11-04: #152 ends ~02:20 and they are back ~10:00, so fill the idle box). After #152:
#  1 the STS check flagged in #151: gen 6.5's full STS d10 9,620 / STS300 @ 249k 1,890 sit below gen 6.4's 10-30 record
#    (#117: 10,063 / 1,944; STS300 d10 2,003). Gen 6.4's knobs on today's Cython module, same three benches (deterministic).
#    Registered: if they reproduce #117 exactly, the setup is unchanged and the move came from gen 6.5's seven search fixes;
#    if not, something else changed since 10-30, and the five batch-7/8 speed knobs at 0 are run next to locate it.
#  2 ENABLE_TT_FLAG_FIX replicate, fresh seeds 153111-153115, 1,000 g on gen 6.6 (knob on vs off), ONLY if #152's first
#    read passed (z > -1). A replicate after a pass can only revoke it, never rescue a fail. Registered: pooled 2,000 g stay
#    above -1 sigma.
#  3 F15's real-play read (owner 11-04: pass now, re-check later): gen 6.6 built with the hardening kept (HARDEN=1, own PGO
#    profile) vs production; fingerprint, build-identity games, pinned bench, then 600 g at 10+0.1 for the per-engine nps.
#    Registered: identical games; production faster in real play, guess +1 ... +3% (pinned bench #147: +2.4%).
# Nothing is installed. Deadline 2026-11-05 10:00. Never edit this file while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
PROD="$ND/build_uci/chess_engine_uci_g66pgo"
HARD="$ND/build_uci/chess_engine_uci_g66hpgo"
P=/mnt/e/chess_data/pgo_g66h
STOP=/mnt/e/chess_data/q153.stop
PAUSE=/mnt/e/chess_data/q153.pause
DEADLINE=$(date -d '2026-11-05 10:00' +%s)
G64="V2_PRESET=shipped FIX_TT_MATE=0 FIX_MDP=0 ENABLE_NULL_MATE_CLAMP=0 FIX_MAX_NULL=0 NULL_NPM_GUARD=0 FIX_MATE_50=0 FIX_HALFMOVE=0 FIX_REP_KEY=0 FIX_KILLER_SPLICE=0 FIX_MATE_GATE=0"
SPEED0="KS_FAST=0 PAWN_HASH=0 SCRATCH_REUSE=0 PIN_FAST=0 REP_FILTER=0"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"
echo "[q153] $(date) waiting for queue #152 (stop: $STOP)"
until grep -qE 'QUEUE 152 (DONE|ABORTED)' /mnt/e/chess_data/q152.log 2>/dev/null; do
  [ -e "$STOP" ] && { echo "[q153] stop file present — QUEUE 153 DONE (nothing run)"; exit 0; }
  sleep 30
done
sleep 20
pgrep -f "fastchess-linux-x86-64/fastchess|vs_sf.py" >/dev/null && { echo "[q153] a match is running — QUEUE 153 ABORTED"; exit 1; }
START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[q153] $START start"
cmp -s build_uci/chess_engine_uci "$PROD" && echo "[q153] production == chess_engine_uci_g66pgo" \
  || { echo "[q153] ☠️ production differs from chess_engine_uci_g66pgo — QUEUE 153 ABORTED"; exit 1; }

hold() {   # $1 tag, $2 seconds needed
  if [ -e "$PAUSE" ]; then
    echo "[q153] $(date) paused before $1 (remove $PAUSE to resume)"
    while [ -e "$PAUSE" ] && [ ! -e "$STOP" ]; do sleep 30; done
    echo "[q153] $(date) resumed"
  fi
  [ -e "$STOP" ] && { echo "[q153] $(date) skip $1: stop file present"; return 1; }
  local left=$(( DEADLINE - $(date +%s) ))
  [ $left -ge $2 ] || { echo "[q153] $(date) skip $1: only ${left}s before the deadline (needs $2)"; return 1; }
}
COMMON="-repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
match() {   # $1 tag, $2 seed base, $3 arm binary, $4 arm knobs (or -), $5 ctl binary, $6 chunks of 200 games
  local T=$1 c ak=""
  [ "$4" != "-" ] && ak="option.Knobs=$4"
  rm -f $O/${T}_c[1-5].pgn $O/${T}_c[1-5].log
  for c in $(seq 1 $6); do
    hold "$T chunk $c" 2400 || break
    nice -n 5 "$FC" -engine cmd="$3" name=arm option.OwnBook=false $ak tc=10+0.1 timemargin=250 \
      -engine cmd="$5" name=ctl option.OwnBook=false tc=10+0.1 timemargin=250 \
      -openings file=$O/openings_uho.epd format=epd order=random $COMMON -srand $(( $2 + c )) -rounds 100 -games 2 \
      -pgnout file=$O/${T}_c$c.pgn notation=san nodes=true append=false > $O/${T}_c$c.log 2>&1
    echo "[q153] $(date) $T chunk $c: $(grep -E '^Elo:' $O/${T}_c$c.log | tail -1)  disconnects/stalls: $(grep -ciE 'disconnect|stall|not responsive' $O/${T}_c$c.log)  time forfeits: $(grep -c 'loses on time' $O/${T}_c$c.pgn)"
  done
}
report() {   # pooled report over the given pgns
  echo "[q153] $(date) $1 pooled:"; shift
  timeout 600 bash "$R" pyrun selfplay/fastchess_report.py "$@" --p1 arm 2>&1 | sed -n '/vs the field/,$p' | sed 's/^/[q153]   /'
}

# ── 1 STS check: gen 6.4 knobs on today's Cython module ──────────────────────────────────────────────────────────────
echo "[q153] $(date) == STS check, gen 6.4 knobs (#117 record: STS300 d10 2,003 · full d10 10,063 · STS300 @249k 1,944)"
echo "[q153] g64k WAC d10 (want 249 / 36991901): $(bash "$R" wac fp_g64k $G64 2>&1 | grep -E '^SOLVED|^NODES|^EBF' | tr '\n' ' ')"
s300=$(bash "$R" sts rec_g64k $G64 | tail -1)
sfull=$(bash "$R" sts_full recf_g64k $G64 | grep 'STS score' | tail -1)
seqn=$(bash "$R" sts sts_eqn_g64k $G64 MAX_DEPTH=64 NODE_LIMIT=249014 | tail -1)
echo "[q153] g64k STS300 d10: $s300"
echo "[q153] g64k STS full d10: $sfull"
echo "[q153] g64k STS300 @249k: $seqn"
if echo "$sfull" | grep -q 'STS score: 10063/' && echo "$seqn" | grep -q 'STS score: 1944/'; then
  echo "[q153] ⇒ reproduces #117: setup unchanged; gen 6.5's move came from its seven search fixes"
else
  echo "[q153] ⇒ does NOT reproduce #117; the batch-7/8 speed knobs at 0 next ($SPEED0)"
  echo "[q153] g64k+speed0 WAC d10: $(bash "$R" wac fp_g64s $G64 $SPEED0 2>&1 | grep -E '^SOLVED|^NODES|^EBF' | tr '\n' ' ')"
  echo "[q153] g64k+speed0 STS full d10: $(bash "$R" sts_full recf_g64s $G64 $SPEED0 | grep 'STS score' | tail -1)"
  echo "[q153] g64k+speed0 STS300 @249k: $(bash "$R" sts sts_eqn_g64s $G64 $SPEED0 MAX_DEPTH=64 NODE_LIMIT=249014 | tail -1)"
fi

# ── 2 ENABLE_TT_FLAG_FIX replicate (only after a first-read pass) ────────────────────────────────────────────────────
L=$(sed -n '/ttflag_uho pooled:/,$p' /mnt/e/chess_data/q152.log | grep -m1 -E 'Elo [-+]?[0-9]')
N=$(sed -n '/ttflag_uho pooled:/,$p' /mnt/e/chess_data/q152.log | grep -m1 -oE 'field: [0-9]+ games' | grep -oE '[0-9]+')
read e lo hi < <(echo "$L" | sed -E 's/.*Elo ([-+]?[0-9.]+) +\(95% ([-+]?[0-9.]+) [^0-9+-]+ ([-+]?[0-9.]+)\).*/\1 \2 \3/')
z=$(awk -v e="$e" -v lo="$lo" -v hi="$hi" 'BEGIN{ s=(hi-lo)/3.92; if (s > 0) printf "%.2f", e/s; else print "nan" }')
echo "[q153] $(date) == ENABLE_TT_FLAG_FIX first read (#152): ${N:-?} games, Elo ${e:-?} (${lo:-?} … ${hi:-?}), z ${z}"
if [ "${N:-0}" -ge 800 ] && awk -v z="$z" 'BEGIN{ exit !(z != "nan" && z > -1) }'; then
  echo "[q153] ⇒ first read passed: replicate on fresh seeds"
  match ttflag_rep 153110 "$PROD" ENABLE_TT_FLAG_FIX=1 "$PROD" 5
  f=$(ls $O/ttflag_rep_c[1-5].pgn 2>/dev/null)
  [ -n "$f" ] && report "ttflag_rep" $f
  f0=$(ls $O/ttflag_uho_c[1-5].pgn 2>/dev/null)
  [ -n "$f" ] && report "ttflag first read + replicate" $f0 $f
else
  echo "[q153] ⇒ first read did not pass (or is incomplete): no replicate"
fi

# ── 3 F15 real-play read: hardened gen 6.6 vs production ─────────────────────────────────────────────────────────────
if hold "F15 build" 3600; then
  echo "[q153] $(date) == F15: gen 6.6 with the hardening kept (HARDEN=1), own PGO profile"
  rm -rf "$P"; mkdir -p "$P"
  if HARDEN=1 OUT=chess_engine_uci_g66hgen PGO=gen PGO_DIR=$P bash selfplay/build_uci.sh; then
    nice -n 5 "$FC" -engine cmd="$ND/build_uci/chess_engine_uci_g66hgen" name=a option.OwnBook=false st=0.25 \
      -engine cmd="$ND/build_uci/chess_engine_uci_g66hgen" name=b option.OwnBook=false st=0.25 \
      -openings file=$O/openings_uho.epd format=epd order=random -srand 65 -rounds 4 -games 2 -concurrency 1 > /tmp/pgo_g66h_train.log 2>&1
    echo "[q153] training: $(grep -E '^Games:' /tmp/pgo_g66h_train.log | tail -1) · profile files $(find "$P" -name '*.gcda' | wc -l)"
    HARDEN=1 OUT=chess_engine_uci_g66hpgo PGO=use PGO_DIR=$P bash selfplay/build_uci.sh || echo "[q153] ☠️ hardened use build failed"
  else
    echo "[q153] ☠️ hardened gen build failed"
  fi
  if [ -x "$HARD" ]; then
    h=$(printf 'bench\nquit\n' | "$HARD" 2>/dev/null | grep '^bench')
    echo "[q153] hardened default (want 252 / 39257155): $h"
    if echo "$h" | grep -q "solved 252/300 nodes 39257155 "; then
      echo "[q153] $(date) build identity: hardened vs production, fixed 30k nodes, 40 starts per set, one process per side"
      bash "$R" pyrun diagnostics/_fix_firecount.py BIN="$HARD" BIN_ON="$PROD" ON= N=40 NODES=30000 PLIES=200 SIDES=2 JOBS=2 2>&1 \
        | grep -E '===|diverged|CRASH' | sed 's/^/[q153]   /'
      echo "[q153] $(date) pinned bench, hardened vs production, alternating x6"
      for r in 1 2 3 4 5 6; do
        echo "[q153]   hard $(printf 'bench\nquit\n' | taskset -c 5 "$HARD" 2>/dev/null | grep '^bench')"
        echo "[q153]   prod $(printf 'bench\nquit\n' | taskset -c 5 "$PROD" 2>/dev/null | grep '^bench')"
      done
      echo "[q153] $(date) == F15 real play: production (arm) vs hardened (ctl), 600 g"
      match f15_rp 153210 "$PROD" - "$HARD" 3
      f=$(ls $O/f15_rp_c[1-3].pgn 2>/dev/null)
      [ -n "$f" ] && report "f15_rp" $f
    else
      echo "[q153] ☠️ hardened fingerprint differs — F15 read skipped"
    fi
  fi
fi

echo "[q153] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | sed 's/^/[q153]   /'
echo "[q153] $(date) QUEUE 153 DONE"
