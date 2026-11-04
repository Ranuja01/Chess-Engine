#!/bin/bash
# 2026-11-04 morning, queue #144 (owner 11-03: start the next items' runs after #143): FIX_TT_R50 where it fires.
# FIX_TT_R50 (no TT cutoffs at halfmove clock >= 90, where a stored score may not hold on this path) is a narrow item: it is
# byte-identical on the bench and almost never fires from opening starts, so it is judged on a venue built to reach it:
#  0 waits for queue #143 to finish.
#  1 venue: every start of selfplay/eg_critical_fens.txt (real-game positions a few plies before a critical point) with the
#    clocks set to halfmove 80, move 40 (the 50-move draw is 10 plies of no captures or pawn moves away; the move number
#    also gives the clock rule a realistic game ply). PGN openings for fastchess, FENs for the fire count.
#  2 fire count: _fix_firecount.py ON=FIX_TT_R50=1 on 40 venue starts at 30k nodes; 0 divergent games => the match is
#    skipped (a vacuous test).
#  3 preflight: 0 option warnings, the arm echoes its knob, and the engines receive the 80 / 40 counters.
#  4 arm FIX_TT_R50=1 vs control, both the production binary, 10+0.1, 1,000 games as five pausable 200-game chunks
#    (no draw adjudication, so the 50-move rule decides; resign adjudication kept). Registered: ship only at >= +2 sigma here
#    (the narrow-item bar); expected 0 ... +15; elsewhere it is byte-identical. Timed chunks must finish by 14:00.
# Pause: touch /mnt/e/chess_data/q144.pause (waits after the current chunk); stop: touch /mnt/e/chess_data/q144.stop.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
PY=/home/ranuja/anaconda3/bin/python
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
STOP=/mnt/e/chess_data/q144.stop
PAUSE=/mnt/e/chess_data/q144.pause
Q143=/mnt/e/chess_data/q143.log
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
PROD="$ND/build_uci/chess_engine_uci_g65b8pgo"
VFEN=$O/eg_r50_fens.txt
VPGN=$O/eg_r50.pgn
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP" "$PAUSE"
DEADLINE=$(date -d '2026-11-04 14:00' +%s)
echo "[q144] $(date) waiting for queue #143 to finish (stop: $STOP)"
until grep -qE 'QUEUE 143 DONE|QUEUE 143 ABORTED' "$Q143" 2>/dev/null; do
  [ -e "$STOP" ] && { echo "[q144] $(date) stop file present — QUEUE 144 DONE (nothing run)"; exit 0; }
  sleep 60
done
sleep 30
pgrep -f "fastchess" >/dev/null && { echo "[q144] matches are still running after #143 — QUEUE 144 ABORTED"; exit 1; }
START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[q144] $START start"
d=$(printf 'bench\nquit\n' | "$PROD" 2>/dev/null | grep '^bench')
r=$(printf 'setoption name Knobs value FIX_TT_R50=1\nbench\nquit\n' | "$PROD" 2>/dev/null | grep '^bench')
echo "[q144] production (want 252 / 36960934): $d"
echo "[q144] FIX_TT_R50=1 (bench never reaches halfmove 90: want the same): $r"
echo "$d" | grep -q "nodes 36960934 " || { echo "[q144] ☠️ production fingerprint failed — QUEUE 144 ABORTED"; exit 1; }

# ── 1 venue ─────────────────────────────────────────────────────────────────────────────────────────────────────────
"$PY" - "$VFEN" "$VPGN" <<'EOF'
import sys, chess
out_fen, out_pgn = sys.argv[1], sys.argv[2]
fens = []
for raw in open('selfplay/eg_critical_fens.txt'):
    line = raw.split(';')[0].strip()
    if not line or line.startswith('#'):
        continue
    if line.split()[3] != '-':   # an en passant square means the last move was a pawn push: no halfmove 80
        continue
    b = chess.Board(' '.join(line.split()[:4]) + ' 80 40')
    if b.is_valid() and not b.is_game_over():
        fens.append(b.fen())
open(out_fen, 'w').write('\n'.join(fens) + '\n')
with open(out_pgn, 'w') as f:
    for i, fen in enumerate(fens):
        f.write('[Event "eg_r50 %d"]\n[SetUp "1"]\n[FEN "%s"]\n\n*\n\n' % (i, fen))
print('[q144] venue: %d starts (halfmove 80, move 40)' % len(fens))
EOF

# ── 2 fire count ────────────────────────────────────────────────────────────────────────────────────────────────────
fc=$(bash "$R" pyrun diagnostics/_fix_firecount.py BIN="$PROD" N=40 NODES=30000 PLIES=120 SETS=r50:$VFEN ON=FIX_TT_R50=1 JOBS=2 2>&1)
echo "$fc" | grep -E 'diverged|r50|CRASH' | sed 's/^/[q144]   /'
div=$(echo "$fc" | sed -n 's/.*diverged: \([0-9]*\) \/.*/\1/p' | head -1)
[ -n "$div" ] && [ "$div" -gt 0 ] || { echo "[q144] ☠️ FIX_TT_R50 did not change a single game on the venue — match skipped (vacuous). QUEUE 144 DONE"; exit 0; }

# ── 3 preflight ─────────────────────────────────────────────────────────────────────────────────────────────────────
rm -f /tmp/q144_pre.engine.log
timeout 180 "$FC" -engine cmd="$PROD" name=arm option.OwnBook=false option.Knobs=FIX_TT_R50=1 st=0.05 \
  -engine cmd="$PROD" name=ctl option.OwnBook=false st=0.05 \
  -openings file=$VPGN format=pgn order=random -rounds 1 -games 2 -concurrency 1 \
  -log file=/tmp/q144_pre.engine.log level=info engine=true > /tmp/q144_pre.log 2>&1
W=$(grep -c 'have option' /tmp/q144_pre.log)
K=$(grep -c 'knobs FIX_TT_R50=1' /tmp/q144_pre.engine.log)
C=$(grep -cE 'position fen .* 8[0-9] 4[0-9]( moves|$)' /tmp/q144_pre.engine.log)
echo "[q144] preflight: $(grep -E '^Games:' /tmp/q144_pre.log | tail -1) · warnings $W · arm knob echoes $K · position commands with the 80/40 counters $C"
[ "$W" = "0" ] && [ "$K" -ge 1 ] && [ "$C" -ge 1 ] || { echo "[q144] ☠️ PREFLIGHT FAILED — match skipped. QUEUE 144 DONE"; exit 0; }

# ── 4 the match ─────────────────────────────────────────────────────────────────────────────────────────────────────
COMMON="-repeat -concurrency 3 -recover -resign movecount=3 score=1000"
hold() {   # between chunks: honour the pause file, then the stop file and the deadline; $1 tag, $2 seconds needed
  if [ -e "$PAUSE" ]; then
    echo "[q144] $(date) paused before $1 (remove $PAUSE to resume)"
    while [ -e "$PAUSE" ] && [ ! -e "$STOP" ]; do sleep 30; done
    echo "[q144] $(date) resumed"
  fi
  [ -e "$STOP" ] && { echo "[q144] $(date) skip $1: stop file present"; return 1; }
  local left=$(( DEADLINE - $(date +%s) ))
  [ $left -ge $2 ] || { echo "[q144] $(date) skip $1: only ${left}s before the deadline (needs $2)"; return 1; }
}
T=r50_eg
rm -f $O/${T}_c[1-5].pgn $O/${T}_c[1-5].log
for c in 1 2 3 4 5; do
  hold "$T chunk $c" 3000 || break
  nice -n 5 "$FC" -engine cmd="$PROD" name=arm option.OwnBook=false option.Knobs=FIX_TT_R50=1 tc=10+0.1 timemargin=250 \
    -engine cmd="$PROD" name=ctl option.OwnBook=false tc=10+0.1 timemargin=250 \
    -openings file=$VPGN format=pgn order=random $COMMON -srand $(( 1441 * 10 + c )) -rounds 100 -games 2 \
    -pgnout file=$O/${T}_c$c.pgn notation=san nodes=true append=false > $O/${T}_c$c.log 2>&1
  echo "[q144] $(date) $T chunk $c: $(grep -E '^Elo:' $O/${T}_c$c.log | tail -1)  disconnects/stalls: $(grep -ciE 'disconnect|stall|not responsive' $O/${T}_c$c.log)  time forfeits: $(grep -c 'loses on time' $O/${T}_c$c.pgn)"
done
f=$(ls $O/${T}_c[1-5].pgn 2>/dev/null)
if [ -n "$f" ]; then
  echo "[q144] $(date) $T pooled (arm = FIX_TT_R50=1, ctl = production):"
  timeout 600 bash "$R" pyrun selfplay/fastchess_report.py $f --p1 arm 2>&1 | sed -n '/vs the field/,$p' | sed 's/^/[q144]   /'
fi
echo "[q144] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | sed 's/^/[q144]   /'
echo "[q144] $(date) QUEUE 144 DONE"
