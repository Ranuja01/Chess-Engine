#!/bin/bash
# 2026-11-05 ~13:15, queue #159: the time manager's low-clock gap (TIME-MANAGER-RESEARCH 10-25, "known gap"): with no
# increment and about 1-3 s left, the overhead reserve (30 ms x 52) takes the whole clock and the move is played after the
# first iteration. TM_LOWCLOCK=1 keeps at least half of the remaining time (same idea as Stockfish shrinking its horizon
# under 1 s, in a form that scales with Move Overhead). Increment games are unchanged by construction (an increment above
# the overhead makes the reserve negative), so the item is judged where it fires: sudden death.
#  1 plain build chess_engine_uci_tmlc from the repo sources; bench must stay 251 / 36,859,850 (the clock rule is not in it).
#  2 probe: a middlegame position at wtime/btime 1500, no increment, knob off vs on (depth reached, time used).
#  3 self-play 10+0 UHO, arm Knobs=TM_LOWCLOCK=1 vs ctl, same binary, 4 x 200 g, seeds 159111-159114.
#    Registered: +0 ... +30; ship (narrow item) at >= +2 sigma here, with 0 time forfeits for the arm.
# Stop: touch /mnt/e/chess_data/q159.stop (checked between steps and chunks). Never edit this file or the sources while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
BIN="$ND/build_uci/chess_engine_uci_tmlc"
STOP=/mnt/e/chess_data/q159.stop
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"
show() { sed "s/^/[q159]   /"; }
pgrep -f "fastchess-linux-x86-64/fastchess|vs_sf.py" >/dev/null && { echo "[q159] a match is running — QUEUE 159 ABORTED"; exit 1; }
START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[q159] $START start"

# ── 1 build and fingerprint ───────────────────────────────────────────────────────────────────────────────────────────
OUT=chess_engine_uci_tmlc bash selfplay/build_uci.sh || { echo "[q159] build failed — QUEUE 159 ABORTED"; exit 1; }
d=$(printf 'bench\nquit\n' | "$BIN" 2>/dev/null | grep '^bench')
echo "[q159] default bench (want 251 / 36859850): $d"
echo "$d" | grep -q "solved 251/300 nodes 36859850 " || { echo "[q159] ☠️ fingerprint differs — QUEUE 159 ABORTED"; exit 1; }

# ── 2 probe at 1.5 s, no increment ────────────────────────────────────────────────────────────────────────────────────
FEN="r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP1B1PPP/R2QKB1R w KQ - 0 9"
for k in 0 1; do
  for ms in 3000 1500 800; do
    out=$( (printf "setoption name Knobs value TM_LOWCLOCK=$k\nposition fen $FEN\ngo wtime $ms btime $ms\n"; sleep 4; printf 'quit\n') \
      | timeout 20 "$BIN" 2>/dev/null | grep -E '^info depth|^bestmove' | tail -2 | tr '\n' ' ')
    echo "[q159] TM_LOWCLOCK=$k wtime $ms: $(echo "$out" | grep -oE 'info depth [0-9]+' | tail -1) $(echo "$out" | grep -oE 'time [0-9]+' | tail -1) $(echo "$out" | grep -oE 'bestmove [a-h1-8qrbn]+')"
  done
done

# ── 3 self-play 10+0 ──────────────────────────────────────────────────────────────────────────────────────────────────
[ -e "$STOP" ] && { echo "[q159] stop file present before the match — QUEUE 159 DONE"; exit 0; }
rm -f /tmp/q159_pre.engine.log
timeout 180 "$FC" -engine cmd="$BIN" name=arm option.OwnBook=false option.Knobs=TM_LOWCLOCK=1 st=0.05 \
  -engine cmd="$BIN" name=ctl option.OwnBook=false st=0.05 \
  -openings file=$O/openings_uho.epd format=epd order=random -rounds 1 -games 2 -concurrency 1 \
  -log file=/tmp/q159_pre.engine.log level=info engine=true > /tmp/q159_pre.log 2>&1
W=$(grep -c 'have option' /tmp/q159_pre.log)
E=$(grep -c "knobs TM_LOWCLOCK=1" /tmp/q159_pre.engine.log)
echo "[q159] preflight: $(grep -E '^Games:' /tmp/q159_pre.log | tail -1) · warnings $W · arm echoes $E"
if [ "$W" = 0 ] && [ "$E" -ge 1 ]; then
  COMMON="-repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
  T=tmlc10; rm -f $O/${T}_c[1-5].pgn $O/${T}_c[1-5].log
  for c in 1 2 3 4; do
    [ -e "$STOP" ] && { echo "[q159] $(date) skip $T chunk $c: stop file present"; break; }
    nice -n 5 "$FC" -engine cmd="$BIN" name=arm option.OwnBook=false option.Knobs=TM_LOWCLOCK=1 tc=10+0 timemargin=250 \
      -engine cmd="$BIN" name=ctl option.OwnBook=false tc=10+0 timemargin=250 \
      -openings file=$O/openings_uho.epd format=epd order=random $COMMON -srand $(( 159110 + c )) -rounds 100 -games 2 \
      -pgnout file=$O/${T}_c$c.pgn notation=san nodes=true append=false > $O/${T}_c$c.log 2>&1
    echo "[q159] $(date) $T chunk $c: $(grep -E '^Elo:' $O/${T}_c$c.log | tail -1)  disconnects/stalls: $(grep -ciE 'disconnect|stall|not responsive' $O/${T}_c$c.log)  time forfeits: $(grep -c 'loses on time' $O/${T}_c$c.pgn)"
  done
  f=$(ls $O/${T}_c[1-4].pgn 2>/dev/null)
  if [ -n "$f" ]; then
    echo "[q159] $(date) $T pooled:"
    timeout 600 bash "$R" pyrun selfplay/fastchess_report.py $f --p1 arm 2>&1 | sed -n '/vs the field/,$p' | show
    echo "[q159] time forfeits by side:"
    for s in arm ctl; do
      echo "[q159]   $s: $(cat $f | awk -v s=$s '/^\[White /{w=$2; gsub(/[]"[]/,"",w)} /^\[Black /{b=$2; gsub(/[]"[]/,"",b)} /loses on time/{ if ((index($0,"White loses")>0 && w==s) || (index($0,"Black loses")>0 && b==s)) n++ } END {print n+0}')"
    done
  fi
else
  echo "[q159] ☠️ preflight failed — match skipped"
fi
echo "[q159] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | show
echo "[q159] $(date) QUEUE 159 DONE"
