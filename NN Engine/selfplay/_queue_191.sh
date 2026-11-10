#!/bin/bash
# 2026-11-10 ~01:10, queue #191: replays #189's lost stage-2 chunk (c2r vs gen 7.1, seed 189222) after #190, so c2r's
# registered read is on 2,000 games as #189 registered it. The chunk was lost to a bug in #189's helpers: sf_block's loop
# variable `a` was not local, and bash locals are dynamically scoped, so the evening SF18 block (run from wait_timed inside
# that chunk's tchunk) left tchunk's local `a` = ctl, kn returned "" and fastchess refused "option.Knobs=" (no games, 00:00).
# #190's helpers keep their variables local and do not share it.
# Registered: #189's c2r bars, unchanged (gain at >= +2 sigma; not harmful at >= -1 sigma; c2r - c2q >= +10 = the tighter
# margin pays, a screen). The read is the same fastchess_report over all 8 c2r chunks.
# Timed: the chunk starts only before 10:00 on 11-10 (#189's window); later, it does not run and c2r stays at 1,750 games.
# Stop: touch /mnt/e/chess_data/q191.stop. It does not run if #189 or #190 was stopped.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
BIN="$ND/build_uci/chess_engine_uci_lazy4"
C2R="CORR2=1,CORR2_WP=88,CORR2_WN=100,RFP_MARGIN=500"
STOP=/mnt/e/chess_data/q191.stop
END=$(date -d '2026-11-10 10:00' +%s)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"
show() { sed "s/^/[q191]   /"; }
busy() { pgrep -f "fastchess-linux-x86-64/fastchess|vs_sf.py|_depth_residual_pass|_build_regret_set|_retune_fit|sts_test|tactical_test|_theme_movetime|chess_engine_uci_lzprobe2" >/dev/null; }

echo "[q191] $(date '+%Y-%m-%d %H:%M:%S') queued; waiting for #189 and #190"
while pgrep -f "_queue_189.sh" >/dev/null || pgrep -f "_queue_190.sh" >/dev/null; do
  [ -e "$STOP" ] && { echo "[q191] $(date) stopped while waiting: QUEUE 191 NOT RUN"; exit 0; }
  sleep 60
done
while busy; do sleep 30; done
echo "[q191] $(date '+%Y-%m-%d %H:%M:%S') start"
for s in /mnt/e/chess_data/q189.stop /mnt/e/chess_data/q190.stop "$STOP"; do
  [ -e "$s" ] && { echo "[q191] $s present: QUEUE 191 NOT RUN"; exit 0; }
done
[ $(date +%s) -lt $END ] || { echo "[q191] past 10:00: QUEUE 191 NOT RUN (c2r stays at 1,750 games)"; exit 0; }

F=$(printf 'setoption name Knobs value %s\nbench\nquit\n' "$C2R" | "$BIN" 2>/dev/null | grep -E '^bench|lazy eval' | sed -E 's/ time [0-9]+ ms nps [0-9]+//')
echo "[q191] c2r (want 245 / 36768650, failed to cut 0):"; echo "$F" | show
echo "$F" | grep -q 'solved 245/300 nodes 36768650$' && echo "$F" | grep -q 'failed to cut 0$' \
  || { echo "[q191] ☠️ fingerprint or the failed-to-cut check — QUEUE 191 ABORTED"; exit 1; }

T=c2r_10p1; SEED=189222
nice -n 5 "$FC" -engine "cmd=$BIN" name=arm option.OwnBook=false "option.Knobs=$C2R" tc=10+0.1 timemargin=250 \
  -engine "cmd=$BIN" name=ctl option.OwnBook=false tc=10+0.1 timemargin=250 \
  -openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10 \
  -srand $SEED -rounds 125 -games 2 -pgnout file=$O/${T}_s$SEED.pgn notation=san nodes=true append=false > $O/${T}_s$SEED.log 2>&1
echo "[q191] $(date) $T seed $SEED: $(grep -E '^Elo:' $O/${T}_s$SEED.log | tail -1) · $(grep -E '^Games:' $O/${T}_s$SEED.log | tail -1 | cut -d, -f1) · warnings $(grep -c 'have option' $O/${T}_s$SEED.log) · disconnects/stalls $(grep -ciE 'disconnect|stall|not responsive' $O/${T}_s$SEED.log) · time forfeits $(grep -c 'loses on time' $O/${T}_s$SEED.pgn 2>/dev/null)"

tread() {  # tag, label, rule
  local f out
  f=$(ls $O/$1_s*.pgn 2>/dev/null)
  [ -n "$f" ] || { echo "[q191] $2: no games"; ELO=""; return; }
  out=$(timeout 900 bash "$R" pyrun selfplay/fastchess_report.py $f --p1 arm 2>&1 | sed -n '/vs the field/,$p')
  echo "[q191] $(date) REGISTERED READ, $2 ($3):"
  echo "$out" | show
  ELO=$(echo "$out" | grep -m1 -E 'Elo [-+]?[0-9.]+' | awk '{gsub(/[()]/,""); gsub(/\+/,""); print $2}')
  echo "$out" | grep -m1 -E 'Elo [-+]?[0-9.]+' | awk '{gsub(/[()]/,""); gsub(/\+/,""); printf "[q191]   z = %.2f\n", $2 / (($6 - $4) / 3.92)}'
}
ELO=""
tread c2r_10p1 "c2r vs gen 7.1 at 10+0.1, all 8 chunks" "gain at +2 sigma, not harmful at >= -1 sigma"; ER=$ELO
tread c2q_10p1 "c2q vs gen 7.1 at 10+0.1, #189's games read again" "for the screen"; EQ=$ELO
[ -n "$ER" ] && [ -n "$EQ" ] && awk -v r="$ER" -v q="$EQ" 'BEGIN { printf "[q191]   c2r - c2q = %+.1f Elo (the tighter margin pays if >= +10; a screen)\n", r - q }'
echo "[q191] $(date) QUEUE 191 DONE"
