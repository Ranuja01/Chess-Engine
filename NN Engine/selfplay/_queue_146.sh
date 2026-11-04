#!/bin/bash
# 2026-11-04 morning, queue #146: idle-box external read of today's production (gen 6.5 + speed batches 7 and 8) against
# Stockfish 1.1 at the standing 3x time odds (us 60+0.6, it 20+0.2), the same match and seed as #108 / #118 (seed 1082 = the
# same openings). Series so far: gen 6.1 22.5% · 6.2 21.0% · 6.3 29.25% · 6.4 29.5% (#118, plain thr2 build).
# Registered: production scores >= 26.5% (gen 6.4 - 1 sigma), expected 31 ... 37% (production is +13% nps over gen 6.4's PGO
# build in real play and PGO over the plain build #118 used). A series point, not a ship gate; 200 games resolve about +-6 pp.
# One fastchess run of ~2 h 15 min; it does not start after 08:20 so the box is free by ~10:35. Never edit this file while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
BIN="$ND/build_uci/chess_engine_uci"
SF1="/mnt/e/chess_data/engines/sf1.1/stockfish-1.1"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
STOP=/mnt/e/chess_data/q146.stop
T=prod_sf11_odds
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"
START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[q146] $START start"
pgrep -f "fastchess|vs_sf.py" >/dev/null && { echo "[q146] another match is running — QUEUE 146 ABORTED"; exit 1; }
[ "$(date +%H%M)" -le 0820 ] || { echo "[q146] too late to finish before ~10:35 — QUEUE 146 DONE (nothing run)"; exit 0; }
fp=$(printf "bench\nquit\n" | "$BIN" 2>/dev/null | grep '^bench')
echo "[q146] production (want 252 / 36960934): $fp"
echo "$fp" | grep -q "solved 252/300 nodes 36960934 " || { echo "[q146] ☠️ fingerprint failed — QUEUE 146 ABORTED"; exit 1; }

COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
SF1OPT="option.Threads=1 option.Ponder=false option.OwnBook=false option.Hash=128"
# Preflight: both engines start, play, and accept every option (0 "have option" warnings).
timeout 180 "$FC" -engine cmd="$BIN" name=ours option.OwnBook=false st=0.05 \
  -engine cmd="$SF1" name=sf1.1 $SF1OPT st=0.05 \
  -openings file=$O/openings_uho.epd format=epd order=random -rounds 1 -games 2 -concurrency 1 > /tmp/q146_pre.log 2>&1
W=$(grep -c "have option" /tmp/q146_pre.log)
echo "[q146] preflight: $(grep -E '^Games:' /tmp/q146_pre.log | tail -1) · warnings $W"
[ "$W" = "0" ] && grep -qE '^Games: 2' /tmp/q146_pre.log || { echo "[q146] ☠️ PREFLIGHT FAILED — QUEUE 146 ABORTED"; exit 1; }
[ -e "$STOP" ] && { echo "[q146] $(date) stop file present — QUEUE 146 DONE (nothing run)"; exit 0; }

echo "[q146] $(date) $T (200 games: ours 60+0.6, sf1.1 20+0.2, seed 1082)"
nice -n 5 "$FC" -engine cmd="$BIN" name=ours option.OwnBook=false tc=60+0.6 timemargin=250 restart=on \
  -engine cmd="$SF1" name=sf1.1 $SF1OPT tc=20+0.2 timemargin=250 restart=on $COMMON -srand 1082 -rounds 100 -games 2 \
  -pgnout file=$O/$T.pgn notation=san nodes=true append=false > $O/$T.log 2>&1
echo "[q146] $(date) $T done: $(grep -E '^Games:|^Elo:' $O/$T.log | tail -2 | tr '\n' ' ')"
echo "[q146]   $(grep -E '^Ptnml' $O/$T.log | tail -1)  disconnects/stalls: $(grep -ciE 'disconnect|stall|not responsive' $O/$T.log)  time forfeits: $(grep -c 'loses on time' $O/$T.pgn)"
timeout 600 bash "$R" pyrun selfplay/fastchess_report.py $O/$T.pgn --p1 ours 2>&1 | sed -n '/vs the field/,$p' | sed 's/^/[q146]   /'
echo "[q146] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | sed 's/^/[q146]   /'
echo "[q146] $(date) QUEUE 146 DONE"
