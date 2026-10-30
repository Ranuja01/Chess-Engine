#!/bin/bash
# 2026-10-30, queue #120: FIX_LMR_EG (search correctness, SEARCH-AUDIT-REDUCTIONS-EXTENSIONS G4). In light endgames
# (phase_score >= 117: pawn endings, minor endings, a single rook on the board) the reduction ignored move number and remaining
# depth and used one absolute target from the iteration depth, capped at ply 12; the fix applies the normal schedule at phase
# scale 250 (continuing the 150 / 175 / 200 / 225 ladder). Binary chess_engine_uci_lmreg (gen 6.4 + the knob, default off):
# bench 249 / 36,991,901 off, 249 / 36,739,004 at 250. Chained after #119.
#  1 lmreg_eg   fix vs gen 6.4, 10+0.1, eg_critical starts (464, 2-4 moves before an endgame critical point; fires by construction)
#  2 lmreg_uho  fix vs gen 6.4, 10+0.1, 1,000 g UHO (no-harm check), skipped if it cannot finish before 20:55 (owner's games)
# Registered: 1 −10 … +30 (correctness rule: ships on no harm; +2σ in the venue counts as positive) · 2 −15 … +10.
# Never edit this file while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
BIN="$ND/build_uci/chess_engine_uci_lmreg"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
BASE=THREAT_ORDER_W=100
ARM=$BASE,FIX_LMR_EG=250
cd "$ND" || exit 1
echo "[q120] $(date) waiting for queue 119"
until grep -q 'QUEUE 119 DONE' /mnt/e/chess_data/q119.log 2>/dev/null; do sleep 60; done
while pgrep -f "fastchess|vs_sf.py" >/dev/null; do sleep 60; done
START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[q120] $START start"
nodes() { printf "$2bench\nquit\n" | "$1" 2>/dev/null | grep '^bench' | sed -n 's/.* nodes \([0-9]*\).*/\1/p'; }
nb=$(nodes "$BIN" "setoption name Knobs value $BASE\n"); na=$(nodes "$BIN" "setoption name Knobs value $ARM\n")
echo "[q120] lmreg: base $nb (want 36991901) fix $na (want 36739004)"
[ "$nb" = 36991901 ] && [ "$na" = 36739004 ] || { echo "[q120] ☠️ fingerprints failed — QUEUE 120 ABORTED"; exit 1; }
echo "[q120] eg_critical.epd: $(wc -l < $O/eg_critical.epd) openings"

ADJ="-repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
preflight() {
  timeout 180 "$FC" -engine cmd="$BIN" name=arm option.OwnBook=false option.Knobs=$ARM st=0.05 \
    -engine cmd="$BIN" name=ctl option.OwnBook=false option.Knobs=$BASE st=0.05 \
    -openings file=$O/openings_uho.epd format=epd order=random -rounds 1 -games 2 -concurrency 1 \
    -log file=/tmp/q120_pre.engine.log level=info engine=true > /tmp/q120_pre.log 2>&1
  local W K N; W=$(grep -c "have option" /tmp/q120_pre.log); K=$(grep -cF "knobs $ARM" /tmp/q120_pre.engine.log)
  N=$(grep -oE "knobs [A-Z0-9_=,-]+" /tmp/q120_pre.engine.log | grep -cxF "knobs $BASE")
  echo "[q120] preflight: warnings $W, fix echoes $K, base echoes $N"
  [ "$W" = "0" ] && [ "$K" -ge 1 ] && [ "$N" -ge 1 ] || { echo "[q120] ☠️ PREFLIGHT FAILED — QUEUE 120 ABORTED"; exit 1; }
}
preflight
run() {   # $1 tag, $2 seed, $3 rounds, $4 openings file
  echo "[q120] $(date) $1 ($(( $3 * 2 )) games, 10+0.1)"
  nice -n 5 "$FC" -engine cmd="$BIN" name=lmreg option.OwnBook=false option.Knobs=$ARM tc=10+0.1 timemargin=250 \
    -engine cmd="$BIN" name=g64 option.OwnBook=false option.Knobs=$BASE tc=10+0.1 timemargin=250 \
    -openings file=$4 format=epd order=random $ADJ -srand $2 -rounds $3 -games 2 \
    -pgnout file=$O/$1.pgn notation=san nodes=true append=false > $O/$1.log 2>&1
  echo "[q120] $(date) $1 done: $(grep -E '^Games:|^Elo:' $O/$1.log | tail -2 | tr '\n' ' ')"
  echo "[q120]   $(grep -E '^Ptnml' $O/$1.log | tail -1)  disconnects/stalls: $(grep -ciE 'disconnect|stall|not responsive' $O/$1.log)  time forfeits: $(grep -c 'loses on time' $O/$1.pgn)"
}
run lmreg_eg 1201 $(wc -l < $O/eg_critical.epd) $O/eg_critical.epd
left=$(( $(date -d '20:55' +%s) - $(date +%s) ))
if [ $left -ge 10200 ]; then run lmreg_uho 1202 500 $O/openings_uho.epd
else echo "[q120] $(date) skip lmreg_uho: only ${left}s before 20:55 (needs ~10,000)"; fi
echo "[q120] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | sed 's/^/[q120]   /'
echo "[q120] $(date) QUEUE 120 DONE"
