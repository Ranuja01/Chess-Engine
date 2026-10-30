#!/bin/bash
# 2026-10-30, queue #122 (replaces #121): FIX_LMR_EG=250 failed in its own venue (lmreg_eg −19.8 ± 12.9 at 860 of 928 g on
# eg_critical, registered −10 … +30: reducing late moves by move number in light endgames costs strength), so its UHO no-harm
# leg is dropped and the caps replication moves into this afternoon. Caps at SF15.1's values: OCBX −42 + 3/piece, ROOKE −28,
# QNOQ −27 + 3/minor, LONEMINOR −64. Binary chess_engine_uci_g64pgo (gen 6.4).
#  1 caps_uho2    caps vs gen 6.4, 10+0.1, 1,000 g UHO, seed 1211 (now if it finishes before 20:55, else after 00:05;
#                 pooled with #119 for the +2σ read)
#  2 caps_rfp500  caps + RFP_MARGIN 500 vs gen 6.4, 10+0.1, 1,000 g, seed 1212 (after 00:05; crank arm)
#  3 caps_60      caps vs gen 6.4, 60+0.6, 200 g, seed 1213 (after 00:05; the longer-clock ship leg)
# Registered: 1 −5 … +20 · 2 −5 … +30 (crank present if it beats caps alone by > 1σ) · 3 −10 … +40.
# Never edit this file while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
G64="$ND/build_uci/chess_engine_uci_g64pgo"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
BASE=THREAT_ORDER_W=100
CAPS=$BASE,POT_V2_WIN_OCBX=-42,POT_V2_WIN_OCBX_PC=3,POT_V2_WIN_ROOKE=-28,POT_V2_WIN_QNOQ=-27,POT_V2_WIN_QNOQ_MINOR=3,POT_V2_WIN_LONEMINOR=-64
CRANK=$CAPS,RFP_MARGIN=500
cd "$ND" || exit 1
wait_clear() { while pgrep -f "fastchess|vs_sf.py|_queue_120.sh|_queue_121.sh" >/dev/null; do sleep 60; done; }
night() { until [ "$(date +%H%M)" -ge 0005 ] && [ "$(date +%H%M)" -lt 1200 ]; do sleep 60; done; }
echo "[q122] $(date) waiting for the box to clear"
wait_clear
START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[q122] $START start"
nodes() { printf "$2bench\nquit\n" | "$1" 2>/dev/null | grep '^bench' | sed -n 's/.* nodes \([0-9]*\).*/\1/p'; }
nb=$(nodes "$G64" "setoption name Knobs value $BASE\n"); nc=$(nodes "$G64" "setoption name Knobs value $CAPS\n")
nk=$(nodes "$G64" "setoption name Knobs value $CRANK\n")
echo "[q122] g64pgo: base $nb (want 36991901) caps $nc (want 38998902) caps+rfp500 $nk (must differ from caps)"
[ "$nb" = 36991901 ] && [ "$nc" = 38998902 ] && [ -n "$nk" ] && [ "$nk" != "$nc" ] || { echo "[q122] ☠️ fingerprints failed — QUEUE 122 ABORTED"; exit 1; }

COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
preflight() {   # $1 binary, $2 arm knobs
  timeout 180 "$FC" -engine cmd="$1" name=arm option.OwnBook=false option.Knobs=$2 st=0.05 \
    -engine cmd="$1" name=ctl option.OwnBook=false option.Knobs=$BASE st=0.05 \
    -openings file=$O/openings_uho.epd format=epd order=random -rounds 1 -games 2 -concurrency 1 \
    -log file=/tmp/q122_pre.engine.log level=info engine=true > /tmp/q122_pre.log 2>&1
  local W K N; W=$(grep -c "have option" /tmp/q122_pre.log); K=$(grep -cF "knobs $2" /tmp/q122_pre.engine.log)
  N=$(grep -oE "knobs [A-Z0-9_=,-]+" /tmp/q122_pre.engine.log | grep -cxF "knobs $BASE")
  echo "[q122] preflight: warnings $W, arm echoes $K, base echoes $N"
  [ "$W" = "0" ] && [ "$K" -ge 1 ] && [ "$N" -ge 1 ]
}
run() {   # $1 tag, $2 binary, $3 arm knobs, $4 tc, $5 seed, $6 rounds
  preflight "$2" "$3" || { echo "[q122] ☠️ preflight failed — skipping $1"; return; }
  echo "[q122] $(date) $1 ($(( $6 * 2 )) games, $4)"
  nice -n 5 "$FC" -engine cmd="$2" name=arm option.OwnBook=false option.Knobs=$3 tc=$4 timemargin=250 \
    -engine cmd="$2" name=g64 option.OwnBook=false option.Knobs=$BASE tc=$4 timemargin=250 $COMMON -srand $5 -rounds $6 -games 2 \
    -pgnout file=$O/$1.pgn notation=san nodes=true append=false > $O/$1.log 2>&1
  echo "[q122] $(date) $1 done: $(grep -E '^Games:|^Elo:' $O/$1.log | tail -2 | tr '\n' ' ')"
  echo "[q122]   $(grep -E '^Ptnml' $O/$1.log | tail -1)  disconnects/stalls: $(grep -ciE 'disconnect|stall|not responsive' $O/$1.log)  time forfeits: $(grep -c 'loses on time' $O/$1.pgn)"
}
left=$(( $(date -d '20:55' +%s) - $(date +%s) ))
if [ $left -lt 10200 ]; then echo "[q122] $(date) caps_uho2 cannot finish before 20:55 (${left}s) — waiting for 00:05"; night; wait_clear; fi
run caps_uho2 "$G64" "$CAPS" 10+0.1 1211 500
echo "[q122] $(date) waiting for 00:05"
night; wait_clear
run caps_rfp500 "$G64" "$CRANK" 10+0.1 1212 500
run caps_60 "$G64" "$CAPS" 60+0.6 1213 100
echo "[q122] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | sed 's/^/[q122]   /'
echo "[q122] $(date) QUEUE 122 DONE"
