#!/bin/bash
# 2026-10-27 night, queue #111: threats 2x2, chained after #110 (waits for "QUEUE 110 DONE"). Never edit this file while it runs.
# Binary build_uci/chess_engine_uci_thr (plain build: gen 6.2 code + THREAT_ORDER_W). TM = gen 6.2's time manager knobs, both sides.
# TH = the threats eval term as measured 10-08 (self-play @50k +32 +- 9, SF18 @1000 -5 +- 9).
#  1 thr_ord_10   TM,THREAT_ORDER_W=100       vs TM, 10+0.1, 1,000 g  (threat-aware quiet ordering, SF19 form)
#  2 thr_eval_10  TM,TH                       vs TM, 10+0.1, 1,000 g  (threats in eval)
#  3 thr_both_10  TM,TH,THREAT_ORDER_W=100    vs TM, 10+0.1, 1,000 g
#  4 if any arm's point estimate is positive: the best one again on a fresh seed, 1,000 g; otherwise stop.
# Bench (d10, 300 WAC): default 257 / 46,196,950 (= gen 6.2) · W100 256 / 42,263,170 (-8.5% nodes, time-to-depth ~flat)
# · TH 252 / 48,736,019 · both 254 / 46,276,206.
# Registered: 1 -5 ... +10 (SF annotates threat ordering at ~3 Elo); 2 -10 ... +25; 3 above 2 if search-side threat
# awareness is what the eval term lacked (the 10-08 hypothesis), else about 1 + 2.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
BIN="$ND/build_uci/chess_engine_uci_thr"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
TM=TM_START=175,TM_SINGLE=1,TM_SCALE=7,TM_NORM=95,TM_OPT_PLY=1
TH=THREAT_V2_PCT=100,THREAT_V2_HANGING=1,THREAT_V2_KING=1,THREAT_V2_PAWN_TARGETS=1
cd "$ND" || exit 1
echo "[q111] $(date) waiting for queue 110"
until grep -q 'QUEUE 110 DONE' /mnt/e/chess_data/q110.log 2>/dev/null; do sleep 60; done
START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[q111] $START start"
nodes() { printf "$2bench\nquit\n" | "$1" 2>/dev/null | grep '^bench' | sed -n 's/.* nodes \([0-9]*\).*/\1/p'; }
n0=$(nodes "$BIN" ''); nt=$(nodes "$BIN" "setoption name Knobs value $TM\n")
no=$(nodes "$BIN" "setoption name Knobs value $TM,THREAT_ORDER_W=100\n"); ne=$(nodes "$BIN" "setoption name Knobs value $TM,$TH\n")
nb=$(nodes "$BIN" "setoption name Knobs value $TM,$TH,THREAT_ORDER_W=100\n")
echo "[q111] thr: v2 $n0 (want 46196950) TM $nt (want 46196950) ord $no (want 42263170) eval $ne (want 48736019) both $nb (want 46276206)"
[ "$n0" = 46196950 ] && [ "$nt" = 46196950 ] && [ "$no" = 42263170 ] && [ "$ne" = 48736019 ] && [ "$nb" = 46276206 ] \
  || { echo "[q111] ☠️ fingerprints failed — QUEUE 111 ABORTED"; exit 1; }

COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
report() {   # $1 tag
  echo "[q111] $(date) $1 done: $(grep -E '^Games:|^Elo:' $O/$1.log | tail -2 | tr '\n' ' ')"
  echo "[q111]   $(grep -E '^Ptnml' $O/$1.log | tail -1)  disconnects/stalls: $(grep -ciE 'disconnect|stall|not responsive' $O/$1.log)  time forfeits: $(grep -c 'loses on time' $O/$1.pgn)"
}
elo() { grep -E '^Elo:' $O/$1.log | tail -1 | awk '{print $2}' | tr -d ','; }
preflight() {   # $1 tag, $2 arm knobs, $3 control knobs
  timeout 180 "$FC" -engine cmd="$BIN" name=arm option.OwnBook=false option.Knobs=$2 st=0.05 \
    -engine cmd="$BIN" name=ctl option.OwnBook=false option.Knobs=$3 st=0.05 \
    -openings file=$O/openings_uho.epd format=epd order=random -rounds 1 -games 2 -concurrency 1 \
    -log file=/tmp/q111_pre.engine.log level=info engine=true > /tmp/q111_pre.log 2>&1
  local W K N; W=$(grep -c "have option" /tmp/q111_pre.log); K=$(grep -cF "knobs $2" /tmp/q111_pre.engine.log)
  N=$(grep -oE "knobs [A-Z0-9_=,]+" /tmp/q111_pre.engine.log | grep -cxF "knobs $3")
  echo "[q111] $1 preflight: warnings $W, arm echoes $K, control echoes $N"
  [ "$W" = "0" ] && [ "$K" -ge 1 ] && [ "$N" -ge 1 ] || { echo "[q111] ☠️ $1 PREFLIGHT FAILED — skipped"; return 1; }
}
arm() {   # $1 tag, $2 extra knobs, $3 tc, $4 rounds, $5 seed
  preflight $1 "$TM,$2" "$TM" || return
  echo "[q111] $(date) $1 ($((2 * $4)) games, $3)"
  nice -n 5 "$FC" -engine cmd="$BIN" name=arm option.OwnBook=false option.Knobs=$TM,$2 tc=$3 timemargin=250 restart=on \
    -engine cmd="$BIN" name=ctl option.OwnBook=false option.Knobs=$TM tc=$3 timemargin=250 restart=on $COMMON -srand $5 -rounds $4 -games 2 \
    -pgnout file=$O/$1.pgn notation=san nodes=true append=false > $O/$1.log 2>&1
  report $1
}

arm thr_ord_10 THREAT_ORDER_W=100 10+0.1 500 1111
arm thr_eval_10 $TH 10+0.1 500 1112
arm thr_both_10 $TH,THREAT_ORDER_W=100 10+0.1 500 1113

# 4: the best positive arm on a fresh seed
best=""; bestv=0
for t in thr_ord_10:THREAT_ORDER_W=100 thr_eval_10:$TH thr_both_10:$TH,THREAT_ORDER_W=100; do
  tag=${t%%:*}; k=${t#*:}; v=$(elo $tag)
  [ -n "$v" ] && awk -v a="$v" -v b="$bestv" 'BEGIN{exit !(a > b)}' && { best=$tag; bestk=$k; bestv=$v; }
done
if [ -n "$best" ]; then
  echo "[q111] best positive arm: $best ($bestv) — fresh-seed re-run"
  arm ${best}b $bestk 10+0.1 500 1114
else
  echo "[q111] no arm positive — no re-run"
fi

echo "[q111] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | sed 's/^/[q111]   /'
echo "[q111] $(date) QUEUE 111 DONE"
