#!/bin/bash
# 2026-11-02 afternoon, queue #135. Never edit this file while it runs.
# Rotates Stockfish 1.1 (2008) in as an external opponent with a TIME HANDICAP (owner 10-25). At equal time gen 6.1 scored
# 6.5% (-463, queue #103), which cannot show progress; the handicap is calibrated so that we land at ~25-40%.
#  1 fingerprint: production gen 6.4 (PGO) bench 36,991,901
#  2 gen 6.4 at 60+0.6 vs SF 1.1 at 15+0.15 (1/4 time), 100 g
#  3 gen 6.4 at 60+0.6 vs SF 1.1 at 7.5+0.075 (1/8 time), 100 g, only if it can finish by 18:00
# -recover; disconnects and time forfeits counted per engine; new WSL crash dumps listed. Stop file: /mnt/e/chess_data/q135.stop
# Registered (no ship, a calibration): 1/4 time 15 ... 40%; 1/8 time 25 ... 55%; SF 1.1 time forfeits <= 2 per leg at 1/8.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
PROD="$ND/build_uci/chess_engine_uci"
SF1="/mnt/e/chess_data/engines/sf1.1/stockfish-1.1"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
STOP=/mnt/e/chess_data/q135.stop
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
DEADLINE=$(date -d '2026-11-02 18:00' +%s)
cd "$ND" || exit 1
rm -f "$STOP"
START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[q135] $START start (timed legs must finish by $(date -d @$DEADLINE); stop file $STOP)"
nodes() { printf "$2bench\nquit\n" | "$1" 2>/dev/null | grep '^bench' | sed -n 's/.* nodes \([0-9]*\).*/\1/p'; }
p2=$(nodes "$PROD" '')
echo "[q135] production gen 6.4: bench $p2 (want 36991901)"
[ "$p2" = 36991901 ] && [ -x "$SF1" ] || { echo "[q135] ☠️ fingerprint or SF 1.1 binary failed — QUEUE 135 ABORTED"; exit 1; }

COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
SF1OPT="option.Threads=1 option.Ponder=false option.OwnBook=false option.Hash=128"
report() {   # $1 tag
  echo "[q135] $(date) $1 done: $(grep -E '^Games:|^Elo:' $O/$1.log | tail -2 | tr '\n' ' ')"
  echo "[q135]   $(grep -E '^Ptnml' $O/$1.log | tail -1)  disconnects/stalls: $(grep -ciE 'disconnect|stall|not responsive' $O/$1.log)"
  echo "[q135]   time forfeits by engine: $(awk '/^\[White /{w=$2; gsub(/[]"]/,"",w)} /^\[Black /{b=$2; gsub(/[]"]/,"",b)}
    /White loses on time/{c[w]++} /Black loses on time/{c[b]++} END{n=0; for (k in c) {printf "%s %d  ", k, c[k]; n++} if (!n) printf "none"}' $O/$1.pgn)"
}
leg() {   # $1 tag, $2 SF 1.1 time control, $3 seed
  echo "[q135] $(date) $1 (100 games, g64 60+0.6 vs sf1.1 $2)"
  nice -n 5 "$FC" -engine cmd="$PROD" name=g64 option.OwnBook=false tc=60+0.6 timemargin=250 restart=on \
    -engine cmd="$SF1" name=sf1.1 $SF1OPT tc=$2 timemargin=250 restart=on $COMMON -srand $3 -rounds 50 -games 2 \
    -pgnout file=$O/$1.pgn notation=san nodes=true append=false > $O/$1.log 2>&1
  report $1
}

# Preflight: 2 short games, 0 "doesn't have option" warnings, both engines produce moves.
timeout 120 "$FC" -engine cmd="$PROD" name=g64 option.OwnBook=false st=0.1 -engine cmd="$SF1" name=sf1.1 $SF1OPT st=0.1 \
  -openings file=$O/openings_uho.epd format=epd order=random -rounds 1 -games 2 -concurrency 1 -pgnout file=/tmp/q135_pre.pgn > /tmp/q135_pre.log 2>&1
W=$(grep -c "have option" /tmp/q135_pre.log); G=$(grep -c '^\[Result ' /tmp/q135_pre.pgn 2>/dev/null)
echo "[q135] preflight: warnings $W, games $G"
[ "$W" = "0" ] && [ "$G" = "2" ] || { echo "[q135] ☠️ PREFLIGHT FAILED — QUEUE 135 ABORTED (see /tmp/q135_pre.log)"; exit 1; }

# A 100-game leg takes ~70-80 min at concurrency 3; start one only if 90 min remain.
for spec in "g64_sf11_q4 15+0.15 1351" "g64_sf11_q8 7.5+0.075 1352"; do
  set -- $spec
  [ -e "$STOP" ] && { echo "[q135] stop file — skip $1"; continue; }
  [ $(( $(date +%s) + 90 * 60 )) -le "$DEADLINE" ] || { echo "[q135] skip $1: cannot finish by 18:00"; continue; }
  leg "$1" "$2" "$3"
done

echo "[q135] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | sed 's/^/[q135]   /'
echo "[q135] $(date) QUEUE 135 DONE"
