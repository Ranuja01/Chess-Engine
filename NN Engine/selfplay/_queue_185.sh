#!/bin/bash
# 2026-11-08 evening, queue #185 (owner 11-08 "note the best we can do and improve upon it"; replaces #181): the gated lazy eval
# keep_gate (LAZY_V2=7, LAZY_V2_KEEP=1, LAZY_V2_GATE=3: the record fix, guards on, build-1 margins 1500 / 3000, no lazy exit after
# a capture or a null move), on build_uci/chess_engine_uci_lazy3 (no PGO). #184 failed keep_m1000 against SF18 (arm - ctl -4.2 pp,
# z -1.54); the #183 probe put keep_gate's misses at 0.41 per 1,000 asks against keep_m1000's 2.57.
#  0 fingerprints: default 246 / 36,509,830; keep 245 / 38,661,603; keep_m1000 249 / 37,197,972; keep_gate 247 / 39,852,210.
#  1 pinned WAC bench, depth 10, off / keep / keep_gate / keep_m1000 x 3 on core 5: time to depth.
#  2 passed-pawn suites at depth 10 and 12 for keep_gate, paired against #179's per-position files for off / keep / keep_m1000.
#  3 SF18 at 1,100 nodes vs ours at 250,000 nodes (node-limited, safe in the owner's evening), 250-game chunks, concurrency 3:
#    seed 185111 ctl, gate, keep, b1; 185112 b1, keep, gate, ctl; 185113 gate, ctl. No chunk starts after 00:30 on 11-09.
#  4 from 00:45 on 11-09 (or /mnt/e/chess_data/q185.go): timed self-play 10+0.1 UHO, keep_gate vs ctl, seeds 185211-185214,
#    4 x 500 g, concurrency 3; no timed chunk starts after 12:00 on 11-09.
#  5 the critical-situation test at equal time: theme suites + passed-pawn suites + WAC + ksattack, 0.25 s per move, gate and ctl
#    together on two cores.
# Registered before any game:
#   SF18 (step 3): keep_gate PASSES if its score minus the control's is above -1 sigma, the control pooled over #184's 500 games and
#     this queue's 750 (the same engine, lazy off; the fingerprint shows the two binaries agree). keep and b1 are diagnostic only.
#     Prediction: keep_gate - ctl between -2.5 and +2 pp, PASS.
#   Self-play (step 4): NOT HARMFUL at or above -1 sigma, a gain only at +2 sigma. Prediction: 0 to +10 Elo.
#   Passers (step 2): harmful if off beats keep_gate at z >= 2 at one depth with the same sign at the other. Prediction: |z| < 1.
#   A theme suite moving by more than 14 at equal time is flagged for a look either way.
# Nothing is installed. Stop: touch /mnt/e/chess_data/q185.stop (checked between steps and chunks). Never edit this file while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
SF18="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/stockfish_18_linux/stockfish-ubuntu-x86-64-avx2"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
L=/mnt/e/chess_data/lazy
P=/mnt/e/chess_data/puzzles/passer
P179=$L/passer179
OD=$L/passer185
BIN="$ND/build_uci/chess_engine_uci_lazy3"
KG="LAZY_V2=7,LAZY_V2_KEEP=1,LAZY_V2_GATE=3"
STOP=/mnt/e/chess_data/q185.stop
GO=/mnt/e/chess_data/q185.go
SF_LATEST=$(date -d '2026-11-09 00:30' +%s)
EARLIEST=$(date -d '2026-11-09 00:45' +%s)
TIMED_LATEST=$(date -d '2026-11-09 12:00' +%s)
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"
show() { sed "s/^/[q185]   /"; }
busy() { pgrep -f "fastchess-linux-x86-64/fastchess|vs_sf.py|_depth_residual_pass|_build_regret_set|_retune_fit|_ks_footprint_regret|sts_test|tactical_test|_theme_movetime|chess_engine_uci_lzprobe2" >/dev/null; }
kn() {
  case $1 in
    off|ctl) echo "" ;;
    b1) echo "LAZY_V2=7" ;;
    keep) echo "LAZY_V2=7,LAZY_V2_KEEP=1" ;;
    gate) echo "$KG" ;;
    keep_m1000) echo "LAZY_V2=7,LAZY_V2_KEEP=1,LAZY_V2_M=1000,LAZY_V2_M_CHEAP=2000" ;;
  esac
}
echo "[q185] $(date) waiting for a quiet box (stop: $STOP)"
while busy; do [ -e "$STOP" ] && { echo "[q185] stop file while waiting — QUEUE 185 DONE"; exit 0; }; sleep 60; done
START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[q185] $START start"

# ── 0 fingerprints ───────────────────────────────────────────────────────────────────────────────────────────────────
fp() { local k=""; [ -n "$1" ] && k="setoption name Knobs value $1"; printf '%s\nbench\nquit\n' "$k" | "$BIN" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+'; }
F0=$(fp ""); FK=$(fp "$(kn keep)"); FM=$(fp "$(kn keep_m1000)"); FG=$(fp "$KG")
echo "[q185] bench default: $F0 · keep: $FK · keep_m1000: $FM · keep_gate: $FG"
[ "$F0" = "solved 246/300 nodes 36509830" ] && [ "$FK" = "solved 245/300 nodes 38661603" ] && [ "$FM" = "solved 249/300 nodes 37197972" ] \
  && [ "$FG" = "solved 247/300 nodes 39852210" ] || { echo "[q185] ☠️ fingerprints differ — QUEUE 185 ABORTED"; exit 1; }

# ── 1 pinned bench ───────────────────────────────────────────────────────────────────────────────────────────────────
for i in 1 2 3; do
  for v in off keep gate keep_m1000; do
    [ -e "$STOP" ] && break 2
    k=$(kn $v); [ -n "$k" ] && k="setoption name Knobs value $k"
    echo "[q185] pinned WAC d10 $v: $(printf '%s\nbench\nquit\n' "$k" | taskset -c 5 "$BIN" 2>/dev/null | grep -E '^bench' | grep -oE 'time [0-9]+ ms nps [0-9]+')"
  done
done

# ── 2 passed-pawn suites at fixed depth ──────────────────────────────────────────────────────────────────────────────
if [ -e "$STOP" ]; then echo "[q185] stop file before the passer runs"; else
  rm -rf "$OD"; mkdir -p $OD/d10 $OD/d12
  EACH=$OD/d10/gate bash "$R" pyrun diagnostics/_theme_movetime.py "$BIN" 3 d10 "$KG" $P/*.txt > $OD/d10/gate.txt 2>&1 &
  EACH=$OD/d12/gate bash "$R" pyrun diagnostics/_theme_movetime.py "$BIN" 5 d12 "$KG" $P/*.txt > $OD/d12/gate.txt 2>&1 &
  wait
  pair() {  # depth, dir A, name A, dir B, name B
    local d=$1 DA=$2 A=$3 DB=$4 B=$5 a=0 b=0 f x y
    for f in $DA/d$d/$A/*.txt; do
      read x y < <(paste -d' ' "$f" "$DB/d$d/$B/$(basename $f)" | awk '{ if ($2 == 1 && $5 == 0) a++; if ($2 == 0 && $5 == 1) b++ } END { print a + 0, b + 0 }')
      a=$((a + x)); b=$((b + y))
    done
    awk -v d=$d -v A=$A -v B=$B -v a=$a -v b=$b 'BEGIN { printf "  depth %-3s %-11s vs %-11s only the first %4d · only the second %4d · z %+.2f\n", d, A, B, a, b, (a + b > 0 ? (a - b) / sqrt(a + b) : 0) }'
  }
  tot() { awk 'NF == 3 && $2 ~ /^[0-9]+$/ && $3 ~ /^[0-9]+$/ { t += $2 } END { print t + 0 }' "$1"; }
  echo "[q185] $(date) passers solved of 1,500: depth 10 off $(tot $P179/d10/off.txt) keep $(tot $P179/d10/keep.txt) keep_m1000 $(tot $P179/d10/keep_m1000.txt) gate $(tot $OD/d10/gate.txt) · depth 12 off $(tot $P179/d12/off.txt) keep $(tot $P179/d12/keep.txt) keep_m1000 $(tot $P179/d12/keep_m1000.txt) gate $(tot $OD/d12/gate.txt)"
  for d in 10 12; do pair $d $P179 off $OD gate; pair $d $P179 keep $OD gate; pair $d $P179 keep_m1000 $OD gate; done | show
fi

# ── 3 SF18 at fixed nodes ────────────────────────────────────────────────────────────────────────────────────────────
COMMON="-each tc=inf -openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
rm -f /tmp/q185_pre.engine.log
timeout 300 "$FC" -engine "cmd=$BIN" name=ours option.OwnBook=false "option.Knobs=$KG" nodes=250000 \
  -engine "cmd=$SF18" name=sf18 option.Threads=1 option.Hash=16 nodes=1100 \
  -each tc=inf -openings file=$O/openings_uho.epd format=epd order=random -rounds 1 -games 2 -concurrency 1 \
  -log file=/tmp/q185_pre.engine.log level=info engine=true > /tmp/q185_pre.log 2>&1
W=$(grep -c 'have option' /tmp/q185_pre.log)
echo "[q185] SF18 preflight: $(grep -E '^Games:' /tmp/q185_pre.log | tail -1) · warnings $W · arm echoes $(grep -cF "knobs $KG" /tmp/q185_pre.engine.log) · go lines $(grep -cE 'go nodes (250000|1100)' /tmp/q185_pre.engine.log)"
[ "$W" = 0 ] || { echo "[q185] ☠️ SF18 preflight warnings — QUEUE 185 ABORTED"; exit 1; }
T=g18_lzg
chunk() {  # n, arm, seed
  local c=$1 a=$2 seed=$3 k
  [ -e "$STOP" ] && { echo "[q185] stop file before SF18 chunk $c"; return; }
  [ $(date +%s) -gt $SF_LATEST ] && { echo "[q185] $(date) skip SF18 chunk $c (past 00:30)"; return; }
  k=$(kn $a); [ -n "$k" ] && k="option.Knobs=$k"
  nice -n 5 "$FC" -engine "cmd=$BIN" name=ours option.OwnBook=false $k nodes=250000 \
    -engine "cmd=$SF18" name=sf18 option.Threads=1 option.Hash=16 nodes=1100 \
    $COMMON -srand $seed -rounds 125 -games 2 -pgnout file=$O/${T}_c${c}_$a.pgn notation=san nodes=true append=false > $O/${T}_c${c}_$a.log 2>&1
  echo "[q185] $(date) SF18 chunk $c ($a, seed $seed): $(grep -E '^Games:' $O/${T}_c${c}_$a.log | tail -1) · warnings $(grep -c 'have option' $O/${T}_c${c}_$a.log) · disconnects/stalls $(grep -ciE 'disconnect|stall|not responsive' $O/${T}_c${c}_$a.log)"
}
rm -f $O/${T}_c*
chunk 1 ctl 185111; chunk 2 gate 185111; chunk 3 keep 185111; chunk 4 b1 185111
chunk 5 b1 185112; chunk 6 keep 185112; chunk 7 gate 185112; chunk 8 ctl 185112
chunk 9 gate 185113; chunk 10 ctl 185113
sumarm() {  # name, log files...: name n w l d score var-of-mean
  local a=$1; shift
  for f in "$@"; do [ -e "$f" ] && grep -E '^Games:' "$f" | tail -1; done | awk -v a=$a '{
      gsub(/,/, ""); for (i = 1; i <= NF; i++) { if ($i == "Games:") n += $(i+1); if ($i == "Wins:") w += $(i+1); if ($i == "Losses:") l += $(i+1); if ($i == "Draws:") d += $(i+1) } }
    END { s = (n > 0 ? (w + d / 2) / n : 0); v = (n > 0 ? (w + d / 4) / n - s * s : 0); printf "%s %d %d %d %d %.6f %.8f\n", a, n, w, l, d, s, (n > 0 ? v / n : 0) }'
}
{ sumarm ctl_pooled $O/${T}_c*_ctl.log $O/g18_lzm_c1_ctl.log $O/g18_lzm_c4_ctl.log
  sumarm ctl_185 $O/${T}_c*_ctl.log
  sumarm gate $O/${T}_c*_gate.log
  sumarm keep $O/${T}_c*_keep.log
  sumarm b1 $O/${T}_c*_b1.log
  sumarm keep_m1000 $O/g18_lzm_c2_arm.log $O/g18_lzm_c3_arm.log; } > /tmp/q185_read.txt
echo "[q185] $(date) REGISTERED READ, score vs SF18 @ 1,100 nodes (gate passes above -1 sigma against the pooled control):"
awk '{ printf "[q185]   %-11s games %4d  +%d -%d =%d  score %.1f%%  (se %.1f pp)\n", $1, $2, $3, $4, $5, 100 * $6, 100 * sqrt($7) }' /tmp/q185_read.txt
awk '{ s[$1] = $6; v[$1] = $7 } END {
    split("gate keep b1 keep_m1000", A, " ");
    for (i = 1; i <= 4; i++) { a = A[i]; d = s[a] - s["ctl_pooled"]; se = sqrt(v[a] + v["ctl_pooled"]);
      printf "[q185]   %-11s - ctl_pooled %+.2f pp  ± %.2f (95%%)  z %+.2f%s\n", a, 100 * d, 196 * se, (se > 0 ? d / se : 0), (a == "gate" ? (se > 0 && d / se > -1 ? "  -> PASS" : "  -> FAIL") : "  (diagnostic)") }
    d = s["gate"] - s["ctl_185"]; se = sqrt(v["gate"] + v["ctl_185"]);
    printf "[q185]   gate        - ctl_185    %+.2f pp  ± %.2f (95%%)  z %+.2f  (this queue only)\n", 100 * d, 196 * se, (se > 0 ? d / se : 0) }' /tmp/q185_read.txt

# ── 4 timed self-play ────────────────────────────────────────────────────────────────────────────────────────────────
give_up() { [ -e "$STOP" ] || [ $(date +%s) -gt $TIMED_LATEST ]; }
echo "[q185] $(date) waiting for 00:45 on 11-09 (or $GO) and a quiet box for the timed games"
until { [ $(date +%s) -ge $EARLIEST ] || [ -e "$GO" ]; } && ! busy; do
  give_up && { echo "[q185] $(date) stop file or past 12:00 before the timed games — QUEUE 185 DONE"; exit 0; }
  sleep 60
done
sleep 30
TSTART=$(date '+%Y-%m-%d %H:%M:%S')
echo "[q185] $TSTART timed games start"
rm -f /tmp/q185_tpre.engine.log
timeout 180 "$FC" -engine "cmd=$BIN" name=arm option.OwnBook=false "option.Knobs=$KG" st=0.05 \
  -engine "cmd=$BIN" name=ctl option.OwnBook=false st=0.05 \
  -openings file=$O/openings_uho.epd format=epd order=random -rounds 1 -games 2 -concurrency 1 \
  -log file=/tmp/q185_tpre.engine.log level=info engine=true > /tmp/q185_tpre.log 2>&1
W=$(grep -c 'have option' /tmp/q185_tpre.log)
echo "[q185] timed preflight: $(grep -E '^Games:' /tmp/q185_tpre.log | tail -1) · warnings $W · arm echoes $(grep -cF "knobs $KG" /tmp/q185_tpre.engine.log)"
[ "$W" = 0 ] || { echo "[q185] ☠️ timed preflight warnings — QUEUE 185 ABORTED"; exit 1; }
sleep 20
COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
T=lzg_10p1
rm -f $O/${T}_c[1-4].pgn $O/${T}_c[1-4].log
for c in 1 2 3 4; do
  while busy && ! give_up; do sleep 60; done
  give_up && { echo "[q185] $(date) skip $T chunk $c (stop or past 12:00)"; continue; }
  nice -n 5 "$FC" -engine "cmd=$BIN" name=arm option.OwnBook=false "option.Knobs=$KG" tc=10+0.1 timemargin=250 \
    -engine "cmd=$BIN" name=ctl option.OwnBook=false tc=10+0.1 timemargin=250 \
    $COMMON -srand $(( 185210 + c )) -rounds 250 -games 2 \
    -pgnout file=$O/${T}_c$c.pgn notation=san nodes=true append=false > $O/${T}_c$c.log 2>&1
  echo "[q185] $(date) $T chunk $c: $(grep -E '^Elo:' $O/${T}_c$c.log | tail -1) · warnings $(grep -c 'have option' $O/${T}_c$c.log) · disconnects/stalls $(grep -ciE 'disconnect|stall|not responsive' $O/${T}_c$c.log) · time forfeits $(grep -c 'loses on time' $O/${T}_c$c.pgn)"
done
f=$(ls $O/${T}_c[1-4].pgn 2>/dev/null)
if [ -n "$f" ]; then
  out=$(timeout 900 bash "$R" pyrun selfplay/fastchess_report.py $f --p1 arm 2>&1 | sed -n '/vs the field/,$p')
  echo "[q185] $(date) REGISTERED READ, keep_gate vs lazy off, all chunks (not harmful at >= -1 sigma, gain at +2 sigma):"
  echo "$out" | show
  echo "$out" | grep -m1 -E 'Elo [-+]?[0-9.]+' | awk '{gsub(/[()]/,""); gsub(/\+/,""); printf "[q185]   z = %.2f\n", $2 / (($6 - $4) / 3.92)}'
fi

# ── 5 critical-situation test at equal time ──────────────────────────────────────────────────────────────────────────
if [ -e "$STOP" ]; then echo "[q185] stop file before the theme test"; else
  while busy; do sleep 60; done
  TH="$(ls /mnt/e/chess_data/puzzles/themes/*.txt) $(ls $P/*.txt)"; X="$L/wac.txt $L/ksattack.txt"
  echo "[q185] $(date) themes and passers at 0.25 s per move, gate (core 3) and ctl (core 5) together"
  bash "$R" pyrun diagnostics/_theme_movetime.py "$BIN" 3 0.25 "$KG" $TH $X > $L/themes_mt_gate.txt 2>&1 &
  bash "$R" pyrun diagnostics/_theme_movetime.py "$BIN" 5 0.25 - $TH $X > $L/themes_mt_off4.txt 2>&1 &
  wait
  echo "[q185] theme                off   gate  (of n)"
  join <(sort $L/themes_mt_off4.txt) <(sort $L/themes_mt_gate.txt) | awk '{d=$4-$2; flag=(d>14||d<-14)?"  <- flag":""; printf "%-26s %4d %4d  (%d)  %+d%s\n", $1, $2, $4, $3, d, flag; so+=$2; sa+=$4; n+=$3} END {printf "%-26s %4d %4d  (%d)  %+d\n", "TOTAL", so, sa, n, sa-so}' | show
fi
echo "[q185] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | show
echo "[q185] $(date) QUEUE 185 DONE"
