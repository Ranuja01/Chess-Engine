#!/bin/bash
# 2026-11-14 ~23:35, queue #244: games for the best switch from #242/#243 while the owner is away (back ~11:00 Sunday).
# Waits for queue #241 (D7's last games, Mediocre, SF18) to finish, then:
#  1 PGO build chess_engine_uci_s6pgo of today's sources (all audit switches, default 0; profile /mnt/e/chess_data/pgo_s6),
#    built while light queue #245 may still run; gates: default 249 / 37,279,382 · D7 250 / 37,320,382 · D7=2 254 / 35,430,812.
#  2 writes q245.stop and waits for #245 to end (no side runs during timed games), checks no other games run.
#  3 games at 10+0.1, D7=2 (SV2_VERIFY_DEEPER=2) vs D7 (SV2_VERIFY_DEEPER=1), BOTH on s6pgo (same binary: the arms differ
#    only by the knob), seed 2441, up to 1,000 pairs; SIGINT at 12:00 Sunday (weekend-afternoon gaming); finished games kept.
#    REGISTERED: D7=2 - D7 >= +2 sigma = candidate on top of D7 (Mediocre / SF18 next); within ±2 sigma = NEUTRAL (D7=2 is
#    then the cleaner rule at fewer nodes: re-read with more games before choosing); <= -2 sigma = the exact-repeat re-search
#    carries value (window or TT side effects). Prediction: 0 to +10 (bench −5% nodes on WAC, −2-3% on quiet / game).
# Stop: touch /mnt/e/chess_data/q244.stop (between steps); games stop with SIGINT to fastchess.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
W=/mnt/e/chess_data/r13a
P=/mnt/e/chess_data/pgo_s6
STOP=/mnt/e/chess_data/q244.stop
NEW="$ND/build_uci/chess_engine_uci_s6pgo"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"; mkdir -p "$W"
show() { sed "s/^/[q244]   /"; }
stopped() { [ -e "$STOP" ] && { echo "[q244] $(date) stop file present before $1 — QUEUE 244 DONE"; exit 0; }; }
fp() { local k=""; [ -n "$2" ] && k="setoption name Knobs value $2"; printf '%s\nbench\nquit\n' "$k" | "$1" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+'; }
echo "[q244] $(date '+%Y-%m-%d %H:%M:%S') queued; waiting for the end of #241"
until grep -qE 'QUEUE 241 (DONE|ABORTED)|☠' /mnt/e/chess_data/q241.log; do stopped "the start"; sleep 60; done
grep -q 'QUEUE 241 DONE' /mnt/e/chess_data/q241.log || { echo "[q244] #241 did not finish cleanly — QUEUE 244 ABORTED (check it first)"; exit 1; }
sleep 30
echo "[q244] $(date '+%H:%M') start"

# ── 1 PGO build ───────────────────────────────────────────────────────────────────────────────────────────────────────
rm -rf "$P"; mkdir -p "$P"
OUT=chess_engine_uci_s6gen PGO=gen PGO_DIR=$P bash selfplay/build_uci.sh > "$W/gen.out" 2>&1 || { echo "[q244] ☠️ gen build failed — QUEUE 244 ABORTED"; exit 1; }
nice -n 5 "$FC" -engine cmd="$ND/build_uci/chess_engine_uci_s6gen" name=a option.OwnBook=false option.Knobs=SV2_VERIFY_DEEPER=2 st=0.25 \
	-engine cmd="$ND/build_uci/chess_engine_uci_s6gen" name=b option.OwnBook=false option.Knobs=SV2_VERIFY_DEEPER=1 st=0.25 \
	-openings file=$O/openings_uho.epd format=epd order=random -srand 744 -rounds 4 -games 2 -concurrency 1 > "$W/train.log" 2>&1
OUT=chess_engine_uci_s6pgo PGO=use PGO_DIR=$P bash selfplay/build_uci.sh > "$W/use.out" 2>&1 || { echo "[q244] ☠️ use build failed — QUEUE 244 ABORTED"; exit 1; }
d=$(fp "$NEW" ""); d1=$(fp "$NEW" SV2_VERIFY_DEEPER=1); d2=$(fp "$NEW" SV2_VERIFY_DEEPER=2)
echo "[q244] s6pgo: default [$d] · D7 [$d1] · D7=2 [$d2]"
[ "$d" = "solved 249/300 nodes 37279382" ] && [ "$d1" = "solved 250/300 nodes 37320382" ] && [ "$d2" = "solved 254/300 nodes 35430812" ] || { echo "[q244] ☠️ PGO gates failed — QUEUE 244 ABORTED"; exit 1; }

# ── 2 clear the box ───────────────────────────────────────────────────────────────────────────────────────────────────
touch /mnt/e/chess_data/q245.stop
until grep -qE 'QUEUE 245 (DONE|ABORTED)|☠' /mnt/e/chess_data/q245.log 2>/dev/null || [ ! -e /mnt/e/chess_data/q245.log ]; do stopped "the games"; sleep 30; done
sleep 30
if pgrep -f "[f]astchess-linux-x86-64/fastchess" >/dev/null || pgrep -f "[s]ts_test|[_]ks_footprint|[d]epth_nps_bench" >/dev/null; then echo "[q244] ☠️ another job is running — QUEUE 244 ABORTED"; exit 1; fi
stopped "the games"
[ "$(date +%u)" = 7 ] && [ "$(date +%H%M)" -ge 1145 ] && { echo "[q244] $(date) too close to 12:00 Sunday — QUEUE 244 DONE (games next window)"; exit 0; }

# ── 3 games ───────────────────────────────────────────────────────────────────────────────────────────────────────────
rm -f /tmp/q244_pre.log
timeout 180 "$FC" -engine "cmd=$NEW" name=d72 option.OwnBook=false option.Knobs=SV2_VERIFY_DEEPER=2 st=0.05 \
	-engine "cmd=$NEW" name=d7 option.OwnBook=false option.Knobs=SV2_VERIFY_DEEPER=1 st=0.05 \
	-openings file=$O/openings_uho.epd format=epd order=random -rounds 1 -games 2 -concurrency 1 > /tmp/q244_pre.log 2>&1
WN=$(grep -c 'have option' /tmp/q244_pre.log)
echo "[q244] preflight: $(grep -E '^Games:' /tmp/q244_pre.log | tail -1) · warnings $WN"
[ "$WN" = 0 ] || { echo "[q244] ☠️ preflight warnings — QUEUE 244 ABORTED"; exit 1; }
COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
echo "[q244] $(date '+%H:%M') games D7=2 vs D7 (SIGINT at Sunday 12:00)"
( until [ "$(date +%u)" = 7 ] && [ "$(date +%H%M)" -ge 1200 ]; do sleep 20; done; pkill -INT -f "[f]astchess-linux-x86-64/fastchess" && echo "[q244] $(date '+%H:%M') 12:00 guard: games interrupted" ) &
GUARD=$!
nice -n 5 "$FC" -engine "cmd=$NEW" name=d72 option.OwnBook=false "option.Knobs=SV2_VERIFY_DEEPER=2" tc=10+0.1 timemargin=250 \
	-engine "cmd=$NEW" name=d7 option.OwnBook=false "option.Knobs=SV2_VERIFY_DEEPER=1" tc=10+0.1 timemargin=250 \
	$COMMON -srand 2441 -rounds 1000 -games 2 -pgnout file=$W/d72a.pgn notation=san nodes=true append=false > $W/d72a.log 2>&1
kill $GUARD 2>/dev/null
echo "[q244] d72a: $(grep -E '^Games:' $W/d72a.log | tail -1) · warnings $(grep -c 'have option' $W/d72a.log) · disconnects/stalls $(grep -ciE 'disconnect|stall|not responsive' $W/d72a.log) · time forfeits $(grep -c 'loses on time' $W/d72a.pgn)"
timeout 900 bash "$R" pyrun selfplay/fastchess_report.py $W/d72a.pgn --p1 d72 2>&1 | sed -n '/vs the field/,$p' | show
echo "[q244] $(date) QUEUE 244 DONE"
