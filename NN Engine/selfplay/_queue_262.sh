#!/bin/bash
# 2026-11-16 ~21:55, queue #262: C4's self-play on the gen 7.5 base, chained after #257 (PROMOTE) so the night and Tuesday's
# daytime are used (owner 11-16: nights always free, weekdays usually; evening block 20:50-00:15 kept).
# Binary chess_engine_uci_c4pgo (#261: = production at default, 252 / 35,235,460; =2 35,235,446; =1 36,659,901). Arms against
# the same binary at default: c4g = SV2_HIST_ALLNODE_GATE=1 (the gate; STS1500 +99, z +0.47), c4b = 2 (the bound alone; +48,
# z +0.47). 10+0.1, 150-pair chunks alternating c4g / c4b, seeds 2621+, to 2,000 games each (1,000 pairs).
# REGISTERED (self-play stage): lower 95% bound > 0 = pass (then SF18 + Mediocre legs in a later queue); within ±2σ = NEUTRAL;
# upper bound < 0 = harm. Prediction: c4g −10 … +10, c4b −5 … +5 (the bound changes 8.8% of STS moves by tiny amounts).
# If PROMOTE ships in the morning (gen 7.6), the owner decides whether C4 continues on the gen 7.5 base: touch q262.stop.
# Timed-window handling waits instead of skipping (memory evening-block-cut-falls-through-to-skipped-legs): an evening-block
# cut pauses until the window reopens; only the cutoff (Tue 20:50) ends the queue.
# Stop: touch /mnt/e/chess_data/q262.stop · pause: q262.pause · run through the evening block: q262.go.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
GW=/mnt/e/chess_data/r16c
BIN="$ND/build_uci/chess_engine_uci_c4pgo"
STOP=/mnt/e/chess_data/q262.stop
PAUSE=/mnt/e/chess_data/q262.pause
GO=/mnt/e/chess_data/q262.go
LOG257=/mnt/e/chess_data/q257.log
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP" "$PAUSE" "$GO"; mkdir -p "$GW"
show() { sed "s/^/[q262]   /"; }
stopped() { [ -e "$STOP" ] && { echo "[q262] $(date) stop file before $1 — QUEUE 262 DONE"; exit 0; }; }
kn() { case $1 in c4g) echo "SV2_HIST_ALLNODE_GATE=1" ;; c4b) echo "SV2_HIST_ALLNODE_GATE=2" ;; esac; }
busy() { pgrep -f "[f]astchess-linux-x86-64/fastchess" >/dev/null || pgrep -f "[s]ts_test|[t]actical_test|[v]algrind|[b]uild_uci.sh" >/dev/null; }
finished() { local c; c=$(grep -cE '^\[Result "(1-0|0-1|1/2-1/2)"\]' "$1" 2>/dev/null); echo "${c:-0}"; }
sp_count() { local n=0 f; for f in "$GW/$1"_s*.pgn; do [ -e "$f" ] && n=$((n + $(finished "$f"))); done; echo $n; }

echo "[q262] $(date '+%Y-%m-%d %H:%M:%S') queued; waiting for #257 to end"
until grep -qE 'QUEUE 257 (DONE|ABORTED)|☠' "$LOG257"; do stopped "the start"; sleep 60; done
sleep 60
stopped "the start"
busy && { echo "[q262] ☠️ another job is running — QUEUE 262 ABORTED"; exit 1; }
START=$(date '+%Y-%m-%d %H:%M:%S')
# the evening block of the day the queue is in, and the cutoff Tuesday 20:50 (this queue starts after midnight, Tue)
CUT=$(date -d 'today 20:50' +%s); [ "$(date +%H)" -ge 21 ] && CUT=$(date -d 'tomorrow 20:50' +%s)
GS=$(date -d 'today 20:50' +%s); GE=$(date -d "$(date -d tomorrow +%F) 00:15" +%s)
timed_ok() { local t; t=$(date +%s); [ ! -e "$PAUSE" ] && [ "$t" -lt "$CUT" ] && ! { [ "$t" -ge "$GS" ] && [ "$t" -lt "$GE" ] && [ ! -e "$GO" ]; }; }
guard() { ( until ! timed_ok; do sleep 15; done; pkill -INT -f "[f]astchess-linux-x86-64/fastchess" && echo "[q262] $(date '+%H:%M') guard: games interrupted" ) & GUARD=$!; }
echo "[q262] $START start · cutoff $(date -d @$CUT '+%a %H:%M') · #257 ended: $(grep -E 'QUEUE 257' "$LOG257" | tail -1 | sed 's/^\[q257\] //')"
D=$(printf 'bench\nquit\n' | "$BIN" 2>&1 | grep -oE 'solved [0-9]+/300 nodes [0-9]+')
G1=$(printf 'setoption name Knobs value SV2_HIST_ALLNODE_GATE=1\nbench\nquit\n' | "$BIN" 2>&1 | grep -oE 'solved [0-9]+/300 nodes [0-9]+')
G2=$(printf 'setoption name Knobs value SV2_HIST_ALLNODE_GATE=2\nbench\nquit\n' | "$BIN" 2>&1 | grep -oE 'solved [0-9]+/300 nodes [0-9]+')
echo "[q262] c4pgo default [$D] · =1 [$G1] · =2 [$G2]"
[ "$D" = "solved 252/300 nodes 35235460" ] && [ "$G1" = "solved 252/300 nodes 36659901" ] && [ "$G2" = "solved 252/300 nodes 35235446" ] \
	|| { echo "[q262] ☠️ c4pgo fingerprints moved — QUEUE 262 ABORTED"; exit 1; }
rm -f /tmp/q262_pre.engine.log
timeout 180 "$FC" -engine "cmd=$BIN" name=c4g option.OwnBook=false "option.Knobs=$(kn c4g)" st=0.05 \
	-engine "cmd=$BIN" name=ctl option.OwnBook=false st=0.05 \
	-openings file=$O/openings_uho.epd format=epd order=random -rounds 1 -games 2 -concurrency 1 \
	-log file=/tmp/q262_pre.engine.log level=info engine=true > /tmp/q262_pre.log 2>&1
WN=$(grep -c 'have option' /tmp/q262_pre.log)
echo "[q262] preflight: $(grep -E '^Games:' /tmp/q262_pre.log | tail -1) · warnings $WN · echoes $(grep -cF "SV2_HIST_ALLNODE_GATE=1" /tmp/q262_pre.engine.log)"
[ "$WN" = 0 ] || { echo "[q262] ☠️ preflight warnings — QUEUE 262 ABORTED"; exit 1; }
COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
SEQ=0
sp_chunk() {  # arm; returns 1 when the arm already has 2,000
	local a=$1 n left pairs seed f
	n=$(sp_count $a); left=$(( (2000 - n + 1) / 2 )); [ $left -le 0 ] && return 1
	pairs=$(( left < 150 ? left : 150 )); SEQ=$((SEQ + 1)); seed=$((2620 + SEQ)); f=$GW/${a}_s$seed
	echo "[q262] $(date '+%H:%M') self-play $a: $n finished, chunk seed $seed, $pairs pairs"
	guard
	nice -n 5 "$FC" -engine "cmd=$BIN" name=$a option.OwnBook=false "option.Knobs=$(kn $a)" tc=10+0.1 timemargin=250 \
		-engine "cmd=$BIN" name=g75 option.OwnBook=false tc=10+0.1 timemargin=250 \
		$COMMON -srand $seed -rounds $pairs -games 2 -pgnout file=$f.pgn notation=san nodes=true append=false > $f.log 2>&1
	kill $GUARD 2>/dev/null
	echo "[q262] $(date '+%H:%M') $a s$seed: $(grep -E '^Games:' $f.log | tail -1) · warnings $(grep -c 'have option' $f.log) · disconnects/stalls $(grep -ciE 'disconnect|stall|not responsive' $f.log) · time forfeits $(grep -c 'loses on time' $f.pgn) · unfinished $(grep -c '^\[Result "\*"' $f.pgn)"
	return 0
}
# wait-not-skip: keep going until both arms reach 2,000 or the cutoff passes
while [ "$(date +%s)" -lt "$CUT" ]; do
	stopped "a self-play chunk"
	if ! timed_ok; then sleep 30; continue; fi
	complete=0
	for a in c4g c4b; do
		stopped "a self-play chunk"
		timed_ok || break
		sp_chunk $a || complete=$((complete + 1))
	done
	# both arms at 2,000 ends the loop; a pass cut short by the timed window only waits for it to reopen
	[ "$(sp_count c4g)" -ge 1999 ] && [ "$(sp_count c4b)" -ge 1999 ] && break
done
for a in c4g c4b; do
	[ "$(sp_count $a)" -gt 0 ] || continue
	echo "[q262] pooled self-play $a vs gen 7.5 ($(sp_count $a) finished):"
	timeout 900 bash "$R" pyrun selfplay/fastchess_report.py "$GW/${a}"_s*.pgn --p1 $a 2>&1 | sed -n '/vs the field/,$p' | show
done
echo "[q262] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | show
echo "[q262] $(date) QUEUE 262 DONE"
