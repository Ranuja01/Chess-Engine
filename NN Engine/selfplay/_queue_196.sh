#!/bin/bash
# 2026-11-10 ~20:45, queue #196: the night after #195 (owner 11-10: "if there's extra things to fill the night i don't mind
# it"). Waits for #195. One process group at a time.
#  1 QUIET-BOX RE-RUN of checkpoint #3's two timed record legs (wac_timed_depth, wac_speed 3 reps). #194's readings
#    (20:10-20:17) overlapped two side benches of mine, so these replace them.
#  2 THE TRANSITION END TO END: gen 7.1 (production) vs gen 4 (10-19, the ship the transition started from;
#    selfplay/gen4_uci.sh = the gen 5 binary with V2_SHIP_GEN=4, bench 258 / 47,689,413) at 60+0.6, 300 g, UHO, seed 196001.
#    One direct number for the whole arc in place of a sum of per-ship reads (they do not add). Gen 4's binary is not PGO
#    and has the old time manager; both are transition work, so they count.
#    REGISTERED: a summary, not a gate. Prediction +120 to +220 Elo for gen 7.1.
#    Starts only before 04:00 on 11-11 (about 5 h of games; the owner may be back in the morning), else skipped.
#  3 checkpoint #3's critical-move leg on a SECOND set: our d10 and d12 moves on game_regret_set_v2 (11,940 rows, fixed
#    depth, 3 workers; ~80 min). Also test B's replication rows: the registered MISS on set 1 stands, this only narrows
#    the interval (read by hand with _pot_mg_fit.py MODE=depthgain).
# Stop: touch /mnt/e/chess_data/q196.stop (checked between steps). Never edit this file or gen4_uci.sh while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
BIN="$ND/build_uci/chess_engine_uci"
G4="$ND/selfplay/gen4_uci.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
RD=/mnt/e/chess_data/checkpoint3
RB=/mnt/e/chess_data/s0_testB_v2
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
STOP=/mnt/e/chess_data/q196.stop
LATEST=$(date -d '2026-11-11 04:00' +%s)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"; mkdir -p "$RD" "$RB"
echo "[q196] $(date '+%Y-%m-%d %H:%M:%S') queued; waiting for #195"
until grep -q "QUEUE 195 DONE" /mnt/e/chess_data/q195.log 2>/dev/null || ! pgrep -f "_queue_195.sh" >/dev/null; do
	[ -e "$STOP" ] && { echo "[q196] stopped while waiting"; exit 0; }
	sleep 60
done
while pgrep -f "fastchess-linux-x86-64/fastchess|vs_sf.py|_ks_footprint_regret|_case_triage|_mopup_conversion" >/dev/null; do sleep 30; done
START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[q196] $START start"
fp=$(printf "bench\nquit\n" | "$BIN" 2>/dev/null | grep '^bench')
echo "[q196] production (want 247 / 39852210): $fp"
echo "$fp" | grep -q "solved 247/300 nodes 39852210 " || { echo "[q196] ☠️ fingerprint failed — QUEUE 196 ABORTED"; exit 1; }

# ── 1 quiet-box re-run of the timed record legs ─────────────────────────────────────────────────────────────────────────
for sub in "wac_timed_depth rec3tq_v2" "wac_speed rec3sq_v2 3"; do
	[ -e "$STOP" ] && break
	set -- $sub
	echo "[q196] $(date) record (quiet box): $sub"
	bash "$R" $sub V2_PRESET=shipped > "$RD/$1_quiet.log" 2>&1
	case $1 in
		wac_timed_depth) grep -m6 -E 'depth|DEPTH|mean|median' "$RD/$1_quiet.log" | sed 's/^/[q196]   /' ;;
		wac_speed)       grep -E 'NODES:|MAX_NPS|MEDIAN|SPREAD' "$RD/$1_quiet.log" | sed 's/^/[q196]   /' ;;
	esac
done

# ── 2 the transition end to end: gen 7.1 vs gen 4 at 60+0.6 ─────────────────────────────────────────────────────────────
T=g71_vs_g4_60
COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
if [ ! -e "$STOP" ] && [ $(date +%s) -lt $LATEST ]; then
	f4=$(printf "bench\nquit\n" | "$G4" 2>/dev/null | grep '^bench')
	echo "[q196] gen 4 wrapper (want 258 / 47689413): $f4"
	timeout 180 "$FC" -engine "cmd=$BIN" name=g71 option.OwnBook=false st=0.05 \
	  -engine "cmd=$G4" name=g4 option.OwnBook=false st=0.05 \
	  -openings file=$O/openings_uho.epd format=epd order=random -rounds 1 -games 2 -concurrency 1 > /tmp/q196_pre.log 2>&1
	W=$(grep -c "have option" /tmp/q196_pre.log)
	echo "[q196] preflight: $(grep -E '^Games:' /tmp/q196_pre.log | tail -1) · warnings $W"
	if echo "$f4" | grep -q "solved 258/300 nodes 47689413 " && [ "$W" = "0" ] && grep -qE '^Games: 2' /tmp/q196_pre.log; then
		echo "[q196] $(date) $T (300 games at 60+0.6, seed 196001)"
		nice -n 5 "$FC" -engine "cmd=$BIN" name=g71 option.OwnBook=false tc=60+0.6 timemargin=250 \
		  -engine "cmd=$G4" name=g4 option.OwnBook=false tc=60+0.6 timemargin=250 \
		  $COMMON -srand 196001 -rounds 150 -games 2 -pgnout file=$O/$T.pgn notation=san nodes=true append=false > $O/$T.log 2>&1
		echo "[q196] $(date) $T done: $(grep -E '^Games:|^Elo:' $O/$T.log | tail -2 | tr '\n' ' ')"
		echo "[q196]   $(grep -E '^Ptnml' $O/$T.log | tail -1)  disconnects/stalls: $(grep -ciE 'disconnect|stall|not responsive' $O/$T.log)  time forfeits: $(grep -c 'loses on time' $O/$T.pgn)"
		timeout 900 bash "$R" pyrun selfplay/fastchess_report.py $O/$T.pgn --p1 g71 2>&1 | sed -n '/vs the field/,$p' | sed 's/^/[q196]   /'
	else
		echo "[q196] ☠️ gen 4 fingerprint or preflight failed — the end-to-end match skipped"
	fi
else
	echo "[q196] $(date) end-to-end match skipped (stop, or past 04:00)"
fi

# ── 3 the critical-move leg on a second set (fixed depth) ───────────────────────────────────────────────────────────────
export RESIGN_THRESHOLD=-99999999   # a resigning worker returns no move in a lost position
regret_pass() {  # depth
	local dep=$1 i pids=""
	[ -e "$STOP" ] && return
	echo "[q196] $(date) regret pass on game_regret_set_v2, depth $dep, 3 workers"
	for i in 0 1 2; do
		env WORKER=1 SET=diagnostics/ks_sets/game_regret_set_v2.csv DEPTH=$dep SLICE=$i/3 SPLIT=all OUT=$RB/d${dep}_s$i.csv \
			V2_PRESET=shipped bash "$R" pyrun diagnostics/_ks_footprint_regret.py > $RB/d${dep}_s$i.log 2>&1 &
		pids="$pids $!"
	done
	wait $pids
	echo "[q196] $(date) depth $dep rows: $(cat $RB/d${dep}_s*.csv 2>/dev/null | wc -l) · mean regret $(cat $RB/d${dep}_s*.csv | awk -F, '{ s += $3; n++ } END { if (n) printf "%.3f", s / n }')"
}
regret_pass 10
regret_pass 12

echo "[q196] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | sed 's/^/[q196]   /'
echo "[q196] $(date) QUEUE 196 DONE"
