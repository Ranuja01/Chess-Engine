#!/bin/bash
# 2026-11-10 ~18:45, queue #194: BENCH CHECKPOINT #3, the transition's formal last step (10-19 list; owner 11-10 "ok" to
# the close-out order). Waits for #193 (fixed-depth S0 work). Production gen 7.1 is unchanged by the close-out (E1 parked,
# F24 / F26 neutral), so the legs already in hand are reused: SF18 at equal nodes 56.9% over 2,250 g (#189 / #192 control
# arms), Mediocre 81.3% over 300 g (#189 control), static ladder (#169), depth ladder (search v2 plan §6.1), critical
# move test (#193's depth-12 regret pass). This queue runs the missing legs, one process at a time:
#  1 the record queue for v2 (`V2_PRESET=shipped`, Cython path; template #87): WAC d10, STS300 d10, STS full d10 (fixed
#    depth), then WAC at LIGHTNING (depth reached per move) and WAC speed (3 reps) on a quiet box. v1's record (10-20)
#    stands: v1 is frozen.
#  2 the Stockfish 1.1 series point (template #146, identical protocol: seed 1082, same openings, ours 60+0.6 vs SF1.1
#    20+0.2, 200 g, concurrency 3). Series: gen 6.1 22.5% · 6.2 21.0% · 6.3 29.25% · 6.4 29.5% · production 11-04 36.75%.
#    REGISTERED: a series point, not a gate; >= 33.6% (11-04 − 1σ, σ ≈ 3.1 pp at 200 g) = no regression. Prediction
#    37-44% (gen 6.7 +23, gen 7 +16, gen 7.1 +13 in self-play since 11-04; external gains about half).
# Stop: touch /mnt/e/chess_data/q194.stop (checked between steps). Never edit this file while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
BIN="$ND/build_uci/chess_engine_uci"
SF1="/mnt/e/chess_data/engines/sf1.1/stockfish-1.1"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
RD=/mnt/e/chess_data/checkpoint3
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
STOP=/mnt/e/chess_data/q194.stop
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"; mkdir -p "$RD"
echo "[q194] $(date '+%Y-%m-%d %H:%M:%S') queued; waiting for #193"
until grep -q "QUEUE 193 DONE" /mnt/e/chess_data/q193.log 2>/dev/null || ! pgrep -f "_queue_193.sh" >/dev/null; do
	[ -e "$STOP" ] && { echo "[q194] stopped while waiting"; exit 0; }
	sleep 60
done
while pgrep -f "fastchess-linux-x86-64/fastchess|vs_sf.py|_ks_footprint_regret|_case_triage" >/dev/null; do sleep 30; done
START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[q194] $START start"
fp=$(printf "bench\nquit\n" | "$BIN" 2>/dev/null | grep '^bench')
echo "[q194] production (want 247 / 39852210): $fp"
echo "$fp" | grep -q "solved 247/300 nodes 39852210 " || { echo "[q194] ☠️ fingerprint failed — QUEUE 194 ABORTED"; exit 1; }

# ── 1 the record queue, v2 ──────────────────────────────────────────────────────────────────────────────────────────────
V2="V2_PRESET=shipped"
for sub in "wac rec3_v2" "sts rec3_v2" "sts_full rec3f_v2" "wac_timed_depth rec3t_v2" "wac_speed rec3s_v2 3"; do
	[ -e "$STOP" ] && break
	set -- $sub
	echo "[q194] $(date) record: $sub"
	bash "$R" $sub $V2 > "$RD/$1.log" 2>&1
	case $1 in
		wac)             grep -m3 -E 'solved|nodes|EBF' "$RD/$1.log" | sed 's/^/[q194]   /' ;;
		sts)             tail -2 "$RD/$1.log" | sed 's/^/[q194]   /' ;;
		sts_full)        tail -1 "$RD/$1.log" | sed 's/^/[q194]   /' ;;
		wac_timed_depth) grep -m6 -E 'depth|DEPTH|mean|median' "$RD/$1.log" | sed 's/^/[q194]   /' ;;
		wac_speed)       grep -E 'NODES:|MAX_NPS|MEDIAN|SPREAD' "$RD/$1.log" | sed 's/^/[q194]   /' ;;
	esac
done

# ── 2 the Stockfish 1.1 series point (#146 protocol) ────────────────────────────────────────────────────────────────────
T=g71_sf11_odds
COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
SF1OPT="option.Threads=1 option.Ponder=false option.OwnBook=false option.Hash=128"
if [ ! -e "$STOP" ]; then
	timeout 180 "$FC" -engine cmd="$BIN" name=ours option.OwnBook=false st=0.05 \
	  -engine cmd="$SF1" name=sf1.1 $SF1OPT st=0.05 \
	  -openings file=$O/openings_uho.epd format=epd order=random -rounds 1 -games 2 -concurrency 1 > /tmp/q194_pre.log 2>&1
	W=$(grep -c "have option" /tmp/q194_pre.log)
	echo "[q194] preflight: $(grep -E '^Games:' /tmp/q194_pre.log | tail -1) · warnings $W"
	if [ "$W" = "0" ] && grep -qE '^Games: 2' /tmp/q194_pre.log; then
		echo "[q194] $(date) $T (200 games: ours 60+0.6, sf1.1 20+0.2, seed 1082)"
		nice -n 5 "$FC" -engine cmd="$BIN" name=ours option.OwnBook=false tc=60+0.6 timemargin=250 restart=on \
		  -engine cmd="$SF1" name=sf1.1 $SF1OPT tc=20+0.2 timemargin=250 restart=on $COMMON -srand 1082 -rounds 100 -games 2 \
		  -pgnout file=$O/$T.pgn notation=san nodes=true append=false > $O/$T.log 2>&1
		echo "[q194] $(date) $T done: $(grep -E '^Games:|^Elo:' $O/$T.log | tail -2 | tr '\n' ' ')"
		echo "[q194]   $(grep -E '^Ptnml' $O/$T.log | tail -1)  disconnects/stalls: $(grep -ciE 'disconnect|stall|not responsive' $O/$T.log)  time forfeits: $(grep -c 'loses on time' $O/$T.pgn)"
		timeout 600 bash "$R" pyrun selfplay/fastchess_report.py $O/$T.pgn --p1 ours 2>&1 | sed -n '/vs the field/,$p' | sed 's/^/[q194]   /'
	else
		echo "[q194] ☠️ SF1.1 preflight failed — series point skipped"
	fi
fi
echo "[q194] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | sed 's/^/[q194]   /'
echo "[q194] $(date) QUEUE 194 DONE"
