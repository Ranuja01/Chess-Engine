#!/bin/bash
# 2026-11-18 ~00:35, queue #270: the night after #269 (RFP_THREAT_GATE screen). Owner back ~11:00.
#  1 [fixed] waits for #269; reads its screen (games for each gate arm with STS1500 >= base − 70 and WAC solved >= base − 3).
#  2 [fixed] PGO build from the current sources (gen 7.6 + SORT_KEYED + RFP_THREAT_GATE + the C6 counters). With the owner's
#    yes to ship SORT_KEYED (flag file q270.ship_sk), the preset gets SORT_KEYED = 1 and the build is chess_engine_uci_g76spgo;
#    otherwise a test build chess_engine_uci_n18pgo. Gates: default 246 / 35,138,434 == SORT_KEYED=0; gate =1 / =2 reproduce
#    #269 (257 / 38,712,861 · 253 / 43,958,749); v1 250 / 35,310,775-778.
#  3 TIMED from here: waits for q270.go (the owner's good night) or 03:00. With the ship flag: install (backup g76pgo), Cython,
#    pinned bench ×9 g76s / g76 (ledger).
#  4 C6 liveness: 100 games at 10+0.1, ASP_KEEP_FAILHIGH=1 (+ Debug, FIX_COUNT_LOG=1 through a stderr wrapper) vs default;
#    counts asp_keep_used / asp_keep_differs per searched move. REGISTERED: live = used >= 1 per 1,000 moves; worth games =
#    differs >= 1 per 2,000 moves (decided in the morning).
#  5 gate self-play vs default on the same binary, 10+0.1, 150-pair chunks alternating the arms that passed, seeds 2701+, wait
#    on q270.pause, cutoff 10:40. REGISTERED: lower 95% bound > 0 = pass (then SF18 + Mediocre); prediction rtg1 −10 … +15,
#    rtg2 −20 … +15 (its tree is +25% at equal depth).
# Stop: q270.stop · pause the timed games: q270.pause. Never edit the sources before "build done".
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
W=/mnt/e/chess_data/q270
GW=/mnt/e/chess_data/r18a
P=/mnt/e/chess_data/pgo_n18
PROD="$ND/build_uci/chess_engine_uci"
G76="$ND/build_uci/chess_engine_uci_g76pgo"
SE="$ND/search_engine.cpp"
Q269=/mnt/e/chess_data/q269.log
LOG=/mnt/e/chess_data/q270.log
SHIP=/mnt/e/chess_data/q270.ship_sk
GO=/mnt/e/chess_data/q270.go
STOP=/mnt/e/chess_data/q270.stop
PAUSE=/mnt/e/chess_data/q270.pause
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP" "$PAUSE"; mkdir -p "$W" "$GW"
show() { sed "s/^/[q270]   /"; }
stopped() { [ -e "$STOP" ] && { echo "[q270] $(date) stop file before $1 — QUEUE 270 DONE"; exit 0; }; }
fp() { local k=""; [ -n "$2" ] && k="setoption name Knobs value $2"; printf '%s\nbench\nquit\n' "$k" | "$1" 2>&1 | grep -oE 'solved [0-9]+/300 nodes [0-9]+' | head -1; }
busy() { pgrep -f "[f]astchess-linux-x86-64/fastchess" >/dev/null || pgrep -f "[s]ts_test|[v]algrind|[b]uild_uci.sh|[_]quiet_threat" >/dev/null; }
finished() { local c; c=$(grep -cE '^\[Result "(1-0|0-1|1/2-1/2)"\]' "$1" 2>/dev/null); echo "${c:-0}"; }
sp_count() { local n=0 f; for f in "$GW/$1"_s*.pgn; do [ -e "$f" ] && n=$((n + $(finished "$f"))); done; echo $n; }
echo "[q270] $(date '+%Y-%m-%d %H:%M:%S') queued; waiting for #269"
until grep -qE 'QUEUE 269 (DONE|ABORTED)' "$Q269" || ! pgrep -f "[_]queue_269.sh" >/dev/null; do stopped "the start"; sleep 60; done
sleep 30

# ── 1 the screen ──────────────────────────────────────────────────────────────────────────────────────────────────────────
sts() { grep -E "^\[q269\] [0-9:]+ $1: STS score" "$Q269" | tail -1 | grep -oE 'STS score: [0-9]+' | grep -oE '[0-9]+$'; }
wsol() { grep -E '^\[q269\] WAC default' "$Q269" | tail -1 | grep -oE "$1 \[SOLVED: Solved [0-9]+" | grep -oE '[0-9]+$'; }
SB=$(sts g76); S1=$(sts rtg1); S2=$(sts rtg2)
WB=$(grep -E '^\[q269\] WAC default' "$Q269" | grep -oE 'default \[SOLVED: Solved [0-9]+' | grep -oE '[0-9]+$'); W1=$(wsol '=1'); W2=$(wsol '=2')
ARMS=""
[ -n "$SB" ] && [ -n "$S1" ] && [ "$S1" -ge $((SB - 70)) ] && [ -n "$W1" ] && [ "$W1" -ge $((WB - 3)) ] && ARMS="rtg1"
[ -n "$SB" ] && [ -n "$S2" ] && [ "$S2" -ge $((SB - 70)) ] && [ -n "$W2" ] && [ "$W2" -ge $((WB - 3)) ] && ARMS="$ARMS rtg2"
echo "[q270] $(date '+%H:%M') screen: STS1500 base [$SB] rtg1 [$S1] rtg2 [$S2] · WAC base [$WB] =1 [$W1] =2 [$W2] → games for [${ARMS# }]"

# ── 2 build ───────────────────────────────────────────────────────────────────────────────────────────────────────────────
busy && { echo "[q270] ☠️ another job is running — QUEUE 270 ABORTED"; exit 1; }
if [ -e "$SHIP" ]; then
	[ "$(grep -c '^ *Config::SV2_VERIFY_PROMOTE = 1;$' "$SE")" = 1 ] && [ "$(grep -c 'Config::SORT_KEYED = 1;' "$SE")" = 0 ] \
		|| { echo "[q270] ☠️ the sources are not as expected for the SORT_KEYED preset — QUEUE 270 ABORTED"; exit 1; }
	cp "$SE" "$W/search_engine.cpp.g76"
	cat > "$W/preset.txt" <<'EOF'
                    // ── GEN 7.6 speed (2026-11-18, result-identical): the move-ordering sort on unique keys with a sorting
                    // network and branch-free merges; pinned bench -1.8% time (12 of 15 rounds), same moves. SORT_KEYED=0
                    // restores the previous sort.
                    Config::SORT_KEYED = 1;
EOF
	sed -i "/^ *Config::SV2_VERIFY_PROMOTE = 1;\$/r $W/preset.txt" "$SE"
	[ "$(grep -c '^ *Config::SORT_KEYED = 1;$' "$SE")" = 1 ] || { echo "[q270] ☠️ preset edit did not apply — QUEUE 270 ABORTED"; cp "$W/search_engine.cpp.g76" "$SE"; exit 1; }
	NAME=chess_engine_uci_g76spgo; echo "[q270] owner's yes to ship SORT_KEYED: preset written (+4 lines after SV2_VERIFY_PROMOTE 1)"
else
	NAME=chess_engine_uci_n18pgo; echo "[q270] no ship flag: test build (production stays g76pgo)"
fi
NEW="$ND/build_uci/$NAME"
rm -rf "$P"; mkdir -p "$P"
if OUT=chess_engine_uci_n18gen PGO=gen PGO_DIR=$P nice -n 10 bash selfplay/build_uci.sh > "$W/gen.out" 2>&1; then
	nice -n 10 "$FC" -engine cmd="$ND/build_uci/chess_engine_uci_n18gen" name=a option.OwnBook=false st=0.25 \
		-engine cmd="$ND/build_uci/chess_engine_uci_n18gen" name=b option.OwnBook=false st=0.25 \
		-openings file=$O/openings_uho.epd format=epd order=random -srand 770 -rounds 4 -games 2 -concurrency 1 > "$W/train.log" 2>&1
	OUT=$NAME PGO=use PGO_DIR=$P nice -n 10 bash selfplay/build_uci.sh > "$W/use.out" 2>&1 || { echo "[q270] ☠️ PGO use build failed — QUEUE 270 ABORTED"; grep -iE 'error' "$W/use.out" | head -8 | show; [ -e "$SHIP" ] && cp "$W/search_engine.cpp.g76" "$SE"; exit 1; }
else
	echo "[q270] ☠️ PGO gen build failed — QUEUE 270 ABORTED"; grep -iE 'error' "$W/gen.out" | head -8 | show; [ -e "$SHIP" ] && cp "$W/search_engine.cpp.g76" "$SE"; exit 1
fi
echo "[q270] $(date '+%H:%M') build done ($NAME) — the sources are free to edit"
a=$(fp "$NEW" ""); s0=$(fp "$NEW" "SORT_KEYED=0"); g1=$(fp "$NEW" "RFP_THREAT_GATE=1"); g2=$(fp "$NEW" "RFP_THREAT_GATE=2")
v1=$(printf 'setoption name EvalArm value 0\nbench\nquit\n' | "$NEW" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+')
echo "[q270] $NAME default [$a] · SORT_KEYED=0 [$s0] (both 246 / 35138434) · =1 [$g1] (257 / 38712861) · =2 [$g2] (253 / 43958749) · v1 [$v1]"
[ "$a" = "solved 246/300 nodes 35138434" ] && [ "$s0" = "$a" ] && [ "$g1" = "solved 257/300 nodes 38712861" ] && [ "$g2" = "solved 253/300 nodes 43958749" ] \
	&& echo "$v1" | grep -qE 'solved 250/300 nodes 3531077[5-8]' || { echo "[q270] ☠️ a gate differs — nothing installed, no games — QUEUE 270 DONE"; [ -e "$SHIP" ] && cp "$W/search_engine.cpp.g76" "$SE"; exit 1; }
echo "[q270] ✅ every gate equal"

# ── 3 timed from here ─────────────────────────────────────────────────────────────────────────────────────────────────────
LATEST=$(date -d 'today 03:00' +%s); [ "$(date +%H)" -ge 12 ] && LATEST=$(date -d 'tomorrow 03:00' +%s)
CUT=$(date -d 'today 10:40' +%s); [ "$(date +%H)" -ge 12 ] && CUT=$(date -d 'tomorrow 10:40' +%s)
echo "[q270] $(date '+%H:%M') waiting for q270.go (the owner's good night) or $(date -d @$LATEST '+%H:%M')"
until [ -e "$GO" ] || [ "$(date +%s)" -ge "$LATEST" ]; do stopped "the timed work"; sleep 30; done
echo "[q270] $(date '+%H:%M') timed work starts (go file: $([ -e "$GO" ] && echo yes || echo no, fallback time))"
if [ -e "$SHIP" ]; then
	cmp -s "$PROD" "$G76" || { echo "[q270] ☠️ production is not g76pgo — not installed"; }
	if cmp -s "$PROD" "$G76"; then
		cp "$NEW" "$PROD" && cmp -s "$NEW" "$PROD" && echo "[q270] $(date '+%H:%M') INSTALLED: production == $NAME: bench [$(fp "$PROD" "")]" \
			|| { echo "[q270] ☠️ install copy failed — restoring g76pgo"; cp "$G76" "$PROD"; }
		nice -n 5 bash "$R" build 2>&1 | grep -iE 'error|copying|FAILED' | tail -n 3 | show
		echo "[q270] Cython: $(bash "$R" wac q270_fp V2_PRESET=shipped 2>&1 | grep -E '^SOLVED|^NODES|^EBF' | tr '\n' ' ') (want 246 / 35138434)"
		for i in 1 2 3 4 5 6 7 8 9; do
			if [ $((i % 2)) = 1 ]; then o="g76s g76"; else o="g76 g76s"; fi
			for x in $o; do b=$G76; [ $x = g76s ] && b=$NEW; echo "[q270]   r$i $x $(printf 'bench\nquit\n' | taskset -c 5 "$b" 2>/dev/null | grep '^bench')"; done
		done
		bt() { grep -E "^\[q270\]   r[0-9]+ $1 bench" "$LOG" | awk '{print $2, $10}' | sort; }
		echo "[q270] within-round g76s / g76: time median $(join <(bt g76s) <(bt g76) | awk '{print $2 / $3}' | sort -g | awk '{a[NR]=$1} END {print a[int((NR+1)/2)]}') (g76s faster in $(join <(bt g76s) <(bt g76) | awk '$2 < $3 {n++} END {print n + 0}') of 9)"
	fi
fi

# ── 4 C6 liveness ─────────────────────────────────────────────────────────────────────────────────────────────────────────
stopped "C6"
rm -f "$W"/c6err_*.log
printf '#!/bin/bash\nexec "%s" 2>>"%s/c6err_$$.log"\n' "$NEW" "$W" > "$W/c6_wrap.sh"; chmod +x "$W/c6_wrap.sh"
COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
nice -n 5 "$FC" -engine "cmd=$W/c6_wrap.sh" name=c6 option.OwnBook=false option.Debug=true "option.Knobs=ASP_KEEP_FAILHIGH=1,FIX_COUNT_LOG=1" tc=10+0.1 timemargin=250 \
	-engine "cmd=$NEW" name=ctl option.OwnBook=false tc=10+0.1 timemargin=250 \
	$COMMON -srand 2700 -rounds 50 -games 2 -pgnout file=$W/c6.pgn notation=san append=false > $W/c6.log 2>&1
echo "[q270] $(date '+%H:%M') C6 games: $(grep -E '^Games:' $W/c6.log | tail -1) · warnings $(grep -c 'have option' $W/c6.log)"
for f in "$W"/c6err_*.log; do grep -E '^TMCNT' "$f" | tail -1; done > "$W/c6_tmcnt.txt"
awk '{ for (i = 1; i <= NF; i++) { split($i, kv, "="); s[kv[1]] += kv[2] } } END { printf "moves %d · time cut %d · partial_used %d (differs %d) · asp_keep_used %d (differs %d) · per 1,000 moves: used %.2f, differs %.2f\n", s["moves"], s["cut"], s["partial_used"], s["partial_differs"], s["asp_keep_used"], s["asp_keep_differs"], 1000 * s["asp_keep_used"] / (s["moves"] ? s["moves"] : 1), 1000 * s["asp_keep_differs"] / (s["moves"] ? s["moves"] : 1) }' "$W/c6_tmcnt.txt" | sed 's/^/[q270] C6 counts (engine processes: '"$(wc -l < "$W/c6_tmcnt.txt")"'): /'
echo "[q270] registered: live = used >= 1 per 1,000 moves; worth games = differs >= 1 per 2,000 moves"

# ── 5 gate self-play ──────────────────────────────────────────────────────────────────────────────────────────────────────
[ -n "${ARMS# }" ] || { echo "[q270] no gate arm passed the screen — no gate games — QUEUE 270 DONE"; exit 0; }
kn() { case $1 in rtg1) echo "RFP_THREAT_GATE=1" ;; rtg2) echo "RFP_THREAT_GATE=2" ;; esac; }
timed_ok() { [ ! -e "$PAUSE" ] && [ "$(date +%s)" -lt "$CUT" ]; }
guard() { ( until ! timed_ok; do sleep 15; done; pkill -INT -f "[f]astchess-linux-x86-64/fastchess" && echo "[q270] $(date '+%H:%M') guard: games interrupted" ) & GUARD=$!; }
SEQ=0
sp_chunk() {
	local a=$1 n left pairs seed f
	n=$(sp_count $a); left=$(( (2000 - n + 1) / 2 )); [ $left -le 0 ] && return 1
	pairs=$(( left < 150 ? left : 150 )); SEQ=$((SEQ + 1)); seed=$((2700 + SEQ)); f=$GW/${a}_s$seed
	echo "[q270] $(date '+%H:%M') self-play $a: $n finished, chunk seed $seed, $pairs pairs"
	guard
	nice -n 5 "$FC" -engine "cmd=$NEW" name=$a option.OwnBook=false "option.Knobs=$(kn $a)" tc=10+0.1 timemargin=250 \
		-engine "cmd=$NEW" name=g76 option.OwnBook=false tc=10+0.1 timemargin=250 \
		$COMMON -srand $seed -rounds $pairs -games 2 -pgnout file=$f.pgn notation=san nodes=true append=false > $f.log 2>&1
	kill $GUARD 2>/dev/null
	echo "[q270] $(date '+%H:%M') $a s$seed: $(grep -E '^Games:' $f.log | tail -1) · warnings $(grep -c 'have option' $f.log) · time forfeits $(grep -c 'loses on time' $f.pgn) · unfinished $(grep -c '^\[Result "\*"' $f.pgn)"
	return 0
}
while [ "$(date +%s)" -lt "$CUT" ]; do
	stopped "a self-play chunk"
	if ! timed_ok; then sleep 30; continue; fi
	done_all=1
	for a in $ARMS; do stopped "a self-play chunk"; timed_ok || break; sp_chunk $a && done_all=0; done
	[ $done_all = 1 ] && break
done
for a in $ARMS; do
	[ "$(sp_count $a)" -gt 0 ] || continue
	echo "[q270] pooled self-play $a vs gen 7.6 default ($(sp_count $a) finished):"
	timeout 900 bash "$R" pyrun selfplay/fastchess_report.py "$GW/${a}"_s*.pgn --p1 $a 2>&1 | sed -n '/vs the field/,$p' | show
done
echo "[q270] $(date) QUEUE 270 DONE"
