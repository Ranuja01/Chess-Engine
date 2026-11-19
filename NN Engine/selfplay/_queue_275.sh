#!/bin/bash
# 2026-11-18 ~14:50, queue #275: #274's safety net, then the night's leftover box time. Chained after #274 (waits for it).
# Why: #274's step 4 waits for a timed window only before its second self-play pass. If an afternoon pause pulls the SF18 legs
# forward, that pass resumes ~20:00, the 20:50 block cuts it, and the Mediocre legs then SKIP (timed_ok fails) instead of
# waiting: the ship-rule set would end incomplete and the night idle. Every timed step here WAITS for a window, never skips.
# Pause / good night: the same files as #274 (q274.pause holds every timed step, q274.go lifts the 20:50 to 00:15 block).
#  1 Complete #274: the seed's self-play to 2,000 games (fresh chunk seeds 2761+), every Mediocre leg without a complete
#    150-game log (re-run whole), then #274's registered reads on the complete set.
#  2 SORT_KEYED pinned bench on a quiet box: g76spgo (gen 7.6 + SORT_KEYED preset, #271) vs g76pgo, x21 on core 5, alternating.
#    REGISTERED (as #271): faster if the within-round time median < 0.995; prediction 0.975 … 0.990. Report only (no install).
#  3 C6 re-count: 100 g at 10+0.1, s18pgo with ASP_KEEP_FAILHIGH=1 + Debug + FIX_COUNT_LOG=1, stderr on /tmp (never /mnt), vs
#    the same binary at defaults. REGISTERED: C6 is live (worth games) if asp_keep_used >= 1 per 1,000 moves; prediction
#    0.5 … 2 per 1,000 (the clock cuts 5.2 iterations per 1,000 moves, #273).
#  4 C5 SV2_MALUS_SEARCHED_ONLY=1 vs gen 7.6, self-play 10+0.1 in 150-pair chunks (fresh seeds 2781+) to 2,000 games or the
#    cutoff. #249's STS1500 −261 (z −2.04) is not a verdict (#272: STS does not screen search changes). REGISTERED: lower 95%
#    > 0 ⇒ SF18 + Mediocre next; within ±2σ ⇒ neutral, parked; upper 95% < 0 ⇒ rejected. Prediction −15 … +8.
# Cutoff Thu 11-19 10:40. Stop: touch /mnt/e/chess_data/q275.stop (between steps).
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
MED="$ND/selfplay/mediocre_uci.sh"
O=/mnt/e/chess_data/fastchess
W=/mnt/e/chess_data/q275
T=/tmp/q275
GW=/mnt/e/chess_data/r18b
NEW="$ND/build_uci/chess_engine_uci_s18pgo"
G76="$ND/build_uci/chess_engine_uci_g76pgo"
G76S="$ND/build_uci/chess_engine_uci_g76spgo"
KS="ENABLE_QSTANDPAT_SEED=1"
KM="SV2_MALUS_SEARCHED_ONLY=1"
LOG=/mnt/e/chess_data/q275.log
STOP=/mnt/e/chess_data/q275.stop
PAUSE=/mnt/e/chess_data/q274.pause
GO=/mnt/e/chess_data/q274.go
GS=$(date -d '2026-11-18 20:50' +%s); GE=$(date -d '2026-11-19 00:15' +%s); CUT=$(date -d '2026-11-19 10:40' +%s)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"; mkdir -p "$W"; rm -rf "$T"; mkdir -p "$T"
show() { sed "s/^/[q275]   /"; }
stopped() { [ -e "$STOP" ] && { echo "[q275] $(date) stop file before $1 — QUEUE 275 DONE"; exit 0; }; }
fp() { local k=""; [ -n "$2" ] && k="setoption name Knobs value $2"; printf '%s\nbench\nquit\n' "$k" | "$1" 2>&1 | grep -oE 'solved [0-9]+/300 nodes [0-9]+' | head -1; }
busy() { pgrep -f "[f]astchess-linux-x86-64/fastchess" >/dev/null || pgrep -f "[s]ts_test|[t]actical_test|[v]algrind|[_]ks_footprint|[b]uild_uci.sh" >/dev/null; }
past_cut() { [ "$(date +%s)" -ge "$CUT" ]; }
timed_ok() { local t; t=$(date +%s); [ ! -e "$PAUSE" ] && [ "$t" -lt "$CUT" ] && ! { [ "$t" -ge "$GS" ] && [ "$t" -lt "$GE" ] && [ ! -e "$GO" ]; }; }
wait_timed() { until timed_ok || past_cut; do stopped "$1"; sleep 30; done; ! past_cut; }   # false once past the cutoff
guard() { ( until ! timed_ok; do sleep 15; done; pkill -INT -f "[f]astchess-linux-x86-64/fastchess" && echo "[q275] $(date '+%H:%M') guard: games interrupted" ) & GUARD=$!; }
finished() { local c; c=$(grep -cE '^\[Result "(1-0|0-1|1/2-1/2)"\]' "$1" 2>/dev/null); echo "${c:-0}"; }
count() { local n=0 f; for f in "$GW/$1"_s*.pgn; do [ -e "$f" ] && n=$((n + $(finished "$f"))); done; echo $n; }
COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"

echo "[q275] $(date '+%Y-%m-%d %H:%M:%S') queued; waiting for #274"
until grep -q 'QUEUE 274' /mnt/e/chess_data/q274.log 2>/dev/null || ! pgrep -f "[_]queue_274.sh" >/dev/null; do stopped "the start"; sleep 30; done
sleep 20
busy && { echo "[q275] ☠️ another job is running — QUEUE 275 ABORTED"; exit 1; }
echo "[q275] $(date '+%H:%M') start (#274: $(grep -oE 'QUEUE 274 [A-Z]+' /mnt/e/chess_data/q274.log | tail -1))"
[ -x "$NEW" ] && [ -x "$G76" ] && [ -x "$G76S" ] || { echo "[q275] ☠️ a binary is missing — QUEUE 275 ABORTED"; exit 1; }
a=$(fp "$NEW" "")
echo "[q275] s18pgo default [$a] (246 / 35138434)"
[ "$a" = "solved 246/300 nodes 35138434" ] || { echo "[q275] ☠️ the default bench differs — QUEUE 275 ABORTED"; exit 1; }

# self-play chunk: arm name, knobs, seed base; returns 1 when the arm has 2,000 finished games
SEQ=0
sp_chunk() {
	local a=$1 k=$2 base=$3 n left pairs seed f
	n=$(count $a); left=$(( (2000 - n + 1) / 2 )); [ $left -le 0 ] && return 1
	pairs=$(( left < 150 ? left : 150 )); SEQ=$((SEQ + 1)); seed=$((base + SEQ)); f=$GW/${a}_s$seed
	echo "[q275] $(date '+%H:%M') self-play $a: $n finished, chunk seed $seed, $pairs pairs"
	guard
	nice -n 5 "$FC" -engine "cmd=$NEW" name=$a option.OwnBook=false "option.Knobs=$k" tc=10+0.1 timemargin=250 \
		-engine "cmd=$NEW" name=g76 option.OwnBook=false tc=10+0.1 timemargin=250 \
		$COMMON -srand $seed -rounds $pairs -games 2 -pgnout file=$f.pgn notation=san nodes=true append=false > $f.log 2>&1
	kill $GUARD 2>/dev/null
	echo "[q275] $(date '+%H:%M') $a s$seed: $(grep -E '^Games:' $f.log | tail -1) · warnings $(grep -c 'have option' $f.log) · disconnects/stalls $(grep -ciE 'disconnect|stall|not responsive' $f.log) · time forfeits $(grep -c 'loses on time' $f.pgn) · unfinished $(grep -c '^\[Result "\*"' $f.pgn)"
	return 0
}
sp_run() { while wait_timed "$1 self-play"; do stopped "$1 self-play"; sp_chunk "$1" "$2" "$3" || return 0; done; }
pooled() {
	[ "$(count $1)" -gt 0 ] || return
	echo "[q275] pooled self-play $1 vs gen 7.6 ($(count $1) finished):"
	timeout 900 bash "$R" pyrun selfplay/fastchess_report.py "$GW/$1"_s*.pgn --p1 $1 2>&1 | sed -n '/vs the field/,$p' | show
}

# ── 1 complete #274 ──────────────────────────────────────────────────────────────────────────────────────────────────────
echo "[q275] #274 self-play seed: $(count seed) finished"
SEQ=0; sp_run seed "$KS" 2760
med_done() { grep -E '^Games:' "$1" 2>/dev/null | tail -1 | grep -qE '^Games: 150,'; }
med_chunk() {  # seed, arm (the file names #274's reads expect)
	local seed=$1 a=$2 k f=$O/med_q274_s$1_$2.log try=0
	until med_done "$f"; do
		stopped "Mediocre $seed $a"
		try=$((try + 1)); [ $try -gt 4 ] && { echo "[q275] ☠️ Mediocre $seed $a incomplete after 4 tries — left out"; return; }
		wait_timed "Mediocre $seed $a" || { echo "[q275] $(date) Mediocre $seed $a not run (cutoff)"; return; }
		k=""; [ "$a" = seed ] && k="option.Knobs=$KS"
		guard
		nice -n 5 "$FC" -engine "cmd=$NEW" name=ours option.OwnBook=false $k tc=10+0.1 timemargin=250 \
			-engine "cmd=$MED" name=mediocre tc=10+0.1 timemargin=1000 \
			$COMMON -srand $seed -rounds 75 -games 2 -pgnout file=$O/med_q274_s${seed}_$a.pgn notation=san append=false > "$f" 2>&1
		kill $GUARD 2>/dev/null
		echo "[q275] $(date '+%H:%M') Mediocre seed $seed $a: $(grep -E '^Games:' "$f" | tail -1) · warnings $(grep -c 'have option' "$f") · disconnects/stalls $(grep -ciE 'disconnect|stall|not responsive' "$f") · time losses $(grep -c 'loses on time' "$f")"
	done
}
for x in ctl seed; do med_chunk 274311 $x; done
for x in seed ctl; do med_chunk 274312 $x; done
row() {  # seed, arm, log: complete chunks only
	grep -E '^Games:' "$3" 2>/dev/null | tail -1 | awk -v s=$1 -v a=$2 '{ gsub(/,/, ""); for (i = 1; i <= NF; i++) { if ($i == "Games:") n = $(i+1); if ($i == "Wins:") w = $(i+1); if ($i == "Losses:") l = $(i+1); if ($i == "Draws:") d = $(i+1) } if (n > 0) print s, a, n, w, l, d }'
}
cmpr() {  # A, B, rows file, label: A - B over the seeds both played
	awk -v A=$1 -v B=$2 -v lab="$4" '
		{ if ($2 == A) { hA[$1] = 1; nA[$1] = $3; wA[$1] = $4; dA[$1] = $6 } if ($2 == B) { hB[$1] = 1; nB[$1] = $3; wB[$1] = $4; dB[$1] = $6 } }
		END {
			for (k in hA) if (hB[k]) { na += nA[k]; wa += wA[k]; da += dA[k]; nb += nB[k]; wb += wB[k]; db += dB[k]; seeds++ }
			if (na == 0 || nb == 0) { printf "  %s %-5s - %-5s: no shared seeds\n", lab, A, B; exit }
			sa = (wa + da / 2) / na; va = ((wa + da / 4) / na - sa * sa) / na
			sb = (wb + db / 2) / nb; vb = ((wb + db / 4) / nb - sb * sb) / nb
			d = sa - sb; se = sqrt(va + vb)
			printf "  %s %-5s - %-5s (%d seeds, %d / %d games): %.1f%% vs %.1f%%  %+.2f pp  ± %.2f (95%%)  z %+.2f\n", lab, A, B, seeds, na, nb, 100 * sa, 100 * sb, 100 * d, 196 * se, d / se
		}' $3
}
pooled seed
for f in $O/g18_q274_s*_*.log; do [ -e "$f" ] || continue; b=$(basename $f .log); b=${b#g18_q274_s}; row ${b%%_*} ${b#*_} $f; done > $T/sf.txt
for f in $O/med_q274_s*_*.log; do [ -e "$f" ] || continue; b=$(basename $f .log); b=${b#med_q274_s}; row ${b%%_*} ${b#*_} $f; done > $T/med.txt
echo "[q275] #274 per chunk (seed arm games W L D):"; { sed 's/^/Mediocre /' $T/med.txt; sed 's/^/SF18 /' $T/sf.txt; } | show
echo "[q275] #274 REGISTERED READS (seed: self-play lower 95% > 0, SF18 seed - ctl z > -1, Mediocre seed - ctl >= 0; W1: SF18 ctl - lazy0 z > -1 keeps lazy eval):"
{ cmpr seed ctl $T/sf.txt SF18; cmpr seed ctl $T/med.txt Mediocre; cmpr ctl lazy0 $T/sf.txt "SF18 W1"; } | show
echo "[q275] $(date '+%H:%M') #274 COMPLETE"

# ── 2 SORT_KEYED pinned bench (quiet box) ────────────────────────────────────────────────────────────────────────────────
stopped "the pinned bench"
if wait_timed "the pinned bench"; then
	busy && echo "[q275] ☠️ another job is running — the pinned bench below is disturbed"
	bench_one() { local b; case $2 in g76s) b="$G76S" ;; g76) b="$G76" ;; esac; echo "[q275]   r$1 $2 $(printf 'bench\nquit\n' | taskset -c 5 "$b" 2>/dev/null | grep '^bench')"; }
	bt() { grep -E "^\[q275\]   r[0-9]+ $1 bench" "$LOG" | awk '{print $2, $10}' | sort; }
	bn() { grep -E "^\[q275\]   r[0-9]+ $1 bench" "$LOG" | awk '{print $2, $13}' | sort; }
	med() { sort -g | awk '{a[NR] = $1} END { if (NR % 2) print a[(NR + 1) / 2]; else print (a[NR / 2] + a[NR / 2 + 1]) / 2 }'; }
	echo "[q275] $(date '+%H:%M') == pinned bench x21 on core 5, g76s / g76 alternating"
	for i in $(seq 1 21); do
		stopped "bench round $i"
		if [ $((i % 2)) = 1 ]; then o="g76s g76"; else o="g76 g76s"; fi
		for x in $o; do bench_one $i $x; done
	done
	for x in g76s g76; do echo "[q275] median $x: $(bt $x | awk '{print $2}' | med) ms · nps $(bn $x | awk '{print $2}' | med) over $(bt $x | wc -l) runs"; done
	RT=$(join <(bt g76s) <(bt g76) | awk '{print $2 / $3}' | med)
	NF=$(join <(bt g76s) <(bt g76) | awk '$2 < $3 {n++} END {print n + 0}')
	echo "[q275] within-round g76s / g76: time median $RT (g76s faster in $NF of 21) · nps median $(join <(bn g76s) <(bn g76) | awk '{print $2 / $3}' | med)"
	echo "[q275] registered: faster if the time median < 0.995; prediction 0.975 … 0.990 (report only, no install)"
else
	echo "[q275] pinned bench not run (cutoff)"
fi

# ── 3 C6 re-count ────────────────────────────────────────────────────────────────────────────────────────────────────────
stopped "the C6 count"
printf '#!/bin/bash\nexec "%s" 2>>"%s/err_$$.log"\n' "$NEW" "$T" > "$T/wrap.sh"; chmod +x "$T/wrap.sh"
c6_done() { grep -E '^Games:' "$W/c6.log" 2>/dev/null | tail -1 | grep -qE '^Games: 100,'; }
try=0
until c6_done; do
	try=$((try + 1)); [ $try -gt 4 ] && { echo "[q275] ☠️ C6 leg incomplete after 4 tries — no count"; break; }
	wait_timed "the C6 count" || { echo "[q275] C6 count not run (cutoff)"; break; }
	rm -f "$T"/err_*.log
	guard
	nice -n 5 "$FC" -engine "cmd=$T/wrap.sh" name=c6 option.OwnBook=false option.Debug=true "option.Knobs=ASP_KEEP_FAILHIGH=1,FIX_COUNT_LOG=1" tc=10+0.1 timemargin=250 \
		-engine "cmd=$NEW" name=ctl option.OwnBook=false tc=10+0.1 timemargin=250 \
		$COMMON -srand 2750 -rounds 50 -games 2 -pgnout file=$W/c6.pgn notation=san append=false > $W/c6.log 2>&1
	kill $GUARD 2>/dev/null
	echo "[q275] $(date '+%H:%M') C6 leg: $(grep -E '^Games:' $W/c6.log | tail -1) · warnings $(grep -c 'have option' $W/c6.log)"
done
if c6_done; then
	for f in "$T"/err_*.log; do grep -E '^TMCNT' "$f" | tail -1; done > "$W/tmcnt.txt"
	du -sh "$T" | show
	awk '{ for (i = 1; i <= NF; i++) { split($i, kv, "="); s[kv[1]] += kv[2] } } END { mv = (s["moves"] ? s["moves"] : 1); printf "moves %d · time cut %d (differs %d) · asp_keep_used %d (differs %d) · root clock exits %d · per 1,000 moves: cut %.2f, asp_keep_used %.2f\n", s["moves"], s["cut"], s["cut_differs"], s["asp_keep_used"], s["asp_keep_differs"], s["root_clock_m0"] + s["root_clock_later"], 1000 * s["cut"] / mv, 1000 * s["asp_keep_used"] / mv }' "$W/tmcnt.txt" | sed 's/^/[q275] C6 TMCNT (engine processes: '"$(wc -l < "$W/tmcnt.txt")"'): /'
	echo "[q275] registered: C6 live (worth games) if asp_keep_used >= 1 per 1,000 moves; prediction 0.5 … 2"
fi

# ── 4 C5 self-play ───────────────────────────────────────────────────────────────────────────────────────────────────────
stopped "the C5 self-play"
m=$(fp "$NEW" "$KM")
echo "[q275] C5 gate: $KM [$m] (must differ from the default [$a])"
if [ -n "$m" ] && [ "$m" != "$a" ]; then
	SEQ=0; sp_run malus "$KM" 2780
	pooled malus
else
	echo "[q275] ☠️ the C5 knob does not move the bench — no C5 games"
fi
echo "[q275] registered C5: lower 95% > 0 ⇒ SF18 + Mediocre next; within ±2σ ⇒ neutral, parked; upper 95% < 0 ⇒ rejected; prediction −15 … +8"
echo "[q275] $(date) QUEUE 275 DONE"
