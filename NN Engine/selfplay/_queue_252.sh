#!/bin/bash
# 2026-11-15 ~17:50, queue #252: phase H, the night after PROMOTE's games. Starts when #251 ends (~19:00 at #251's pace).
# Production stays g74pgo (gen 7.4, 250 / 37,320,382); nothing is built here.
#  A Pinned bench re-run (quiet box, before 20:30 only): #251's 5 rounds were disturbed (rounds 3-4 slow for every arm), so 9
#    rounds, arm order rotating each round, core 5: production g74pgo · e12pgo batch ON (EVAL_MAT_CACHE=1,EVAL_FAST=15) · e12pgo
#    batch OFF. REGISTERED: the eval speed batch is a speed-ship candidate when the median of the 9 within-round ratios
#    on / g74 is below 0.99, on beats g74 in at least 7 of 9 rounds, and the median on / off ratio is below 1. Prediction −3 to −6%.
#    If the window is missed, the bench runs after the morning's games instead.
#  B Timed self-play (10+0.1, vs production), C2 of phase H, alternating 150-pair chunks, fresh seed per chunk from 2521:
#      c2a = FIX_RESEARCH=1 (STS1500 +589 z +3.28, WAC 255 at +27% nodes)
#      c2c = FIX_RESEARCH=1,FIX_MIN_SCOUT=1,FIX_MIN_RD1=1 (STS1500 +210 z +0.52, WAC 252 at −5.6% nodes)
#    Runs until 20:50 (the owner games ~21:00 to midnight), resumes at 00:15 (or earlier on touch q252.go), until each arm
#    has 2,000 finished games or the Monday 20:50 cutoff. A1-A3 have NO valid game evidence: the 10-19/20 matches were void
#    (Knob_ options were never sent). This is their first real test, now on search v2 with D7 in place.
#  C SF18 at equal nodes (node-limited, safe while the owner games): ours 250k, SF18 1,100, 250 g per arm per seed, fresh
#    seeds 252111-252116, arm order rotating; arms ctl (production), c2a, c2c.
#  D Mediocre at 10+0.1 (timed, after self-play): 150 g per arm per seed, seeds 252311-252312, arms ctl, c2a, c2c.
# REGISTERED before any game (ship rule: self-play >= +2 sigma, SF18 arm − ctl z > −1, Mediocre arm − ctl >= 0 pp):
#   self-play pooled 2,000 per arm: lower 95% bound > 0 = passes; within ±2 sigma = NEUTRAL; upper bound < 0 = costs more
#   than it buys. Predictions: c2a 0 to +30 (fixed depth flatters its +27% nodes: about 0.2 ply at EBF 3.74), c2c 0 to +20.
#   If both pass, c2a vs c2c head to head decides between them (c2c is c2a plus the symmetric pair).
# Stop: touch /mnt/e/chess_data/q252.stop (between chunks). Hold timed work: touch /mnt/e/chess_data/q252.pause (a running
# timed chunk is interrupted; remove the file to resume). Never edit this file or the binaries while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
SF18="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/stockfish_18_linux/stockfish-ubuntu-x86-64-avx2"
MED="$ND/selfplay/mediocre_uci.sh"
O=/mnt/e/chess_data/fastchess
W=/mnt/e/chess_data/r15a
LOG=/mnt/e/chess_data/q252.log
PROD="$ND/build_uci/chess_engine_uci"
G74="$ND/build_uci/chess_engine_uci_g74pgo"
NEW="$ND/build_uci/chess_engine_uci_e12pgo"
K="EVAL_MAT_CACHE=1,EVAL_FAST=15"
STOP=/mnt/e/chess_data/q252.stop
PAUSE=/mnt/e/chess_data/q252.pause
GO=/mnt/e/chess_data/q252.go
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP" "$PAUSE" "$GO"; mkdir -p "$W"
show() { sed "s/^/[q252]   /"; }
stopped() { [ -e "$STOP" ] && { echo "[q252] $(date) stop file before $1 — QUEUE 252 DONE"; exit 0; }; }
kn() { case $1 in ctl) echo "" ;; c2a) echo "FIX_RESEARCH=1" ;; c2c) echo "FIX_RESEARCH=1,FIX_MIN_SCOUT=1,FIX_MIN_RD1=1" ;; esac; }
fp() { local k=""; [ -n "$2" ] && k="setoption name Knobs value $2"; printf '%s\nbench\nquit\n' "$k" | "$1" 2>&1 | grep -oE 'solved [0-9]+/300 nodes [0-9]+|mismatch.*' | head -1; }
busy() { pgrep -f "[f]astchess-linux-x86-64/fastchess" >/dev/null || pgrep -f "[s]ts_test|[t]actical_test|[v]algrind|[_]ks_footprint" >/dev/null; }
# timed work: never in the owner's evening (20:50 to 00:15 unless q252.go), never while q252.pause exists, never past the cutoff
timed_ok() { local t; t=$(date +%s); [ ! -e "$PAUSE" ] && [ "$t" -lt "$CUT" ] && ! { [ "$t" -ge "$GS" ] && [ "$t" -lt "$GE" ] && [ ! -e "$GO" ]; }; }
guard() { ( until ! timed_ok; do sleep 15; done; pkill -INT -f "[f]astchess-linux-x86-64/fastchess" && echo "[q252] $(date '+%H:%M') guard: games interrupted" ) & GUARD=$!; }
finished() { local c; c=$(grep -cE '^\[Result "(1-0|0-1|1/2-1/2)"\]' "$1" 2>/dev/null); echo "${c:-0}"; }
sp_count() { local n=0 f; for f in "$W/$1"_s*.pgn; do [ -e "$f" ] && n=$((n + $(finished "$f"))); done; echo $n; }

echo "[q252] $(date '+%Y-%m-%d %H:%M:%S') queued; waiting for #251 to end"
until grep -qE 'QUEUE 251 (DONE|ABORTED)|☠' /mnt/e/chess_data/q251.log; do stopped "the start"; sleep 20; done
sleep 30
START=$(date '+%Y-%m-%d %H:%M:%S')
if [ "$(date +%H)" -ge 12 ]; then
	GS=$(date -d 'today 20:50' +%s); GE=$(date -d "$(date -d tomorrow +%F) 00:15" +%s); CUT=$(date -d "$(date -d tomorrow +%F) 20:50" +%s)
else
	GS=$(date +%s); GE=$GS; CUT=$(date -d 'today 20:50' +%s)
fi
echo "[q252] $START start (no timed work $(date -d @$GS '+%a %H:%M') to $(date -d @$GE '+%a %H:%M') unless q252.go; cutoff $(date -d @$CUT '+%a %H:%M'))"
busy && { echo "[q252] ☠️ another games or bench job is running — QUEUE 252 ABORTED"; exit 1; }
cmp -s "$PROD" "$G74" || { echo "[q252] ☠️ production is not g74pgo — QUEUE 252 ABORTED"; exit 1; }

# ── 0 fingerprints and the knob preflight ─────────────────────────────────────────────────────────────────────────────────
F0=$(fp "$PROD" ""); FA=$(fp "$PROD" "$(kn c2a)"); FC2=$(fp "$PROD" "$(kn c2c)")
echo "[q252] production [$F0] (want 250 / 37320382) · c2a [$FA] (Cython WAC read 255 / 47417794) · c2c [$FC2] (Cython 252 / 35235460)"
[ "$F0" = "solved 250/300 nodes 37320382" ] || { echo "[q252] ☠️ production fingerprint — QUEUE 252 ABORTED"; exit 1; }
[ -n "$FA" ] && [ "$FA" != "$F0" ] && [ -n "$FC2" ] && [ "$FC2" != "$F0" ] && [ "$FA" != "$FC2" ] || { echo "[q252] ☠️ a C2 knob does not move the bench — QUEUE 252 ABORTED"; exit 1; }
rm -f /tmp/q252_pre.engine.log
timeout 180 "$FC" -engine "cmd=$PROD" name=c2a option.OwnBook=false "option.Knobs=$(kn c2a)" st=0.05 \
	-engine "cmd=$PROD" name=c2c option.OwnBook=false "option.Knobs=$(kn c2c)" st=0.05 \
	-openings file=$O/openings_uho.epd format=epd order=random -rounds 1 -games 2 -concurrency 1 \
	-log file=/tmp/q252_pre.engine.log level=info engine=true > /tmp/q252_pre.log 2>&1
WN=$(grep -c 'have option' /tmp/q252_pre.log)
echo "[q252] preflight: $(grep -E '^Games:' /tmp/q252_pre.log | tail -1) · warnings $WN · echoes FIX_RESEARCH=1 $(grep -cF 'FIX_RESEARCH=1' /tmp/q252_pre.engine.log), FIX_MIN_RD1=1 $(grep -cF 'FIX_MIN_RD1=1' /tmp/q252_pre.engine.log)"
[ "$WN" = 0 ] || { echo "[q252] ☠️ preflight warnings — QUEUE 252 ABORTED"; exit 1; }

# ── A pinned bench ────────────────────────────────────────────────────────────────────────────────────────────────────────
bench_one() {  # round, arm
	local out
	case $2 in
		g74) out=$(printf 'bench\nquit\n' | taskset -c 5 "$G74" 2>/dev/null | grep '^bench') ;;
		e12on) out=$(printf 'setoption name Knobs value %s\nbench\nquit\n' "$K" | taskset -c 5 "$NEW" 2>/dev/null | grep '^bench') ;;
		e12of) out=$(printf 'bench\nquit\n' | taskset -c 5 "$NEW" 2>/dev/null | grep '^bench') ;;
	esac
	echo "[q252]   r$1 $2 $out"
}
bt() { grep -E "^\[q252\]   r[0-9]+ $1 bench" "$LOG" | awk '{print $2, $10}' | sort; }   # round, ms
med() { sort -g | awk '{a[NR] = $1} END { if (NR % 2) print a[(NR + 1) / 2]; else print (a[NR / 2] + a[NR / 2 + 1]) / 2 }'; }
bench_done=0
do_bench() {
	local i o
	echo "[q252] $(date '+%H:%M') == TIMED: pinned bench x9 on core 5, arm order rotating"
	for i in 1 2 3 4 5 6 7 8 9; do
		stopped "bench round $i"
		case $((i % 3)) in 1) o="g74 e12on e12of" ;; 2) o="e12on e12of g74" ;; 0) o="e12of g74 e12on" ;; esac
		for a in $o; do bench_one $i $a; done
	done
	for a in g74 e12on e12of; do echo "[q252] median $a: $(bt $a | awk '{print $2}' | med) ms over $(bt $a | wc -l) runs"; done
	echo "[q252] within-round on / g74: median $(join <(bt e12on) <(bt g74) | awk '{print $2 / $3}' | med) · on faster in $(join <(bt e12on) <(bt g74) | awk '$2 < $3 {n++} END {print n + 0}') of $(join <(bt e12on) <(bt g74) | wc -l)"
	echo "[q252] within-round on / off: median $(join <(bt e12on) <(bt e12of) | awk '{print $2 / $3}' | med) · off / g74: median $(join <(bt e12of) <(bt g74) | awk '{print $2 / $3}' | med)"
	echo "[q252] REGISTERED: candidate when on / g74 median < 0.99, on faster in >= 7 of 9, and on / off median < 1"
	bench_done=1
}
stopped "the bench"
if [ $(( $(date +%s) + 1200 )) -lt "$GS" ] && [ ! -e "$PAUSE" ]; then do_bench; else echo "[q252] bench window missed — it runs after the morning's games"; fi

# ── B self-play chunks ────────────────────────────────────────────────────────────────────────────────────────────────────
COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
SEQ=0
sp_chunk() {  # arm; returns 1 when the arm already has 2,000
	local a=$1 n left pairs seed f
	n=$(sp_count $a); left=$(( (2000 - n + 1) / 2 )); [ $left -le 0 ] && return 1
	pairs=$(( left < 150 ? left : 150 )); SEQ=$((SEQ + 1)); seed=$((2520 + SEQ)); f=$W/${a}_s$seed
	echo "[q252] $(date '+%H:%M') self-play $a: $n finished, chunk seed $seed, $pairs pairs"
	guard
	nice -n 5 "$FC" -engine "cmd=$PROD" name=$a option.OwnBook=false "option.Knobs=$(kn $a)" tc=10+0.1 timemargin=250 \
		-engine "cmd=$PROD" name=g74 option.OwnBook=false tc=10+0.1 timemargin=250 \
		$COMMON -srand $seed -rounds $pairs -games 2 -pgnout file=$f.pgn notation=san nodes=true append=false > $f.log 2>&1
	kill $GUARD 2>/dev/null
	echo "[q252] $(date '+%H:%M') $a s$seed: $(grep -E '^Games:' $f.log | tail -1) · warnings $(grep -c 'have option' $f.log) · disconnects/stalls $(grep -ciE 'disconnect|stall|not responsive' $f.log) · time forfeits $(grep -c 'loses on time' $f.pgn) · unfinished $(grep -c '^\[Result "\*"' $f.pgn)"
	return 0
}
sp_loop() {
	local ran
	while timed_ok; do
		ran=0
		for a in c2a c2c; do stopped "a self-play chunk"; timed_ok || break; sp_chunk $a && ran=1; done
		[ $ran = 0 ] && break
	done
}
pooled() {
	local a
	for a in c2a c2c; do
		[ "$(sp_count $a)" -gt 0 ] || continue
		echo "[q252] pooled self-play $a vs gen 7.4 ($(sp_count $a) finished):"
		timeout 900 bash "$R" pyrun selfplay/fastchess_report.py "$W/$a"_s*.pgn --p1 $a 2>&1 | sed -n '/vs the field/,$p' | show
	done
}
sp_loop
pooled

# ── C SF18 at equal nodes (node-limited) ──────────────────────────────────────────────────────────────────────────────────
rm -f $O/g18_q252_s* $O/med_q252_s*
sf_chunk() {  # seed, arm
	local seed=$1 a=$2 k f=$O/g18_q252_s$1_$2.log
	stopped "SF18 $seed $a"
	k=$(kn $a); [ -n "$k" ] && k="option.Knobs=$k"
	nice -n 5 "$FC" -engine "cmd=$PROD" name=ours option.OwnBook=false $k nodes=250000 \
		-engine "cmd=$SF18" name=sf18 option.Threads=1 option.Hash=16 nodes=1100 \
		-each tc=inf $COMMON -srand $seed -rounds 125 -games 2 -pgnout file=$O/g18_q252_s${seed}_$a.pgn notation=san nodes=true append=false > "$f" 2>&1
	echo "[q252] $(date '+%H:%M') SF18 seed $seed $a: $(grep -E '^Games:' "$f" | tail -1) · warnings $(grep -c 'have option' "$f") · disconnects/stalls $(grep -ciE 'disconnect|stall|not responsive' "$f")"
}
echo "[q252] $(date '+%H:%M') SF18 legs (node-limited)"
for s in 1 2 3 4 5 6; do
	case $((s % 3)) in 1) o="ctl c2a c2c" ;; 2) o="c2a c2c ctl" ;; 0) o="c2c ctl c2a" ;; esac
	for a in $o; do sf_chunk $((252110 + s)) $a; done
done

# ── B again: the rest of the self-play ────────────────────────────────────────────────────────────────────────────────────
until timed_ok || [ "$(date +%s)" -ge "$CUT" ]; do stopped "the morning self-play"; sleep 30; done
sp_loop
pooled

# ── D Mediocre (timed) ────────────────────────────────────────────────────────────────────────────────────────────────────
med_chunk() {  # seed, arm
	local seed=$1 a=$2 k f=$O/med_q252_s$1_$2.log
	stopped "Mediocre $seed $a"
	timed_ok || { echo "[q252] $(date) skip Mediocre $seed $a (cutoff or pause)"; return; }
	k=$(kn $a); [ -n "$k" ] && k="option.Knobs=$k"
	guard
	nice -n 5 "$FC" -engine "cmd=$PROD" name=ours option.OwnBook=false $k tc=10+0.1 timemargin=250 \
		-engine "cmd=$MED" name=mediocre tc=10+0.1 timemargin=1000 \
		$COMMON -srand $seed -rounds 75 -games 2 -pgnout file=$O/med_q252_s${seed}_$a.pgn notation=san append=false > "$f" 2>&1
	kill $GUARD 2>/dev/null
	echo "[q252] $(date '+%H:%M') Mediocre seed $seed $a: $(grep -E '^Games:' "$f" | tail -1) · warnings $(grep -c 'have option' "$f") · disconnects/stalls $(grep -ciE 'disconnect|stall|not responsive' "$f") · time losses $(grep -c 'loses on time' "$f")"
}
for a in ctl c2a c2c; do med_chunk 252311 $a; done
for a in c2c c2a ctl; do med_chunk 252312 $a; done

# ── A again, if the evening window was missed ─────────────────────────────────────────────────────────────────────────────
if [ $bench_done = 0 ]; then
	if timed_ok && ! busy; then do_bench; else echo "[q252] bench skipped (cutoff or pause)"; fi
fi

# ── reads ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────
row() {  # seed, arm, log: complete chunks only
	grep -E '^Games:' "$3" 2>/dev/null | tail -1 | awk -v s=$1 -v a=$2 '{ gsub(/,/, ""); for (i = 1; i <= NF; i++) { if ($i == "Games:") n = $(i+1); if ($i == "Wins:") w = $(i+1); if ($i == "Losses:") l = $(i+1); if ($i == "Draws:") d = $(i+1) } if (n > 0) print s, a, n, w, l, d }'
}
cmpr() {  # A, B, rows file, label: A - B over the seeds both played
	awk -v A=$1 -v B=$2 -v lab="$4" '
		{ if ($2 == A) { hA[$1] = 1; nA[$1] = $3; wA[$1] = $4; dA[$1] = $6 } if ($2 == B) { hB[$1] = 1; nB[$1] = $3; wB[$1] = $4; dB[$1] = $6 } }
		END {
			for (k in hA) if (hB[k]) { na += nA[k]; wa += wA[k]; da += dA[k]; nb += nB[k]; wb += wB[k]; db += dB[k]; seeds++ }
			if (na == 0 || nb == 0) { printf "  %s %-4s - %-4s: no shared seeds\n", lab, A, B; exit }
			sa = (wa + da / 2) / na; va = ((wa + da / 4) / na - sa * sa) / na
			sb = (wb + db / 2) / nb; vb = ((wb + db / 4) / nb - sb * sb) / nb
			d = sa - sb; se = sqrt(va + vb)
			printf "  %s %-4s - %-4s (%d seeds, %d / %d games): %.1f%% vs %.1f%%  %+.2f pp  ± %.2f (95%%)  z %+.2f\n", lab, A, B, seeds, na, nb, 100 * sa, 100 * sb, 100 * d, 196 * se, d / se
		}' $3
}
for f in $O/g18_q252_s*_*.log; do [ -e "$f" ] || continue; b=$(basename $f .log); b=${b#g18_q252_s}; row ${b%%_*} ${b#*_} $f; done > /tmp/q252_sf.txt
for f in $O/med_q252_s*_*.log; do [ -e "$f" ] || continue; b=$(basename $f .log); b=${b#med_q252_s}; row ${b%%_*} ${b#*_} $f; done > /tmp/q252_med.txt
echo "[q252] $(date) per chunk (seed arm games W L D):"; { sed 's/^/Mediocre /' /tmp/q252_med.txt; sed 's/^/SF18 /' /tmp/q252_sf.txt; } | show
echo "[q252] REGISTERED READS (SF18: arm - ctl z > -1 = pass; Mediocre: arm - ctl >= 0 = pass):"
{ cmpr c2a ctl /tmp/q252_sf.txt SF18; cmpr c2c ctl /tmp/q252_sf.txt SF18; cmpr c2a ctl /tmp/q252_med.txt Mediocre; cmpr c2c ctl /tmp/q252_med.txt Mediocre; } | show
echo "[q252] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | show
echo "[q252] $(date) QUEUE 252 DONE"
