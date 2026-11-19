#!/bin/bash
# 2026-11-19 ~11:45, queue #280: S2-3 — a futility-pruned move's bound (eval + margin) counts toward the node's fail-low bound
# (SV2_FUT_BOUND=1, SF's bestValue = futilityValue), on gen 7.7. Chained after #279 (the SF1.1 judge's calibration).
# Why: #277 counted no-cut exits whose futility-pruned value beats the returned bound: 20.2% (WAC), 26.4% (game12), mean
# ~0.4 pawn; the parent stores the too-low upper bound at full depth. Sweep-2 order: S2-3 first.
# ★ FIRST RUN OF THE STREAMLINED JUDGING (owner 11-19: "sure, lets try it and observe if it truly saves time while also still
# being accurate"). Self-play runs first and alone; the outside judges play only if self-play passes, because a candidate that
# fails self-play cannot ship whatever they say. The outside judges are 3 SF18 seeds + 3 SF1.1 seeds per arm, read as ONE pooled
# "not clearly worse" check (the old form: 6 SF18 seeds and 3 third-judge seeds every time). Every stage is timed, and the
# report prints this run's box hours beside what the old form would have cost.
#  1 Wait for #279; production must be g77pgo; nothing else running. PGO test build chess_engine_uci_s3pgo from the current
#    sources (gen 7.7 + the SV2_FUT_BOUND knob, default 0). GATES: default 254 / 35,922,538 == g77pgo; SV2_FUT_BOUND=1 must
#    move the bench; v1 250 / 35,310,775-778; game12 d10 default == g77pgo (7/12 2,412,001).
#  2 Self-play fut vs gen 7.7, 10+0.1, 150-pair chunks, seeds 2801+, to 2,000 games. No timed work 20:50 to 00:15 unless
#    q280.go; q280.pause holds; cutoff tomorrow 10:40.
#  3 ONLY IF self-play passes (2,000 games, lower 95% > 0): SF18 (ours 250k, SF18 1,100 nodes) seeds 280111-280113 and SF1.1
#    (ours 250k, SF1.1 at the #279 rung nearest 50%) seeds 280211-280213, ctl and fut, 250 g each, node-limited.
# REGISTERED before any game:
#  - self-play lower 95% > 0 = pass; within ±2σ = NEUTRAL (node direction unclear, so parked with a trigger, not rejected);
#    upper 95% < 0 = fails. Prediction −10 … +15.
#  - outside judges, pooled SF18 + SF1.1: fut − ctl z > −1 (not clearly worse). Each judge is also reported alone.
#  - Bench: SV2_FUT_BOUND=1 moves it; prediction nodes within ±4% of 35,922,538.
# Stop: touch /mnt/e/chess_data/q280.stop (between steps). Never edit this file, the binaries or the sources before "build done".
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
SF18="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/stockfish_18_linux/stockfish-ubuntu-x86-64-avx2"
SF1=/mnt/e/chess_data/engines/sf1.1/stockfish-1.1
O=/mnt/e/chess_data/fastchess
W=/mnt/e/chess_data/q280
GW=/mnt/e/chess_data/r280
P=/mnt/e/chess_data/pgo_s3
G=/mnt/e/chess_data/sv2_speed/game12.txt
PROD="$ND/build_uci/chess_engine_uci"
G77="$ND/build_uci/chess_engine_uci_g77pgo"
NEW="$ND/build_uci/chess_engine_uci_s3pgo"
KS="SV2_FUT_BOUND=1"
LOG=/mnt/e/chess_data/q280.log
Q279=/mnt/e/chess_data/q279.log
NFILE=/mnt/e/chess_data/sf11_judge_nodes.txt
STOP=/mnt/e/chess_data/q280.stop
PAUSE=/mnt/e/chess_data/q280.pause
GO=/mnt/e/chess_data/q280.go
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
SF18_LEG_MIN=13   # an SF18 leg (250 g) in #274: 12.5-13.5 min
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP" "$PAUSE" "$GO"; mkdir -p "$W" "$GW"
rm -f "$GW"/fut_s*.pgn "$GW"/fut_s*.log $O/g18_q280_s* $O/s11_q280_s*
show() { sed "s/^/[q280]   /"; }
stopped() { [ -e "$STOP" ] && { echo "[q280] $(date) stop file before $1 — QUEUE 280 DONE"; exit 0; }; }
kn() { case $1 in ctl) echo "" ;; fut) echo "$KS" ;; esac; }
fp() { local k=""; [ -n "$2" ] && k="setoption name Knobs value $2"; printf '%s\nbench\nquit\n' "$k" | "$1" 2>&1 | grep -oE 'solved [0-9]+/300 nodes [0-9]+|mismatch.*' | head -1; }
gfp() { local k=""; [ -n "$2" ] && k="setoption name Knobs value $2"; printf "%s\nbench $G 10\nquit\n" "$k" | "$1" 2>&1 | grep -oE 'solved [0-9]+/[0-9]+ nodes [0-9]+|mismatch.*' | head -1; }
busy() { pgrep -f "[f]astchess-linux-x86-64/fastchess" >/dev/null || pgrep -f "[s]ts_test|[t]actical_test|[v]algrind|[_]ks_footprint|[b]uild_uci.sh" >/dev/null; }
timed_ok() { local t; t=$(date +%s); [ ! -e "$PAUSE" ] && [ "$t" -lt "$CUT" ] && ! { [ "$t" -ge "$GS" ] && [ "$t" -lt "$GE" ] && [ ! -e "$GO" ]; }; }
guard() { ( until ! timed_ok; do sleep 15; done; pkill -INT -f "[f]astchess-linux-x86-64/fastchess" && echo "[q280] $(date '+%H:%M') guard: games interrupted" ) & GUARD=$!; }
finished() { local c; c=$(grep -cE '^\[Result "(1-0|0-1|1/2-1/2)"\]' "$1" 2>/dev/null); echo "${c:-0}"; }
sp_count() { local n=0 f; for f in "$GW/$1"_s*.pgn; do [ -e "$f" ] && n=$((n + $(finished "$f"))); done; echo $n; }
hrs() { awk -v s=$1 'BEGIN { printf "%.2f h", s / 3600 }'; }
echo "[q280] $(date '+%Y-%m-%d %H:%M:%S') queued; waiting for #279"
until grep -qE 'QUEUE 279 (DONE|ABORTED)' "$Q279" 2>/dev/null || ! pgrep -f "[_]queue_279.sh" >/dev/null; do stopped "the start"; sleep 30; done
sleep 20
cmp -s "$PROD" "$G77" || { echo "[q280] ☠️ production is not g77pgo — QUEUE 280 ABORTED"; exit 1; }
busy && { echo "[q280] ☠️ another job is running — QUEUE 280 ABORTED"; exit 1; }
START=$(date '+%Y-%m-%d %H:%M:%S'); T0=$(date +%s)
echo "[q280] $START start"

# ── 1 build + gates ──────────────────────────────────────────────────────────────────────────────────────────────────────
rm -rf "$P"; mkdir -p "$P"
if OUT=chess_engine_uci_s3gen PGO=gen PGO_DIR=$P nice -n 10 bash selfplay/build_uci.sh > "$W/gen.out" 2>&1; then
	nice -n 10 "$FC" -engine cmd="$ND/build_uci/chess_engine_uci_s3gen" name=a option.OwnBook=false st=0.25 \
		-engine cmd="$ND/build_uci/chess_engine_uci_s3gen" name=b option.OwnBook=false st=0.25 \
		-openings file=$O/openings_uho.epd format=epd order=random -srand 780 -rounds 4 -games 2 -concurrency 1 > "$W/train.log" 2>&1
	OUT=chess_engine_uci_s3pgo PGO=use PGO_DIR=$P nice -n 10 bash selfplay/build_uci.sh > "$W/use.out" 2>&1 || { echo "[q280] ☠️ PGO use build failed — QUEUE 280 ABORTED"; grep -iE 'error' "$W/use.out" | head -8 | show; exit 1; }
else
	echo "[q280] ☠️ PGO gen build failed — QUEUE 280 ABORTED"; grep -iE 'error' "$W/gen.out" | head -8 | show; exit 1
fi
echo "[q280] $(date '+%H:%M') build done (chess_engine_uci_s3pgo) — the sources are free to edit"
a=$(fp "$NEW" ""); b=$(fp "$G77" ""); k=$(fp "$NEW" "$KS")
v1=$(printf 'setoption name EvalArm value 0\nbench\nquit\n' | "$NEW" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+')
ga=$(gfp "$NEW" ""); gb=$(gfp "$G77" ""); gk=$(gfp "$NEW" "$KS")
echo "[q280] s3pgo default [$a] == g77pgo [$b] (254 / 35922538) · $KS [$k] (must differ; prediction nodes ±4%) · v1 [$v1] · game12 d10 default [$ga] == g77 [$gb] · $KS [$gk]"
[ "$a" = "solved 254/300 nodes 35922538" ] && [ "$b" = "$a" ] && [ -n "$k" ] && [ "$k" != "$a" ] \
	&& echo "$v1" | grep -qE 'solved 250/300 nodes 3531077[5-8]' && [ -n "$ga" ] && [ "$ga" = "$gb" ] || { echo "[q280] ☠️ a gate failed — no games — QUEUE 280 ABORTED"; exit 1; }
echo "[q280] ✅ every gate equal"
rm -f /tmp/q280_pre.engine.log
timeout 180 "$FC" -engine "cmd=$NEW" name=fut option.OwnBook=false "option.Knobs=$KS" st=0.05 \
	-engine "cmd=$NEW" name=ctl option.OwnBook=false st=0.05 \
	-openings file=$O/openings_uho.epd format=epd order=random -rounds 1 -games 2 -concurrency 1 \
	-log file=/tmp/q280_pre.engine.log level=info engine=true > /tmp/q280_pre.log 2>&1
WN=$(grep -c 'have option' /tmp/q280_pre.log)
echo "[q280] preflight: $(grep -E '^Games:' /tmp/q280_pre.log | tail -1) · warnings $WN · echoes $KS $(grep -cF "$KS" /tmp/q280_pre.engine.log)"
[ "$WN" = 0 ] || { echo "[q280] ☠️ preflight warnings — no games — QUEUE 280 DONE"; exit 1; }
T_BUILD=$(( $(date +%s) - T0 ))

# ── 2 self-play, alone and first ─────────────────────────────────────────────────────────────────────────────────────────
if [ "$(date +%H)" -ge 12 ]; then
	GS=$(date -d 'today 20:50' +%s); GE=$(date -d "$(date -d tomorrow +%F) 00:15" +%s); CUT=$(date -d "$(date -d tomorrow +%F) 10:40" +%s)
else
	GS=$(date -d 'today 20:50' +%s); GE=$(date -d "$(date -d tomorrow +%F) 00:15" +%s); CUT=$(date -d 'today 20:50' +%s)
fi
echo "[q280] $(date '+%H:%M') self-play fut vs gen 7.7 (no timed work $(date -d @$GS '+%a %H:%M') to $(date -d @$GE '+%a %H:%M') unless q280.go; cutoff $(date -d @$CUT '+%a %H:%M'))"
COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
SEQ=0; T_SP=0; T_WAIT=0
sp_chunk() {  # arm; returns 1 when the arm already has 2,000
	local a=$1 n left pairs seed f t
	n=$(sp_count $a); left=$(( (2000 - n + 1) / 2 )); [ $left -le 0 ] && return 1
	pairs=$(( left < 150 ? left : 150 )); SEQ=$((SEQ + 1)); seed=$((2800 + SEQ)); f=$GW/${a}_s$seed
	echo "[q280] $(date '+%H:%M') self-play $a: $n finished, chunk seed $seed, $pairs pairs"
	guard; t=$(date +%s)
	nice -n 5 "$FC" -engine "cmd=$NEW" name=$a option.OwnBook=false "option.Knobs=$(kn $a)" tc=10+0.1 timemargin=250 \
		-engine "cmd=$NEW" name=g77 option.OwnBook=false tc=10+0.1 timemargin=250 \
		$COMMON -srand $seed -rounds $pairs -games 2 -pgnout file=$f.pgn notation=san nodes=true append=false > $f.log 2>&1
	kill $GUARD 2>/dev/null; T_SP=$(( T_SP + $(date +%s) - t ))
	echo "[q280] $(date '+%H:%M') $a s$seed: $(grep -E '^Games:' $f.log | tail -1) · warnings $(grep -c 'have option' $f.log) · disconnects/stalls $(grep -ciE 'disconnect|stall|not responsive' $f.log) · time forfeits $(grep -c 'loses on time' $f.pgn) · unfinished $(grep -c '^\[Result "\*"' $f.pgn)"
	return 0
}
sp_loop() { while timed_ok; do stopped "a self-play chunk"; sp_chunk fut || break; done; }
while [ "$(sp_count fut)" -lt 2000 ] && [ "$(date +%s)" -lt "$CUT" ]; do
	t=$(date +%s); until timed_ok || [ "$(date +%s)" -ge "$CUT" ]; do stopped "the self-play"; sleep 30; done; T_WAIT=$(( T_WAIT + $(date +%s) - t ))
	sp_loop
done
SPN=$(sp_count fut)
echo "[q280] pooled self-play fut vs gen 7.7 ($SPN finished):"
timeout 900 bash "$R" pyrun selfplay/fastchess_report.py "$GW/fut"_s*.pgn --p1 fut 2>&1 | sed -n '/vs the field/,$p' > "$W/pooled.txt"
show < "$W/pooled.txt"
LO=$(grep -oE '\(95% [+-]?[0-9.]+' "$W/pooled.txt" | head -1 | sed 's/(95% //')
HI=$(grep -oE '… [+-]?[0-9.]+\)' "$W/pooled.txt" | head -1 | sed 's/… //; s/)//')
if [ "$SPN" -lt 2000 ]; then VERDICT="INCOMPLETE ($SPN of 2,000 by the cutoff)"
elif awk -v l="$LO" 'BEGIN { exit !(l != "" && l + 0 > 0) }'; then VERDICT="PASS"
elif awk -v h="$HI" 'BEGIN { exit !(h != "" && h + 0 < 0) }'; then VERDICT="FAIL"
else VERDICT="NEUTRAL"; fi
echo "[q280] REGISTERED self-play read: $VERDICT (95% $LO … $HI; pass = lower > 0, neutral = within ±2σ → parked, fail = upper < 0)"

# SF1.1's node count: an override file, else the #279 rung nearest 50% if it lies in 35-65%
N11=$(cat "$NFILE" 2>/dev/null | tr -dc '0-9')
[ -z "$N11" ] && N11=$(grep -E '^\[q279\] +SF1\.1 +[0-9]+ nodes:' "$Q279" | awk '{ gsub(/%/, "", $7); d = $7 - 50; if (d < 0) d = -d; if (best == "" || d < bd) { best = $3; bd = d; sc = $7 } } END { if (best != "" && sc >= 35 && sc <= 65) print best }')
# minutes #279 took for 100 games at N11 (for the time accounting)
RUNG_MIN=$(awk -v n="$N11" '$0 ~ "rung SF1.1 nodes=" n " " { split($2, a, ":"); s = a[1] * 60 + a[2] } $0 ~ "[]] [0-9:]+ nodes=" n ":" { split($2, b, ":"); e = b[1] * 60 + b[2] } END { if (s != "" && e != "") print e - s }' "$Q279")
S11_LEG_MIN=$(awk -v r="${RUNG_MIN:-10}" 'BEGIN { printf "%.0f", r * 2.5 }')
echo "[q280] SF1.1 judge: ${N11:-none} nodes (100 g took ${RUNG_MIN:-?} min in #279 ⇒ ~$S11_LEG_MIN min per 250-game leg)"

# ── 3 the outside judges, only when self-play passes ─────────────────────────────────────────────────────────────────────
T_EX=0
ext_chunk() {  # judge (g18 | s11), seed, arm
	local j=$1 seed=$2 a=$3 k f=$O/${1}_q280_s$2_$3.log opp t
	stopped "$j $seed $a"
	k=$(kn $a); [ -n "$k" ] && k="option.Knobs=$k"
	case $j in
		g18) opp=(-engine "cmd=$SF18" name=sf18 option.Threads=1 option.Hash=16 nodes=1100) ;;
		s11) opp=(-engine "cmd=$SF1" name=sf1.1 option.Threads=1 option.Ponder=false option.OwnBook=false option.Hash=16 nodes=$N11) ;;
	esac
	t=$(date +%s)
	nice -n 5 "$FC" -engine "cmd=$NEW" name=ours option.OwnBook=false $k nodes=250000 "${opp[@]}" \
		-each tc=inf $COMMON -srand $seed -rounds 125 -games 2 -pgnout file=$O/${j}_q280_s${seed}_$a.pgn notation=san nodes=true append=false > "$f" 2>&1
	T_EX=$(( T_EX + $(date +%s) - t ))
	echo "[q280] $(date '+%H:%M') $j seed $seed $a: $(grep -E '^Games:' "$f" | tail -1) · warnings $(grep -c 'have option' "$f") · disconnects/stalls $(grep -ciE 'disconnect|stall|not responsive' "$f") · $(( $(date +%s) - t ))s"
}
if [ "$VERDICT" = "PASS" ]; then
	echo "[q280] $(date '+%H:%M') self-play passed ⇒ outside judges: SF18 3 seeds, SF1.1 ${N11:-(not calibrated: skipped)} 3 seeds, ctl and fut"
	for s in 1 2 3; do
		if [ $((s % 2)) = 1 ]; then o="ctl fut"; else o="fut ctl"; fi
		for x in $o; do ext_chunk g18 $((280110 + s)) $x; done
	done
	if [ -n "$N11" ]; then
		for s in 1 2 3; do
			if [ $((s % 2)) = 1 ]; then o="ctl fut"; else o="fut ctl"; fi
			for x in $o; do ext_chunk s11 $((280210 + s)) $x; done
		done
	fi
else
	echo "[q280] self-play $VERDICT ⇒ the outside judges are skipped (the candidate cannot ship without a self-play pass)"
fi

# ── reads ────────────────────────────────────────────────────────────────────────────────────────────────────────────────
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
if [ "$VERDICT" = "PASS" ]; then
	for j in g18 s11; do
		for f in $O/${j}_q280_s*_*.log; do [ -e "$f" ] || continue; b=$(basename $f .log); b=${b#${j}_q280_s}; row ${b%%_*} ${b#*_} $f; done > /tmp/q280_$j.txt
	done
	cat /tmp/q280_g18.txt /tmp/q280_s11.txt > /tmp/q280_ext.txt
	echo "[q280] $(date) per chunk (seed arm games W L D):"; { sed 's/^/SF18 /' /tmp/q280_g18.txt; sed "s/^/SF1.1@$N11 /" /tmp/q280_s11.txt; } | show
	echo "[q280] REGISTERED outside read (pooled SF18 + SF1.1: fut - ctl z > -1), and each judge alone:"
	{ cmpr fut ctl /tmp/q280_ext.txt "POOLED"; cmpr fut ctl /tmp/q280_g18.txt "SF18  "; cmpr fut ctl /tmp/q280_s11.txt "SF1.1 "; } | show
fi

# ── the time accounting (owner: "observe if it truly saves time") ────────────────────────────────────────────────────────
TOT=$(( $(date +%s) - T0 ))
OLD_EX=$(( (12 * SF18_LEG_MIN + 6 * S11_LEG_MIN) * 60 ))
NEW_EX_FULL=$(( (6 * SF18_LEG_MIN + 6 * S11_LEG_MIN) * 60 ))
echo "[q280] TIME: build + gates $(hrs $T_BUILD) · self-play $(hrs $T_SP) (+ $(hrs $T_WAIT) waiting out the gaming window) · outside judges $(hrs $T_EX) · total $(hrs $TOT)"
if [ "$VERDICT" = "PASS" ]; then
	echo "[q280] TIME: the old form's outside judges (6 SF18 seeds + 3 SF1.1 seeds per arm) ≈ $(hrs $OLD_EX) at these leg times ⇒ saved ≈ $(hrs $(( OLD_EX - T_EX )))"
else
	echo "[q280] TIME: outside judges skipped ⇒ saved ≈ $(hrs $OLD_EX) against the old form (12 SF18 legs x ${SF18_LEG_MIN} min + 6 SF1.1 legs x ~${S11_LEG_MIN} min)"
fi
echo "[q280] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | show
echo "[q280] $(date) QUEUE 280 DONE"
