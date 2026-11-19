#!/bin/bash
# 2026-11-19 ~17:10, queue #281: S2-9 — a lazy eval estimate is returned as what the lazy exit proved, the estimate less the
# margin it cleared its bound by (LAZY_FAIL_HARD=2), on gen 7.7. Chained after #280 (S2-3). Owner 11-19: S2-9 "sounds like a
# more serious problem", so it goes before S2-26 (the RFP return's blend form, #282), which changes the same return.
# Why: a lazy estimate (up to ~2.6 pawns off at p99) proves that its site cuts, not by how much, yet reverse futility returns it
# and the parent stores it in the TT at full depth (38% / 33% of RFP returns, #277), and qsearch's stand-pat cut returns it into
# the q-cache (~5% of q-cache stores). The lazy exit fires only when the estimate clears its bound by a margin M (1.5 pawns with
# the parent's positional part, 3 without, times the site scale), so "estimate − M" is what it proved; mode 2 returns that
# (never below beta). The lazy exits themselves are unchanged.
# Form, measured before this queue (16:35-17:03, plain check build chess_engine_uci_s6chk, every gate equal): the first form,
# mode 1 (return beta), costs +28.9% bench nodes (249 / 46,312,304; predicted +0.5 … +5%); +24.3% from reverse futility alone,
# +20.6% from the stand-pat cut alone, +26.8% with the near-miss re-search off, so the cost is the lost fail-soft information
# itself. Mode 2: 249 / 37,325,964 (+3.9%; predicted +5 … +15%), game12 d10 9/12 at 2,785,346 (+15%) vs 7/12 (mode 1 9/12 at
# 3,221,293). Mode 2 is the form played; mode 1 is parked (its node cost would mask any gain).
# Third run of the STREAMLINED JUDGING trial (owner 11-19), with two changes from #280's form:
#  - one shared pause file, /mnt/e/chess_data/gaming.pause, beside q281.pause (either holds timed work);
#  - while timed work is held (a gaming pause or the 20:50 to 00:15 block) and self-play is unfinished, the box plays the
#    outside judges' node-limited legs instead of idling. If self-play then passes they count; if not, they are reported as
#    information only. A leg in progress when the hold lifts finishes first (≤ ~25 min), and the log says so.
#  1 Wait for #280; production must be g77pgo; nothing else running. PGO test build chess_engine_uci_s6pgo from the current
#    sources (gen 7.7 + the knobs SV2_FUT_BOUND, ROOT_M0_CUT (S2-6), PS_TT_FLAG_FIX (S2-4), LAZY_FAIL_HARD (S2-9), all default
#    0): the binary for the next queues too. GATES (the plain build's values; PGO keeps results): default 254 / 35,922,538 ==
#    g77pgo; LAZY_FAIL_HARD=2 == 249 / 37,325,964; LAZY_FAIL_HARD=1 == 249 / 46,312,304; SV2_FUT_BOUND=1 == #280's
#    252 / 36,716,126; v1 250 / 35,310,775-778; game12 d10 default == g77pgo (7/12 2,412,001). ROOT_M0_CUT=1, PS_TT_FLAG_FIX=1
#    and RFP_RETURN_BLEND=67 are recorded (liveness for their own queues), not gated.
#  2 Self-play lfh vs gen 7.7, 10+0.1, 150-pair chunks, seeds 2811+, to 2,000 games. No timed work 20:50 to 00:15 unless
#    q281.go; q281.pause or gaming.pause holds; cutoff tomorrow 10:40. Held time plays outside-judge legs (above).
#  3 IF self-play passes (2,000 games, lower 95% > 0): the outside-judge legs not yet played: SF18 (ours 250k, SF18 1,100
#    nodes) seeds 281111-281113 and SF1.1 (ours 250k, SF1.1 at the #279 rung nearest 50%, 320k) seeds 281211-281213, ctl and
#    lfh, 250 g each, node-limited.
# REGISTERED before any game:
#  - self-play lower 95% > 0 = pass; within ±2σ = NEUTRAL (a node-adding fix: parked with a node-saver partner such as S2-6,
#    not rejected); upper 95% < 0 = fails. Prediction −6 … +12.
#  - outside judges, pooled SF18 + SF1.1: lfh − ctl z > −1 (not clearly worse). Each judge is also reported alone.
# Stop: touch /mnt/e/chess_data/q281.stop (between steps). Never edit this file, the binaries or the sources before "build done".
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
SF18="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/stockfish_18_linux/stockfish-ubuntu-x86-64-avx2"
SF1=/mnt/e/chess_data/engines/sf1.1/stockfish-1.1
O=/mnt/e/chess_data/fastchess
W=/mnt/e/chess_data/q281
GW=/mnt/e/chess_data/r281
P=/mnt/e/chess_data/pgo_s6
G=/mnt/e/chess_data/sv2_speed/game12.txt
PROD="$ND/build_uci/chess_engine_uci"
G77="$ND/build_uci/chess_engine_uci_g77pgo"
NEW="$ND/build_uci/chess_engine_uci_s6pgo"
KS="LAZY_FAIL_HARD=2"
Q280=/mnt/e/chess_data/q280.log
LOG=/mnt/e/chess_data/q281.log
Q279=/mnt/e/chess_data/q279.log
NFILE=/mnt/e/chess_data/sf11_judge_nodes.txt
STOP=/mnt/e/chess_data/q281.stop
PAUSE=/mnt/e/chess_data/q281.pause
GPAUSE=/mnt/e/chess_data/gaming.pause
GO=/mnt/e/chess_data/q281.go
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
SF18_LEG_MIN=13   # an SF18 leg (250 g) in #274: 12.5-13.5 min
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP" "$GO"; mkdir -p "$W" "$GW"   # the pause files are the owner's: never removed here
rm -f "$GW"/lfh_s*.pgn "$GW"/lfh_s*.log $O/g18_q281_s* $O/s11_q281_s*
show() { sed "s/^/[q281]   /"; }
stopped() { [ -e "$STOP" ] && { echo "[q281] $(date) stop file before $1 — QUEUE 281 DONE"; exit 0; }; }
kn() { case $1 in ctl) echo "" ;; lfh) echo "$KS" ;; esac; }
fp() { local k=""; [ -n "$2" ] && k="setoption name Knobs value $2"; printf '%s\nbench\nquit\n' "$k" | "$1" 2>&1 | grep -oE 'solved [0-9]+/300 nodes [0-9]+|mismatch.*' | head -1; }
gfp() { local k=""; [ -n "$2" ] && k="setoption name Knobs value $2"; printf "%s\nbench $G 10\nquit\n" "$k" | "$1" 2>&1 | grep -oE 'solved [0-9]+/[0-9]+ nodes [0-9]+|mismatch.*' | head -1; }
busy() { pgrep -f "[f]astchess-linux-x86-64/fastchess" >/dev/null || pgrep -f "[s]ts_test|[t]actical_test|[v]algrind|[_]ks_footprint|[b]uild_uci.sh" >/dev/null; }
timed_ok() { local t; t=$(date +%s); [ ! -e "$PAUSE" ] && [ ! -e "$GPAUSE" ] && [ "$t" -lt "$CUT" ] && ! { [ "$t" -ge "$GS" ] && [ "$t" -lt "$GE" ] && [ ! -e "$GO" ]; }; }
guard() { ( until ! timed_ok; do sleep 15; done; pkill -INT -f "[f]astchess-linux-x86-64/fastchess" && echo "[q281] $(date '+%H:%M') guard: games interrupted" ) & GUARD=$!; }
finished() { local c; c=$(grep -cE '^\[Result "(1-0|0-1|1/2-1/2)"\]' "$1" 2>/dev/null); echo "${c:-0}"; }
sp_count() { local n=0 f; for f in "$GW/$1"_s*.pgn; do [ -e "$f" ] && n=$((n + $(finished "$f"))); done; echo $n; }
hrs() { awk -v s=$1 'BEGIN { printf "%.2f h", s / 3600 }'; }
echo "[q281] $(date '+%Y-%m-%d %H:%M:%S') queued; waiting for #280"
until grep -qE 'QUEUE 280 (DONE|ABORTED)' "$Q280" 2>/dev/null || ! pgrep -f "[_]queue_280.sh" >/dev/null; do stopped "the start"; sleep 30; done
sleep 20
cmp -s "$PROD" "$G77" || { echo "[q281] ☠️ production is not g77pgo — QUEUE 281 ABORTED"; exit 1; }
busy && { echo "[q281] ☠️ another job is running — QUEUE 281 ABORTED"; exit 1; }
START=$(date '+%Y-%m-%d %H:%M:%S'); T0=$(date +%s)
echo "[q281] $START start"

# ── 1 build + gates ──────────────────────────────────────────────────────────────────────────────────────────────────────
rm -rf "$P"; mkdir -p "$P"
if OUT=chess_engine_uci_s6gen PGO=gen PGO_DIR=$P nice -n 10 bash selfplay/build_uci.sh > "$W/gen.out" 2>&1; then
	nice -n 10 "$FC" -engine cmd="$ND/build_uci/chess_engine_uci_s6gen" name=a option.OwnBook=false st=0.25 \
		-engine cmd="$ND/build_uci/chess_engine_uci_s6gen" name=b option.OwnBook=false st=0.25 \
		-openings file=$O/openings_uho.epd format=epd order=random -srand 781 -rounds 4 -games 2 -concurrency 1 > "$W/train.log" 2>&1
	OUT=chess_engine_uci_s6pgo PGO=use PGO_DIR=$P nice -n 10 bash selfplay/build_uci.sh > "$W/use.out" 2>&1 || { echo "[q281] ☠️ PGO use build failed — QUEUE 281 ABORTED"; grep -iE 'error' "$W/use.out" | head -8 | show; exit 1; }
else
	echo "[q281] ☠️ PGO gen build failed — QUEUE 281 ABORTED"; grep -iE 'error' "$W/gen.out" | head -8 | show; exit 1
fi
echo "[q281] $(date '+%H:%M') build done (chess_engine_uci_s6pgo) — the sources are free to edit"
a=$(fp "$NEW" ""); b=$(fp "$G77" ""); k=$(fp "$NEW" "$KS"); k1=$(fp "$NEW" "LAZY_FAIL_HARD=1"); f3=$(fp "$NEW" "SV2_FUT_BOUND=1")
v1=$(printf 'setoption name EvalArm value 0\nbench\nquit\n' | "$NEW" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+')
ga=$(gfp "$NEW" ""); gb=$(gfp "$G77" ""); gk=$(gfp "$NEW" "$KS")
echo "[q281] s6pgo default [$a] == g77pgo [$b] (254 / 35922538) · $KS [$k] (== 249 / 37325964) · LAZY_FAIL_HARD=1 [$k1] (== 249 / 46312304) · SV2_FUT_BOUND=1 [$f3] (== #280's 252 / 36716126) · v1 [$v1] · game12 d10 default [$ga] == g77 [$gb] · $KS [$gk] (== 9/12 2785346)"
[ "$a" = "solved 254/300 nodes 35922538" ] && [ "$b" = "$a" ] && [ "$k" = "solved 249/300 nodes 37325964" ] && [ "$k1" = "solved 249/300 nodes 46312304" ] \
	&& [ "$f3" = "solved 252/300 nodes 36716126" ] && echo "$v1" | grep -qE 'solved 250/300 nodes 3531077[5-8]' && [ -n "$ga" ] && [ "$ga" = "$gb" ] \
	|| { echo "[q281] ☠️ a gate failed — no games — QUEUE 281 ABORTED"; exit 1; }
echo "[q281] liveness for the next queues (recorded, not gated): ROOT_M0_CUT=1 [$(fp "$NEW" "ROOT_M0_CUT=1")] · game12 [$(gfp "$NEW" "ROOT_M0_CUT=1")] · PS_TT_FLAG_FIX=1 [$(fp "$NEW" "PS_TT_FLAG_FIX=1")] · game12 [$(gfp "$NEW" "PS_TT_FLAG_FIX=1")] · RFP_RETURN_BLEND=67 [$(fp "$NEW" "RFP_RETURN_BLEND=67")] · game12 [$(gfp "$NEW" "RFP_RETURN_BLEND=67")] (default [$a] · [$ga])"
echo "[q281] ✅ every gate equal"
rm -f /tmp/q281_pre.engine.log
timeout 180 "$FC" -engine "cmd=$NEW" name=lfh option.OwnBook=false "option.Knobs=$KS" st=0.05 \
	-engine "cmd=$NEW" name=ctl option.OwnBook=false st=0.05 \
	-openings file=$O/openings_uho.epd format=epd order=random -rounds 1 -games 2 -concurrency 1 \
	-log file=/tmp/q281_pre.engine.log level=info engine=true > /tmp/q281_pre.log 2>&1
WN=$(grep -c 'have option' /tmp/q281_pre.log)
echo "[q281] preflight: $(grep -E '^Games:' /tmp/q281_pre.log | tail -1) · warnings $WN · echoes $KS $(grep -cF "$KS" /tmp/q281_pre.engine.log)"
[ "$WN" = 0 ] || { echo "[q281] ☠️ preflight warnings — no games — QUEUE 281 DONE"; exit 1; }
T_BUILD=$(( $(date +%s) - T0 ))

# SF1.1's node count: an override file, else the #279 rung nearest 50% if it lies in 35-65%
N11=$(cat "$NFILE" 2>/dev/null | tr -dc '0-9')
[ -z "$N11" ] && N11=$(grep -E '^\[q279\] +SF1\.1 +[0-9]+ nodes:' "$Q279" | awk '{ gsub(/%/, "", $7); d = $7 - 50; if (d < 0) d = -d; if (best == "" || d < bd) { best = $3; bd = d; sc = $7 } } END { if (best != "" && sc >= 35 && sc <= 65) print best }')
# minutes #279 took for 100 games at N11 (for the time accounting)
RUNG_MIN=$(awk -v n="$N11" '$0 ~ "rung SF1.1 nodes=" n " " { split($2, a, ":"); s = a[1] * 60 + a[2] } $0 ~ "[]] [0-9:]+ nodes=" n ":" { split($2, b, ":"); e = b[1] * 60 + b[2] } END { if (s != "" && e != "") print e - s }' "$Q279")
S11_LEG_MIN=$(awk -v r="${RUNG_MIN:-10}" 'BEGIN { printf "%.0f", r * 2.5 }')
echo "[q281] SF1.1 judge: ${N11:-none} nodes (100 g took ${RUNG_MIN:-?} min in #279 ⇒ ~$S11_LEG_MIN min per 250-game leg)"

# ── the outside judges' legs (played while timed work is held, and after a self-play pass) ──────────────────────────────
COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
# judge:seed:arm, alternating which arm plays first and the judges, so a partial set stays paired and balanced
LEGS="g18:1:ctl g18:1:lfh s11:1:lfh s11:1:ctl g18:2:lfh g18:2:ctl s11:2:ctl s11:2:lfh g18:3:ctl g18:3:lfh s11:3:lfh s11:3:ctl"
T_EX=0; T_EXP=0; N_EXP=0; FAILED=" "
leg_seed() { case $1 in g18) echo $((281110 + $2)) ;; s11) echo $((281210 + $2)) ;; esac; }
leg_done() { grep -qE '^Games: 250,' "$O/${1}_q281_s$(leg_seed $1 $2)_$3.log" 2>/dev/null; }
next_leg() {  # prints "judge index arm" of the first leg not yet complete and not given up on, or nothing
	local L j s a
	for L in $LEGS; do
		IFS=: read -r j s a <<< "$L"
		[ "$j" = s11 ] && [ -z "$N11" ] && continue
		case "$FAILED" in *" $L "*) continue ;; esac
		leg_done $j $s $a || { echo "$j $s $a"; return; }
	done
}
run_leg() {  # judge, index, arm: plays the leg; a leg that ends incomplete twice is given up on (logged)
	ext_chunk $1 $(leg_seed $1 $2) $3
	leg_done $1 $2 $3 && return 0
	case "$TRIED" in
		*" $1:$2:$3 "*) FAILED="$FAILED$1:$2:$3 "; echo "[q281] ☠️ leg $1 $2 $3 incomplete twice — given up" ;;
		*) TRIED="$TRIED$1:$2:$3 " ;;
	esac
}
TRIED=" "
ext_chunk() {  # judge (g18 | s11), seed, arm
	local j=$1 seed=$2 a=$3 k f=$O/${1}_q281_s$2_$3.log opp t
	stopped "$j $seed $a"
	k=$(kn $a); [ -n "$k" ] && k="option.Knobs=$k"
	case $j in
		g18) opp=(-engine "cmd=$SF18" name=sf18 option.Threads=1 option.Hash=16 nodes=1100) ;;
		s11) opp=(-engine "cmd=$SF1" name=sf1.1 option.Threads=1 option.Ponder=false option.OwnBook=false option.Hash=16 nodes=$N11) ;;
	esac
	t=$(date +%s)
	nice -n 5 "$FC" -engine "cmd=$NEW" name=ours option.OwnBook=false $k nodes=250000 "${opp[@]}" \
		-each tc=inf $COMMON -srand $seed -rounds 125 -games 2 -pgnout file=$O/${j}_q281_s${seed}_$a.pgn notation=san nodes=true append=false > "$f" 2>&1
	LEG_S=$(( $(date +%s) - t ))
	echo "[q281] $(date '+%H:%M') $j seed $seed $a: $(grep -E '^Games:' "$f" | tail -1) · warnings $(grep -c 'have option' "$f") · disconnects/stalls $(grep -ciE 'disconnect|stall|not responsive' "$f") · ${LEG_S}s"
}

# ── 2 self-play first; held time plays outside-judge legs ────────────────────────────────────────────────────────────────
if [ "$(date +%H)" -ge 12 ]; then
	GS=$(date -d 'today 20:50' +%s); GE=$(date -d "$(date -d tomorrow +%F) 00:15" +%s); CUT=$(date -d "$(date -d tomorrow +%F) 10:40" +%s)
else
	GS=$(date -d 'today 20:50' +%s); GE=$(date -d "$(date -d tomorrow +%F) 00:15" +%s); CUT=$(date -d 'today 20:50' +%s)
fi
echo "[q281] $(date '+%H:%M') self-play lfh vs gen 7.7 (no timed work $(date -d @$GS '+%a %H:%M') to $(date -d @$GE '+%a %H:%M') unless q281.go; q281.pause or gaming.pause holds; cutoff $(date -d @$CUT '+%a %H:%M'))"
SEQ=0; T_SP=0; T_WAIT=0
sp_chunk() {  # arm; returns 1 when the arm already has 2,000
	local a=$1 n left pairs seed f t
	n=$(sp_count $a); left=$(( (2000 - n + 1) / 2 )); [ $left -le 0 ] && return 1
	pairs=$(( left < 150 ? left : 150 )); SEQ=$((SEQ + 1)); seed=$((2810 + SEQ)); f=$GW/${a}_s$seed
	echo "[q281] $(date '+%H:%M') self-play $a: $n finished, chunk seed $seed, $pairs pairs"
	guard; t=$(date +%s)
	nice -n 5 "$FC" -engine "cmd=$NEW" name=$a option.OwnBook=false "option.Knobs=$(kn $a)" tc=10+0.1 timemargin=250 \
		-engine "cmd=$NEW" name=g77 option.OwnBook=false tc=10+0.1 timemargin=250 \
		$COMMON -srand $seed -rounds $pairs -games 2 -pgnout file=$f.pgn notation=san nodes=true append=false > $f.log 2>&1
	kill $GUARD 2>/dev/null; T_SP=$(( T_SP + $(date +%s) - t ))
	echo "[q281] $(date '+%H:%M') $a s$seed: $(grep -E '^Games:' $f.log | tail -1) · warnings $(grep -c 'have option' $f.log) · disconnects/stalls $(grep -ciE 'disconnect|stall|not responsive' $f.log) · time forfeits $(grep -c 'loses on time' $f.pgn) · unfinished $(grep -c '^\[Result "\*"' $f.pgn)"
	return 0
}
sp_loop() { while timed_ok; do stopped "a self-play chunk"; sp_chunk lfh || break; done; }
held() {  # waits out a hold on timed work, playing outside-judge legs meanwhile
	local t L
	t=$(date +%s)
	until timed_ok || [ "$(date +%s)" -ge "$CUT" ]; do
		stopped "the self-play"
		L=$(next_leg)
		if [ -n "$L" ]; then
			echo "[q281] $(date '+%H:%M') timed work held ⇒ an outside-judge leg on the free box: $L"
			run_leg $L; T_EXP=$(( T_EXP + LEG_S )); N_EXP=$((N_EXP + 1))
			timed_ok && echo "[q281] $(date '+%H:%M') the hold lifted during that leg; self-play resumes now"
		else
			sleep 30
		fi
	done
	T_WAIT=$(( T_WAIT + $(date +%s) - t ))
}
while [ "$(sp_count lfh)" -lt 2000 ] && [ "$(date +%s)" -lt "$CUT" ]; do
	held
	sp_loop
done
SPN=$(sp_count lfh)
echo "[q281] pooled self-play lfh vs gen 7.7 ($SPN finished):"
timeout 900 bash "$R" pyrun selfplay/fastchess_report.py "$GW/lfh"_s*.pgn --p1 lfh 2>&1 | sed -n '/vs the field/,$p' > "$W/pooled.txt"
show < "$W/pooled.txt"
LO=$(grep -oE '\(95% [+-]?[0-9.]+' "$W/pooled.txt" | head -1 | sed 's/(95% //')
HI=$(grep -oE '… [+-]?[0-9.]+\)' "$W/pooled.txt" | head -1 | sed 's/… //; s/)//')
if [ "$SPN" -lt 2000 ]; then VERDICT="INCOMPLETE ($SPN of 2,000 by the cutoff)"
elif awk -v l="$LO" 'BEGIN { exit !(l != "" && l + 0 > 0) }'; then VERDICT="PASS"
elif awk -v h="$HI" 'BEGIN { exit !(h != "" && h + 0 < 0) }'; then VERDICT="FAIL"
else VERDICT="NEUTRAL"; fi
echo "[q281] REGISTERED self-play read: $VERDICT (95% $LO … $HI; pass = lower > 0, neutral = within ±2σ → parked, fail = upper < 0)"

# ── 3 the remaining outside-judge legs, only when self-play passes ───────────────────────────────────────────────────────
if [ "$VERDICT" = "PASS" ]; then
	echo "[q281] $(date '+%H:%M') self-play passed ⇒ outside judges: SF18 3 seeds, SF1.1 ${N11:-(not calibrated: skipped)} 3 seeds, ctl and lfh ($N_EXP legs already played while timed work was held)"
	L=$(next_leg)
	while [ -n "$L" ]; do run_leg $L; T_EX=$(( T_EX + LEG_S )); L=$(next_leg); done
else
	echo "[q281] self-play $VERDICT ⇒ no further outside-judge legs (the candidate cannot ship without a self-play pass); $N_EXP legs played while timed work was held are information only"
fi

# ── reads ────────────────────────────────────────────────────────────────────────────────────────────────────────────────
row() {  # seed, arm, log: complete legs only
	grep -E '^Games:' "$3" 2>/dev/null | tail -1 | awk -v s=$1 -v a=$2 '{ gsub(/,/, ""); for (i = 1; i <= NF; i++) { if ($i == "Games:") n = $(i+1); if ($i == "Wins:") w = $(i+1); if ($i == "Losses:") l = $(i+1); if ($i == "Draws:") d = $(i+1) } if (n == 250) print s, a, n, w, l, d }'
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
for j in g18 s11; do
	for f in $O/${j}_q281_s*_*.log; do [ -e "$f" ] || continue; b=$(basename $f .log); b=${b#${j}_q281_s}; row ${b%%_*} ${b#*_} $f; done > /tmp/q281_$j.txt
done
cat /tmp/q281_g18.txt /tmp/q281_s11.txt > /tmp/q281_ext.txt
if [ -s /tmp/q281_ext.txt ]; then
	if [ "$VERDICT" = "PASS" ]; then TAG="REGISTERED outside read"; else TAG="outside read, INFORMATION ONLY (self-play $VERDICT)"; fi
	echo "[q281] $(date) per leg (seed arm games W L D):"; { sed 's/^/SF18 /' /tmp/q281_g18.txt; sed "s/^/SF1.1@$N11 /" /tmp/q281_s11.txt; } | show
	echo "[q281] $TAG (pooled SF18 + SF1.1: lfh - ctl z > -1), and each judge alone:"
	{ cmpr lfh ctl /tmp/q281_ext.txt "POOLED"; cmpr lfh ctl /tmp/q281_g18.txt "SF18  "; cmpr lfh ctl /tmp/q281_s11.txt "SF1.1 "; } | show
fi

# ── the time accounting (owner: "observe if it truly saves time") ────────────────────────────────────────────────────────
TOT=$(( $(date +%s) - T0 ))
OLD_EX=$(( (12 * SF18_LEG_MIN + 6 * S11_LEG_MIN) * 60 ))
echo "[q281] TIME: build + gates $(hrs $T_BUILD) · self-play $(hrs $T_SP) · timed work held $(hrs $T_WAIT), of which outside-judge legs $(hrs $T_EXP) ($N_EXP legs) · outside judges after self-play $(hrs $T_EX) · total $(hrs $TOT)"
echo "[q281] TIME: the old form's outside judges (6 SF18 seeds + 3 SF1.1 seeds per arm, after self-play) ≈ $(hrs $OLD_EX) at these leg times ⇒ box time this queue spent on judges outside held time $(hrs $T_EX), saved ≈ $(hrs $(( OLD_EX - T_EX )))"
echo "[q281] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | show
echo "[q281] $(date) QUEUE 281 DONE"
