#!/bin/bash
# 2026-11-19 ~19:40, queue #282: S2-26 — reverse futility returns what it proved, not the raw eval: RFP_RETURN_BLEND=67, SF18's
# beta + (eval − beta) / 3. Chained after #281 (S2-9). The knob was built in October and never played; this tests its FORM once
# at SF's value (owner 11-19: settle forms now, tune constants near the end).
# Why: RFP returns the raw static eval and the parent stores it as a full-depth lower bound, so a later probe cuts for any beta
# up to the eval (reverse futility with margin 0, register S2-26).
# The BASE follows #281, because S2-9 and S2-26 change the same return (search changes are antagonistic, not additive): with
# LAZY_FAIL_HARD=2 a lazy RFP return gives "estimate − M" and the blend touches only the full-eval returns; on gen 7.7 the blend
# also pulls the lazy returns (38% / 33% of RFP returns, #277) toward beta, so it would claim part of S2-9's job.
#  - #281 self-play PASS and its pooled outside read z > −1 (its registered criteria) ⇒ BASE = LAZY_FAIL_HARD=2: both arms
#    carry it and the read is S2-26 on top of S2-9. The owner's ship yes is still pending; the log says so.
#  - otherwise ⇒ BASE = gen 7.7 (no knobs).
# Fourth run of the STREAMLINED JUDGING trial, in #281's form: self-play first; gaming.pause or q282.pause holds timed work; any
# hold (gaming, or the 20:50 to 00:15 block) mid-self-play plays the outside judges' node-limited legs, which count only after a
# self-play pass (information only otherwise).
#  1 Wait for #281; production must be g77pgo; nothing else running. NO REBUILD: #281's PGO build chess_engine_uci_s6pgo (the
#    sources are unchanged since 254d01f). GATES: default 254 / 35,922,538 == g77pgo; RFP_RETURN_BLEND=67 == 256 / 36,044,508 (the
#    plain build's value and #281's liveness line); v1 250 / 35,310,775-778; game12 d10 default == g77pgo (7/12 2,412,001). With
#    the S2-9 base also: LAZY_FAIL_HARD=2 == 249 / 37,325,964, and base + blend must differ from it (recorded).
#  2 Self-play blend vs the base, 10+0.1, 150-pair chunks, seeds 2821+, to 2,000 games. No timed work 20:50 to 00:15 unless
#    q282.go; cutoff 10:40 the morning after the next evening block.
#  3 IF self-play passes: the remaining outside-judge legs, SF18 (ours 250k, SF18 1,100 nodes) seeds 282111-282113 and SF1.1
#    (ours 250k, SF1.1 at the #279 rung nearest 50%, 320k) seeds 282211-282213, ctl (= the base) and blend, 250 g each.
# REGISTERED before any game:
#  - self-play lower 95% > 0 = pass; within ±2σ = NEUTRAL (parked: trigger the constants pass near the end, or RFP reshaped);
#    upper 95% < 0 = fails. Prediction on gen 7.7 −5 … +20 (SF adopted it); on the S2-9 base −5 … +12 (fewer returns to touch).
#  - outside judges, pooled SF18 + SF1.1: blend − ctl z > −1 (not clearly worse). Each judge is also reported alone.
#  - Bench: the draft predicted +1 … +6% nodes; measured 256 / 36,044,508 = +0.34% (a miss: smaller stored bounds cost little).
#    On the S2-9 base: within −2 … +3% of 37,325,964.
# Stop: touch /mnt/e/chess_data/q282.stop (between steps). Never edit this file or the binaries while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
SF18="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/stockfish_18_linux/stockfish-ubuntu-x86-64-avx2"
SF1=/mnt/e/chess_data/engines/sf1.1/stockfish-1.1
O=/mnt/e/chess_data/fastchess
W=/mnt/e/chess_data/q282
GW=/mnt/e/chess_data/r282
G=/mnt/e/chess_data/sv2_speed/game12.txt
PROD="$ND/build_uci/chess_engine_uci"
G77="$ND/build_uci/chess_engine_uci_g77pgo"
NEW="$ND/build_uci/chess_engine_uci_s6pgo"
KS="RFP_RETURN_BLEND=67"
S29="LAZY_FAIL_HARD=2"
Q281=/mnt/e/chess_data/q281.log
LOG=/mnt/e/chess_data/q282.log
Q279=/mnt/e/chess_data/q279.log
NFILE=/mnt/e/chess_data/sf11_judge_nodes.txt
STOP=/mnt/e/chess_data/q282.stop
PAUSE=/mnt/e/chess_data/q282.pause
GPAUSE=/mnt/e/chess_data/gaming.pause
GO=/mnt/e/chess_data/q282.go
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
SF18_LEG_MIN=13   # an SF18 leg (250 g) in #274: 12.5-13.5 min
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP" "$GO"; mkdir -p "$W" "$GW"   # the pause files are the owner's: never removed here
# A resume (the 11-20 04:14 restart by Windows Update) keeps the complete chunks: q282.resume holds the self-play seconds they took.
RESUME=/mnt/e/chess_data/q282.resume
if [ -e "$RESUME" ]; then SP_PRIOR=$(tr -dc '0-9' < "$RESUME"); else SP_PRIOR=0; rm -f "$GW"/blend_s*.pgn "$GW"/blend_s*.log $O/g18_q282_s* $O/s11_q282_s*; fi
BASE=""
show() { sed "s/^/[q282]   /"; }
stopped() { [ -e "$STOP" ] && { echo "[q282] $(date) stop file before $1 — QUEUE 282 DONE"; exit 0; }; }
kn() { case $1 in ctl) echo "$BASE" ;; blend) echo "${BASE:+$BASE,}$KS" ;; esac; }
fp() { local k=""; [ -n "$2" ] && k="setoption name Knobs value $2"; printf '%s\nbench\nquit\n' "$k" | "$1" 2>&1 | grep -oE 'solved [0-9]+/300 nodes [0-9]+|mismatch.*' | head -1; }
gfp() { local k=""; [ -n "$2" ] && k="setoption name Knobs value $2"; printf "%s\nbench $G 10\nquit\n" "$k" | "$1" 2>&1 | grep -oE 'solved [0-9]+/[0-9]+ nodes [0-9]+|mismatch.*' | head -1; }
busy() { pgrep -f "[f]astchess-linux-x86-64/fastchess" >/dev/null || pgrep -f "[s]ts_test|[t]actical_test|[v]algrind|[_]ks_footprint|[b]uild_uci.sh" >/dev/null; }
timed_ok() { local t; t=$(date +%s); [ ! -e "$PAUSE" ] && [ ! -e "$GPAUSE" ] && [ "$t" -lt "$CUT" ] && ! { [ "$t" -ge "$GS" ] && [ "$t" -lt "$GE" ] && [ ! -e "$GO" ]; }; }
guard() { ( until ! timed_ok; do sleep 15; done; pkill -INT -f "[f]astchess-linux-x86-64/fastchess" && echo "[q282] $(date '+%H:%M') guard: games interrupted" ) & GUARD=$!; }
finished() { local c; c=$(grep -cE '^\[Result "(1-0|0-1|1/2-1/2)"\]' "$1" 2>/dev/null); echo "${c:-0}"; }
sp_count() { local n=0 f; for f in "$GW/$1"_s*.pgn; do [ -e "$f" ] && n=$((n + $(finished "$f"))); done; echo $n; }
hrs() { awk -v s=$1 'BEGIN { printf "%.2f h", s / 3600 }'; }
echo "[q282] $(date '+%Y-%m-%d %H:%M:%S') queued; waiting for #281"
until grep -qE 'QUEUE 281 (DONE|ABORTED)' "$Q281" 2>/dev/null || ! pgrep -f "[_]queue_281.sh" >/dev/null; do stopped "the start"; sleep 30; done
sleep 20
cmp -s "$PROD" "$G77" || { echo "[q282] ☠️ production is not g77pgo — QUEUE 282 ABORTED"; exit 1; }
[ -x "$NEW" ] || { echo "[q282] ☠️ $NEW missing — QUEUE 282 ABORTED"; exit 1; }
busy && { echo "[q282] ☠️ another job is running — QUEUE 282 ABORTED"; exit 1; }
START=$(date '+%Y-%m-%d %H:%M:%S'); T0=$(date +%s)
echo "[q282] $START start"

# ── the base, from #281's registered reads ──────────────────────────────────────────────────────────────────────────────
PZ=$(grep -E 'POOLED +lfh' "$Q281" | grep -oE 'z [+-]?[0-9.]+' | tail -1 | sed 's/z //')
if grep -qE 'REGISTERED self-play read: PASS' "$Q281" && grep -qE 'REGISTERED outside read' "$Q281" \
	&& awk -v z="$PZ" 'BEGIN { exit !(z != "" && z + 0 > -1) }'; then
	BASE="$S29"
	echo "[q282] #281 passed (self-play PASS, pooled outside z $PZ > −1) ⇒ BASE = $BASE in both arms: S2-26 on top of S2-9. The owner's ship yes is still pending; if S2-9 is not shipped, this read is on that base, not on production"
else
	echo "[q282] #281 did not pass ($(grep -oE 'REGISTERED self-play read: [A-Z]+' "$Q281" | tail -1 | sed 's/REGISTERED //'); pooled outside z ${PZ:-none}) ⇒ BASE = gen 7.7 (no knobs)"
fi
CN=${BASE:+base}; CN=${CN:-g77}
CK=""; [ -n "$BASE" ] && CK="option.Knobs=$BASE"

# ── 1 gates (no rebuild) ─────────────────────────────────────────────────────────────────────────────────────────────────
a=$(fp "$NEW" ""); b=$(fp "$G77" ""); k=$(fp "$NEW" "$KS")
v1=$(printf 'setoption name EvalArm value 0\nbench\nquit\n' | "$NEW" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+')
ga=$(gfp "$NEW" ""); gb=$(gfp "$G77" ""); gk=$(gfp "$NEW" "$KS")
echo "[q282] s6pgo default [$a] == g77pgo [$b] (254 / 35922538) · $KS [$k] (== 256 / 36044508) · v1 [$v1] · game12 d10 default [$ga] == g77 [$gb] · $KS [$gk]"
[ "$a" = "solved 254/300 nodes 35922538" ] && [ "$b" = "$a" ] && [ "$k" = "solved 256/300 nodes 36044508" ] \
	&& echo "$v1" | grep -qE 'solved 250/300 nodes 3531077[5-8]' && [ -n "$ga" ] && [ "$ga" = "$gb" ] \
	|| { echo "[q282] ☠️ a gate failed — no games — QUEUE 282 ABORTED"; exit 1; }
if [ -n "$BASE" ]; then
	kb=$(fp "$NEW" "$BASE"); kk=$(fp "$NEW" "$(kn blend)"); gkb=$(gfp "$NEW" "$BASE"); gkk=$(gfp "$NEW" "$(kn blend)")
	echo "[q282] base $BASE [$kb] (== 249 / 37325964) · base + blend [$kk] (must differ; prediction −2 … +3% of 37,325,964) · game12 base [$gkb] · base + blend [$gkk]"
	[ "$kb" = "solved 249/300 nodes 37325964" ] && [ -n "$kk" ] && [ "$kk" != "$kb" ] \
		|| { echo "[q282] ☠️ a base gate failed — no games — QUEUE 282 ABORTED"; exit 1; }
fi
echo "[q282] ✅ every gate equal"
rm -f /tmp/q282_pre.engine.log
timeout 180 "$FC" -engine "cmd=$NEW" name=blend option.OwnBook=false "option.Knobs=$(kn blend)" st=0.05 \
	-engine "cmd=$NEW" name=$CN option.OwnBook=false $CK st=0.05 \
	-openings file=$O/openings_uho.epd format=epd order=random -rounds 1 -games 2 -concurrency 1 \
	-log file=/tmp/q282_pre.engine.log level=info engine=true > /tmp/q282_pre.log 2>&1
WN=$(grep -c 'have option' /tmp/q282_pre.log)
echo "[q282] preflight: $(grep -E '^Games:' /tmp/q282_pre.log | tail -1) · warnings $WN · echoes $KS $(grep -cF "$KS" /tmp/q282_pre.engine.log)${BASE:+ · echoes $BASE $(grep -cF "$BASE" /tmp/q282_pre.engine.log)}"
[ "$WN" = 0 ] || { echo "[q282] ☠️ preflight warnings — no games — QUEUE 282 DONE"; exit 1; }
T_BUILD=$(( $(date +%s) - T0 ))

# SF1.1's node count: an override file, else the #279 rung nearest 50% if it lies in 35-65%
N11=$(cat "$NFILE" 2>/dev/null | tr -dc '0-9')
[ -z "$N11" ] && N11=$(grep -E '^\[q279\] +SF1\.1 +[0-9]+ nodes:' "$Q279" | awk '{ gsub(/%/, "", $7); d = $7 - 50; if (d < 0) d = -d; if (best == "" || d < bd) { best = $3; bd = d; sc = $7 } } END { if (best != "" && sc >= 35 && sc <= 65) print best }')
RUNG_MIN=$(awk -v n="$N11" '$0 ~ "rung SF1.1 nodes=" n " " { split($2, a, ":"); s = a[1] * 60 + a[2] } $0 ~ "[]] [0-9:]+ nodes=" n ":" { split($2, b, ":"); e = b[1] * 60 + b[2] } END { if (s != "" && e != "") print e - s }' "$Q279")
S11_LEG_MIN=$(awk -v r="${RUNG_MIN:-10}" 'BEGIN { printf "%.0f", r * 2.5 }')
echo "[q282] SF1.1 judge: ${N11:-none} nodes (100 g took ${RUNG_MIN:-?} min in #279 ⇒ ~$S11_LEG_MIN min per 250-game leg)"

# ── the outside judges' legs (played while timed work is held, and after a self-play pass) ──────────────────────────────
COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
LEGS="g18:1:ctl g18:1:blend s11:1:blend s11:1:ctl g18:2:blend g18:2:ctl s11:2:ctl s11:2:blend g18:3:ctl g18:3:blend s11:3:blend s11:3:ctl"
T_EX=0; T_EXP=0; N_EXP=0; FAILED=" "; TRIED=" "
leg_seed() { case $1 in g18) echo $((282110 + $2)) ;; s11) echo $((282210 + $2)) ;; esac; }
leg_done() { grep -qE '^Games: 250,' "$O/${1}_q282_s$(leg_seed $1 $2)_$3.log" 2>/dev/null; }
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
		*" $1:$2:$3 "*) FAILED="$FAILED$1:$2:$3 "; echo "[q282] ☠️ leg $1 $2 $3 incomplete twice — given up" ;;
		*) TRIED="$TRIED$1:$2:$3 " ;;
	esac
}
ext_chunk() {  # judge (g18 | s11), seed, arm
	local j=$1 seed=$2 a=$3 k f=$O/${1}_q282_s$2_$3.log opp t
	stopped "$j $seed $a"
	k=$(kn $a); [ -n "$k" ] && k="option.Knobs=$k"
	case $j in
		g18) opp=(-engine "cmd=$SF18" name=sf18 option.Threads=1 option.Hash=16 nodes=1100) ;;
		s11) opp=(-engine "cmd=$SF1" name=sf1.1 option.Threads=1 option.Ponder=false option.OwnBook=false option.Hash=16 nodes=$N11) ;;
	esac
	t=$(date +%s)
	nice -n 5 "$FC" -engine "cmd=$NEW" name=ours option.OwnBook=false $k nodes=250000 "${opp[@]}" \
		-each tc=inf $COMMON -srand $seed -rounds 125 -games 2 -pgnout file=$O/${j}_q282_s${seed}_$a.pgn notation=san nodes=true append=false > "$f" 2>&1
	LEG_S=$(( $(date +%s) - t ))
	echo "[q282] $(date '+%H:%M') $j seed $seed $a: $(grep -E '^Games:' "$f" | tail -1) · warnings $(grep -c 'have option' "$f") · disconnects/stalls $(grep -ciE 'disconnect|stall|not responsive' "$f") · ${LEG_S}s"
}

# ── 2 self-play first; held time plays outside-judge legs ────────────────────────────────────────────────────────────────
# the next evening block (or the one in progress before 00:15), and the cutoff 10:40 the morning after it
NOW=$(date +%s); GS=$(date -d 'today 20:50' +%s)
[ "$NOW" -lt "$(date -d 'today 00:15' +%s)" ] && GS=$(date -d 'yesterday 20:50' +%s)
GE=$(( GS + 12300 )); CUT=$(( GE + 37500 ))
echo "[q282] $(date '+%H:%M') self-play blend vs ${BASE:-gen 7.7} (no timed work $(date -d @$GS '+%a %H:%M') to $(date -d @$GE '+%a %H:%M') unless q282.go; q282.pause or gaming.pause holds; cutoff $(date -d @$CUT '+%a %H:%M'))"
SEQ=$(ls "$GW"/blend_s*.pgn 2>/dev/null | wc -l); T_SP=${SP_PRIOR:-0}; T_WAIT=0
[ -e "$RESUME" ] && echo "[q282] RESUMED after the 04:14 restart by Windows Update (the 26H2 feature update, installing from 03:50): $(sp_count blend) games kept from $SEQ complete chunks (01:08-03:40, part 1 log q282_part1.log); chunk 2824 (193 games, 03:40-04:14, overlapping the install) set aside unread in /mnt/e/chess_data/r282_aside_winupdate and replayed whole; the next chunk seed is $((2821 + SEQ))"
sp_chunk() {  # arm; returns 1 when the arm already has 2,000
	local a=$1 n left pairs seed f t
	n=$(sp_count $a); left=$(( (2000 - n + 1) / 2 )); [ $left -le 0 ] && return 1
	pairs=$(( left < 150 ? left : 150 )); SEQ=$((SEQ + 1)); seed=$((2820 + SEQ)); f=$GW/${a}_s$seed
	echo "[q282] $(date '+%H:%M') self-play $a: $n finished, chunk seed $seed, $pairs pairs"
	guard; t=$(date +%s)
	nice -n 5 "$FC" -engine "cmd=$NEW" name=$a option.OwnBook=false "option.Knobs=$(kn $a)" tc=10+0.1 timemargin=250 \
		-engine "cmd=$NEW" name=$CN option.OwnBook=false $CK tc=10+0.1 timemargin=250 \
		$COMMON -srand $seed -rounds $pairs -games 2 -pgnout file=$f.pgn notation=san nodes=true append=false > $f.log 2>&1
	kill $GUARD 2>/dev/null; T_SP=$(( T_SP + $(date +%s) - t ))
	echo "[q282] $(date '+%H:%M') $a s$seed: $(grep -E '^Games:' $f.log | tail -1) · warnings $(grep -c 'have option' $f.log) · disconnects/stalls $(grep -ciE 'disconnect|stall|not responsive' $f.log) · time forfeits $(grep -c 'loses on time' $f.pgn) · unfinished $(grep -c '^\[Result "\*"' $f.pgn)"
	return 0
}
sp_loop() { while timed_ok; do stopped "a self-play chunk"; sp_chunk blend || break; done; }
held() {  # waits out a hold on timed work, playing outside-judge legs meanwhile
	local t L
	t=$(date +%s)
	until timed_ok || [ "$(date +%s)" -ge "$CUT" ]; do
		stopped "the self-play"
		L=$(next_leg)
		if [ -n "$L" ]; then
			echo "[q282] $(date '+%H:%M') timed work held ⇒ an outside-judge leg on the free box: $L"
			run_leg $L; T_EXP=$(( T_EXP + LEG_S )); N_EXP=$((N_EXP + 1))
			timed_ok && echo "[q282] $(date '+%H:%M') the hold lifted during that leg; self-play resumes now"
		else
			sleep 30
		fi
	done
	T_WAIT=$(( T_WAIT + $(date +%s) - t ))
}
while [ "$(sp_count blend)" -lt 2000 ] && [ "$(date +%s)" -lt "$CUT" ]; do
	held
	sp_loop
done
SPN=$(sp_count blend)
echo "[q282] pooled self-play blend vs ${BASE:-gen 7.7} ($SPN finished):"
timeout 900 bash "$R" pyrun selfplay/fastchess_report.py "$GW/blend"_s*.pgn --p1 blend 2>&1 | sed -n '/vs the field/,$p' > "$W/pooled.txt"
show < "$W/pooled.txt"
LO=$(grep -oE '\(95% [+-]?[0-9.]+' "$W/pooled.txt" | head -1 | sed 's/(95% //')
HI=$(grep -oE '… [+-]?[0-9.]+\)' "$W/pooled.txt" | head -1 | sed 's/… //; s/)//')
if [ "$SPN" -lt 2000 ]; then VERDICT="INCOMPLETE ($SPN of 2,000 by the cutoff)"
elif awk -v l="$LO" 'BEGIN { exit !(l != "" && l + 0 > 0) }'; then VERDICT="PASS"
elif awk -v h="$HI" 'BEGIN { exit !(h != "" && h + 0 < 0) }'; then VERDICT="FAIL"
else VERDICT="NEUTRAL"; fi
echo "[q282] REGISTERED self-play read: $VERDICT (95% $LO … $HI; pass = lower > 0, neutral = within ±2σ → parked, fail = upper < 0)"

# ── 3 the remaining outside-judge legs, only when self-play passes ───────────────────────────────────────────────────────
if [ "$VERDICT" = "PASS" ]; then
	echo "[q282] $(date '+%H:%M') self-play passed ⇒ outside judges: SF18 3 seeds, SF1.1 ${N11:-(not calibrated: skipped)} 3 seeds, ctl and blend ($N_EXP legs already played while timed work was held)"
	L=$(next_leg)
	while [ -n "$L" ]; do run_leg $L; T_EX=$(( T_EX + LEG_S )); L=$(next_leg); done
else
	echo "[q282] self-play $VERDICT ⇒ no further outside-judge legs (the candidate cannot ship without a self-play pass); $N_EXP legs played while timed work was held are information only"
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
	for f in $O/${j}_q282_s*_*.log; do [ -e "$f" ] || continue; b=$(basename $f .log); b=${b#${j}_q282_s}; row ${b%%_*} ${b#*_} $f; done > /tmp/q282_$j.txt
done
cat /tmp/q282_g18.txt /tmp/q282_s11.txt > /tmp/q282_ext.txt
if [ -s /tmp/q282_ext.txt ]; then
	if [ "$VERDICT" = "PASS" ]; then TAG="REGISTERED outside read"; else TAG="outside read, INFORMATION ONLY (self-play $VERDICT)"; fi
	echo "[q282] $(date) per leg (seed arm games W L D; ctl = ${BASE:-gen 7.7}):"; { sed 's/^/SF18 /' /tmp/q282_g18.txt; sed "s/^/SF1.1@$N11 /" /tmp/q282_s11.txt; } | show
	echo "[q282] $TAG (pooled SF18 + SF1.1: blend - ctl z > -1), and each judge alone:"
	{ cmpr blend ctl /tmp/q282_ext.txt "POOLED"; cmpr blend ctl /tmp/q282_g18.txt "SF18  "; cmpr blend ctl /tmp/q282_s11.txt "SF1.1 "; } | show
fi

# ── the time accounting (owner: "observe if it truly saves time") ────────────────────────────────────────────────────────
TOT=$(( $(date +%s) - T0 ))
OLD_EX=$(( (12 * SF18_LEG_MIN + 6 * S11_LEG_MIN) * 60 ))
echo "[q282] TIME: gates $(hrs $T_BUILD) · self-play $(hrs $T_SP) · timed work held $(hrs $T_WAIT), of which outside-judge legs $(hrs $T_EXP) ($N_EXP legs) · outside judges after self-play $(hrs $T_EX) · total $(hrs $TOT)"
echo "[q282] TIME: the old form's outside judges (6 SF18 seeds + 3 SF1.1 seeds per arm, after self-play) ≈ $(hrs $OLD_EX) at these leg times ⇒ box time this queue spent on judges outside held time $(hrs $T_EX), saved ≈ $(hrs $(( OLD_EX - T_EX )))"
echo "[q282] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | show
rm -f "$RESUME"
echo "[q282] $(date) QUEUE 282 DONE"
