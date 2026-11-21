#!/bin/bash
# 2026-11-21 ~14:20, queue #289: S2-1's form 1 — the plain fix (CHECK_EXT_FORM=1: extend a discovered or double check, a
# checker nothing attacks, else today's SEE test), the rule the filter's header comment says was meant ("the opponent cannot
# win its checker"), against today's contested-check rule (gen 7.7). s7pgo. Chained after #288 (form 3).
# Why: today's SEE filter reads an unattacked checker at its full value, so it never extends a safe check; the fix extends
# 31-47% of eligible checks instead of 9-10% (#277's counts). The high-variance arm: more nodes. A FORM item: the +2σ rule.
# If it reads negative, read game depth before parking: lower depth = a node cost (park with a partner such as the check
# cap, or the C7 retune); equal depth = the selection is wrong (form 3 is the answer).
# STREAMLINED JUDGING (#281's form): self-play first; gaming.pause or q289.pause holds timed work; any hold mid-self-play
# plays the outside judges' node-limited legs, which count only after a self-play pass (information only otherwise).
# RESUME MODE: before relaunching after an interruption, copy q289.log to q289_part1.log, move any chunk that did not finish
# to a set-aside folder unread, and write the self-play seconds already played into /mnt/e/chess_data/q289.resume; complete
# chunks are kept, SEQ comes from their files, and q289.resume is removed at DONE.
#  1 Wait for #288; production must be g77pgo; nothing else running. GATES on s7pgo: default 254 / 35,922,538 == g77pgo;
#    CHECK_EXT_FORM=1 == the value the s7 build recorded (WAC and game12) and ≠ default; v1 250 / 35,310,775-778; game12
#    default == g77pgo.
#  2 Self-play cx1 vs gen 7.7, 10+0.1 UHO, 150-pair chunks, seeds 2891+, to 2,000 games.
#  3 IF self-play passes: the outside judges, SF18 (1,100 nodes) seeds 289111-3 and SF1.1 (320k) seeds 289211-3, ctl and cx1.
# REGISTERED before any game (the sweep-3 report's table, §5.3; vs today's rule):
#  - self-play −12 … +10. PASS = lower 95% > 0; NEUTRAL = within ±2σ; FAIL = upper < 0.
#  - bench WAC d10 nodes +8 … +20%; games: depth −0.06 … −0.14 ply, time per move unchanged.
#  - outside judges at equal nodes: pooled cx1 − ctl z > −1.
# Stop: touch /mnt/e/chess_data/q289.stop (between steps). Never edit this file or the binaries while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
SF18="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/stockfish_18_linux/stockfish-ubuntu-x86-64-avx2"
SF1=/mnt/e/chess_data/engines/sf1.1/stockfish-1.1
O=/mnt/e/chess_data/fastchess
W=/mnt/e/chess_data/q289
GW=/mnt/e/chess_data/r289
G=/mnt/e/chess_data/sv2_speed/game12.txt
PROD="$ND/build_uci/chess_engine_uci"
G77="$ND/build_uci/chess_engine_uci_g77pgo"
NEW="$ND/build_uci/chess_engine_uci_s7pgo"
KS="CHECK_EXT_FORM=1"
QPREV=/mnt/e/chess_data/q288.log
BLOG=/mnt/e/chess_data/build_s7.log
LOG=/mnt/e/chess_data/q289.log
Q279=/mnt/e/chess_data/q279.log
NFILE=/mnt/e/chess_data/sf11_judge_nodes.txt
STOP=/mnt/e/chess_data/q289.stop
PAUSE=/mnt/e/chess_data/q289.pause
GPAUSE=/mnt/e/chess_data/gaming.pause
GO=/mnt/e/chess_data/q289.go
RESUME=/mnt/e/chess_data/q289.resume
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
SF18_LEG_MIN=13   # an SF18 leg (250 g) in #274: 12.5-13.5 min
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP" "$GO"; mkdir -p "$W" "$GW"   # the pause files are the owner's: never removed here
if [ -e "$RESUME" ]; then SP_PRIOR=$(tr -dc '0-9' < "$RESUME"); else SP_PRIOR=""; rm -f "$GW"/cx1_s*.pgn "$GW"/cx1_s*.log; fi   # complete outside-judge legs (pre-played during holds) are kept and reused
BASE=""
show() { sed "s/^/[q289]   /"; }
stopped() { [ -e "$STOP" ] && { echo "[q289] $(date) stop file before $1 — QUEUE 289 DONE"; exit 0; }; }
kn() { case $1 in ctl) echo "$BASE" ;; cx1) echo "${BASE:+$BASE,}$KS" ;; esac; }
fp() { local k=""; [ -n "$2" ] && k="setoption name Knobs value $2"; printf '%s\nbench\nquit\n' "$k" | "$1" 2>&1 | grep -oE 'solved [0-9]+/300 nodes [0-9]+|mismatch.*' | head -1; }
gfp() { local k=""; [ -n "$2" ] && k="setoption name Knobs value $2"; printf "%s\nbench $G 10\nquit\n" "$k" | "$1" 2>&1 | grep -oE 'solved [0-9]+/[0-9]+ nodes [0-9]+|mismatch.*' | head -1; }
busy() { pgrep -f "[f]astchess-linux-x86-64/fastchess" >/dev/null || pgrep -f "[s]ts_test|[t]actical_test|[v]algrind|[_]ks_footprint|[b]uild_uci.sh" >/dev/null; }
timed_ok() { local t; t=$(date +%s); [ ! -e "$PAUSE" ] && [ ! -e "$GPAUSE" ] && [ "$t" -lt "$CUT" ] && ! { [ "$t" -ge "$GS" ] && [ "$t" -lt "$GE" ] && [ ! -e "$GO" ]; }; }
guard() { ( until ! timed_ok; do sleep 15; done; pkill -INT -f "[f]astchess-linux-x86-64/fastchess" && echo "[q289] $(date '+%H:%M') guard: games interrupted" ) & GUARD=$!; }
finished() { local c; c=$(grep -cE '^\[Result "(1-0|0-1|1/2-1/2)"\]' "$1" 2>/dev/null); echo "${c:-0}"; }
sp_count() { local n=0 f; for f in "$GW/$1"_s*.pgn; do [ -e "$f" ] && n=$((n + $(finished "$f"))); done; echo $n; }
hrs() { awk -v s=$1 'BEGIN { printf "%.2f h", s / 3600 }'; }
passed() {  # queue log, arm: its registered reads passed (self-play PASS, pooled outside z > -1)
	local z; z=$(grep -E "POOLED +$2 " "$1" 2>/dev/null | grep -oE 'z [+-]?[0-9.]+' | tail -1 | sed 's/z //')
	grep -qE 'REGISTERED self-play read: PASS' "$1" 2>/dev/null && grep -qE 'REGISTERED outside read' "$1" \
		&& awk -v z="$z" 'BEGIN { exit !(z != "" && z + 0 > -1) }'
}
verdict() { echo "$(grep -oE 'REGISTERED self-play read: [A-Z]+' "$1" 2>/dev/null | tail -1 | sed 's/REGISTERED self-play read: //'), pooled outside z $(grep -E "POOLED +$2 " "$1" 2>/dev/null | grep -oE 'z [+-]?[0-9.]+' | tail -1 | sed 's/z //')"; }
echo "[q289] $(date '+%Y-%m-%d %H:%M:%S') queued; waiting for #288, the s7 build and the 11-21 side script"
until { grep -qE 'QUEUE 288 (DONE|ABORTED)' "$QPREV" 2>/dev/null || ! pgrep -f "[_]queue_288.sh" >/dev/null; } && ! pgrep -f "[_]side_1121.sh|[_]build_s7.sh" >/dev/null; do stopped "the start"; sleep 30; done
sleep 20
cmp -s "$PROD" "$G77" || { echo "[q289] ☠️ production is not g77pgo — QUEUE 289 ABORTED"; exit 1; }
[ -x "$NEW" ] || { echo "[q289] ☠️ $NEW missing — QUEUE 289 ABORTED"; exit 1; }
grep -q 'every gate passed — s7pgo usable' "$BLOG" 2>/dev/null || { echo "[q289] ☠️ the s7 build did not pass its gates — QUEUE 289 ABORTED"; exit 1; }
EXP_W=$(grep -E '^\[s7\] +CHECK_EXT_FORM=1:' "$BLOG" | grep -oE 'WAC \[solved [0-9]+/300 nodes [0-9]+' | sed 's/WAC \[//')
EXP_G=$(grep -E '^\[s7\] +CHECK_EXT_FORM=1:' "$BLOG" | grep -oE 'game12 \[solved [0-9]+/[0-9]+ nodes [0-9]+' | sed 's/game12 \[//')
[ -n "$EXP_W" ] && [ -n "$EXP_G" ] || { echo "[q289] ☠️ no recorded s7 liveness value for $KS — QUEUE 289 ABORTED"; exit 1; }
busy && { echo "[q289] ☠️ another job is running — QUEUE 289 ABORTED"; exit 1; }
START=$(date '+%Y-%m-%d %H:%M:%S'); T0=$(date +%s)
echo "[q289] $START start"

# ── the base: gen 7.7 ────────────────────────────────────────────────────────────────────
echo "[q289] BASE = gen 7.7 (no knobs): the check-extension forms are alternatives, each read against today's rule"
CN=${BASE:+base}; CN=${CN:-g77}
CK=""; [ -n "$BASE" ] && CK="option.Knobs=$BASE"

# ── 1 gates (no rebuild) ─────────────────────────────────────────────────────────────────────────────────────────────────
a=$(fp "$NEW" ""); b=$(fp "$G77" ""); k=$(fp "$NEW" "$KS")
v1=$(printf 'setoption name EvalArm value 0\nbench\nquit\n' | "$NEW" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+')
ga=$(gfp "$NEW" ""); gb=$(gfp "$G77" ""); gk=$(gfp "$NEW" "$KS")
echo "[q289] s7pgo default [$a] == g77pgo [$b] (254 / 35922538) · $KS [$k] (== the s7 build's $EXP_W) · v1 [$v1] · game12 d10 default [$ga] == g77 [$gb] · $KS [$gk] (== $EXP_G)"
[ "$a" = "solved 254/300 nodes 35922538" ] && [ "$b" = "$a" ] && [ "$k" = "$EXP_W" ] && [ "$k" != "$a" ] \
	&& echo "$v1" | grep -qE 'solved 250/300 nodes 3531077[5-8]' && [ -n "$ga" ] && [ "$ga" = "$gb" ] \
	&& [ "$gk" = "$EXP_G" ] \
	|| { echo "[q289] ☠️ a gate failed — no games — QUEUE 289 ABORTED"; exit 1; }
if [ -n "$BASE" ]; then
	kb=$(fp "$NEW" "$BASE"); kk=$(fp "$NEW" "$(kn cx1)"); gkb=$(gfp "$NEW" "$BASE"); gkk=$(gfp "$NEW" "$(kn cx1)")
	echo "[q289] base $BASE [$kb] (≠ default) · base + cx1 [$kk] (≠ base) · game12 base [$gkb] · base + cx1 [$gkk]"
	[ -n "$kb" ] && [ "$kb" != "$a" ] && [ -n "$kk" ] && [ "$kk" != "$kb" ] \
		|| { echo "[q289] ☠️ a base gate failed — no games — QUEUE 289 ABORTED"; exit 1; }
fi
echo "[q289] ✅ every gate equal"
rm -f /tmp/q289_pre.engine.log
timeout 180 "$FC" -engine "cmd=$NEW" name=cx1 option.OwnBook=false "option.Knobs=$(kn cx1)" st=0.05 \
	-engine "cmd=$NEW" name=$CN option.OwnBook=false $CK st=0.05 \
	-openings file=$O/openings_uho.epd format=epd order=random -rounds 1 -games 2 -concurrency 1 \
	-log file=/tmp/q289_pre.engine.log level=info engine=true > /tmp/q289_pre.log 2>&1
WN=$(grep -c 'have option' /tmp/q289_pre.log)
echo "[q289] preflight: $(grep -E '^Games:' /tmp/q289_pre.log | tail -1) · warnings $WN · echoes $KS $(grep -cF "$KS" /tmp/q289_pre.engine.log)${BASE:+ · echoes $BASE $(grep -cF "$BASE" /tmp/q289_pre.engine.log)}"
[ "$WN" = 0 ] || { echo "[q289] ☠️ preflight warnings — no games — QUEUE 289 DONE"; exit 1; }
T_BUILD=$(( $(date +%s) - T0 ))

# SF1.1's node count: an override file, else the #279 rung nearest 50% if it lies in 35-65%
N11=$(cat "$NFILE" 2>/dev/null | tr -dc '0-9')
[ -z "$N11" ] && N11=$(grep -E '^\[q279\] +SF1\.1 +[0-9]+ nodes:' "$Q279" | awk '{ gsub(/%/, "", $7); d = $7 - 50; if (d < 0) d = -d; if (best == "" || d < bd) { best = $3; bd = d; sc = $7 } } END { if (best != "" && sc >= 35 && sc <= 65) print best }')
RUNG_MIN=$(awk -v n="$N11" '$0 ~ "rung SF1.1 nodes=" n " " { split($2, a, ":"); s = a[1] * 60 + a[2] } $0 ~ "[]] [0-9:]+ nodes=" n ":" { split($2, b, ":"); e = b[1] * 60 + b[2] } END { if (s != "" && e != "") print e - s }' "$Q279")
S11_LEG_MIN=$(awk -v r="${RUNG_MIN:-10}" 'BEGIN { printf "%.0f", r * 2.5 }')
echo "[q289] SF1.1 judge: ${N11:-none} nodes (100 g took ${RUNG_MIN:-?} min in #279 ⇒ ~$S11_LEG_MIN min per 250-game leg)"

# ── the outside judges' legs (played while timed work is held, and after a self-play pass) ──────────────────────────────
COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
LEGS="g18:1:ctl g18:1:cx1 s11:1:cx1 s11:1:ctl g18:2:cx1 g18:2:ctl s11:2:ctl s11:2:cx1 g18:3:ctl g18:3:cx1 s11:3:cx1 s11:3:ctl"
T_EX=0; T_EXP=0; N_EXP=0; FAILED=" "; TRIED=" "
leg_seed() { case $1 in g18) echo $((289110 + $2)) ;; s11) echo $((289210 + $2)) ;; esac; }
leg_done() { grep -qE '^Games: 250,' "$O/${1}_q289_s$(leg_seed $1 $2)_$3.log" 2>/dev/null; }
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
		*" $1:$2:$3 "*) FAILED="$FAILED$1:$2:$3 "; echo "[q289] ☠️ leg $1 $2 $3 incomplete twice — given up" ;;
		*) TRIED="$TRIED$1:$2:$3 " ;;
	esac
}
ext_chunk() {  # judge (g18 | s11), seed, arm
	local j=$1 seed=$2 a=$3 k f=$O/${1}_q289_s$2_$3.log opp t
	stopped "$j $seed $a"
	k=$(kn $a); [ -n "$k" ] && k="option.Knobs=$k"
	case $j in
		g18) opp=(-engine "cmd=$SF18" name=sf18 option.Threads=1 option.Hash=16 nodes=1100) ;;
		s11) opp=(-engine "cmd=$SF1" name=sf1.1 option.Threads=1 option.Ponder=false option.OwnBook=false option.Hash=16 nodes=$N11) ;;
	esac
	t=$(date +%s)
	nice -n 5 "$FC" -engine "cmd=$NEW" name=ours option.OwnBook=false $k nodes=250000 "${opp[@]}" \
		-each tc=inf $COMMON -srand $seed -rounds 125 -games 2 -pgnout file=$O/${j}_q289_s${seed}_$a.pgn notation=san nodes=true append=false > "$f" 2>&1
	LEG_S=$(( $(date +%s) - t ))
	echo "[q289] $(date '+%H:%M') $j seed $seed $a: $(grep -E '^Games:' "$f" | tail -1) · warnings $(grep -c 'have option' "$f") · disconnects/stalls $(grep -ciE 'disconnect|stall|not responsive' "$f") · ${LEG_S}s"
}

# ── 2 self-play first; held time plays outside-judge legs ────────────────────────────────────────────────────────────────
# the next evening block (or the one in progress before 00:15), and the cutoff 10:40 the morning after it
NOW=$(date +%s); GS=$(date -d 'today 20:50' +%s)
[ "$NOW" -lt "$(date -d 'today 00:15' +%s)" ] && GS=$(date -d 'yesterday 20:50' +%s)
GE=$(( GS + 12300 )); CUT=$(( GE + 37500 ))
echo "[q289] $(date '+%H:%M') self-play cx1 vs ${BASE:-gen 7.7} (no timed work $(date -d @$GS '+%a %H:%M') to $(date -d @$GE '+%a %H:%M') unless q289.go; q289.pause or gaming.pause holds; cutoff $(date -d @$CUT '+%a %H:%M'))"
SEQ=$(ls "$GW"/cx1_s*.pgn 2>/dev/null | wc -l); T_SP=${SP_PRIOR:-0}; T_WAIT=0
[ -e "$RESUME" ] && echo "[q289] RESUMED: $(sp_count cx1) games kept from $SEQ chunk files, ${SP_PRIOR}s of self-play already played (part 1 log q289_part1.log); the next chunk seed is $((2891 + SEQ))"
sp_chunk() {  # arm; returns 1 when the arm already has 2,000
	local a=$1 n left pairs seed f t
	n=$(sp_count $a); left=$(( (2000 - n + 1) / 2 )); [ $left -le 0 ] && return 1
	pairs=$(( left < 150 ? left : 150 )); SEQ=$((SEQ + 1)); seed=$((2890 + SEQ)); f=$GW/${a}_s$seed
	echo "[q289] $(date '+%H:%M') self-play $a: $n finished, chunk seed $seed, $pairs pairs"
	guard; t=$(date +%s)
	nice -n 5 "$FC" -engine "cmd=$NEW" name=$a option.OwnBook=false "option.Knobs=$(kn $a)" tc=10+0.1 timemargin=250 \
		-engine "cmd=$NEW" name=$CN option.OwnBook=false $CK tc=10+0.1 timemargin=250 \
		$COMMON -srand $seed -rounds $pairs -games 2 -pgnout file=$f.pgn notation=san nodes=true append=false > $f.log 2>&1
	kill $GUARD 2>/dev/null; T_SP=$(( T_SP + $(date +%s) - t ))
	echo "[q289] $(date '+%H:%M') $a s$seed: $(grep -E '^Games:' $f.log | tail -1) · warnings $(grep -c 'have option' $f.log) · disconnects/stalls $(grep -ciE 'disconnect|stall|not responsive' $f.log) · time forfeits $(grep -c 'loses on time' $f.pgn) · unfinished $(grep -c '^\[Result "\*"' $f.pgn)"
	return 0
}
sp_loop() { while timed_ok; do stopped "a self-play chunk"; sp_chunk cx1 || break; done; }
held() {  # waits out a hold on timed work, playing outside-judge legs meanwhile
	local t L
	t=$(date +%s)
	until timed_ok || [ "$(date +%s)" -ge "$CUT" ]; do
		stopped "the self-play"
		L=$(next_leg)
		if [ -n "$L" ]; then
			echo "[q289] $(date '+%H:%M') timed work held ⇒ an outside-judge leg on the free box: $L"
			run_leg $L; T_EXP=$(( T_EXP + LEG_S )); N_EXP=$((N_EXP + 1))
			timed_ok && echo "[q289] $(date '+%H:%M') the hold lifted during that leg; self-play resumes now"
		else
			sleep 30
		fi
	done
	T_WAIT=$(( T_WAIT + $(date +%s) - t ))
}
while [ "$(sp_count cx1)" -lt 2000 ] && [ "$(date +%s)" -lt "$CUT" ]; do
	held
	sp_loop
done
SPN=$(sp_count cx1)
echo "[q289] pooled self-play cx1 vs ${BASE:-gen 7.7} ($SPN finished):"
timeout 900 bash "$R" pyrun selfplay/fastchess_report.py "$GW/cx1"_s*.pgn --p1 cx1 2>&1 | sed -n '/vs the field/,$p' > "$W/pooled.txt"
show < "$W/pooled.txt"
LO=$(grep -oE '\(95% [+-]?[0-9.]+' "$W/pooled.txt" | head -1 | sed 's/(95% //')
HI=$(grep -oE '… [+-]?[0-9.]+\)' "$W/pooled.txt" | head -1 | sed 's/… //; s/)//')
if [ "$SPN" -lt 2000 ]; then VERDICT="INCOMPLETE ($SPN of 2,000 by the cutoff)"
elif awk -v l="$LO" 'BEGIN { exit !(l != "" && l + 0 > 0) }'; then VERDICT="PASS"
elif awk -v h="$HI" 'BEGIN { exit !(h != "" && h + 0 < 0) }'; then VERDICT="FAIL"
else VERDICT="NEUTRAL"; fi
echo "[q289] REGISTERED self-play read: $VERDICT (95% $LO … $HI; pass = lower > 0, neutral = within ±2σ → parked, fail = upper < 0)"

# ── 3 the remaining outside-judge legs, only when self-play passes ───────────────────────────────────────────────────────
if [ "$VERDICT" = "PASS" ]; then
	echo "[q289] $(date '+%H:%M') self-play passed ⇒ outside judges: SF18 3 seeds, SF1.1 ${N11:-(not calibrated: skipped)} 3 seeds, ctl and cx1 ($N_EXP legs already played while timed work was held)"
	L=$(next_leg)
	while [ -n "$L" ]; do run_leg $L; T_EX=$(( T_EX + LEG_S )); L=$(next_leg); done
else
	echo "[q289] self-play $VERDICT ⇒ no further outside-judge legs (the candidate cannot ship without a self-play pass); $N_EXP legs played while timed work was held are information only"
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
	for f in $O/${j}_q289_s*_*.log; do [ -e "$f" ] || continue; b=$(basename $f .log); b=${b#${j}_q289_s}; row ${b%%_*} ${b#*_} $f; done > /tmp/q289_$j.txt
done
cat /tmp/q289_g18.txt /tmp/q289_s11.txt > /tmp/q289_ext.txt
if [ -s /tmp/q289_ext.txt ]; then
	if [ "$VERDICT" = "PASS" ]; then TAG="REGISTERED outside read"; else TAG="outside read, INFORMATION ONLY (self-play $VERDICT)"; fi
	echo "[q289] $(date) per leg (seed arm games W L D; ctl = ${BASE:-gen 7.7}):"; { sed 's/^/SF18 /' /tmp/q289_g18.txt; sed "s/^/SF1.1@$N11 /" /tmp/q289_s11.txt; } | show
	echo "[q289] $TAG (pooled SF18 + SF1.1: cx1 - ctl z > -1), and each judge alone:"
	{ cmpr cx1 ctl /tmp/q289_ext.txt "POOLED"; cmpr cx1 ctl /tmp/q289_g18.txt "SF18  "; cmpr cx1 ctl /tmp/q289_s11.txt "SF1.1 "; } | show
fi

# ── the time accounting (owner: "observe if it truly saves time") ────────────────────────────────────────────────────────
TOT=$(( $(date +%s) - T0 ))
OLD_EX=$(( (12 * SF18_LEG_MIN + 6 * S11_LEG_MIN) * 60 ))
echo "[q289] TIME: gates $(hrs $T_BUILD) · self-play $(hrs $T_SP)${SP_PRIOR:+ (incl. ${SP_PRIOR}s before a resume)} · timed work held $(hrs $T_WAIT), of which outside-judge legs $(hrs $T_EXP) ($N_EXP legs) · outside judges after self-play $(hrs $T_EX) · total $(hrs $TOT) (this run's wall only)"
echo "[q289] TIME: the old form's outside judges (6 SF18 seeds + 3 SF1.1 seeds per arm, after self-play) ≈ $(hrs $OLD_EX) at these leg times ⇒ box time this queue spent on judges outside held time $(hrs $T_EX), saved ≈ $(hrs $(( OLD_EX - T_EX )))"
echo "[q289] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | show
rm -f "$RESUME"
echo "[q289] $(date) QUEUE 289 DONE"
