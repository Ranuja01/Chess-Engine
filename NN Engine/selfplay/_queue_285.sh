#!/bin/bash
# 2026-11-20 ~20:30, queue #285: the MATE BUNDLE — FIX_TT_MATE=1 (mate scores stored in the TT as a distance from the stored
# position), FIX_MDP=1 (mate distance pruning), ENABLE_NULL_MATE_CLAMP=1 (an unproven null-move mate returns the bound).
# Chained after #284 (S2-4). No rebuild: all three knobs exist in s6pgo.
# Why: with FIX_TT_MATE=0 no mate-range score is ever stored in the TT (cache_management.h tt_score_storable), so a mate proof
# gets no transpositions. Measured 11-20 19:58 (_side_1120_fixed.sh A, g77pgo, bench at fixed depth): UI game 56's 47.f5
# position at depth 22 costs 167,029,193 nodes in production (the game: 172M) and 16,041,732 with FIX_TT_MATE=1 — 10.4x fewer,
# same move. The bundle was played once, as gen 6.6 on gen 6.5 (11-04/05: self-play ≈ −5.7 ± 7.0, SF18 −3.60 pp z −1.43, not
# kept); the main-search TT flag fix (ENABLE_TT_FLAG_FIX) shipped after that, in gen 6.7, and was suspected of interacting with
# stored mates (#152). The re-test trigger fired 11-19 (register §10, triage).
# HYGIENE item under the owner's 11-20 rule (JUDGING-SEARCH-IDEAS §4): each knob restores a statable invariant (a mate value is
# a distance, so it must be stored relative to the stored position; a mate found by a null move is not proven). Bar: self-play
# z > −1, then the targeted legs (information: where it fires) and the outside judges pooled z > −1, then the owner's yes; it
# would then join the hygiene confirmation bundle.
# Owner 11-20: targeted items get venue legs where they fire ("endgame starts, smaller but longer runs to see behaviour").
#  1 Wait for #284; production must be g77pgo; nothing else running. GATES on s6pgo: default 254 / 35,922,538 == g77pgo; the
#    bundle 252 / 35,107,909 and game12 d10 8/12 2,252,010 (measured 11-20 20:05); v1 250 / 35,310,775-778; game12 default ==
#    g77pgo. Per knob (recorded): FIX_TT_MATE 254 / 36,281,627 · 9/12 2,385,769; FIX_MDP 252 / 36,331,189 · 7/12 2,221,587;
#    ENABLE_NULL_MATE_CLAMP byte-identical on WAC, game12 and both game-56 positions (it fires only on a mate-range null score;
#    carried as a correctness guard, firing unproven).
#  2 Self-play mate vs the base, 10+0.1 UHO, 150-pair chunks, seeds 2851+, to 2,000 games.
#  3 IF self-play PASS or z > −1: (a) endgame starts, eg_critical.epd (464 endgame starts), 10+0.1, 1,000 g, seeds 285501+;
#    (b) a smaller longer run, UHO 60+0.6, 200 g in 50-pair chunks, seeds 285601+, read for behaviour (time per move mean /
#    p95 / max, depth); (c) the outside judges, SF18 (1,100 nodes) seeds 285111-3 and SF1.1 (320k) seeds 285211-3, ctl and mate.
#    No timed work 20:50 to 00:15 unless q285.go; gaming.pause or q285.pause holds timed work, and any hold plays judge legs.
# REGISTERED before any game:
#  - self-play −5 … +8 (gen 6.6 read −5.7 ± 7.0 without the flag fix; mates are rare at 10+0.1 UHO). Verdict: PASS = lower
#    95% > 0; ELIGIBLE (hygiene) = z > −1; PARKED = z ≤ −1 with upper > 0; FAIL = upper < 0.
#  - endgame starts: −5 … +15 (faster mate conversion; the venue resolves ±13-14 per 1,000 g, INSTRUMENT-MAP).
#  - 60+0.6: max time per move lower for mate (registered: max −0 … −30%), mean and p95 within ±3%, depth within ±0.1 ply.
#  - outside judges pooled mate − ctl z > −1.
# RESUME MODE: before relaunching after an interruption, copy q285.log to q285_part1.log, set aside any chunk that did not
# finish (unread), and write the timed seconds already played into /mnt/e/chess_data/q285.resume; complete chunks are kept,
# each leg's SEQ comes from its files, and q285.resume is removed at DONE.
# Stop: touch /mnt/e/chess_data/q285.stop (between steps). Never edit this file or the binaries while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
SF18="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/stockfish_18_linux/stockfish-ubuntu-x86-64-avx2"
SF1=/mnt/e/chess_data/engines/sf1.1/stockfish-1.1
O=/mnt/e/chess_data/fastchess
W=/mnt/e/chess_data/q285
GW=/mnt/e/chess_data/r285
G=/mnt/e/chess_data/sv2_speed/game12.txt
EG=$O/eg_critical.epd
PROD="$ND/build_uci/chess_engine_uci"
G77="$ND/build_uci/chess_engine_uci_g77pgo"
NEW="$ND/build_uci/chess_engine_uci_s6pgo"
KS="FIX_TT_MATE=1,FIX_MDP=1,ENABLE_NULL_MATE_CLAMP=1"
Q284=/mnt/e/chess_data/q284.log
Q279=/mnt/e/chess_data/q279.log
NFILE=/mnt/e/chess_data/sf11_judge_nodes.txt
STOP=/mnt/e/chess_data/q285.stop
PAUSE=/mnt/e/chess_data/q285.pause
GPAUSE=/mnt/e/chess_data/gaming.pause
GO=/mnt/e/chess_data/q285.go
RESUME=/mnt/e/chess_data/q285.resume
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
SF18_LEG_MIN=13   # an SF18 leg (250 g) in #274: 12.5-13.5 min
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP" "$GO"; mkdir -p "$W" "$GW"   # the pause files are the owner's: never removed here
if [ -e "$RESUME" ]; then T_PRIOR=$(tr -dc '0-9' < "$RESUME"); else T_PRIOR=""; rm -f "$GW"/*_s*.pgn "$GW"/*_s*.log $O/g18_q285_s* $O/s11_q285_s*; fi
BASE=""
show() { sed "s/^/[q285]   /"; }
stopped() { [ -e "$STOP" ] && { echo "[q285] $(date) stop file before $1 — QUEUE 285 DONE"; exit 0; }; }
kn() { case $1 in ctl) echo "$BASE" ;; mate) echo "${BASE:+$BASE,}$KS" ;; esac; }
fp() { local k=""; [ -n "$2" ] && k="setoption name Knobs value $2"; printf '%s\nbench\nquit\n' "$k" | "$1" 2>&1 | grep -oE 'solved [0-9]+/300 nodes [0-9]+|mismatch.*' | head -1; }
gfp() { local k=""; [ -n "$2" ] && k="setoption name Knobs value $2"; printf "%s\nbench $G 10\nquit\n" "$k" | "$1" 2>&1 | grep -oE 'solved [0-9]+/[0-9]+ nodes [0-9]+|mismatch.*' | head -1; }
busy() { pgrep -f "[f]astchess-linux-x86-64/fastchess" >/dev/null || pgrep -f "[s]ts_test|[t]actical_test|[v]algrind|[_]ks_footprint|[b]uild_uci.sh" >/dev/null; }
timed_ok() { local t; t=$(date +%s); [ ! -e "$PAUSE" ] && [ ! -e "$GPAUSE" ] && [ "$t" -lt "$CUT" ] && ! { [ "$t" -ge "$GS" ] && [ "$t" -lt "$GE" ] && [ ! -e "$GO" ]; }; }
guard() { ( until ! timed_ok; do sleep 15; done; pkill -INT -f "[f]astchess-linux-x86-64/fastchess" && echo "[q285] $(date '+%H:%M') guard: games interrupted" ) & GUARD=$!; }
finished() { local c; c=$(grep -cE '^\[Result "(1-0|0-1|1/2-1/2)"\]' "$1" 2>/dev/null); echo "${c:-0}"; }
count() { local n=0 f; for f in "$GW/$1"_s*.pgn; do [ -e "$f" ] && n=$((n + $(finished "$f"))); done; echo $n; }
hrs() { awk -v s=$1 'BEGIN { printf "%.2f h", s / 3600 }'; }
passed() {  # queue log, arm: its registered reads passed (self-play PASS, pooled outside z > -1)
	local z; z=$(grep -E "POOLED +$2 " "$1" 2>/dev/null | grep -oE 'z [+-]?[0-9.]+' | tail -1 | sed 's/z //')
	grep -qE 'REGISTERED self-play read: PASS' "$1" 2>/dev/null && grep -qE 'REGISTERED outside read' "$1" \
		&& awk -v z="$z" 'BEGIN { exit !(z != "" && z + 0 > -1) }'
}
verdict() { echo "$(grep -oE 'REGISTERED self-play read: [A-Z]+' "$1" 2>/dev/null | tail -1 | sed 's/REGISTERED self-play read: //'), pooled outside z $(grep -E "POOLED +$2 " "$1" 2>/dev/null | grep -oE 'z [+-]?[0-9.]+' | tail -1 | sed 's/z //')"; }
echo "[q285] $(date '+%Y-%m-%d %H:%M:%S') queued; waiting for #284"
until grep -qE 'QUEUE 284 (DONE|ABORTED)' "$Q284" 2>/dev/null || ! pgrep -f "[_]queue_284.sh" >/dev/null; do stopped "the start"; sleep 30; done
sleep 20
cmp -s "$PROD" "$G77" || { echo "[q285] ☠️ production is not g77pgo — QUEUE 285 ABORTED"; exit 1; }
[ -x "$NEW" ] || { echo "[q285] ☠️ $NEW missing — QUEUE 285 ABORTED"; exit 1; }
[ -s "$EG" ] || { echo "[q285] ☠️ $EG missing — QUEUE 285 ABORTED"; exit 1; }
busy && { echo "[q285] ☠️ another job is running — QUEUE 285 ABORTED"; exit 1; }
START=$(date '+%Y-%m-%d %H:%M:%S'); T0=$(date +%s)
echo "[q285] $START start"

# ── the base: #284 if it passed its registered reads (a hygiene-eligible item goes to the confirmation bundle, not the base) ─
passed "$Q284" ptt && BASE="PS_TT_FLAG_FIX=1"
echo "[q285] #284 S2-4: $(verdict "$Q284" ptt) ⇒ BASE = ${BASE:-gen 7.7 (no knobs)}${BASE:+ in both arms (shipping it still waits on the owner; if not shipped, this read is on that base, not on production)}"
CN=${BASE:+base}; CN=${CN:-g77}
CK=""; [ -n "$BASE" ] && CK="option.Knobs=$BASE"

# ── 1 gates (no rebuild) ─────────────────────────────────────────────────────────────────────────────────────────────────
a=$(fp "$NEW" ""); b=$(fp "$G77" ""); k=$(fp "$NEW" "$KS")
v1=$(printf 'setoption name EvalArm value 0\nbench\nquit\n' | "$NEW" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+')
ga=$(gfp "$NEW" ""); gb=$(gfp "$G77" ""); gk=$(gfp "$NEW" "$KS")
echo "[q285] s6pgo default [$a] == g77pgo [$b] (254 / 35922538) · bundle [$k] (== 252 / 35107909) · v1 [$v1] · game12 d10 default [$ga] == g77 [$gb] · bundle [$gk] (== 8/12 2252010)"
[ "$a" = "solved 254/300 nodes 35922538" ] && [ "$b" = "$a" ] && [ "$k" = "solved 252/300 nodes 35107909" ] \
	&& echo "$v1" | grep -qE 'solved 250/300 nodes 3531077[5-8]' && [ -n "$ga" ] && [ "$ga" = "$gb" ] \
	&& [ "$gk" = "solved 8/12 nodes 2252010" ] \
	|| { echo "[q285] ☠️ a gate failed — no games — QUEUE 285 ABORTED"; exit 1; }
if [ -n "$BASE" ]; then
	kb=$(fp "$NEW" "$BASE"); kk=$(fp "$NEW" "$(kn mate)"); gkb=$(gfp "$NEW" "$BASE"); gkk=$(gfp "$NEW" "$(kn mate)")
	echo "[q285] base $BASE [$kb] (≠ default) · base + mate [$kk] (≠ base) · game12 base [$gkb] · base + mate [$gkk]"
	[ -n "$kb" ] && [ "$kb" != "$a" ] && [ -n "$kk" ] && [ "$kk" != "$kb" ] \
		|| { echo "[q285] ☠️ a base gate failed — no games — QUEUE 285 ABORTED"; exit 1; }
fi
echo "[q285] ✅ every gate equal"
rm -f /tmp/q285_pre.engine.log
timeout 180 "$FC" -engine "cmd=$NEW" name=mate option.OwnBook=false "option.Knobs=$(kn mate)" st=0.05 \
	-engine "cmd=$NEW" name=$CN option.OwnBook=false $CK st=0.05 \
	-openings file=$O/openings_uho.epd format=epd order=random -rounds 1 -games 2 -concurrency 1 \
	-log file=/tmp/q285_pre.engine.log level=info engine=true > /tmp/q285_pre.log 2>&1
WN=$(grep -c 'have option' /tmp/q285_pre.log)
echo "[q285] preflight: $(grep -E '^Games:' /tmp/q285_pre.log | tail -1) · warnings $WN · echoes $KS $(grep -cF "$KS" /tmp/q285_pre.engine.log)${BASE:+ · echoes $BASE $(grep -cF "$BASE" /tmp/q285_pre.engine.log)}"
[ "$WN" = 0 ] || { echo "[q285] ☠️ preflight warnings — no games — QUEUE 285 DONE"; exit 1; }
T_BUILD=$(( $(date +%s) - T0 ))

# SF1.1's node count: an override file, else the #279 rung nearest 50% if it lies in 35-65%
N11=$(cat "$NFILE" 2>/dev/null | tr -dc '0-9')
[ -z "$N11" ] && N11=$(grep -E '^\[q279\] +SF1\.1 +[0-9]+ nodes:' "$Q279" | awk '{ gsub(/%/, "", $7); d = $7 - 50; if (d < 0) d = -d; if (best == "" || d < bd) { best = $3; bd = d; sc = $7 } } END { if (best != "" && sc >= 35 && sc <= 65) print best }')
RUNG_MIN=$(awk -v n="$N11" '$0 ~ "rung SF1.1 nodes=" n " " { split($2, a, ":"); s = a[1] * 60 + a[2] } $0 ~ "[]] [0-9:]+ nodes=" n ":" { split($2, b, ":"); e = b[1] * 60 + b[2] } END { if (s != "" && e != "") print e - s }' "$Q279")
S11_LEG_MIN=$(awk -v r="${RUNG_MIN:-10}" 'BEGIN { printf "%.0f", r * 2.5 }')
echo "[q285] SF1.1 judge: ${N11:-none} nodes (100 g took ${RUNG_MIN:-?} min in #279 ⇒ ~$S11_LEG_MIN min per 250-game leg)"

# ── the outside judges' legs (played while timed work is held, and after self-play reads PASS or ELIGIBLE) ────────────────
COMMON="-repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
UHO="-openings file=$O/openings_uho.epd format=epd order=random"
LEGS="g18:1:ctl g18:1:mate s11:1:mate s11:1:ctl g18:2:mate g18:2:ctl s11:2:ctl s11:2:mate g18:3:ctl g18:3:mate s11:3:mate s11:3:ctl"
T_EX=0; T_EXP=0; N_EXP=0; FAILED=" "; TRIED=" "
leg_seed() { case $1 in g18) echo $((285110 + $2)) ;; s11) echo $((285210 + $2)) ;; esac; }
leg_done() { grep -qE '^Games: 250,' "$O/${1}_q285_s$(leg_seed $1 $2)_$3.log" 2>/dev/null; }
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
		*" $1:$2:$3 "*) FAILED="$FAILED$1:$2:$3 "; echo "[q285] ☠️ leg $1 $2 $3 incomplete twice — given up" ;;
		*) TRIED="$TRIED$1:$2:$3 " ;;
	esac
}
ext_chunk() {  # judge (g18 | s11), seed, arm
	local j=$1 seed=$2 a=$3 k f=$O/${1}_q285_s$2_$3.log opp t
	stopped "$j $seed $a"
	k=$(kn $a); [ -n "$k" ] && k="option.Knobs=$k"
	case $j in
		g18) opp=(-engine "cmd=$SF18" name=sf18 option.Threads=1 option.Hash=16 nodes=1100) ;;
		s11) opp=(-engine "cmd=$SF1" name=sf1.1 option.Threads=1 option.Ponder=false option.OwnBook=false option.Hash=16 nodes=$N11) ;;
	esac
	t=$(date +%s)
	nice -n 5 "$FC" -engine "cmd=$NEW" name=ours option.OwnBook=false $k nodes=250000 "${opp[@]}" \
		-each tc=inf $UHO $COMMON -srand $seed -rounds 125 -games 2 -pgnout file=$O/${j}_q285_s${seed}_$a.pgn notation=san nodes=true append=false > "$f" 2>&1
	LEG_S=$(( $(date +%s) - t ))
	echo "[q285] $(date '+%H:%M') $j seed $seed $a: $(grep -E '^Games:' "$f" | tail -1) · warnings $(grep -c 'have option' "$f") · disconnects/stalls $(grep -ciE 'disconnect|stall|not responsive' "$f") · ${LEG_S}s"
}

# ── timed legs: self-play (UHO 10+0.1), endgame starts (10+0.1), the longer run (UHO 60+0.6) ────────────────────────────
# the next evening block (or the one in progress before 00:15), and the cutoff 10:40 the morning after it
NOW=$(date +%s); GS=$(date -d 'today 20:50' +%s)
[ "$NOW" -lt "$(date -d 'today 00:15' +%s)" ] && GS=$(date -d 'yesterday 20:50' +%s)
GE=$(( GS + 12300 )); CUT=$(( GE + 37500 ))
echo "[q285] $(date '+%H:%M') self-play mate vs ${BASE:-gen 7.7} (no timed work $(date -d @$GS '+%a %H:%M') to $(date -d @$GE '+%a %H:%M') unless q285.go; q285.pause or gaming.pause holds; cutoff $(date -d @$CUT '+%a %H:%M'))"
T_SP=${T_PRIOR:-0}; T_WAIT=0
[ -e "$RESUME" ] && echo "[q285] RESUMED: kept self-play $(count sp) · endgame $(count eg) · 60+0.6 $(count long) games, ${T_PRIOR}s of timed play already (part 1 log q285_part1.log)"
tchunk() {  # tag (sp | eg | long): plays the next chunk of that leg; returns 1 when the leg has its games
	local tag=$1 target book tc maxp base n left pairs seq seed f t
	case $tag in
		sp)   target=2000; book="$UHO"; tc=10+0.1; maxp=150; base=2850 ;;
		eg)   target=1000; book="-openings file=$EG format=epd order=random"; tc=10+0.1; maxp=150; base=285500 ;;
		long) target=200; book="$UHO"; tc=60+0.6; maxp=50; base=285600 ;;
	esac
	n=$(count $tag); left=$(( (target - n + 1) / 2 )); [ $left -le 0 ] && return 1
	pairs=$(( left < maxp ? left : maxp )); seq=$(( $(ls "$GW/${tag}"_s*.pgn 2>/dev/null | wc -l) + 1 )); seed=$((base + seq)); f=$GW/${tag}_s$seed
	echo "[q285] $(date '+%H:%M') $tag: $n finished, chunk seed $seed, $pairs pairs, $tc"
	guard; t=$(date +%s)
	nice -n 5 "$FC" -engine "cmd=$NEW" name=mate option.OwnBook=false "option.Knobs=$(kn mate)" tc=$tc timemargin=250 \
		-engine "cmd=$NEW" name=$CN option.OwnBook=false $CK tc=$tc timemargin=250 \
		$book $COMMON -srand $seed -rounds $pairs -games 2 -pgnout file=$f.pgn notation=san nodes=true append=false > $f.log 2>&1
	kill $GUARD 2>/dev/null; T_SP=$(( T_SP + $(date +%s) - t ))
	echo "[q285] $(date '+%H:%M') $tag s$seed: $(grep -E '^Games:' $f.log | tail -1) · warnings $(grep -c 'have option' $f.log) · disconnects/stalls $(grep -ciE 'disconnect|stall|not responsive' $f.log) · time forfeits $(grep -c 'loses on time' $f.pgn) · unfinished $(grep -c '^\[Result "\*"' $f.pgn)"
	return 0
}
held() {  # waits out a hold on timed work, playing outside-judge legs meanwhile
	local t L
	t=$(date +%s)
	until timed_ok || [ "$(date +%s)" -ge "$CUT" ]; do
		stopped "the timed legs"
		L=$(next_leg)
		if [ -n "$L" ]; then
			echo "[q285] $(date '+%H:%M') timed work held ⇒ an outside-judge leg on the free box: $L"
			run_leg $L; T_EXP=$(( T_EXP + LEG_S )); N_EXP=$((N_EXP + 1))
			timed_ok && echo "[q285] $(date '+%H:%M') the hold lifted during that leg; timed play resumes now"
		else
			sleep 30
		fi
	done
	T_WAIT=$(( T_WAIT + $(date +%s) - t ))
}
play() {  # tag: plays that leg to its target, holding as needed, until the cutoff
	local tag=$1 target
	case $tag in sp) target=2000 ;; eg) target=1000 ;; long) target=200 ;; esac
	while [ "$(count $tag)" -lt "$target" ] && [ "$(date +%s)" -lt "$CUT" ]; do
		held
		while timed_ok; do stopped "a $tag chunk"; tchunk $tag || break; done
	done
}
report() {  # tag, label: the pooled read of a timed leg into $W/<tag>.txt
	timeout 900 bash "$R" pyrun selfplay/fastchess_report.py "$GW/$1"_s*.pgn --p1 mate 2>&1 | sed -n '/vs the field/,$p' > "$W/$1.txt"
	echo "[q285] pooled $2 ($(count $1) finished):"; show < "$W/$1.txt"
}
play sp
SPN=$(count sp)
report sp "self-play mate vs ${BASE:-gen 7.7}, UHO 10+0.1"
ELO=$(grep -oE 'Elo [+-]?[0-9.]+' "$W/sp.txt" | head -1 | sed 's/Elo //')
LO=$(grep -oE '\(95% [+-]?[0-9.]+' "$W/sp.txt" | head -1 | sed 's/(95% //')
HI=$(grep -oE '… [+-]?[0-9.]+\)' "$W/sp.txt" | head -1 | sed 's/… //; s/)//')
Z=$(awk -v e="$ELO" -v l="$LO" -v h="$HI" 'BEGIN { s = (h - l) / 3.92; if (e == "" || s <= 0) print ""; else printf "%+.2f", e / s }')
if [ "$SPN" -lt 2000 ]; then VERDICT="INCOMPLETE ($SPN of 2,000 by the cutoff)"
elif awk -v l="$LO" 'BEGIN { exit !(l != "" && l + 0 > 0) }'; then VERDICT="PASS"
elif awk -v h="$HI" 'BEGIN { exit !(h != "" && h + 0 < 0) }'; then VERDICT="FAIL"
elif awk -v z="$Z" 'BEGIN { exit !(z != "" && z + 0 > -1) }'; then VERDICT="ELIGIBLE"
else VERDICT="PARKED"; fi
echo "[q285] REGISTERED self-play read: $VERDICT (Elo $ELO, 95% $LO … $HI, z $Z; PASS = lower > 0, ELIGIBLE (hygiene) = z > −1, PARKED = z ≤ −1, FAIL = upper < 0)"

# ── 3 the targeted legs and the outside judges, when self-play reads PASS or ELIGIBLE ────────────────────────────────────
if [ "$VERDICT" = "PASS" ] || [ "$VERDICT" = "ELIGIBLE" ]; then
	echo "[q285] $(date '+%H:%M') self-play $VERDICT ⇒ endgame starts (1,000 g), the 60+0.6 run (200 g), then the outside judges ($N_EXP legs already played while timed work was held)"
	play eg; report eg "endgame starts, eg_critical.epd 10+0.1 (registered −5 … +15)"
	play long; report long "the longer run, UHO 60+0.6 (registered: max time per move −0 … −30% for mate; mean, p95 ±3%; depth ±0.1)"
	L=$(next_leg)
	while [ -n "$L" ]; do run_leg $L; T_EX=$(( T_EX + LEG_S )); L=$(next_leg); done
else
	echo "[q285] self-play $VERDICT ⇒ no targeted legs and no further outside-judge legs; $N_EXP legs played while timed work was held are information only"
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
	for f in $O/${j}_q285_s*_*.log; do [ -e "$f" ] || continue; b=$(basename $f .log); b=${b#${j}_q285_s}; row ${b%%_*} ${b#*_} $f; done > /tmp/q285_$j.txt
done
cat /tmp/q285_g18.txt /tmp/q285_s11.txt > /tmp/q285_ext.txt
if [ -s /tmp/q285_ext.txt ]; then
	if [ "$VERDICT" = "PASS" ] || [ "$VERDICT" = "ELIGIBLE" ]; then TAG="REGISTERED outside read"; else TAG="outside read, INFORMATION ONLY (self-play $VERDICT)"; fi
	echo "[q285] $(date) per leg (seed arm games W L D; ctl = ${BASE:-gen 7.7}):"; { sed 's/^/SF18 /' /tmp/q285_g18.txt; sed "s/^/SF1.1@$N11 /" /tmp/q285_s11.txt; } | show
	echo "[q285] $TAG (pooled SF18 + SF1.1: mate - ctl z > -1), and each judge alone:"
	{ cmpr mate ctl /tmp/q285_ext.txt "POOLED"; cmpr mate ctl /tmp/q285_g18.txt "SF18  "; cmpr mate ctl /tmp/q285_s11.txt "SF1.1 "; } | show
fi

# ── the time accounting ──────────────────────────────────────────────────────────────────────────────────────────────────
TOT=$(( $(date +%s) - T0 ))
OLD_EX=$(( (12 * SF18_LEG_MIN + 6 * S11_LEG_MIN) * 60 ))
echo "[q285] TIME: gates $(hrs $T_BUILD) · timed legs $(hrs $T_SP)${T_PRIOR:+ (incl. ${T_PRIOR}s before a resume)} · timed work held $(hrs $T_WAIT), of which outside-judge legs $(hrs $T_EXP) ($N_EXP legs) · outside judges after the timed legs $(hrs $T_EX) · total $(hrs $TOT) (this run's wall only)"
echo "[q285] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | show
rm -f "$RESUME"
echo "[q285] $(date) QUEUE 285 DONE"
