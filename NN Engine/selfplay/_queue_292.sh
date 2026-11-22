#!/bin/bash
# 2026-11-22 ~17:45, queue #292: S2-2 arm 3 — endgame searches stop taking the static eval at a horizon whose result stands
# on its own (QPREC_FIX=3: a horizon node in check, or inside a null-move subtree, always takes qsearch; the cheap static
# scouts elsewhere stay), against today's gate (gen 7.7). s7pgo. Chained after #291 (S2-5 mode 1); the owner is away until
# Monday ~11:00 (11-22 ~17:40: "can you keep things going until then?").
# Why (dev_notes/active/SWEEP3-S2-5-S2-2-2026-11-20.md §2): under an endgame root (use_q_precautions, phase <= 6) a horizon
# whose line target is below 6 returns the static eval, even in check and even where a null-move cut stands on it. Arm 3
# makes every result that decides something real and keeps the static scouts that only order (its own node saver).
# A NARROW ITEM, judged where it fires (owner 11-20, register §10 item 5: "an endgame-start leg is the primary read"); UHO
# self-play is the not-harmful check (only games reaching phase <= 6 exercise it). Arms 1 and 8 wait for this read.
# No rebuild: QPREC_FIX is in s7pgo.
#  1 Wait for #291; production must be g77pgo; nothing else running. GATES on s7pgo: default 254 / 35,922,538 == g77pgo;
#    QPREC_FIX=3 == the value the s7 build recorded (WAC and game12) and ≠ default; v1 250 / 35,310,775-778; game12
#    default == g77pgo.
#  2 (a) PRIMARY: endgame starts, eg_critical.epd (464 starts), 10+0.1, 2,000 g in 150-pair chunks, seeds 292501+;
#    (b) the not-harmful check: UHO 10+0.1, 2,000 g in 150-pair chunks, seeds 292601+.
#    No timed work Mon 20:50 to Tue 00:15 unless q292.go; gaming.pause or q292.pause holds timed work; nothing is played
#    during a hold.
# REGISTERED before any game (the S2-2 report's table, §2.5; vs today's gate):
#  - endgame starts: −4 … +12, centre +3 (the venue resolves about ±7.8 per 1,000 g, #287; ±5.5 at 2,000).
#  - UHO self-play: −3 … +8 (the report's range).
#  - bench WAC d10 nodes +0.2 … +1% (the s7 build recorded +0.3%: inside); endgame depth at equal time 0 … −0.05 ply.
#  Reading (for the owner, not automatic): endgame lower 95% > 0 with UHO z > −1 ⇒ a candidate, then the outside judges from
#  the same endgame starts and arm 8 against it; endgame inside ±2σ ⇒ parked as a narrow null at this venue's resolution;
#  endgame upper < 0 ⇒ read endgame depth (a node cost: pair with S2-10 / S2-6, the report's partners).
# RESUME MODE: before relaunching after an interruption, copy q292.log to q292_part1.log, set aside any chunk that did not
# finish (unread), and write the timed seconds already played into /mnt/e/chess_data/q292.resume; complete chunks are kept,
# each leg's SEQ comes from its files, and q292.resume is removed at DONE.
# Stop: touch /mnt/e/chess_data/q292.stop (between steps). Never edit this file or the binaries while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
W=/mnt/e/chess_data/q292
GW=/mnt/e/chess_data/r292
G=/mnt/e/chess_data/sv2_speed/game12.txt
EG=$O/eg_critical.epd
PROD="$ND/build_uci/chess_engine_uci"
G77="$ND/build_uci/chess_engine_uci_g77pgo"
NEW="$ND/build_uci/chess_engine_uci_s7pgo"
KS="QPREC_FIX=3"
QPREV=/mnt/e/chess_data/q291.log
BLOG=/mnt/e/chess_data/build_s7.log
STOP=/mnt/e/chess_data/q292.stop
PAUSE=/mnt/e/chess_data/q292.pause
GPAUSE=/mnt/e/chess_data/gaming.pause
GO=/mnt/e/chess_data/q292.go
RESUME=/mnt/e/chess_data/q292.resume
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP" "$GO"; mkdir -p "$W" "$GW"   # the pause files are the owner's: never removed here
if [ -e "$RESUME" ]; then T_PRIOR=$(tr -dc '0-9' < "$RESUME"); else T_PRIOR=""; rm -f "$GW"/*_s*.pgn "$GW"/*_s*.log; fi
show() { sed "s/^/[q292]   /"; }
stopped() { [ -e "$STOP" ] && { echo "[q292] $(date) stop file before $1 — QUEUE 292 DONE"; exit 0; }; }
fp() { local k=""; [ -n "$2" ] && k="setoption name Knobs value $2"; printf '%s\nbench\nquit\n' "$k" | "$1" 2>&1 | grep -oE 'solved [0-9]+/300 nodes [0-9]+|mismatch.*' | head -1; }
gfp() { local k=""; [ -n "$2" ] && k="setoption name Knobs value $2"; printf "%s\nbench $G 10\nquit\n" "$k" | "$1" 2>&1 | grep -oE 'solved [0-9]+/[0-9]+ nodes [0-9]+|mismatch.*' | head -1; }
busy() { pgrep -f "[f]astchess-linux-x86-64/fastchess" >/dev/null || pgrep -f "[s]ts_test|[t]actical_test|[v]algrind|[_]ks_footprint|[b]uild_uci.sh" >/dev/null; }
timed_ok() { local t; t=$(date +%s); [ ! -e "$PAUSE" ] && [ ! -e "$GPAUSE" ] && [ "$t" -lt "$CUT" ] && ! { [ "$t" -ge "$GS" ] && [ "$t" -lt "$GE" ] && [ ! -e "$GO" ]; }; }
guard() { ( until ! timed_ok; do sleep 15; done; pkill -INT -f "[f]astchess-linux-x86-64/fastchess" && echo "[q292] $(date '+%H:%M') guard: games interrupted" ) & GUARD=$!; }
finished() { local c; c=$(grep -cE '^\[Result "(1-0|0-1|1/2-1/2)"\]' "$1" 2>/dev/null); echo "${c:-0}"; }
count() { local n=0 f; for f in "$GW/$1"_s*.pgn; do [ -e "$f" ] && n=$((n + $(finished "$f"))); done; echo $n; }
hrs() { awk -v s=$1 'BEGIN { printf "%.2f h", s / 3600 }'; }
echo "[q292] $(date '+%Y-%m-%d %H:%M:%S') queued; waiting for #291"
until { grep -qE 'QUEUE 291 (DONE|ABORTED)' "$QPREV" 2>/dev/null || ! pgrep -f "[_]queue_291.sh" >/dev/null; } && ! pgrep -f "[_]queue_290.sh" >/dev/null; do stopped "the start"; sleep 30; done
sleep 20
cmp -s "$PROD" "$G77" || { echo "[q292] ☠️ production is not g77pgo — QUEUE 292 ABORTED"; exit 1; }
[ -x "$NEW" ] || { echo "[q292] ☠️ $NEW missing — QUEUE 292 ABORTED"; exit 1; }
[ -s "$EG" ] || { echo "[q292] ☠️ $EG missing — QUEUE 292 ABORTED"; exit 1; }
EXP_W=$(grep -E '^\[s7\] +QPREC_FIX=3:' "$BLOG" | grep -oE 'WAC \[solved [0-9]+/300 nodes [0-9]+' | sed 's/WAC \[//')
EXP_G=$(grep -E '^\[s7\] +QPREC_FIX=3:' "$BLOG" | grep -oE 'game12 \[solved [0-9]+/[0-9]+ nodes [0-9]+' | sed 's/game12 \[//')
[ -n "$EXP_W" ] && [ -n "$EXP_G" ] || { echo "[q292] ☠️ no recorded s7 liveness value for $KS — QUEUE 292 ABORTED"; exit 1; }
busy && { echo "[q292] ☠️ another job is running — QUEUE 292 ABORTED"; exit 1; }
START=$(date '+%Y-%m-%d %H:%M:%S'); T0=$(date +%s)
echo "[q292] $START start (S2-2 arm 3; ctl = gen 7.7, no knobs)"

# ── 1 gates (no rebuild) ─────────────────────────────────────────────────────────────────────────────────────────────────
a=$(fp "$NEW" ""); b=$(fp "$G77" ""); k=$(fp "$NEW" "$KS")
v1=$(printf 'setoption name EvalArm value 0\nbench\nquit\n' | "$NEW" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+')
ga=$(gfp "$NEW" ""); gb=$(gfp "$G77" ""); gk=$(gfp "$NEW" "$KS")
echo "[q292] s7pgo default [$a] == g77pgo [$b] (254 / 35922538) · $KS [$k] (== the s7 build's $EXP_W) · v1 [$v1] · game12 d10 default [$ga] == g77 [$gb] · $KS [$gk] (== $EXP_G)"
[ "$a" = "solved 254/300 nodes 35922538" ] && [ "$b" = "$a" ] && [ "$k" = "$EXP_W" ] && [ "$k" != "$a" ] \
	&& echo "$v1" | grep -qE 'solved 250/300 nodes 3531077[5-8]' && [ -n "$ga" ] && [ "$ga" = "$gb" ] \
	&& [ "$gk" = "$EXP_G" ] \
	|| { echo "[q292] ☠️ a gate failed — no games — QUEUE 292 ABORTED"; exit 1; }
echo "[q292] ✅ every gate equal"
rm -f /tmp/q292_pre.engine.log
timeout 180 "$FC" -engine "cmd=$NEW" name=qp3 option.OwnBook=false "option.Knobs=$KS" st=0.05 \
	-engine "cmd=$NEW" name=g77 option.OwnBook=false st=0.05 \
	-openings file=$EG format=epd order=random -rounds 1 -games 2 -concurrency 1 \
	-log file=/tmp/q292_pre.engine.log level=info engine=true > /tmp/q292_pre.log 2>&1
WN=$(grep -c 'have option' /tmp/q292_pre.log)
echo "[q292] preflight: $(grep -E '^Games:' /tmp/q292_pre.log | tail -1) · warnings $WN · echoes $KS $(grep -cF "$KS" /tmp/q292_pre.engine.log)"
[ "$WN" = 0 ] && grep -qE '^Games: 2,' /tmp/q292_pre.log || { echo "[q292] ☠️ preflight failed — no games — QUEUE 292 ABORTED"; exit 1; }
T_BUILD=$(( $(date +%s) - T0 ))

COMMON="-repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
UHO="-openings file=$O/openings_uho.epd format=epd order=random"

# ── 2 timed legs: endgame starts (primary), then UHO (the not-harmful check) ─────────────────────────────────────────────
# the next evening block (or the one in progress before 00:15), and the cutoff 10:40 the morning after it
NOW=$(date +%s); GS=$(date -d 'today 20:50' +%s)
[ "$NOW" -lt "$(date -d 'today 00:15' +%s)" ] && GS=$(date -d 'yesterday 20:50' +%s)
GE=$(( GS + 12300 )); CUT=$(( GE + 37500 ))
echo "[q292] $(date '+%H:%M') timed legs (no timed work $(date -d @$GS '+%a %H:%M') to $(date -d @$GE '+%a %H:%M') unless q292.go; q292.pause or gaming.pause holds; cutoff $(date -d @$CUT '+%a %H:%M'))"
T_SP=${T_PRIOR:-0}; T_WAIT=0
[ -e "$RESUME" ] && echo "[q292] RESUMED: kept endgame $(count eg) · UHO $(count uho) games, ${T_PRIOR}s of timed play already (part 1 log q292_part1.log)"
tchunk() {  # tag (eg | uho): plays the next chunk of that leg; returns 1 when the leg has its games
	local tag=$1 target book maxp=150 base n left pairs seq seed f t
	case $tag in
		eg)  target=2000; book="-openings file=$EG format=epd order=random"; base=292500 ;;
		uho) target=2000; book="$UHO"; base=292600 ;;
	esac
	n=$(count $tag); left=$(( (target - n + 1) / 2 )); [ $left -le 0 ] && return 1
	pairs=$(( left < maxp ? left : maxp )); seq=$(( $(ls "$GW/${tag}"_s*.pgn 2>/dev/null | wc -l) + 1 )); seed=$((base + seq)); f=$GW/${tag}_s$seed
	echo "[q292] $(date '+%H:%M') $tag: $n finished, chunk seed $seed, $pairs pairs, 10+0.1"
	guard; t=$(date +%s)
	nice -n 5 "$FC" -engine "cmd=$NEW" name=qp3 option.OwnBook=false "option.Knobs=$KS" tc=10+0.1 timemargin=250 \
		-engine "cmd=$NEW" name=g77 option.OwnBook=false tc=10+0.1 timemargin=250 \
		$book $COMMON -srand $seed -rounds $pairs -games 2 -pgnout file=$f.pgn notation=san nodes=true append=false > $f.log 2>&1
	kill $GUARD 2>/dev/null; T_SP=$(( T_SP + $(date +%s) - t ))
	echo "[q292] $(date '+%H:%M') $tag s$seed: $(grep -E '^Games:' $f.log | tail -1) · warnings $(grep -c 'have option' $f.log) · disconnects/stalls $(grep -ciE 'disconnect|stall|not responsive' $f.log) · time forfeits $(grep -c 'loses on time' $f.pgn) · unfinished $(grep -c '^\[Result "\*"' $f.pgn)"
	return 0
}
held() {  # waits out a hold on timed work; nothing is played meanwhile
	local t
	t=$(date +%s)
	timed_ok || echo "[q292] $(date '+%H:%M') timed work held (gaming.pause, q292.pause or the evening block); waiting"
	until timed_ok || [ "$(date +%s)" -ge "$CUT" ]; do stopped "the timed legs"; sleep 30; done
	T_WAIT=$(( T_WAIT + $(date +%s) - t ))
}
play() {  # tag: plays that leg to its target, holding as needed, until the cutoff
	local tag=$1
	while [ "$(count $tag)" -lt 2000 ] && [ "$(date +%s)" -lt "$CUT" ]; do
		held
		while timed_ok; do stopped "a $tag chunk"; tchunk $tag || break; done
	done
}
report() {  # tag, label: the pooled read of a timed leg into $W/<tag>.txt
	timeout 900 bash "$R" pyrun selfplay/fastchess_report.py "$GW/$1"_s*.pgn --p1 qp3 2>&1 | sed -n '/vs the field/,$p' > "$W/$1.txt"
	echo "[q292] pooled $2 ($(count $1) finished):"; show < "$W/$1.txt"
}
play eg; report eg "PRIMARY: endgame starts, eg_critical.epd 10+0.1 (registered −4 … +12, centre +3)"
play uho; report uho "the not-harmful check: UHO 10+0.1 (registered −3 … +8)"

# ── the time accounting ──────────────────────────────────────────────────────────────────────────────────────────────────
TOT=$(( $(date +%s) - T0 ))
echo "[q292] TIME: gates $(hrs $T_BUILD) · timed legs $(hrs $T_SP)${T_PRIOR:+ (incl. ${T_PRIOR}s before a resume)} · timed work held $(hrs $T_WAIT) · total $(hrs $TOT) (this run's wall only)"
echo "[q292] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | show
rm -f "$RESUME"
echo "[q292] $(date) QUEUE 292 DONE"
