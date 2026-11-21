#!/bin/bash
# 2026-11-21 ~11:00, queue #287: the MATE BUNDLE's targeted legs, read for INFORMATION — FIX_TT_MATE=1, FIX_MDP=1,
# ENABLE_NULL_MATE_CLAMP=1 against gen 7.7. Owner 11-21 ~10:55 ("sure"): play them after #286 although #285's self-play read
# PARKED (−7.3, 95% −18.7 … +4.1, z −1.26), because these are the venues where the bundle was expected to matter (long thinks
# such as UI game 56's 150 s move; mate conversion in won endings). They do NOT change #285's verdict; they decide whether the
# bundle is worth a re-test and where (its re-test triggers in the register).
# No rebuild: all three knobs exist in s6pgo. Chained after #286 (S2-1's off arm).
#  1 Wait for #286; production must be g77pgo; nothing else running. GATES on s6pgo (as #285): default 254 / 35,922,538 ==
#    g77pgo; the bundle 252 / 35,107,909 and game12 d10 8/12 2,252,010; v1 250 / 35,310,775-778; game12 default == g77pgo.
#  2 (a) endgame starts, eg_critical.epd (464 endgame starts), 10+0.1, 1,000 g in 150-pair chunks, seeds 287501+;
#    (b) the longer run, UHO 60+0.6, 200 g in 50-pair chunks, seeds 287601+, read for behaviour (time per move mean / p95 /
#    max, depth).
#    No timed work Sat 20:50 to Sun 00:15 unless q287.go; gaming.pause or q287.pause holds timed work (the owner may game from
#    ~14:00 today); nothing is played during a hold (the box is left to the owner and to any build step).
# REGISTERED (as in #285, written before #285's self-play; restated here before these games):
#  - endgame starts: −5 … +15 (faster mate conversion; the venue resolves ±13-14 per 1,000 g, INSTRUMENT-MAP).
#  - 60+0.6: max time per move lower for mate (max −0 … −30%), mean and p95 within ±3%, depth within ±0.1 ply.
#  Reading: a clear endgame gain or a long-think cut says the bundle's value lives where 10+0.1 UHO self-play cannot see it ⇒
#  a re-test trigger with a venue leg; both null ⇒ the bundle stays parked for the end-of-arc tuning pass.
# RESUME MODE: before relaunching after an interruption, copy q287.log to q287_part1.log, set aside any chunk that did not
# finish (unread), and write the timed seconds already played into /mnt/e/chess_data/q287.resume; complete chunks are kept,
# each leg's SEQ comes from its files, and q287.resume is removed at DONE.
# Stop: touch /mnt/e/chess_data/q287.stop (between steps). Never edit this file or the binaries while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
W=/mnt/e/chess_data/q287
GW=/mnt/e/chess_data/r287
G=/mnt/e/chess_data/sv2_speed/game12.txt
EG=$O/eg_critical.epd
PROD="$ND/build_uci/chess_engine_uci"
G77="$ND/build_uci/chess_engine_uci_g77pgo"
NEW="$ND/build_uci/chess_engine_uci_s6pgo"
KS="FIX_TT_MATE=1,FIX_MDP=1,ENABLE_NULL_MATE_CLAMP=1"
Q286=/mnt/e/chess_data/q286.log
STOP=/mnt/e/chess_data/q287.stop
PAUSE=/mnt/e/chess_data/q287.pause
GPAUSE=/mnt/e/chess_data/gaming.pause
GO=/mnt/e/chess_data/q287.go
RESUME=/mnt/e/chess_data/q287.resume
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP" "$GO"; mkdir -p "$W" "$GW"   # the pause files are the owner's: never removed here
if [ -e "$RESUME" ]; then T_PRIOR=$(tr -dc '0-9' < "$RESUME"); else T_PRIOR=""; rm -f "$GW"/*_s*.pgn "$GW"/*_s*.log; fi
show() { sed "s/^/[q287]   /"; }
stopped() { [ -e "$STOP" ] && { echo "[q287] $(date) stop file before $1 — QUEUE 287 DONE"; exit 0; }; }
fp() { local k=""; [ -n "$2" ] && k="setoption name Knobs value $2"; printf '%s\nbench\nquit\n' "$k" | "$1" 2>&1 | grep -oE 'solved [0-9]+/300 nodes [0-9]+|mismatch.*' | head -1; }
gfp() { local k=""; [ -n "$2" ] && k="setoption name Knobs value $2"; printf "%s\nbench $G 10\nquit\n" "$k" | "$1" 2>&1 | grep -oE 'solved [0-9]+/[0-9]+ nodes [0-9]+|mismatch.*' | head -1; }
busy() { pgrep -f "[f]astchess-linux-x86-64/fastchess" >/dev/null || pgrep -f "[s]ts_test|[t]actical_test|[v]algrind|[_]ks_footprint|[b]uild_uci.sh" >/dev/null; }
timed_ok() { local t; t=$(date +%s); [ ! -e "$PAUSE" ] && [ ! -e "$GPAUSE" ] && [ "$t" -lt "$CUT" ] && ! { [ "$t" -ge "$GS" ] && [ "$t" -lt "$GE" ] && [ ! -e "$GO" ]; }; }
guard() { ( until ! timed_ok; do sleep 15; done; pkill -INT -f "[f]astchess-linux-x86-64/fastchess" && echo "[q287] $(date '+%H:%M') guard: games interrupted" ) & GUARD=$!; }
finished() { local c; c=$(grep -cE '^\[Result "(1-0|0-1|1/2-1/2)"\]' "$1" 2>/dev/null); echo "${c:-0}"; }
count() { local n=0 f; for f in "$GW/$1"_s*.pgn; do [ -e "$f" ] && n=$((n + $(finished "$f"))); done; echo $n; }
hrs() { awk -v s=$1 'BEGIN { printf "%.2f h", s / 3600 }'; }
echo "[q287] $(date '+%Y-%m-%d %H:%M:%S') queued; waiting for #286"
until grep -qE 'QUEUE 286 (DONE|ABORTED)' "$Q286" 2>/dev/null || ! pgrep -f "[_]queue_286.sh" >/dev/null; do stopped "the start"; sleep 30; done
sleep 20
cmp -s "$PROD" "$G77" || { echo "[q287] ☠️ production is not g77pgo — QUEUE 287 ABORTED"; exit 1; }
[ -x "$NEW" ] || { echo "[q287] ☠️ $NEW missing — QUEUE 287 ABORTED"; exit 1; }
[ -s "$EG" ] || { echo "[q287] ☠️ $EG missing — QUEUE 287 ABORTED"; exit 1; }
busy && { echo "[q287] ☠️ another job is running — QUEUE 287 ABORTED"; exit 1; }
START=$(date '+%Y-%m-%d %H:%M:%S'); T0=$(date +%s)
echo "[q287] $START start (information legs for #285's mate bundle; ctl = gen 7.7, no knobs)"

# ── 1 gates (no rebuild) ─────────────────────────────────────────────────────────────────────────────────────────────────
a=$(fp "$NEW" ""); b=$(fp "$G77" ""); k=$(fp "$NEW" "$KS")
v1=$(printf 'setoption name EvalArm value 0\nbench\nquit\n' | "$NEW" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+')
ga=$(gfp "$NEW" ""); gb=$(gfp "$G77" ""); gk=$(gfp "$NEW" "$KS")
echo "[q287] s6pgo default [$a] == g77pgo [$b] (254 / 35922538) · bundle [$k] (== 252 / 35107909) · v1 [$v1] · game12 d10 default [$ga] == g77 [$gb] · bundle [$gk] (== 8/12 2252010)"
[ "$a" = "solved 254/300 nodes 35922538" ] && [ "$b" = "$a" ] && [ "$k" = "solved 252/300 nodes 35107909" ] \
	&& echo "$v1" | grep -qE 'solved 250/300 nodes 3531077[5-8]' && [ -n "$ga" ] && [ "$ga" = "$gb" ] \
	&& [ "$gk" = "solved 8/12 nodes 2252010" ] \
	|| { echo "[q287] ☠️ a gate failed — no games — QUEUE 287 ABORTED"; exit 1; }
echo "[q287] ✅ every gate equal"
rm -f /tmp/q287_pre.engine.log
timeout 180 "$FC" -engine "cmd=$NEW" name=mate option.OwnBook=false "option.Knobs=$KS" st=0.05 \
	-engine "cmd=$NEW" name=g77 option.OwnBook=false st=0.05 \
	-openings file=$EG format=epd order=random -rounds 1 -games 2 -concurrency 1 \
	-log file=/tmp/q287_pre.engine.log level=info engine=true > /tmp/q287_pre.log 2>&1
WN=$(grep -c 'have option' /tmp/q287_pre.log)
echo "[q287] preflight: $(grep -E '^Games:' /tmp/q287_pre.log | tail -1) · warnings $WN · echoes $KS $(grep -cF "$KS" /tmp/q287_pre.engine.log)"
[ "$WN" = 0 ] || { echo "[q287] ☠️ preflight warnings — no games — QUEUE 287 DONE"; exit 1; }
T_BUILD=$(( $(date +%s) - T0 ))

COMMON="-repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
UHO="-openings file=$O/openings_uho.epd format=epd order=random"

# ── 2 timed legs: endgame starts (10+0.1), the longer run (UHO 60+0.6) ────────────────────────────────────────────────────
# the next evening block (or the one in progress before 00:15), and the cutoff 10:40 the morning after it
NOW=$(date +%s); GS=$(date -d 'today 20:50' +%s)
[ "$NOW" -lt "$(date -d 'today 00:15' +%s)" ] && GS=$(date -d 'yesterday 20:50' +%s)
GE=$(( GS + 12300 )); CUT=$(( GE + 37500 ))
echo "[q287] $(date '+%H:%M') timed legs (no timed work $(date -d @$GS '+%a %H:%M') to $(date -d @$GE '+%a %H:%M') unless q287.go; q287.pause or gaming.pause holds; cutoff $(date -d @$CUT '+%a %H:%M'))"
T_SP=${T_PRIOR:-0}; T_WAIT=0
[ -e "$RESUME" ] && echo "[q287] RESUMED: kept endgame $(count eg) · 60+0.6 $(count long) games, ${T_PRIOR}s of timed play already (part 1 log q287_part1.log)"
tchunk() {  # tag (eg | long): plays the next chunk of that leg; returns 1 when the leg has its games
	local tag=$1 target book tc maxp base n left pairs seq seed f t
	case $tag in
		eg)   target=1000; book="-openings file=$EG format=epd order=random"; tc=10+0.1; maxp=150; base=287500 ;;
		long) target=200; book="$UHO"; tc=60+0.6; maxp=50; base=287600 ;;
	esac
	n=$(count $tag); left=$(( (target - n + 1) / 2 )); [ $left -le 0 ] && return 1
	pairs=$(( left < maxp ? left : maxp )); seq=$(( $(ls "$GW/${tag}"_s*.pgn 2>/dev/null | wc -l) + 1 )); seed=$((base + seq)); f=$GW/${tag}_s$seed
	echo "[q287] $(date '+%H:%M') $tag: $n finished, chunk seed $seed, $pairs pairs, $tc"
	guard; t=$(date +%s)
	nice -n 5 "$FC" -engine "cmd=$NEW" name=mate option.OwnBook=false "option.Knobs=$KS" tc=$tc timemargin=250 \
		-engine "cmd=$NEW" name=g77 option.OwnBook=false tc=$tc timemargin=250 \
		$book $COMMON -srand $seed -rounds $pairs -games 2 -pgnout file=$f.pgn notation=san nodes=true append=false > $f.log 2>&1
	kill $GUARD 2>/dev/null; T_SP=$(( T_SP + $(date +%s) - t ))
	echo "[q287] $(date '+%H:%M') $tag s$seed: $(grep -E '^Games:' $f.log | tail -1) · warnings $(grep -c 'have option' $f.log) · disconnects/stalls $(grep -ciE 'disconnect|stall|not responsive' $f.log) · time forfeits $(grep -c 'loses on time' $f.pgn) · unfinished $(grep -c '^\[Result "\*"' $f.pgn)"
	return 0
}
held() {  # waits out a hold on timed work; nothing is played meanwhile
	local t
	t=$(date +%s)
	timed_ok || echo "[q287] $(date '+%H:%M') timed work held (gaming.pause, q287.pause or the evening block); waiting"
	until timed_ok || [ "$(date +%s)" -ge "$CUT" ]; do stopped "the timed legs"; sleep 30; done
	T_WAIT=$(( T_WAIT + $(date +%s) - t ))
}
play() {  # tag: plays that leg to its target, holding as needed, until the cutoff
	local tag=$1 target
	case $tag in eg) target=1000 ;; long) target=200 ;; esac
	while [ "$(count $tag)" -lt "$target" ] && [ "$(date +%s)" -lt "$CUT" ]; do
		held
		while timed_ok; do stopped "a $tag chunk"; tchunk $tag || break; done
	done
}
report() {  # tag, label: the pooled read of a timed leg into $W/<tag>.txt
	timeout 900 bash "$R" pyrun selfplay/fastchess_report.py "$GW/$1"_s*.pgn --p1 mate 2>&1 | sed -n '/vs the field/,$p' > "$W/$1.txt"
	echo "[q287] pooled $2 ($(count $1) finished):"; show < "$W/$1.txt"
}
play eg; report eg "endgame starts, eg_critical.epd 10+0.1 (registered −5 … +15)"
play long; report long "the longer run, UHO 60+0.6 (registered: max time per move −0 … −30% for mate; mean, p95 ±3%; depth ±0.1)"

# ── the time accounting ──────────────────────────────────────────────────────────────────────────────────────────────────
TOT=$(( $(date +%s) - T0 ))
echo "[q287] TIME: gates $(hrs $T_BUILD) · timed legs $(hrs $T_SP)${T_PRIOR:+ (incl. ${T_PRIOR}s before a resume)} · timed work held $(hrs $T_WAIT) · total $(hrs $TOT) (this run's wall only)"
echo "[q287] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | show
rm -f "$RESUME"
echo "[q287] $(date) QUEUE 287 DONE"
