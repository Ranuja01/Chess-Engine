#!/bin/bash
# 2026-11-21 ~13:00: the s7 test build — the sweep-3 knobs (dev_notes/active/BUILD-S7-KNOBS-2026-11-21.md): CHECK_EXT_FORM,
# DECAY_FIX / DECAY_MG, QPREC_FIX, PS_SCOUT_FIX, PS_TT_PROBE_FIX, TM_ITER_CAP / TM_ITER_FLOOR, FIX_NULL_LOSS_BETA, HYG_FIX, and
# the count-only sweep-2 counters moved behind ENABLE_LMR_COUNTERS (result-identical). Every new knob defaults to 0.
# WHEN: only while no timed game runs — the owner's gaming.pause (#287's timed legs are then held), or after #287 is DONE.
# While it builds it holds #287 with q287.pause (so the owner's "done" cannot restart timed games mid-build) and removes it
# at the end. PGO as #281: a gen build, 8 training games at st=0.25, the use build. Output build_uci/chess_engine_uci_s7pgo;
# production and s6pgo are not touched.
# GATES (the build is usable only if all pass):
#  - default == g77pgo 254 / 35,922,538, game12 d10 == g77pgo (7/12 2,412,001); v1 250 / 35,310,775-778;
#  - knobs that exist in s6pgo reproduce s6pgo: PS_TT_FLAG_FIX=1 252 / 36,566,334; the mate bundle 252 / 35,107,909;
#    CHECK_EXTENSION=0 248 / 35,451,696;
#  - LIVENESS (each must differ from the default bench; recorded): CHECK_EXT_FORM 1, 2, 3, 4; DECAY_FIX 1, 3; QPREC_FIX 1, 3,
#    8; PS_SCOUT_FIX 1; PS_TT_PROBE_FIX 1; PS_TT_FLAG_FIX=1 + PS_TT_PROBE_FIX=1; FIX_NULL_LOSS_BETA 1; HYG_FIX 1, 2, 4, 8, 16.
#    Expected identical by design (recorded, not a failure): DECAY_FIX 2 (fires only at ucinewgame), HYG_FIX 2 and 8 may be
#    (rare), FIX_NULL_LOSS_BETA may be (mate-range windows only). TM_ITER_CAP is not benchable (no soft limit in bench).
#  - speed: default bench time s7 vs s6pgo, same process type, recorded (the counter guards should save a little).
# REGISTERED bench predictions (BUILD-S7 report, from the sweep-3 designs): CHECK_EXT_FORM 1 +8 … +20% nodes, 2 −4 … −0.5%,
#  3 +1 … +8%, 4 +12 … +30%; PS_SCOUT_FIX −1 … −5%; PS_TT_FLAG_FIX + PS_TT_PROBE_FIX below 36,566,334; QPREC_FIX 1 0 … +0.3%,
#  3 +0.2 … +1%, 8 +0.5 … +2%; DECAY_FIX 3 within ±2%; HYG_FIX 1 (S2-8) +0.5 … +2%; HYG_FIX 16 (S2-32) a small decrease.
# Stop: touch /mnt/e/chess_data/build_s7.stop (before the build starts).
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
P=/mnt/e/chess_data/pgo_s7
W=/mnt/e/chess_data/build_s7
G=/mnt/e/chess_data/sv2_speed/game12.txt
G77="$ND/build_uci/chess_engine_uci_g77pgo"
S6="$ND/build_uci/chess_engine_uci_s6pgo"
NEW="$ND/build_uci/chess_engine_uci_s7pgo"
Q287=/mnt/e/chess_data/q287.log
HOLD=/mnt/e/chess_data/q287.pause
GPAUSE=/mnt/e/chess_data/gaming.pause
STOP=/mnt/e/chess_data/build_s7.stop
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
mkdir -p "$W"; rm -f "$STOP"
show() { sed "s/^/[s7]   /"; }
timed_running() { pgrep -f '[f]astchess.*tc=[0-9]' >/dev/null; }
q287_done() { grep -qE 'QUEUE 287 (DONE|ABORTED)' "$Q287" 2>/dev/null || ! pgrep -f "[_]queue_287.sh" >/dev/null; }
fp() { local k=""; [ -n "$2" ] && k="setoption name Knobs value $2"; printf '%s\nbench\nquit\n' "$k" | "$1" 2>&1 | grep -oE 'solved [0-9]+/300 nodes [0-9]+ time [0-9]+|mismatch.*' | head -1; }
gfp() { local k=""; [ -n "$2" ] && k="setoption name Knobs value $2"; printf "%s\nbench $G 10\nquit\n" "$k" | "$1" 2>&1 | grep -oE 'solved [0-9]+/[0-9]+ nodes [0-9]+|mismatch.*' | head -1; }
sn() { echo "$1" | grep -oE 'solved [0-9]+/[0-9]+ nodes [0-9]+'; }   # drops the time field
echo "[s7] $(date '+%Y-%m-%d %H:%M:%S') queued; waiting for gaming.pause (or #287 DONE) with no timed game running"
until { [ -e "$GPAUSE" ] || q287_done; } && ! timed_running; do
	[ -e "$STOP" ] && { echo "[s7] $(date) stop file — BUILD S7 NOT RUN"; exit 0; }
	sleep 30
done
touch "$HOLD"; sleep 20
timed_running && { echo "[s7] ☠️ a timed game is still running — BUILD S7 NOT RUN"; rm -f "$HOLD"; exit 1; }
T0=$(date +%s)
echo "[s7] $(date '+%H:%M') building (q287.pause set: #287's timed legs stay held until the gates are done)"

# ── build ────────────────────────────────────────────────────────────────────────────────────────────────────────────────
rm -rf "$P"; mkdir -p "$P"
if OUT=chess_engine_uci_s7gen PGO=gen PGO_DIR=$P nice -n 10 bash selfplay/build_uci.sh > "$W/gen.out" 2>&1; then
	nice -n 10 "$FC" -engine cmd="$ND/build_uci/chess_engine_uci_s7gen" name=a option.OwnBook=false st=0.25 \
		-engine cmd="$ND/build_uci/chess_engine_uci_s7gen" name=b option.OwnBook=false st=0.25 \
		-openings file=$O/openings_uho.epd format=epd order=random -srand 781 -rounds 4 -games 2 -concurrency 1 > "$W/train.log" 2>&1
	OUT=chess_engine_uci_s7pgo PGO=use PGO_DIR=$P nice -n 10 bash selfplay/build_uci.sh > "$W/use.out" 2>&1 \
		|| { echo "[s7] ☠️ PGO use build failed — BUILD S7 FAILED"; grep -iE 'error' "$W/use.out" | head -12 | show; rm -f "$HOLD"; exit 1; }
else
	echo "[s7] ☠️ PGO gen build failed — BUILD S7 FAILED"; grep -iE 'error' "$W/gen.out" | head -12 | show; rm -f "$HOLD"; exit 1
fi
echo "[s7] $(date '+%H:%M') build done ($(( ($(date +%s) - T0) / 60 )) min) — the sources are free to edit"

# ── gates ────────────────────────────────────────────────────────────────────────────────────────────────────────────────
a=$(fp "$NEW" ""); b=$(fp "$G77" ""); s6=$(fp "$S6" "")
v1=$(printf 'setoption name EvalArm value 0\nbench\nquit\n' | "$NEW" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+')
ga=$(gfp "$NEW" ""); gb=$(gfp "$G77" "")
ptt=$(sn "$(fp "$NEW" "PS_TT_FLAG_FIX=1")"); mate=$(sn "$(fp "$NEW" "FIX_TT_MATE=1,FIX_MDP=1,ENABLE_NULL_MATE_CLAMP=1")"); cxo=$(sn "$(fp "$NEW" "CHECK_EXTENSION=0")")
echo "[s7] default s7 [$a] · g77pgo [$b] · s6pgo [$s6] (== 254 / 35922538; time = the speed read)"
echo "[s7] v1 [$v1] · game12 d10 s7 [$ga] == g77 [$gb] · PS_TT_FLAG_FIX=1 [$ptt] (== 252 / 36566334) · mate bundle [$mate] (== 252 / 35107909) · CHECK_EXTENSION=0 [$cxo] (== 248 / 35451696)"
OK=1
[ "$(sn "$a")" = "solved 254/300 nodes 35922538" ] && [ "$(sn "$b")" = "$(sn "$a")" ] && echo "$v1" | grep -qE 'solved 250/300 nodes 3531077[5-8]' \
	&& [ -n "$ga" ] && [ "$ga" = "$gb" ] && [ "$ptt" = "solved 252/300 nodes 36566334" ] && [ "$mate" = "solved 252/300 nodes 35107909" ] \
	&& [ "$cxo" = "solved 248/300 nodes 35451696" ] || OK=0
D=$(sn "$a")
echo "[s7] liveness (each ≠ default $D unless marked 'by design'):"
for k in CHECK_EXT_FORM=1 CHECK_EXT_FORM=2 CHECK_EXT_FORM=3 CHECK_EXT_FORM=4 DECAY_FIX=1 DECAY_FIX=2 DECAY_FIX=3 QPREC_FIX=1 QPREC_FIX=3 QPREC_FIX=8 \
	PS_SCOUT_FIX=1 PS_TT_PROBE_FIX=1 PS_TT_FLAG_FIX=1,PS_TT_PROBE_FIX=1 PS_SCOUT_FIX=1,PS_TT_PROBE_FIX=1,PS_TT_FLAG_FIX=1 FIX_NULL_LOSS_BETA=1 \
	HYG_FIX=1 HYG_FIX=2 HYG_FIX=4 HYG_FIX=8 HYG_FIX=16 HYG_FIX=31; do
	r=$(sn "$(fp "$NEW" "$k")"); g=$(gfp "$NEW" "$k")
	case $k in DECAY_FIX=2|HYG_FIX=2|HYG_FIX=8|FIX_NULL_LOSS_BETA=1) tag="by design may equal" ;; *) tag="must differ" ;; esac
	if [ "$r" = "$D" ]; then st="SAME"; [ "$tag" = "must differ" ] && OK=0; else st="live"; fi
	echo "[s7]   $k: WAC [$r] · game12 [$g] — $st ($tag)"
done
rm -f "$HOLD"
if [ "$OK" = 1 ]; then echo "[s7] ✅ every gate passed — s7pgo usable"; else echo "[s7] ☠️ a gate failed — s7pgo NOT usable"; fi
echo "[s7] $(date) q287.pause removed · BUILD S7 DONE ($(( ($(date +%s) - T0) / 60 )) min)"
