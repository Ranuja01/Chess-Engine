#!/bin/bash
# 2026-11-18 ~13:00, queue #274: the qsearch STAND-PAT SEED (ENABLE_QSTANDPAT_SEED=1) on gen 7.6, the full ship-rule set, plus
# LAZY-EVAL W1 (the re-run owed after D7, C2c and PROMOTE). Chained after #273. Owner 11-18: "ok" to the seed before C7.
# Why: #273 counted search v2 qsearch nodes that return the mated sentinel because every capture was delta-pruned and no seed
# replaced it: 9.74% of qsearch visits on the WAC bench, 5.46% on game12 d10 (seed: 0). Each hands its parent a mate-sized
# fail-high; the size is stored (q-cache, TT bounds) and read by VERIFY's margin. The seed: WAC bench 254 / 35,922,538 (base 246 /
# 35,138,434, +2.2% nodes), game12 d10 7/12 at +6.2% nodes. Tested only in v1 (08-18: +4% nodes, never in games).
#  1 PGO test build chess_engine_uci_s18pgo from the current sources (gen 7.6 + count-only counters). GATES: default 246 /
#    35,138,434; seed 254 / 35,922,538; LAZY_V2=0 must move the bench; v1 250 / 35,310,775-778; game12 d10 8/12 2,272,158.
#  2 Self-play seed vs default, 10+0.1, 150-pair chunks, seeds 2741+, to 2,000 games. No timed work 20:50 to 00:15 unless
#    q274.go (the owner's good night); q274.pause holds; cutoff tomorrow 10:40.
#  3 SF18 at equal nodes (ours 250k, SF18 1,100), node-limited (safe beside the owner's games): ctl, seed, lazy0 (LAZY_V2=0) on
#    the same openings, seeds 274111-274116, 250 g each.
#  4 The rest of the self-play, then Mediocre 10+0.1 ctl + seed, seeds 274311-274312 (timed).
# REGISTERED before any game:
#  - seed (ship rule): self-play lower 95% bound > 0 = pass, within ±2σ = NEUTRAL; SF18 seed − ctl z > −1; Mediocre seed − ctl
#    >= 0 pp. Prediction self-play −10 … +25.
#  - W1 (LAZY-EVAL.md §7): lazy eval stays if ctl − lazy0 z > −1 on SF18. Prediction −1.5 … +1.5 pp.
# Stop: touch /mnt/e/chess_data/q274.stop (between steps). Never edit this file, the binaries or the sources before "build done".
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
SF18="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/stockfish_18_linux/stockfish-ubuntu-x86-64-avx2"
MED="$ND/selfplay/mediocre_uci.sh"
O=/mnt/e/chess_data/fastchess
W=/mnt/e/chess_data/q274
GW=/mnt/e/chess_data/r18b
P=/mnt/e/chess_data/pgo_s18
G=/mnt/e/chess_data/sv2_speed/game12.txt
NEW="$ND/build_uci/chess_engine_uci_s18pgo"
KS="ENABLE_QSTANDPAT_SEED=1"
LOG=/mnt/e/chess_data/q274.log
STOP=/mnt/e/chess_data/q274.stop
PAUSE=/mnt/e/chess_data/q274.pause
GO=/mnt/e/chess_data/q274.go
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP" "$PAUSE" "$GO"; mkdir -p "$W" "$GW"
show() { sed "s/^/[q274]   /"; }
stopped() { [ -e "$STOP" ] && { echo "[q274] $(date) stop file before $1 — QUEUE 274 DONE"; exit 0; }; }
kn() { case $1 in ctl) echo "" ;; seed) echo "$KS" ;; lazy0) echo "LAZY_V2=0" ;; esac; }
fp() { local k=""; [ -n "$2" ] && k="setoption name Knobs value $2"; printf '%s\nbench\nquit\n' "$k" | "$1" 2>&1 | grep -oE 'solved [0-9]+/300 nodes [0-9]+|mismatch.*' | head -1; }
busy() { pgrep -f "[f]astchess-linux-x86-64/fastchess" >/dev/null || pgrep -f "[s]ts_test|[t]actical_test|[v]algrind|[_]ks_footprint|[b]uild_uci.sh" >/dev/null; }
timed_ok() { local t; t=$(date +%s); [ ! -e "$PAUSE" ] && [ "$t" -lt "$CUT" ] && ! { [ "$t" -ge "$GS" ] && [ "$t" -lt "$GE" ] && [ ! -e "$GO" ]; }; }
guard() { ( until ! timed_ok; do sleep 15; done; pkill -INT -f "[f]astchess-linux-x86-64/fastchess" && echo "[q274] $(date '+%H:%M') guard: games interrupted" ) & GUARD=$!; }
finished() { local c; c=$(grep -cE '^\[Result "(1-0|0-1|1/2-1/2)"\]' "$1" 2>/dev/null); echo "${c:-0}"; }
sp_count() { local n=0 f; for f in "$GW/$1"_s*.pgn; do [ -e "$f" ] && n=$((n + $(finished "$f"))); done; echo $n; }
echo "[q274] $(date '+%Y-%m-%d %H:%M:%S') queued; waiting for #273"
until grep -q 'QUEUE 273' /mnt/e/chess_data/q273.log 2>/dev/null || ! pgrep -f "[_]queue_273.sh" >/dev/null; do stopped "the start"; sleep 30; done
sleep 20
busy && { echo "[q274] ☠️ another job is running — QUEUE 274 ABORTED"; exit 1; }
START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[q274] $START start"

# ── 1 build + gates ──────────────────────────────────────────────────────────────────────────────────────────────────────
rm -rf "$P"; mkdir -p "$P"
if OUT=chess_engine_uci_s18gen PGO=gen PGO_DIR=$P nice -n 10 bash selfplay/build_uci.sh > "$W/gen.out" 2>&1; then
	nice -n 10 "$FC" -engine cmd="$ND/build_uci/chess_engine_uci_s18gen" name=a option.OwnBook=false st=0.25 \
		-engine cmd="$ND/build_uci/chess_engine_uci_s18gen" name=b option.OwnBook=false st=0.25 \
		-openings file=$O/openings_uho.epd format=epd order=random -srand 774 -rounds 4 -games 2 -concurrency 1 > "$W/train.log" 2>&1
	OUT=chess_engine_uci_s18pgo PGO=use PGO_DIR=$P nice -n 10 bash selfplay/build_uci.sh > "$W/use.out" 2>&1 || { echo "[q274] ☠️ PGO use build failed — QUEUE 274 ABORTED"; grep -iE 'error' "$W/use.out" | head -8 | show; exit 1; }
else
	echo "[q274] ☠️ PGO gen build failed — QUEUE 274 ABORTED"; grep -iE 'error' "$W/gen.out" | head -8 | show; exit 1
fi
echo "[q274] $(date '+%H:%M') build done (chess_engine_uci_s18pgo) — the sources are free to edit"
a=$(fp "$NEW" ""); sd=$(fp "$NEW" "$KS"); lz=$(fp "$NEW" "LAZY_V2=0")
v1=$(printf 'setoption name EvalArm value 0\nbench\nquit\n' | "$NEW" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+')
gb=$(printf "bench $G 10\nquit\n" | "$NEW" 2>&1 | grep -oE 'solved [0-9]+/[0-9]+ nodes [0-9]+' | head -1)
echo "[q274] s18pgo default [$a] (246 / 35138434) · seed [$sd] (254 / 35922538) · LAZY_V2=0 [$lz] (must differ) · v1 [$v1] · game12 d10 [$gb] (8/12 2272158)"
[ "$a" = "solved 246/300 nodes 35138434" ] && [ "$sd" = "solved 254/300 nodes 35922538" ] && [ -n "$lz" ] && [ "$lz" != "$a" ] \
	&& echo "$v1" | grep -qE 'solved 250/300 nodes 3531077[5-8]' && [ "$gb" = "solved 8/12 nodes 2272158" ] || { echo "[q274] ☠️ a gate failed — no games — QUEUE 274 ABORTED"; exit 1; }
echo "[q274] ✅ every gate equal"
rm -f /tmp/q274_pre.engine.log
timeout 180 "$FC" -engine "cmd=$NEW" name=seed option.OwnBook=false "option.Knobs=$KS" st=0.05 \
	-engine "cmd=$NEW" name=ctl option.OwnBook=false st=0.05 \
	-openings file=$O/openings_uho.epd format=epd order=random -rounds 1 -games 2 -concurrency 1 \
	-log file=/tmp/q274_pre.engine.log level=info engine=true > /tmp/q274_pre.log 2>&1
WN=$(grep -c 'have option' /tmp/q274_pre.log)
echo "[q274] preflight: $(grep -E '^Games:' /tmp/q274_pre.log | tail -1) · warnings $WN · echoes $KS $(grep -cF "$KS" /tmp/q274_pre.engine.log)"
[ "$WN" = 0 ] || { echo "[q274] ☠️ preflight warnings — no games — QUEUE 274 DONE"; exit 1; }

# ── 2 self-play ──────────────────────────────────────────────────────────────────────────────────────────────────────────
if [ "$(date +%H)" -ge 12 ]; then
	GS=$(date -d 'today 20:50' +%s); GE=$(date -d "$(date -d tomorrow +%F) 00:15" +%s); CUT=$(date -d "$(date -d tomorrow +%F) 10:40" +%s)
else
	GS=$(date +%s); GE=$GS; CUT=$(date -d 'today 10:40' +%s)
fi
echo "[q274] $(date '+%H:%M') self-play seed vs gen 7.6 (no timed work $(date -d @$GS '+%a %H:%M') to $(date -d @$GE '+%a %H:%M') unless q274.go; cutoff $(date -d @$CUT '+%a %H:%M'))"
COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
SEQ=0
sp_chunk() {  # arm; returns 1 when the arm already has 2,000
	local a=$1 n left pairs seed f
	n=$(sp_count $a); left=$(( (2000 - n + 1) / 2 )); [ $left -le 0 ] && return 1
	pairs=$(( left < 150 ? left : 150 )); SEQ=$((SEQ + 1)); seed=$((2740 + SEQ)); f=$GW/${a}_s$seed
	echo "[q274] $(date '+%H:%M') self-play $a: $n finished, chunk seed $seed, $pairs pairs"
	guard
	nice -n 5 "$FC" -engine "cmd=$NEW" name=$a option.OwnBook=false "option.Knobs=$(kn $a)" tc=10+0.1 timemargin=250 \
		-engine "cmd=$NEW" name=g76 option.OwnBook=false tc=10+0.1 timemargin=250 \
		$COMMON -srand $seed -rounds $pairs -games 2 -pgnout file=$f.pgn notation=san nodes=true append=false > $f.log 2>&1
	kill $GUARD 2>/dev/null
	echo "[q274] $(date '+%H:%M') $a s$seed: $(grep -E '^Games:' $f.log | tail -1) · warnings $(grep -c 'have option' $f.log) · disconnects/stalls $(grep -ciE 'disconnect|stall|not responsive' $f.log) · time forfeits $(grep -c 'loses on time' $f.pgn) · unfinished $(grep -c '^\[Result "\*"' $f.pgn)"
	return 0
}
sp_loop() { while timed_ok; do stopped "a self-play chunk"; sp_chunk seed || break; done; }
pooled() {
	[ "$(sp_count seed)" -gt 0 ] || return
	echo "[q274] pooled self-play seed vs gen 7.6 ($(sp_count seed) finished):"
	timeout 900 bash "$R" pyrun selfplay/fastchess_report.py "$GW/seed"_s*.pgn --p1 seed 2>&1 | sed -n '/vs the field/,$p' | show
}
sp_loop
pooled

# ── 3 SF18 at equal nodes (node-limited) ─────────────────────────────────────────────────────────────────────────────────
rm -f $O/g18_q274_s* $O/med_q274_s*
sf_chunk() {  # seed, arm
	local seed=$1 a=$2 k f=$O/g18_q274_s$1_$2.log
	stopped "SF18 $seed $a"
	k=$(kn $a); [ -n "$k" ] && k="option.Knobs=$k"
	nice -n 5 "$FC" -engine "cmd=$NEW" name=ours option.OwnBook=false $k nodes=250000 \
		-engine "cmd=$SF18" name=sf18 option.Threads=1 option.Hash=16 nodes=1100 \
		-each tc=inf $COMMON -srand $seed -rounds 125 -games 2 -pgnout file=$O/g18_q274_s${seed}_$a.pgn notation=san nodes=true append=false > "$f" 2>&1
	echo "[q274] $(date '+%H:%M') SF18 seed $seed $a: $(grep -E '^Games:' "$f" | tail -1) · warnings $(grep -c 'have option' "$f") · disconnects/stalls $(grep -ciE 'disconnect|stall|not responsive' "$f")"
}
echo "[q274] $(date '+%H:%M') SF18 legs (node-limited): ctl, seed, lazy0"
for s in 1 2 3 4 5 6; do
	case $((s % 3)) in 1) o="ctl seed lazy0" ;; 2) o="seed lazy0 ctl" ;; 0) o="lazy0 ctl seed" ;; esac
	for x in $o; do sf_chunk $((274110 + s)) $x; done
done

# ── 4 the rest of the self-play, then Mediocre ───────────────────────────────────────────────────────────────────────────
until timed_ok || [ "$(date +%s)" -ge "$CUT" ]; do stopped "the rest of the self-play"; sleep 30; done
sp_loop
pooled
med_chunk() {  # seed, arm
	local seed=$1 a=$2 k f=$O/med_q274_s$1_$2.log
	stopped "Mediocre $seed $a"
	timed_ok || { echo "[q274] $(date) skip Mediocre $seed $a (cutoff or pause)"; return; }
	k=$(kn $a); [ -n "$k" ] && k="option.Knobs=$k"
	guard
	nice -n 5 "$FC" -engine "cmd=$NEW" name=ours option.OwnBook=false $k tc=10+0.1 timemargin=250 \
		-engine "cmd=$MED" name=mediocre tc=10+0.1 timemargin=1000 \
		$COMMON -srand $seed -rounds 75 -games 2 -pgnout file=$O/med_q274_s${seed}_$a.pgn notation=san append=false > "$f" 2>&1
	kill $GUARD 2>/dev/null
	echo "[q274] $(date '+%H:%M') Mediocre seed $seed $a: $(grep -E '^Games:' "$f" | tail -1) · warnings $(grep -c 'have option' "$f") · disconnects/stalls $(grep -ciE 'disconnect|stall|not responsive' "$f") · time losses $(grep -c 'loses on time' "$f")"
}
for x in ctl seed; do med_chunk 274311 $x; done
for x in seed ctl; do med_chunk 274312 $x; done

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
for f in $O/g18_q274_s*_*.log; do [ -e "$f" ] || continue; b=$(basename $f .log); b=${b#g18_q274_s}; row ${b%%_*} ${b#*_} $f; done > /tmp/q274_sf.txt
for f in $O/med_q274_s*_*.log; do [ -e "$f" ] || continue; b=$(basename $f .log); b=${b#med_q274_s}; row ${b%%_*} ${b#*_} $f; done > /tmp/q274_med.txt
echo "[q274] $(date) per chunk (seed arm games W L D):"; { sed 's/^/Mediocre /' /tmp/q274_med.txt; sed 's/^/SF18 /' /tmp/q274_sf.txt; } | show
echo "[q274] REGISTERED READS (seed: SF18 seed - ctl z > -1, Mediocre seed - ctl >= 0; W1: SF18 ctl - lazy0 z > -1 keeps lazy eval):"
{ cmpr seed ctl /tmp/q274_sf.txt SF18; cmpr seed ctl /tmp/q274_med.txt Mediocre; cmpr ctl lazy0 /tmp/q274_sf.txt "SF18 W1"; } | show
echo "[q274] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | show
echo "[q274] $(date) QUEUE 274 DONE"
