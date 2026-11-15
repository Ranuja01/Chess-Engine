#!/bin/bash
# 2026-11-14 ~21:05, queue #241: D7 (SV2_VERIFY_DEEPER=1) after its self-play read passed the registered bar. #238 pooled
# 1,235 g at 10+0.1 vs production g73spgo: +37.7 (95% +21.2 … +54.3), 0 forfeits, time per move level (0.240 vs 0.242 s),
# nodes per move -4.5%. Binary build_uci/chess_engine_uci_d7pgo (#238's PGO; nothing is built here). Waits for
# /mnt/e/chess_data/q241.go (written when the owner's evening gaming is over), then:
#  1 self-play: the rest of 2,000 (seed 2411, 383 pairs) vs production, pooled with #238's d7a + d7b.
#  2 Mediocre at 10+0.1 (timed), 300 g per arm, fresh seeds 241311-241312 (150 g per chunk, arm order rotating).
#  3 SF18 at equal nodes (node-limited: ours 250k, SF18 1,100), 250 g per arm per seed, fresh seeds 241111-241116, arm order
#    rotating by seed. Node-limited, so safe to run into the owner's Sunday gaming.
#  Timed steps (1, 2) start only before 12:00 tomorrow (weekend afternoon gaming); SF18 has no cutoff.
#  Arms: ctl = no knobs (production: 249 / 37,279,382) · d7 = SV2_VERIFY_DEEPER=1 (250 / 37,320,382).
# REGISTERED before any game (ship rule: self-play >= +2 sigma, SF18 > -1 sigma, Mediocre not negative):
#   self-play pooled 2,000: >= +2 sigma stays a pass (an early lead shrinks; +15 to +30 is the expected landing).
#   SF18: d7 - ctl z > -1 = pass. Mediocre: d7 - ctl >= 0 pp = pass.
#   All pass = D7 is a ship candidate for the owner (the shipped default bench must change to 250 / 37,320,382).
#   Prediction: SF18 0 to +3 pp, Mediocre 0 to +4 pp.
# Stop: touch /mnt/e/chess_data/q241.stop (checked between chunks). Never edit this file or the binaries while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
SF18="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/stockfish_18_linux/stockfish-ubuntu-x86-64-avx2"
MED="$ND/selfplay/mediocre_uci.sh"
O=/mnt/e/chess_data/fastchess
W=/mnt/e/chess_data/r7a
BIN="$ND/build_uci/chess_engine_uci_d7pgo"
PROD="$ND/build_uci/chess_engine_uci"
STOP=/mnt/e/chess_data/q241.stop
GO=/mnt/e/chess_data/q241.go
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"
show() { sed "s/^/[q241]   /"; }
busy() { pgrep -x fastchess-linux-x86-64 >/dev/null || pgrep -f "sts_test|tactical_test|_ebf_per_iter|vs_sf.py" >/dev/null; }
kn() { case $1 in ctl) echo "" ;; d7) echo "SV2_VERIFY_DEEPER=1" ;; esac; }
fp() { local k=""; [ -n "$2" ] && k="setoption name Knobs value $2"; printf '%s\nbench\nquit\n' "$k" | "$1" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+'; }
timed_ok() { [ "$(date +%s)" -lt "$CUT" ]; }
echo "[q241] $(date '+%Y-%m-%d %H:%M:%S') queued; waiting for q241.go"
until [ -e "$GO" ]; do [ -e "$STOP" ] && { echo "[q241] stop file — QUEUE 241 DONE"; exit 0; }; sleep 30; done
START=$(date '+%Y-%m-%d %H:%M:%S')
CUT=$(date -d "$(date -d '+12 hours' +%F) 12:00" +%s); [ $(( CUT - $(date +%s) )) -gt 86400 ] && CUT=$(date -d 'today 12:00' +%s)
echo "[q241] $START go (timed steps start only before $(date -d @$CUT '+%a %H:%M'))"
busy && { echo "[q241] ☠️ a games or bench job is running — QUEUE 241 ABORTED"; exit 1; }

# ── 0 fingerprints and the knob preflight ─────────────────────────────────────────────────────────────────────────────
cmp -s "$PROD" build_uci/chess_engine_uci_g73spgo || { echo "[q241] ☠️ production is not g73spgo — QUEUE 241 ABORTED"; exit 1; }
F0=$(fp "$BIN" ""); F7=$(fp "$BIN" "SV2_VERIFY_DEEPER=1"); FP=$(fp "$PROD" "")
echo "[q241] d7pgo ctl [$F0] (want 249 / 37279382) · d7 [$F7] (want 250 / 37320382) · production [$FP]"
[ "$F0" = "solved 249/300 nodes 37279382" ] && [ "$F7" = "solved 250/300 nodes 37320382" ] && [ "$FP" = "$F0" ] || { echo "[q241] ☠️ fingerprints — QUEUE 241 ABORTED"; exit 1; }
rm -f /tmp/q241_pre.engine.log
timeout 180 "$FC" -engine "cmd=$BIN" name=d7 option.OwnBook=false option.Knobs=SV2_VERIFY_DEEPER=1 st=0.05 \
	-engine "cmd=$BIN" name=ctl option.OwnBook=false st=0.05 \
	-openings file=$O/openings_uho.epd format=epd order=random -rounds 1 -games 2 -concurrency 1 \
	-log file=/tmp/q241_pre.engine.log level=info engine=true > /tmp/q241_pre.log 2>&1
WN=$(grep -c 'have option' /tmp/q241_pre.log)
echo "[q241] preflight: $(grep -E '^Games:' /tmp/q241_pre.log | tail -1) · warnings $WN · knob echoes $(grep -cF 'SV2_VERIFY_DEEPER=1' /tmp/q241_pre.engine.log)"
[ "$WN" = 0 ] || { echo "[q241] ☠️ preflight warnings — QUEUE 241 ABORTED"; exit 1; }

COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"

# ── 1 self-play remainder ─────────────────────────────────────────────────────────────────────────────────────────────
done_games=$(( $(grep -c '^\[Result' $W/d7a.pgn) + $(grep -c '^\[Result' $W/d7b.pgn) ))
left=$(( (2000 - done_games + 1) / 2 ))
if [ -e "$STOP" ]; then :; elif timed_ok && [ $left -gt 0 ]; then
	echo "[q241] $(date '+%H:%M') self-play: $done_games done, $left pairs to go"
	nice -n 5 "$FC" -engine "cmd=$BIN" name=d7 option.OwnBook=false "option.Knobs=SV2_VERIFY_DEEPER=1" tc=10+0.1 timemargin=250 \
		-engine "cmd=$PROD" name=prod option.OwnBook=false tc=10+0.1 timemargin=250 \
		$COMMON -srand 2411 -rounds $left -games 2 -pgnout file=$W/d7c.pgn notation=san nodes=true append=false > $W/d7c.log 2>&1
	echo "[q241] d7c: $(grep -E '^Games:' $W/d7c.log | tail -1) · warnings $(grep -c 'have option' $W/d7c.log) · disconnects/stalls $(grep -ciE 'disconnect|stall|not responsive' $W/d7c.log) · time forfeits $(grep -c 'loses on time' $W/d7c.pgn)"
	echo "[q241] pooled self-play:"
	timeout 900 bash "$R" pyrun selfplay/fastchess_report.py $W/d7a.pgn $W/d7b.pgn $W/d7c.pgn --p1 d7 2>&1 | sed -n '/vs the field/,$p' | show
else
	echo "[q241] $(date) self-play remainder skipped (past the cutoff or nothing left)"
fi

rm -f $O/g18_q241_s* $O/med_q241_s*
# ── 2 Mediocre (timed) ────────────────────────────────────────────────────────────────────────────────────────────────
med_chunk() {  # seed, arm
	local seed=$1 a=$2 k f=$O/med_q241_s$1_$2.log
	[ -e "$STOP" ] && return
	timed_ok || { echo "[q241] $(date) skip Mediocre $seed $a (past the cutoff)"; return; }
	k=$(kn $a); [ -n "$k" ] && k="option.Knobs=$k"
	nice -n 5 "$FC" -engine "cmd=$BIN" name=ours option.OwnBook=false $k tc=10+0.1 timemargin=250 \
		-engine "cmd=$MED" name=mediocre tc=10+0.1 timemargin=1000 \
		$COMMON -srand $seed -rounds 75 -games 2 -pgnout file=$O/med_q241_s${seed}_$a.pgn notation=san append=false > "$f" 2>&1
	echo "[q241] $(date) Mediocre seed $seed $a: $(grep -E '^Games:' "$f" | tail -1) · warnings $(grep -c 'have option' "$f") · disconnects/stalls $(grep -ciE 'disconnect|stall|not responsive' "$f") · time losses $(grep -c 'loses on time' "$f")"
}
med_chunk 241311 ctl; med_chunk 241311 d7
med_chunk 241312 d7; med_chunk 241312 ctl

# ── 3 SF18 at equal nodes (node-limited) ──────────────────────────────────────────────────────────────────────────────
sf_chunk() {  # seed, arm
	local seed=$1 a=$2 k f=$O/g18_q241_s$1_$2.log
	[ -e "$STOP" ] && return
	k=$(kn $a); [ -n "$k" ] && k="option.Knobs=$k"
	nice -n 5 "$FC" -engine "cmd=$BIN" name=ours option.OwnBook=false $k nodes=250000 \
		-engine "cmd=$SF18" name=sf18 option.Threads=1 option.Hash=16 nodes=1100 \
		-each tc=inf $COMMON -srand $seed -rounds 125 -games 2 -pgnout file=$O/g18_q241_s${seed}_$a.pgn notation=san nodes=true append=false > "$f" 2>&1
	echo "[q241] $(date) SF18 seed $seed $a: $(grep -E '^Games:' "$f" | tail -1) · warnings $(grep -c 'have option' "$f") · disconnects/stalls $(grep -ciE 'disconnect|stall|not responsive' "$f")"
}
for s in 1 2 3 4 5 6; do
	if [ $((s % 2)) = 1 ]; then sf_chunk $((241110 + s)) ctl; sf_chunk $((241110 + s)) d7
	else sf_chunk $((241110 + s)) d7; sf_chunk $((241110 + s)) ctl; fi
done

# ── reads ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
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
for f in $O/g18_q241_s*_*.log; do [ -e "$f" ] || continue; b=$(basename $f .log); b=${b#g18_q241_s}; row ${b%%_*} ${b#*_} $f; done > /tmp/q241_sf.txt
for f in $O/med_q241_s*_*.log; do [ -e "$f" ] || continue; b=$(basename $f .log); b=${b#med_q241_s}; row ${b%%_*} ${b#*_} $f; done > /tmp/q241_med.txt
echo "[q241] $(date) per chunk (seed arm games W L D):"; { sed 's/^/Mediocre /' /tmp/q241_med.txt; sed 's/^/SF18 /' /tmp/q241_sf.txt; } | show
echo "[q241] REGISTERED READS (SF18: d7 - ctl z > -1 = pass; Mediocre: d7 - ctl >= 0 = pass):"
{ cmpr d7 ctl /tmp/q241_sf.txt SF18; cmpr d7 ctl /tmp/q241_med.txt Mediocre; } | show
echo "[q241] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | show
echo "[q241] $(date) QUEUE 241 DONE"
