#!/bin/bash
# 2026-11-12 ~10:35, queue #211 (owner 11-12: "we can continue with the tests this morning"): the two remaining ship gates for
# SEE_PER_MOVE=3 as gen 7.3. Self-play already passed: #209 + #210 combined, 2,800 g at 10+0.1, +12.7 (95% +1.2 … +24.2).
# Binary build_uci/sv2_pgo1pe (PGO of the current sources, #209; nothing is built here). Both arms run production's search
# (SEARCH_ARM=0, the default):
#   ctl  = no knobs (gen 7.2: 247 / 39,861,964)
#   see3 = SEE_PER_MOVE=3 (per-move exchange value in capture ordering and the qsearch filter: 249 / 37,279,382)
#  1 Mediocre at 10+0.1 (timed), 300 g per arm, fresh seeds 211311-211312 (150 g per chunk, arm order rotating). Timed chunks
#    start only before 19:00 today (the owner's evening games).
#  2 SF18 at equal nodes (node-limited: ours 250k, SF18 1,100), 250 g per arm per seed, fresh seeds 211111-211116
#    (1,500 g per arm), arm order rotating by seed.
# REGISTERED before any game (ship rule: self-play >= +2 sigma [passed], SF18 > -1 sigma, Mediocre not negative):
#   SF18: see3 - ctl > -1 sigma over the six seeds (z > -1) = pass.
#   Mediocre: see3 - ctl >= 0 pp = pass.
#   Both pass = SEE_PER_MOVE=3 is a ship candidate for the owner (gen 7.3: the shipped default bench must change to
#   249 / 37,279,382). Either fails = report, no ship. Prediction: SF18 0 to +3 pp, Mediocre 0 to +4 pp.
# Stop: touch /mnt/e/chess_data/q211.stop (checked between chunks). Never edit this file or the binary while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
SF18="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/stockfish_18_linux/stockfish-ubuntu-x86-64-avx2"
MED="$ND/selfplay/mediocre_uci.sh"
O=/mnt/e/chess_data/fastchess
BIN="$ND/build_uci/sv2_pgo1pe"
STOP=/mnt/e/chess_data/q211.stop
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
EVE=$(date -d '2026-11-12 19:00' +%s)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"
show() { sed "s/^/[q211]   /"; }
busy() { pgrep -f "fastchess-linux-x86-64/fastchess|vs_sf.py|sts_test|tactical_test|_ebf_per_iter" >/dev/null; }
kn() { case $1 in ctl) echo "" ;; see3) echo "SEE_PER_MOVE=3" ;; esac; }
fp() { local k=""; [ -n "$1" ] && k="setoption name Knobs value $1"; printf '%s\nbench\nquit\n' "$k" | "$BIN" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+'; }
START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[q211] $START start"
busy && { echo "[q211] ☠️ a games or bench job is running — QUEUE 211 ABORTED"; exit 1; }

# ── 0 fingerprints and the knob preflight ─────────────────────────────────────────────────────────────────────────────
F0=$(fp ""); F3=$(fp "SEE_PER_MOVE=3")
echo "[q211] ctl [$F0] (want 247 / 39861964) · see3 [$F3] (want 249 / 37279382)"
[ "$F0" = "solved 247/300 nodes 39861964" ] && [ "$F3" = "solved 249/300 nodes 37279382" ] || { echo "[q211] ☠️ fingerprints — QUEUE 211 ABORTED"; exit 1; }
rm -f /tmp/q211_pre.engine.log
timeout 180 "$FC" -engine "cmd=$BIN" name=see3 option.OwnBook=false option.Knobs=SEE_PER_MOVE=3 st=0.05 \
	-engine "cmd=$BIN" name=ctl option.OwnBook=false st=0.05 \
	-openings file=$O/openings_uho.epd format=epd order=random -rounds 1 -games 2 -concurrency 1 \
	-log file=/tmp/q211_pre.engine.log level=info engine=true > /tmp/q211_pre.log 2>&1
W=$(grep -c 'have option' /tmp/q211_pre.log)
echo "[q211] preflight: $(grep -E '^Games:' /tmp/q211_pre.log | tail -1) · warnings $W · knob echoes $(grep -cF 'SEE_PER_MOVE=3' /tmp/q211_pre.engine.log)"
[ "$W" = 0 ] || { echo "[q211] ☠️ preflight warnings — QUEUE 211 ABORTED"; exit 1; }

COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
rm -f $O/g18_q211_s* $O/med_q211_s*

# ── 1 Mediocre (timed) ────────────────────────────────────────────────────────────────────────────────────────────────
med_chunk() {  # seed, arm
	local seed=$1 a=$2 k f=$O/med_q211_s$1_$2.log
	[ -e "$STOP" ] && return
	[ $(date +%s) -lt $EVE ] || { echo "[q211] $(date) skip Mediocre $seed $a (past 19:00)"; return; }
	k=$(kn $a); [ -n "$k" ] && k="option.Knobs=$k"
	nice -n 5 "$FC" -engine "cmd=$BIN" name=ours option.OwnBook=false $k tc=10+0.1 timemargin=250 \
		-engine "cmd=$MED" name=mediocre tc=10+0.1 timemargin=1000 \
		$COMMON -srand $seed -rounds 75 -games 2 -pgnout file=$O/med_q211_s${seed}_$a.pgn notation=san append=false > "$f" 2>&1
	echo "[q211] $(date) Mediocre seed $seed $a: $(grep -E '^Games:' "$f" | tail -1) · warnings $(grep -c 'have option' "$f") · disconnects/stalls $(grep -ciE 'disconnect|stall|not responsive' "$f") · time losses $(grep -c 'loses on time' "$f")"
}
med_chunk 211311 ctl; med_chunk 211311 see3
med_chunk 211312 see3; med_chunk 211312 ctl

# ── 2 SF18 at equal nodes (node-limited) ──────────────────────────────────────────────────────────────────────────────
sf_chunk() {  # seed, arm
	local seed=$1 a=$2 k f=$O/g18_q211_s$1_$2.log
	[ -e "$STOP" ] && return
	k=$(kn $a); [ -n "$k" ] && k="option.Knobs=$k"
	nice -n 5 "$FC" -engine "cmd=$BIN" name=ours option.OwnBook=false $k nodes=250000 \
		-engine "cmd=$SF18" name=sf18 option.Threads=1 option.Hash=16 nodes=1100 \
		-each tc=inf $COMMON -srand $seed -rounds 125 -games 2 -pgnout file=$O/g18_q211_s${seed}_$a.pgn notation=san nodes=true append=false > "$f" 2>&1
	echo "[q211] $(date) SF18 seed $seed $a: $(grep -E '^Games:' "$f" | tail -1) · warnings $(grep -c 'have option' "$f") · disconnects/stalls $(grep -ciE 'disconnect|stall|not responsive' "$f")"
}
for s in 1 2 3 4 5 6; do
	if [ $((s % 2)) = 1 ]; then sf_chunk $((211110 + s)) ctl; sf_chunk $((211110 + s)) see3
	else sf_chunk $((211110 + s)) see3; sf_chunk $((211110 + s)) ctl; fi
done

# ── reads ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
row() {  # seed, arm, log: complete chunks only
	grep -E '^Games:' "$3" 2>/dev/null | tail -1 | awk -v s=$1 -v a=$2 '{ gsub(/,/, ""); for (i = 1; i <= NF; i++) { if ($i == "Games:") n = $(i+1); if ($i == "Wins:") w = $(i+1); if ($i == "Losses:") l = $(i+1); if ($i == "Draws:") d = $(i+1) } if (n > 0) print s, a, n, w, l, d }'
}
cmp() {  # A, B, rows file, label: A - B over the seeds both played
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
for f in $O/g18_q211_s*_*.log; do [ -e "$f" ] || continue; b=$(basename $f .log); b=${b#g18_q211_s}; row ${b%%_*} ${b#*_} $f; done > /tmp/q211_sf.txt
for f in $O/med_q211_s*_*.log; do [ -e "$f" ] || continue; b=$(basename $f .log); b=${b#med_q211_s}; row ${b%%_*} ${b#*_} $f; done > /tmp/q211_med.txt
echo "[q211] $(date) per chunk (seed arm games W L D):"; { cat /tmp/q211_med.txt | sed 's/^/Mediocre /'; cat /tmp/q211_sf.txt | sed 's/^/SF18 /'; } | show
echo "[q211] REGISTERED READS (SF18: see3 - ctl z > -1 = pass; Mediocre: see3 - ctl >= 0 = pass):"
{ cmp see3 ctl /tmp/q211_sf.txt SF18; cmp see3 ctl /tmp/q211_med.txt Mediocre; } | show
echo "[q211] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | show
echo "[q211] $(date) QUEUE 211 DONE"
