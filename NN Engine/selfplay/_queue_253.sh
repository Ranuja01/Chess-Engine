#!/bin/bash
# 2026-11-15 ~19:00, queue #253: PROMOTE's external legs. #251 pooled self-play vs gen 7.4 (prom_a seed 2481 + prom_b seed 2482,
# 2,001 g, 1,000 pairs): +800 −723 =478, **+13.6 (95% +0.5 … +26.7)**, 0 forfeits, 0 unfinished; per move 0.243 vs 0.244 s,
# depth 11.5 vs 11.6, nodes −0.2% ⇒ passes the registered self-play bar (lower bound > 0). Starts when #252 ends; the control
# arm is #252's ctl (production), so PROMOTE plays the SAME seeds and is compared with those chunks.
#  1 SF18 at equal nodes (node-limited): ours 250k, SF18 1,100, 250 g per seed, seeds 252111-252116 (as #252).
#  2 Mediocre at 10+0.1 (timed, only before the cutoff and outside the owner's evening): 150 g per seed, seeds 252311-252312.
# REGISTERED before any game (ship rule): SF18 prom − ctl z > −1 = pass; Mediocre prom − ctl >= 0 pp = pass. Both pass = a ship
# candidate for the owner (shipped preset SV2_VERIFY_PROMOTE 1; the default bench must change to 251 / 37,302,143).
# Prediction: SF18 −1 to +2 pp, Mediocre 0 to +3 pp.
# Stop: touch /mnt/e/chess_data/q253.stop (between chunks). Never edit this file or the binaries while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
SF18="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/stockfish_18_linux/stockfish-ubuntu-x86-64-avx2"
MED="$ND/selfplay/mediocre_uci.sh"
O=/mnt/e/chess_data/fastchess
PROD="$ND/build_uci/chess_engine_uci"
G74="$ND/build_uci/chess_engine_uci_g74pgo"
KP="SV2_VERIFY_PROMOTE=1"
STOP=/mnt/e/chess_data/q253.stop
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"
show() { sed "s/^/[q253]   /"; }
stopped() { [ -e "$STOP" ] && { echo "[q253] $(date) stop file before $1 — QUEUE 253 DONE"; exit 0; }; }
fp() { local k=""; [ -n "$2" ] && k="setoption name Knobs value $2"; printf '%s\nbench\nquit\n' "$k" | "$1" 2>&1 | grep -oE 'solved [0-9]+/300 nodes [0-9]+|mismatch.*' | head -1; }
busy() { pgrep -f "[f]astchess-linux-x86-64/fastchess" >/dev/null || pgrep -f "[s]ts_test|[t]actical_test|[v]algrind|[_]ks_footprint" >/dev/null; }
timed_ok() { local t h; t=$(date +%s); h=$(date +%H%M); [ "$t" -lt "$CUT" ] && [ "$h" -lt 2050 ] && [ "$h" -ge 0015 ]; }
guard() { ( until ! timed_ok; do sleep 15; done; pkill -INT -f "[f]astchess-linux-x86-64/fastchess" && echo "[q253] $(date '+%H:%M') guard: games interrupted" ) & GUARD=$!; }

echo "[q253] $(date '+%Y-%m-%d %H:%M:%S') queued; waiting for #252 to end"
until grep -qE 'QUEUE 252 (DONE|ABORTED)|☠' /mnt/e/chess_data/q252.log; do stopped "the start"; sleep 60; done
grep -qE 'QUEUE 252 ABORTED|☠' /mnt/e/chess_data/q252.log && { echo "[q253] ☠️ #252 aborted (no ctl chunks to pair with) — QUEUE 253 ABORTED"; exit 1; }
sleep 30
START=$(date '+%Y-%m-%d %H:%M:%S')
CUT=$(date -d 'today 20:50' +%s)
echo "[q253] $START start (timed steps only before $(date -d @$CUT '+%a %H:%M'))"
busy && { echo "[q253] ☠️ another games or bench job is running — QUEUE 253 ABORTED"; exit 1; }
cmp -s "$PROD" "$G74" || { echo "[q253] ☠️ production is not g74pgo — QUEUE 253 ABORTED"; exit 1; }
F0=$(fp "$PROD" ""); FPR=$(fp "$PROD" "$KP")
echo "[q253] production [$F0] (want 250 / 37320382) · prom [$FPR] (want 251 / 37302143)"
[ "$F0" = "solved 250/300 nodes 37320382" ] && [ "$FPR" = "solved 251/300 nodes 37302143" ] || { echo "[q253] ☠️ fingerprints — QUEUE 253 ABORTED"; exit 1; }
COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
rm -f $O/g18_q253_s* $O/med_q253_s*

# ── 1 SF18 at equal nodes (node-limited) ──────────────────────────────────────────────────────────────────────────────────
for s in 1 2 3 4 5 6; do
	seed=$((252110 + s)); f=$O/g18_q253_s${seed}_prom.log
	stopped "SF18 $seed"
	nice -n 5 "$FC" -engine "cmd=$PROD" name=ours option.OwnBook=false "option.Knobs=$KP" nodes=250000 \
		-engine "cmd=$SF18" name=sf18 option.Threads=1 option.Hash=16 nodes=1100 \
		-each tc=inf $COMMON -srand $seed -rounds 125 -games 2 -pgnout file=$O/g18_q253_s${seed}_prom.pgn notation=san nodes=true append=false > "$f" 2>&1
	echo "[q253] $(date '+%H:%M') SF18 seed $seed prom: $(grep -E '^Games:' "$f" | tail -1) · warnings $(grep -c 'have option' "$f") · disconnects/stalls $(grep -ciE 'disconnect|stall|not responsive' "$f")"
done

# ── 2 Mediocre (timed) ────────────────────────────────────────────────────────────────────────────────────────────────────
for seed in 252311 252312; do
	f=$O/med_q253_s${seed}_prom.log
	stopped "Mediocre $seed"
	timed_ok || { echo "[q253] $(date) skip Mediocre $seed (outside the timed window)"; continue; }
	guard
	nice -n 5 "$FC" -engine "cmd=$PROD" name=ours option.OwnBook=false "option.Knobs=$KP" tc=10+0.1 timemargin=250 \
		-engine "cmd=$MED" name=mediocre tc=10+0.1 timemargin=1000 \
		$COMMON -srand $seed -rounds 75 -games 2 -pgnout file=$O/med_q253_s${seed}_prom.pgn notation=san append=false > "$f" 2>&1
	kill $GUARD 2>/dev/null
	echo "[q253] $(date '+%H:%M') Mediocre seed $seed prom: $(grep -E '^Games:' "$f" | tail -1) · warnings $(grep -c 'have option' "$f") · disconnects/stalls $(grep -ciE 'disconnect|stall|not responsive' "$f") · time losses $(grep -c 'loses on time' "$f")"
done

# ── reads: prom against #252's ctl chunks on the same seeds ───────────────────────────────────────────────────────────────
row() {  # seed, arm, log
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
{ for s in 1 2 3 4 5 6; do seed=$((252110 + s)); row $seed ctl $O/g18_q252_s${seed}_ctl.log; row $seed prom $O/g18_q253_s${seed}_prom.log; done; } > /tmp/q253_sf.txt
{ for seed in 252311 252312; do row $seed ctl $O/med_q252_s${seed}_ctl.log; row $seed prom $O/med_q253_s${seed}_prom.log; done; } > /tmp/q253_med.txt
echo "[q253] $(date) per chunk (seed arm games W L D):"; { sed 's/^/Mediocre /' /tmp/q253_med.txt; sed 's/^/SF18 /' /tmp/q253_sf.txt; } | show
echo "[q253] REGISTERED READS (SF18: prom - ctl z > -1 = pass; Mediocre: prom - ctl >= 0 = pass):"
{ cmpr prom ctl /tmp/q253_sf.txt SF18; cmpr prom ctl /tmp/q253_med.txt Mediocre; } | show
echo "[q253] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | show
echo "[q253] $(date) QUEUE 253 DONE"
