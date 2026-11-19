#!/bin/bash
# 2026-11-19 ~10:50, queue #279: CALIBRATE STOCKFISH 1.1 AS THE THIRD JUDGE (owner 11-19: "drop mediocre as a judge from now on
# and use something like SF1.1 since its much closer to our strength and can separate things better"). Chained after #278.
# Why a ladder: a judge separates best near 50% (1 pp ≈ 7 Elo there, ≈ 24 Elo at Mediocre's 92%). The form copies the SF18
# judge: node-limited (safe beside the owner's games, no time manager or machine load in the read), ours at 250k nodes.
#  1 Wait for #278; production must be gen 7.7 (g77pgo, "GEN 7.7 SHIPPED" in q278.log); nothing else running.
#  2 Preflight: 2 games, both engines accept every option (0 "have option" warnings).
#  3 Ladder: ours nodes=250000 vs SF1.1 nodes=N for N = 80k, 160k, 40k, 320k; 100 games each (50 pairs), the same openings
#    (seed 279100) on every rung, concurrency 3; per rung the score and both engines' real nodes / depth per move.
# REGISTERED before any game: the score falls as N rises (each doubling ≈ −10 … −15 pp near 50%); the 50% point lies between
# N = 60k and 200k (low confidence: gen 6.5 scored 36.75% at 3x TIME odds on 11-04, and our 250k nodes ≈ 0.17 s ≈ SF1.1's
# 290k nodes at equal time). The judge's setting = the rung nearest 50% (between 35% and 65% is usable), 3 seeds x 250 g per
# arm per candidate (≈ ±29 Elo at 95%).
# Stop: touch /mnt/e/chess_data/q279.stop (between rungs). Never edit this file while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
SF1=/mnt/e/chess_data/engines/sf1.1/stockfish-1.1
O=/mnt/e/chess_data/fastchess
PROD="$ND/build_uci/chess_engine_uci"
G77="$ND/build_uci/chess_engine_uci_g77pgo"
LOG=/mnt/e/chess_data/q279.log
STOP=/mnt/e/chess_data/q279.stop
Q278=/mnt/e/chess_data/q278.log
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"
show() { sed "s/^/[q279]   /"; }
stopped() { [ -e "$STOP" ] && { echo "[q279] $(date) stop file before $1 — QUEUE 279 DONE"; exit 0; }; }
busy() { pgrep -f "[f]astchess-linux-x86-64/fastchess" >/dev/null || pgrep -f "[s]ts_test|[t]actical_test|[v]algrind|[_]ks_footprint|[b]uild_uci.sh" >/dev/null; }

# ── 1 wait for #278 ──────────────────────────────────────────────────────────────────────────────────────────────────────
echo "[q279] $(date '+%Y-%m-%d %H:%M:%S') queued; waiting for #278"
until grep -qE 'QUEUE 278 (DONE|ABORTED)' "$Q278" 2>/dev/null || ! pgrep -f "[_]queue_278.sh" >/dev/null; do stopped "the start"; sleep 30; done
sleep 20
grep -q 'GEN 7.7 SHIPPED' "$Q278" || { echo "[q279] ☠️ #278 did not ship gen 7.7 — QUEUE 279 ABORTED (calibrate on the production that will judge)"; exit 1; }
cmp -s "$PROD" "$G77" || { echo "[q279] ☠️ production is not g77pgo — QUEUE 279 ABORTED"; exit 1; }
busy && { echo "[q279] ☠️ another job is running — QUEUE 279 ABORTED"; exit 1; }
START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[q279] $START start · production $(printf 'uci\nquit\n' | "$PROD" 2>/dev/null | grep -m1 '^id name') · bench $(printf 'bench\nquit\n' | "$PROD" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+')"

# ── 2 preflight ──────────────────────────────────────────────────────────────────────────────────────────────────────────
COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
SF1OPT="option.Threads=1 option.Ponder=false option.OwnBook=false option.Hash=16"
timeout 300 "$FC" -engine "cmd=$PROD" name=ours option.OwnBook=false nodes=250000 \
	-engine "cmd=$SF1" name=sf1.1 $SF1OPT nodes=80000 -each tc=inf \
	-openings file=$O/openings_uho.epd format=epd order=random -rounds 1 -games 2 -concurrency 1 > /tmp/q279_pre.log 2>&1
WN=$(grep -c "have option" /tmp/q279_pre.log)
echo "[q279] preflight: $(grep -E '^Games:' /tmp/q279_pre.log | tail -1) · warnings $WN"
[ "$WN" = 0 ] && grep -qE '^Games: 2' /tmp/q279_pre.log || { echo "[q279] ☠️ PREFLIGHT FAILED — QUEUE 279 ABORTED"; tail -5 /tmp/q279_pre.log | show; exit 1; }

# ── 3 the ladder ─────────────────────────────────────────────────────────────────────────────────────────────────────────
rung() {  # SF1.1 nodes
	local n=$1 t=sf11j_q279_n$1
	stopped "the $n rung"
	echo "[q279] $(date '+%H:%M') rung SF1.1 nodes=$n (100 g, seed 279100)"
	nice -n 5 "$FC" -engine "cmd=$PROD" name=ours option.OwnBook=false nodes=250000 \
		-engine "cmd=$SF1" name=sf1.1 $SF1OPT nodes=$n \
		-each tc=inf $COMMON -srand 279100 -rounds 50 -games 2 -pgnout file=$O/$t.pgn notation=san nodes=true append=false > $O/$t.log 2>&1
	echo "[q279] $(date '+%H:%M') nodes=$n: $(grep -E '^Games:' $O/$t.log | tail -1) · warnings $(grep -c 'have option' $O/$t.log) · disconnects/stalls $(grep -ciE 'disconnect|stall|not responsive' $O/$t.log)"
	timeout 600 bash "$R" pyrun selfplay/fastchess_report.py $O/$t.pgn --p1 ours 2>&1 | sed -n '/vs the field/,$p' | grep -vE '^\s*$' | show
}
for n in 80000 160000 40000 320000; do rung $n; done

# ── reads ────────────────────────────────────────────────────────────────────────────────────────────────────────────────
echo "[q279] ladder (SF1.1 nodes, games, our score, Elo):"
for n in 40000 80000 160000 320000; do
	grep -E '^Games:' $O/sf11j_q279_n$n.log 2>/dev/null | tail -1 | awk -v n=$n '{ gsub(/,/, ""); for (i = 1; i <= NF; i++) { if ($i == "Games:") g = $(i+1); if ($i == "Wins:") w = $(i+1); if ($i == "Draws:") d = $(i+1) } if (g > 0) { s = (w + d / 2) / g; e = (s > 0 && s < 1) ? -400 * log(1 / s - 1) / log(10) : 0; printf "%d %d %.4f %.1f\n", n, g, s, e } }'
done > /tmp/q279_ladder.txt
awk '{ printf "  SF1.1 %7d nodes: %3d g  %.1f%%  Elo %+.0f\n", $1, $2, 100 * $3, $4 }' /tmp/q279_ladder.txt | show
awk 'NR > 1 && (ps - 0.5) * ($3 - 0.5) <= 0 && ps != $3 { x = log(pn) + (log($1) - log(pn)) * (pe - 0) / (pe - $4); printf "  50%% point ≈ SF1.1 at %.0f nodes (Elo-linear in log nodes between %d and %d)\n", exp(x), pn, $1; f = 1 }
	{ pn = $1; ps = $3; pe = $4 } END { if (!f) print "  no rung pair brackets 50% — extend the ladder" }' /tmp/q279_ladder.txt | show
echo "[q279] registered: score falls as N rises (≈ −10 … −15 pp per doubling); 50% between 60k and 200k; judge = the rung nearest 50% (35-65% usable)"
echo "[q279] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | show
echo "[q279] $(date) QUEUE 279 DONE"
