#!/bin/bash
# 2026-11-13 ~19:40, queue #231: S3 root R3, the second games read — H2 = the first root without the pre-search that is
# complete in its main supplier: A4 (cascading tables: pre-search off, root razor off, provenance tiers + root table) plus
# PLY1_NODE_PRUNE=3 (null move and reverse futility at the reply node of a null-window root move, the giants' cheap dismissal of
# late root moves; S3-ROOT-WITHOUT-PRESEARCH H1). Depth 10 (#230): WAC 253 / 30,359,342 (−18.6% vs today), STS 2083 (−15.6%
# nodes), quiet −5.0%, game set −6.9%. Still UNTUNED, and the time manager's node share still reads differently without the
# pre-search (A4 spent +2.9% / +6.6% clock per move in #228).
#  1 PGO build chess_engine_uci_r3pgo from today's sources (default = the shipped preset incl. the SEARCH_ARM=1 candidate, same
#    nodes as production). Gates: default 249 / 37,279,382; SEARCH_ARM=0 the same; H2 knobs 253 / 30,359,342 (the Cython value).
#  2 Games: H2 vs today's root on that binary, 10+0.1, UHO, seed 231, concurrency 3, 2,000 games (~5.3 h).
#    REGISTERED before any game (H2 − today, pentanomial):
#      ≥ +2σ  the pre-search-free root becomes the root candidate (the ship rule's SF18 / Mediocre legs next);
#      ±2σ    LEVEL: a clean path without the pre-search exists; the time-manager share fix, root reductions and retuning next;
#      ≤ −2σ  BEHIND: read time per move and the counters; the time-manager fix next.
#      Plus 0 time forfeits and 0 disconnects / stalls. Prediction −5 to +15.
# The owner may game: then the games are stopped with SIGINT to fastchess (finished games kept) and resumed afterwards.
# Stop: touch /mnt/e/chess_data/q231.stop (before the games). Never edit this file, the binary or the sources before "build done".
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
W=/mnt/e/chess_data/r3b
P=/mnt/e/chess_data/pgo_r3
B="$ND/build_uci/chess_engine_uci_r3pgo"
STOP=/mnt/e/chess_data/q231.stop
H2K="ENABLE_ROOT_PRESEARCH=0,ENABLE_ROOT_RAZOR=0,ENABLE_PRESEARCH_OFF_SORT=1,PRESEARCH_OFF_SORT_MODE=2,ENABLE_ROOT_TABLE=1,PLY1_NODE_PRUNE=3"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"; mkdir -p "$W"
show() { sed "s/^/[q231]   /"; }
stopped() { [ -e "$STOP" ] && { echo "[q231] $(date) stop file present before $1 — QUEUE 231 DONE"; exit 0; }; }
fp() { local k=""; [ -n "$2" ] && k="setoption name Knobs value $2"; printf '%s\nbench\nquit\n' "$k" | "$1" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+'; }
echo "[q231] $(date '+%Y-%m-%d %H:%M:%S') queued; waiting for #230"
until grep -qE 'QUEUE 230 (DONE|ABORTED)|☠️' /mnt/e/chess_data/q230.log; do stopped "the start"; sleep 30; done
grep -q '☠️' /mnt/e/chess_data/q230.log && { echo "[q231] ☠️ #230 flagged a failure — QUEUE 231 ABORTED"; exit 1; }
echo "[q231] $(date '+%Y-%m-%d %H:%M:%S') start"

# ── 1 build and gates ─────────────────────────────────────────────────────────────────────────────────────────────────
rm -rf "$P"; mkdir -p "$P"
OUT=chess_engine_uci_r3gen PGO=gen PGO_DIR=$P bash selfplay/build_uci.sh > "$W/gen.out" 2>&1 || { echo "[q231] ☠️ gen build failed — QUEUE 231 ABORTED"; tail -5 "$W/gen.out" | show; exit 1; }
nice -n 5 "$FC" -engine cmd="$ND/build_uci/chess_engine_uci_r3gen" name=a option.OwnBook=false st=0.25 \
	-engine cmd="$ND/build_uci/chess_engine_uci_r3gen" name=b option.OwnBook=false st=0.25 \
	-openings file=$O/openings_uho.epd format=epd order=random -srand 736 -rounds 4 -games 2 -concurrency 1 > "$W/train.log" 2>&1
echo "[q231] training: $(grep -E '^Games:' "$W/train.log" | tail -1) · profile files $(find "$P" -name '*.gcda' | wc -l)"
OUT=chess_engine_uci_r3pgo PGO=use PGO_DIR=$P bash selfplay/build_uci.sh > "$W/use.out" 2>&1 || { echo "[q231] ☠️ use build failed — QUEUE 231 ABORTED"; tail -5 "$W/use.out" | show; exit 1; }
echo "[q231] $(date '+%H:%M') build done — the sources are free to edit"
d=$(fp "$B" ""); d0=$(fp "$B" SEARCH_ARM=0); h=$(fp "$B" "$H2K")
echo "[q231] default (want 249 / 37279382):      [$d]"
echo "[q231] SEARCH_ARM=0 (want the same):       [$d0]"
echo "[q231] H2 (want 253 / 30359342):           [$h]"
if [ "$d" != "solved 249/300 nodes 37279382" ] || [ "$d0" != "$d" ] || [ "$h" != "solved 253/300 nodes 30359342" ]; then
	echo "[q231] ☠️ fingerprints differ — QUEUE 231 ABORTED"; exit 1
fi
stopped "the games"

# ── 2 games ───────────────────────────────────────────────────────────────────────────────────────────────────────────
COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
echo "[q231] $(date '+%H:%M') games: H2 (no pre-search, cascading tables, ply-1 null + reverse futility) vs today's root at 10+0.1, 2,000 games, seed 231"
nice -n 5 "$FC" -engine "cmd=$B" name=h2 option.OwnBook=false "option.Knobs=$H2K" tc=10+0.1 timemargin=250 \
	-engine "cmd=$B" name=root0 option.OwnBook=false tc=10+0.1 timemargin=250 \
	$COMMON -srand 231 -rounds 1000 -games 2 -pgnout file=$W/h2.pgn notation=san nodes=true append=false > $W/h2.log 2>&1
echo "[q231] games: $(grep -E '^Games:' $W/h2.log | tail -1) · warnings $(grep -c 'have option' $W/h2.log) · disconnects/stalls $(grep -ciE 'disconnect|stall|not responsive' $W/h2.log) · time forfeits $(grep -c 'loses on time' $W/h2.pgn)"
grep -E 'Elo:|Ptnml' $W/h2.log | tail -2 | show
timeout 900 bash "$R" pyrun selfplay/fastchess_report.py $W/h2.pgn --p1 h2 2>&1 | sed -n '/vs the field/,$p' | show
echo "[q231] $(date) QUEUE 231 DONE"
