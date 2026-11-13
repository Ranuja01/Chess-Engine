#!/bin/bash
# 2026-11-13 ~13:00, queue #225: search v2's interior as the default (SW5, plan §7e; owner 11-13 "ok lets go"). The shipped preset
# sets SEARCH_ARM=1 (search_engine.cpp, the gen-7 block, before the start-up read: SEARCH_ARM=0 restores the previous interior).
# S1 is a one-for-one port: node-for-node identical on the six identity sets, 12 of 12 games identical (SW2), −1.87% instructions,
# −2.86% time to depth 13 (#207b), smoke +9.3 ± 35.0 over 300 g with 0 forfeits.
#  1 Build, now (box free; nothing timed): PGO chess_engine_uci_g73spgo trained on its own default (arm 1), profile
#    /mnt/e/chess_data/pgo_g73s. Gates: default 249 / 37,279,382 and SEARCH_ARM=0 the same; MOVEGEN_FAST=0 the same; gen 7.2 /
#    7.1 / 6.7 knobs; v1 printed. Live check: cachegrind instructions, default vs SEARCH_ARM=0, 12 game positions at depth 8
#    (equal nodes, different instructions). Build identity games against production (g73fpgo), 30k nodes, 40 starts per set:
#    every set "diverged: 0". Ends with "build done": the sources are free to edit from that line on.
#  2 Waits for /mnt/e/chess_data/q225.go (written on the owner's word, after their gaming).
#  3 Pinned bench, production g73fpgo and the candidate alternating x6, on a quiet box.
#  4 Games: candidate (default) vs production (default), 2,000 at 10+0.1, UHO, seed 225, concurrency 3; start only before 04:00.
#    REGISTERED before any game: NOT HARMFUL = Elo above −2σ (pentanomial, the report's 95% lower bound above −2σ's line) with
#    0 time forfeits, 0 disconnects / stalls; then the switch ships on the owner's yes. Prediction 0 to +5 (speed only).
# Stop: touch /mnt/e/chess_data/q225.stop (between steps); the games step is stopped with SIGINT to fastchess (finished games kept).
# Never edit this file, the binaries or the sources while it runs before "build done"; never edit the binaries after.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
W=/mnt/e/chess_data/q225
P=/mnt/e/chess_data/pgo_g73s
PROD="$ND/build_uci/chess_engine_uci"
PRODF="$ND/build_uci/chess_engine_uci_g73fpgo"
NEW="$ND/build_uci/chess_engine_uci_g73spgo"
G=/mnt/e/chess_data/sv2_speed/game12.txt
STOP=/mnt/e/chess_data/q225.stop
GO=/mnt/e/chess_data/q225.go
GAMES_LATEST_HOUR=4
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP" "$GO"; mkdir -p "$W"
show() { sed "s/^/[q225]   /"; }
stopped() { [ -e "$STOP" ] && { echo "[q225] $(date) stop file present before $1 — QUEUE 225 DONE"; exit 0; }; }
busy() { pgrep -f "fastchess-linux-x86-64/fastchess|vs_sf.py|sts_test|tactical_test|_ebf_per_iter|_grade_game|_depth_ladder|chess_ui" >/dev/null; }
wait_quiet() { local t=$(( $(date +%s) + 3600 )); until ! busy; do stopped "$1"; [ $(date +%s) -gt $t ] && { echo "[q225] $(date) box still busy after an hour — $1 skipped"; return 1; }; sleep 20; done; }
fp() { local k=""; [ -n "$2" ] && k="setoption name Knobs value $2"; printf '%s\nbench\nquit\n' "$k" | "$1" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+'; }
irefs() { grep -E 'I *refs' "$1" | grep -oE '[0-9,]+$' | tr -d ,; }
echo "[q225] $(date '+%Y-%m-%d %H:%M:%S') start"
cmp -s "$PROD" "$PRODF" && echo "[q225] production == chess_engine_uci_g73fpgo" || { echo "[q225] ☠️ production is not g73fpgo — QUEUE 225 ABORTED"; exit 1; }

# ── 1 build, gates, live check, identity games ──────────────────────────────────────────────────────────────────────────
rm -rf "$P"; mkdir -p "$P"
OUT=chess_engine_uci_g73sgen PGO=gen PGO_DIR=$P bash selfplay/build_uci.sh > "$W/gen.out" 2>&1 || { echo "[q225] ☠️ gen build failed — QUEUE 225 ABORTED"; tail -5 "$W/gen.out" | show; exit 1; }
nice -n 5 "$FC" -engine cmd="$ND/build_uci/chess_engine_uci_g73sgen" name=a option.OwnBook=false st=0.25 \
	-engine cmd="$ND/build_uci/chess_engine_uci_g73sgen" name=b option.OwnBook=false st=0.25 \
	-openings file=$O/openings_uho.epd format=epd order=random -srand 735 -rounds 4 -games 2 -concurrency 1 > "$W/train.log" 2>&1
echo "[q225] training: $(grep -E '^Games:' "$W/train.log" | tail -1) · profile files $(find "$P" -name '*.gcda' | wc -l)"
OUT=chess_engine_uci_g73spgo PGO=use PGO_DIR=$P bash selfplay/build_uci.sh > "$W/use.out" 2>&1 || { echo "[q225] ☠️ use build failed — QUEUE 225 ABORTED"; tail -5 "$W/use.out" | show; exit 1; }
echo "[q225] $(date '+%H:%M') build done — the sources are free to edit"
d=$(fp "$NEW" ""); d0=$(fp "$NEW" SEARCH_ARM=0); dm=$(fp "$NEW" MOVEGEN_FAST=0); g72=$(fp "$NEW" SEE_PER_MOVE=0); g71=$(fp "$NEW" SEE_PER_MOVE=0,MOPUP_V2=0); g67=$(fp "$NEW" V2_SHIP_GEN=6)
echo "[q225] default (want 249 / 37279382):              [$d]"
echo "[q225] SEARCH_ARM=0 (want the same):               [$d0]"
echo "[q225] MOVEGEN_FAST=0 (want the same):             [$dm]"
echo "[q225] SEE_PER_MOVE=0 = gen 7.2 (247 / 39861964):  [$g72]"
echo "[q225] + MOPUP_V2=0 = gen 7.1 (247 / 39852210):    [$g71]"
echo "[q225] V2_SHIP_GEN=6 = gen 6.7 (251 / 36859850):   [$g67]"
echo "[q225] v1 EvalArm 0 (250 / 35310775-778):          $(printf 'setoption name EvalArm value 0\nbench\nquit\n' | "$NEW" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+')"
echo "[q225] id: $(printf 'uci\nquit\n' | "$NEW" 2>/dev/null | grep '^id name')"
if ! { [ "$d" = "solved 249/300 nodes 37279382" ] && [ "$d0" = "$d" ] && [ "$dm" = "$d" ] && [ "$g72" = "solved 247/300 nodes 39861964" ] \
	&& [ "$g71" = "solved 247/300 nodes 39852210" ] && [ "$g67" = "solved 251/300 nodes 36859850" ]; }; then
	echo "[q225] ☠️ fingerprints differ — QUEUE 225 ABORTED"; exit 1
fi
for a in 1 0; do
	printf "bench $G 8\nquit\n" | env SEARCH_ARM=$a valgrind --tool=cachegrind --cache-sim=no \
		--cachegrind-out-file="$W/cg_arm$a.out" "$NEW" > "$W/bench_arm$a.txt" 2> "$W/cg_arm$a.log"
	echo "[q225] live check SEARCH_ARM=$a: $(grep -oE 'solved [0-9]+/[0-9]+ nodes [0-9]+' "$W/bench_arm$a.txt") · Ir $(irefs "$W/cg_arm$a.log")"
done
echo "[q225] default - SEARCH_ARM=0 instructions: $(awk -v a="$(irefs "$W/cg_arm0.log")" -v b="$(irefs "$W/cg_arm1.log")" 'BEGIN { printf "%+.3f%%", 100 * (b - a) / a }')"
stopped "the identity games"
echo "[q225] $(date '+%H:%M') build identity: production (g73fpgo) vs the candidate, 30k nodes, 40 starts per set, one process per side"
bash "$R" pyrun diagnostics/_fix_firecount.py BIN="$PRODF" BIN_ON="$NEW" ON= N=40 NODES=30000 PLIES=200 SIDES=2 JOBS=2 2>&1 \
	| grep -E '===|diverged|CRASH' | show

# ── 2 the owner's go ──────────────────────────────────────────────────────────────────────────────────────────────────
echo "[q225] $(date '+%H:%M') waiting for the go file (owner's word after gaming)"
until [ -e "$GO" ]; do stopped "the go"; sleep 30; done
echo "[q225] $(date '+%Y-%m-%d %H:%M:%S') go"

# ── 3 pinned bench ────────────────────────────────────────────────────────────────────────────────────────────────────
if wait_quiet "the pinned bench"; then
	echo "[q225] $(date '+%H:%M') == TIMED: UCI bench, production g73fpgo and candidate g73spgo, pinned, alternating x6"
	for i in 1 2 3 4 5 6; do
		echo "[q225]   g73f $(printf 'bench\nquit\n' | taskset -c 5 "$PRODF" 2>/dev/null | grep '^bench')"
		echo "[q225]   g73s $(printf 'bench\nquit\n' | taskset -c 5 "$NEW" 2>/dev/null | grep '^bench')"
	done
fi

# ── 4 games ───────────────────────────────────────────────────────────────────────────────────────────────────────────
stopped "the games"
h=$(date +%H); h=$((10#$h))
if [ $h -ge $GAMES_LATEST_HOUR ] && [ $h -lt 12 ]; then echo "[q225] $(date) past 04:00 — games skipped (run them on the next free night)"; echo "[q225] $(date) QUEUE 225 DONE"; exit 0; fi
COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
echo "[q225] $(date '+%H:%M') games: v2 default (g73spgo) vs production (g73fpgo) at 10+0.1, 2,000 games, seed 225"
nice -n 5 "$FC" -engine "cmd=$NEW" name=v2def option.OwnBook=false tc=10+0.1 timemargin=250 \
	-engine "cmd=$PRODF" name=prod option.OwnBook=false tc=10+0.1 timemargin=250 \
	$COMMON -srand 225 -rounds 1000 -games 2 -pgnout file=$W/sw5.pgn notation=san nodes=true append=false > $W/sw5.log 2>&1
echo "[q225] games: $(grep -E '^Games:' $W/sw5.log | tail -1) · warnings $(grep -c 'have option' $W/sw5.log) · disconnects/stalls $(grep -ciE 'disconnect|stall|not responsive' $W/sw5.log) · time forfeits $(grep -c 'loses on time' $W/sw5.pgn)"
grep -E 'Elo:|Ptnml' $W/sw5.log | tail -2 | show
timeout 900 bash "$R" pyrun selfplay/fastchess_report.py $W/sw5.pgn --p1 v2def 2>&1 | sed -n '/vs the field/,$p' | show
echo "[q225] $(date) QUEUE 225 DONE"
