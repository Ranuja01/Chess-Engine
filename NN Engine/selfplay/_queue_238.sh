#!/bin/bash
# 2026-11-14 ~12:45, queue #238: S5 D7, search v2's near-miss (VERIFY) re-search never replaces a score with a shallower one
# (SV2_VERIFY_DEEPER=1; default 0 byte-identical). Owner's windows today: heavy until ~13:55, gaming ~14:00-17:00 (fixed depth
# only), heavy again until ~21:00, gaming from 21:00.
#  1 Plain UCI test build chess_engine_uci_r7test. Gates: default 249 / 37,279,382, SV2_VERIFY_DEEPER=0 and SEARCH_ARM=0 the same;
#    SV2_VERIFY_DEEPER=1 printed (must differ: the knob is live).
#  2 Cython module rebuilt (production SW5 module backed up to /mnt/e/chess_data/so_backup_sw5/, restored at the end), gated.
#  3 Depth 10 with ENABLE_LMR_COUNTERS=1 on WAC, STS300, quiet (100), game (100): D7 = SV2_VERIFY_DEEPER=1. Reference (not
#    re-run): #233 S0 = WAC 249 / 37,279,382, STS 2037 / 93,222,085, quiet 25,798,338, game 19,969,178.
#    REGISTERED prediction: nodes 3-8% below S0 on every set (the skipped re-searches), WAC within 3 of 249.
#  4 PGO build chess_engine_uci_d7pgo (profile /mnt/e/chess_data/pgo_d7), gated like 1.
#  5 Games, D7 (d7pgo + SV2_VERIFY_DEEPER=1) vs production (g73spgo), 10+0.1, UHO, concurrency 3:
#    part A from the end of 4 until 13:50 (seed 238; SIGINT at 13:50, finished games kept); part B waits for
#    /mnt/e/chess_data/q238.go (written when the owner's gaming is over) and plays the rest of 2,000 (seed 2381), SIGINT at 20:55
#    (the owner games from 21:00); any remainder goes in a later queue. Read pooled.
#    REGISTERED: >= +2 sigma = candidate (SF18 / Mediocre legs next); +-2 sigma = NEUTRAL (knob stays 0, kept for Stack D's
#    VERIFY retune); <= -2 sigma = the shallow re-search is a productive bug (VERIFY_MARGIN / VERIFY_RESEARCH_REDUCTION were
#    tuned with it: retune them together in Stack D). 0 forfeits, 0 disconnects.
# Stop: touch /mnt/e/chess_data/q238.stop (between steps); games stop with SIGINT to fastchess.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
W=/mnt/e/chess_data/r7a
BK=/mnt/e/chess_data/so_backup_sw5
P=/mnt/e/chess_data/pgo_d7
STOP=/mnt/e/chess_data/q238.stop
GO=/mnt/e/chess_data/q238.go
GAMECSV=/mnt/e/chess_data/r1a/game_v2_stratum.csv
T="$ND/build_uci/chess_engine_uci_r7test"
NEW="$ND/build_uci/chess_engine_uci_d7pgo"
PROD="$ND/build_uci/chess_engine_uci"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP" "$GO"; mkdir -p "$W"
show() { sed "s/^/[q238]   /"; }
stopped() { [ -e "$STOP" ] && { echo "[q238] $(date) stop file present before $1 — QUEUE 238 DONE"; exit 0; }; }
fp() { local k=""; [ -n "$2" ] && k="setoption name Knobs value $2"; printf '%s\nbench\nquit\n' "$k" | "$1" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+'; }
gates() {
	local b=$1 d d0 a0 d1
	d=$(fp "$b" ""); d0=$(fp "$b" SV2_VERIFY_DEEPER=0); a0=$(fp "$b" SEARCH_ARM=0); d1=$(fp "$b" SV2_VERIFY_DEEPER=1)
	echo "[q238] $(basename "$b"): default [$d] · SV2_VERIFY_DEEPER=0 [$d0] · SEARCH_ARM=0 [$a0] · SV2_VERIFY_DEEPER=1 [$d1]"
	[ "$d" = "solved 249/300 nodes 37279382" ] && [ "$d0" = "$d" ] && [ "$a0" = "$d" ] && [ -n "$d1" ] && [ "$d1" != "$d" ]
}
echo "[q238] $(date '+%Y-%m-%d %H:%M:%S') start"
cmp -s "$PROD" build_uci/chess_engine_uci_g73spgo || { echo "[q238] ☠️ production is not g73spgo — QUEUE 238 ABORTED"; exit 1; }

# ── 1 test binary ─────────────────────────────────────────────────────────────────────────────────────────────────────
OUT=chess_engine_uci_r7test bash selfplay/build_uci.sh > "$W/build.out" 2>&1 || { echo "[q238] ☠️ test build failed — QUEUE 238 ABORTED"; tail -5 "$W/build.out" | show; exit 1; }
gates "$T" || { echo "[q238] ☠️ gates failed (default moved, or the knob is dead) — QUEUE 238 ABORTED"; exit 1; }

# ── 2 module ──────────────────────────────────────────────────────────────────────────────────────────────────────────
mkdir -p "$BK" && cp ChessAI.*.so "$BK"/ && echo "[q238] production (SW5) module backed up to $BK" || { echo "[q238] ☠️ backup failed — QUEUE 238 ABORTED"; exit 1; }
bash "$R" build 2>&1 | grep -iE 'error|copying|FAILED' | tail -n 3 | show
ls ChessAI.*.so >/dev/null 2>&1 || { echo "[q238] ☠️ no .so after the build — restoring"; cp "$BK"/ChessAI.*.so .; exit 1; }
echo "[q238] $(date '+%H:%M') module built — the sources are free to edit"
f=$(bash "$R" wac r7a_fp V2_PRESET=shipped 2>&1 | grep -E '^SOLVED|^NODES' | tr '\n' ' ')
echo "[q238] module shipped (want 249 / 37279382): $f"
echo "$f" | grep -q 'Solved 249/300 NODES: 37279382' || { echo "[q238] ☠️ the module moved a node — restoring production"; cp "$BK"/ChessAI.*.so .; exit 1; }

# ── 3 screens ─────────────────────────────────────────────────────────────────────────────────────────────────────────
BASE="MAX_DEPTH=10 USE_OPENING_BOOK=0 PRESET=LONG_FORMAT V2_PRESET=shipped"
k="SV2_VERIFY_DEEPER=1 ENABLE_LMR_COUNTERS=1"
sum_nodes() { grep -hoE '\(nodes=[0-9]+' "$1" | grep -oE '[0-9]+' | awk '{s+=$1} END{print s+0}'; }
read_run() { echo "[q238]   $1 nodes=$(sum_nodes "$2")"; grep -E '^\[sv2_lmr\]|^\[sv2_verify\]' "$2" | tail -2 | sed 's/^/[q238]     /'; }
stopped "the screens"
echo "[q238] $(date '+%H:%M') == D7 $k"
echo "[q238]   WAC: $(bash "$R" wac r7a_d7 V2_PRESET=shipped $k 2>&1 | grep -E '^SOLVED|^NODES' | tr '\n' ' ')"
cp /tmp/wac_r7a_d7.err "$W/wac_d7.err" 2>/dev/null; read_run WAC "$W/wac_d7.err"
env $BASE $k bash "$R" pyrun diagnostics/sts_test.py sts300.epd r7a_d7 > "$W/sts_d7.out" 2> "$W/sts_d7.err"
echo "[q238]   STS: $(grep -E 'STS score' "$W/sts_d7.out" | tail -1)"; read_run STS "$W/sts_d7.err"
bash "$R" pyrun diagnostics/depth_nps_bench.py $BASE $k --n 100 > "$W/quiet_d7.out" 2> "$W/quiet_d7.err"; read_run quiet "$W/quiet_d7.err"
bash "$R" pyrun diagnostics/depth_nps_bench.py $BASE $k --n 100 --corpus "$GAMECSV" > "$W/game_d7.out" 2> "$W/game_d7.err"; read_run game "$W/game_d7.err"
cp "$BK"/ChessAI.*.so . && echo "[q238] production module restored: $(bash "$R" wac r7a_fprest V2_PRESET=shipped 2>&1 | grep -E '^SOLVED|^NODES' | tr '\n' ' ')"

# ── 4 PGO build ───────────────────────────────────────────────────────────────────────────────────────────────────────
stopped "the PGO build"
rm -rf "$P"; mkdir -p "$P"
OUT=chess_engine_uci_d7gen PGO=gen PGO_DIR=$P bash selfplay/build_uci.sh > "$W/gen.out" 2>&1 || { echo "[q238] ☠️ gen build failed — QUEUE 238 ABORTED"; exit 1; }
nice -n 5 "$FC" -engine cmd="$ND/build_uci/chess_engine_uci_d7gen" name=a option.OwnBook=false option.Knobs=SV2_VERIFY_DEEPER=1 st=0.25 \
	-engine cmd="$ND/build_uci/chess_engine_uci_d7gen" name=b option.OwnBook=false st=0.25 \
	-openings file=$O/openings_uho.epd format=epd order=random -srand 735 -rounds 4 -games 2 -concurrency 1 > "$W/train.log" 2>&1
OUT=chess_engine_uci_d7pgo PGO=use PGO_DIR=$P bash selfplay/build_uci.sh > "$W/use.out" 2>&1 || { echo "[q238] ☠️ use build failed — QUEUE 238 ABORTED"; exit 1; }
gates "$NEW" || { echo "[q238] ☠️ PGO gates failed — QUEUE 238 ABORTED"; exit 1; }

# ── 5 games ───────────────────────────────────────────────────────────────────────────────────────────────────────────
COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
play() {   # $1 seed, $2 rounds, $3 tag
	nice -n 5 "$FC" -engine "cmd=$NEW" name=d7 option.OwnBook=false "option.Knobs=SV2_VERIFY_DEEPER=1" tc=10+0.1 timemargin=250 \
		-engine "cmd=$PROD" name=prod option.OwnBook=false tc=10+0.1 timemargin=250 \
		$COMMON -srand $1 -rounds $2 -games 2 -pgnout file=$W/$3.pgn notation=san nodes=true append=false > $W/$3.log 2>&1
	echo "[q238] $3: $(grep -E '^Games:' $W/$3.log | tail -1) · warnings $(grep -c 'have option' $W/$3.log) · disconnects/stalls $(grep -ciE 'disconnect|stall|not responsive' $W/$3.log) · time forfeits $(grep -c 'loses on time' $W/$3.pgn)"
}
if [ "$(date +%H%M)" -lt 1345 ]; then
	echo "[q238] $(date '+%H:%M') games part A until 13:50"
	( while [ "$(date +%H%M)" -lt 1350 ]; do sleep 20; done; pkill -INT -f "fastchess-linux-x86-64/fastchess" && echo "[q238] $(date '+%H:%M') 13:50 guard: part A interrupted" ) &
	GUARD=$!
	play 238 1000 d7a
	kill $GUARD 2>/dev/null
fi
done_games=$(grep -c '^\[Result' $W/d7a.pgn 2>/dev/null); done_games=${done_games:-0}
left=$(( (2000 - done_games + 1) / 2 ))
echo "[q238] $(date '+%H:%M') part A: $done_games games; waiting for q238.go for the remaining $left pairs"
until [ -e "$GO" ]; do stopped "part B"; sleep 30; done
echo "[q238] $(date '+%H:%M') go: games part B (SIGINT at 20:55 for the owner's evening gaming; finished games kept)"
( until [ "$(date +%H%M)" -ge 2055 ] && [ "$(date +%H%M)" -lt 2300 ]; do sleep 20; done; pkill -INT -f "fastchess-linux-x86-64/fastchess" && echo "[q238] $(date '+%H:%M') 20:55 guard: part B interrupted" ) &
GUARD=$!
play 2381 $left d7b
kill $GUARD 2>/dev/null
echo "[q238] pooled:"
timeout 900 bash "$R" pyrun selfplay/fastchess_report.py $( [ -s $W/d7a.pgn ] && echo $W/d7a.pgn ) $W/d7b.pgn --p1 d7 2>&1 | sed -n '/vs the field/,$p' | show
echo "[q238] $(date) QUEUE 238 DONE"
