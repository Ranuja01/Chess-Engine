#!/bin/bash
# 2026-11-16 ~21:50, queue #261: the UCI test build for C4's games (production g75pgo predates SV2_HIST_ALLNODE_GATE bit 2).
# PGO build chess_engine_uci_c4pgo from the current sources (gen 7.5 + bit 2, default 0), its own profile /mnt/e/chess_data/pgo_c4.
# Production is NOT touched. GATES: default 252 / 35,235,460 (= production); SV2_HIST_ALLNODE_GATE=2 252 / 35,235,446 and =1
# 252 / 36,659,901 (= the Cython screens #259 / #260); three FIX off = gen 7.4 250 / 37,320,382; v1 250 / 35,310,775-778.
# Fixed work and a short PGO training run (8 games at st=0.25, profile only); done well before #257's timed games at 00:15.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
W=/mnt/e/chess_data/q261
P=/mnt/e/chess_data/pgo_c4
NEW="$ND/build_uci/chess_engine_uci_c4pgo"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
mkdir -p "$W"
show() { sed "s/^/[q261]   /"; }
fp() { local k=""; [ -n "$2" ] && k="setoption name Knobs value $2"; printf '%s\nbench\nquit\n' "$k" | "$1" 2>&1 | grep -oE 'solved [0-9]+/300 nodes [0-9]+|mismatch.*' | head -1; }
echo "[q261] $(date '+%Y-%m-%d %H:%M:%S') start"
pgrep -f "[f]astchess-linux|[s]ts_test|[t]actical_test|[b]uild_uci.sh" >/dev/null && { echo "[q261] ☠️ another job is running — QUEUE 261 ABORTED"; exit 1; }
rm -rf "$P"; mkdir -p "$P"
if OUT=chess_engine_uci_c4gen PGO=gen PGO_DIR=$P nice -n 10 bash selfplay/build_uci.sh > "$W/gen.out" 2>&1; then
	nice -n 10 "$FC" -engine cmd="$ND/build_uci/chess_engine_uci_c4gen" name=a option.OwnBook=false st=0.25 \
		-engine cmd="$ND/build_uci/chess_engine_uci_c4gen" name=b option.OwnBook=false st=0.25 \
		-openings file=$O/openings_uho.epd format=epd order=random -srand 761 -rounds 4 -games 2 -concurrency 1 > "$W/train.log" 2>&1
	OUT=chess_engine_uci_c4pgo PGO=use PGO_DIR=$P nice -n 10 bash selfplay/build_uci.sh > "$W/use.out" 2>&1 || { echo "[q261] ☠️ PGO use build failed — QUEUE 261 ABORTED"; grep -iE 'error' "$W/use.out" | head -8 | show; exit 1; }
else
	echo "[q261] ☠️ PGO gen build failed — QUEUE 261 ABORTED"; grep -iE 'error' "$W/gen.out" | head -8 | show; exit 1
fi
echo "[q261] $(date '+%H:%M') build done; training: $(grep -E '^Games:' "$W/train.log" | tail -1)"
a=$(fp "$NEW" ""); b2=$(fp "$NEW" "SV2_HIST_ALLNODE_GATE=2"); b1=$(fp "$NEW" "SV2_HIST_ALLNODE_GATE=1")
off=$(fp "$NEW" "FIX_RESEARCH=0,FIX_MIN_SCOUT=0,FIX_MIN_RD1=0")
v1=$(printf 'setoption name EvalArm value 0\nbench\nquit\n' | "$NEW" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+')
echo "[q261] default [$a] (252 / 35235460) · =2 [$b2] (252 / 35235446) · =1 [$b1] (252 / 36659901) · three FIX off [$off] (250 / 37320382) · v1 [$v1]"
if [ "$a" = "solved 252/300 nodes 35235460" ] && [ "$b2" = "solved 252/300 nodes 35235446" ] && [ "$b1" = "solved 252/300 nodes 36659901" ] \
	&& [ "$off" = "solved 250/300 nodes 37320382" ] && echo "$v1" | grep -qE 'solved 250/300 nodes 3531077[5-8]'; then
	echo "[q261] ✅ every gate equal — chess_engine_uci_c4pgo is ready for C4's games"
else
	echo "[q261] ☠️ a gate differs — do not use c4pgo"
fi
echo "[q261] $(date) QUEUE 261 DONE"
