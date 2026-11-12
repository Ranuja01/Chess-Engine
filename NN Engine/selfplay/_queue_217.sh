#!/bin/bash
# 2026-11-12 ~15:35, queue #217: SHIP GEN 7.3 (owner 11-12: "so we can properly ship"; done in the hour before the owner's
# gaming). Gen 7.3 = gen 7.2 + SEE_PER_MOVE=3 in the shipped preset (search_engine.cpp, the gen-7 block, set before the
# start-up read). Evidence: self-play +12.7 (95% +1.2 … +24.2) over 2,800 g (#209 + #210); SF18 at equal nodes +2.60 pp ± 3.26
# (z +1.56) and Mediocre +2.33 pp (#211); bench WAC d10 249 / 37,279,382, pinned −7.1% time (#213).
#  1 PGO build chess_engine_uci_g73pgo from the repo sources. Gates: default 249 / 37,279,382; SEE_PER_MOVE=0 = gen 7.2
#    247 / 39,861,964; + MOPUP_V2=0 = gen 7.1 247 / 39,852,210; V2_SHIP_GEN=6 = gen 6.7 251 / 36,859,850; v1 printed.
#    Install as production (gen 7.2 stays as chess_engine_uci_g72pgo).
#  2 Cython module rebuild (gen 7.2's .so is in /mnt/e/chess_data/so_backup_g72/) and its fingerprints.
#  3 pinned bench, gen 7.2 and gen 7.3 production binaries alternating x6, on a quiet box (no UI, no other job), before 16:25;
#    otherwise skipped and run later.
# Stop: touch /mnt/e/chess_data/q217.stop (checked between steps). Never edit this file or the sources while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
P=/mnt/e/chess_data/pgo_g73
OLD="$ND/build_uci/chess_engine_uci_g72pgo"
NEW="$ND/build_uci/chess_engine_uci_g73pgo"
STOP=/mnt/e/chess_data/q217.stop
BENCH_LATEST=$(date -d '2026-11-12 16:25' +%s)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"
show() { sed "s/^/[q217]   /"; }
stopped() { [ -e "$STOP" ] && { echo "[q217] $(date) stop file present before $1 — QUEUE 217 DONE"; exit 0; }; }
busy() { pgrep -f "fastchess-linux-x86-64/fastchess|vs_sf.py|sts_test|tactical_test|_ebf_per_iter|_grade_game|chess_ui" >/dev/null; }
k() { printf 'setoption name Knobs value %s\nbench\nquit\n' "$2" | "$1" 2>/dev/null | grep '^bench'; }
echo "[q217] $(date) start"
cmp -s build_uci/chess_engine_uci "$OLD" && echo "[q217] backup ok: production == chess_engine_uci_g72pgo" \
	|| { echo "[q217] ☠️ production differs from chess_engine_uci_g72pgo — QUEUE 217 ABORTED"; exit 1; }

# ── 1 build, gate, install ────────────────────────────────────────────────────────────────────────────────────────────
rm -rf "$P"; mkdir -p "$P"
OUT=chess_engine_uci_g73gen PGO=gen PGO_DIR=$P bash selfplay/build_uci.sh > /tmp/q217_gen.out 2>&1 || { echo "[q217] ☠️ gen build failed — QUEUE 217 ABORTED"; tail -20 /tmp/q217_gen.out | show; exit 1; }
nice -n 5 "$FC" -engine cmd="$ND/build_uci/chess_engine_uci_g73gen" name=a option.OwnBook=false st=0.25 \
	-engine cmd="$ND/build_uci/chess_engine_uci_g73gen" name=b option.OwnBook=false st=0.25 \
	-openings file=$O/openings_uho.epd format=epd order=random -srand 73 -rounds 4 -games 2 -concurrency 1 > /tmp/pgo_g73_train.log 2>&1
echo "[q217] training: $(grep -E '^Games:' /tmp/pgo_g73_train.log | tail -1) · profile files $(find "$P" -name '*.gcda' | wc -l)"
OUT=chess_engine_uci_g73pgo PGO=use PGO_DIR=$P bash selfplay/build_uci.sh > /tmp/q217_use.out 2>&1 || { echo "[q217] ☠️ use build failed — QUEUE 217 ABORTED"; tail -20 /tmp/q217_use.out | show; exit 1; }
d=$(printf 'bench\nquit\n' | "$NEW" 2>/dev/null | grep '^bench')
g72=$(k "$NEW" SEE_PER_MOVE=0)
g71=$(k "$NEW" SEE_PER_MOVE=0,MOPUP_V2=0)
g67=$(k "$NEW" V2_SHIP_GEN=6)
echo "[q217] gen 7.3 default (want 249 / 37279382):                $d"
echo "[q217] SEE_PER_MOVE=0 = gen 7.2 (want 247 / 39861964):         $g72"
echo "[q217] + MOPUP_V2=0 = gen 7.1 (want 247 / 39852210):           $g71"
echo "[q217] V2_SHIP_GEN=6 = gen 6.7 (want 251 / 36859850):          $g67"
echo "[q217] v1 EvalArm 0 (want 250 / 35310775-778):                 $(printf 'setoption name EvalArm value 0\nbench\nquit\n' | "$NEW" 2>/dev/null | grep '^bench')"
echo "[q217] id: $(printf 'uci\nquit\n' | "$NEW" 2>/dev/null | grep '^id name')"
echo "$d" | grep -q "solved 249/300 nodes 37279382 " && echo "$g72" | grep -q "solved 247/300 nodes 39861964 " \
	&& echo "$g71" | grep -q "solved 247/300 nodes 39852210 " && echo "$g67" | grep -q "solved 251/300 nodes 36859850 " \
	|| { echo "[q217] ☠️ fingerprints differ — production NOT replaced, QUEUE 217 ABORTED"; exit 1; }
cp "$NEW" build_uci/.chess_engine_uci.tmp && mv -f build_uci/.chess_engine_uci.tmp build_uci/chess_engine_uci \
	&& cmp -s build_uci/chess_engine_uci "$NEW" && echo "[q217] $(date) INSTALLED: build_uci/chess_engine_uci == chess_engine_uci_g73pgo" \
	|| { echo "[q217] ☠️ install failed — QUEUE 217 ABORTED"; exit 1; }

# ── 2 Cython module ───────────────────────────────────────────────────────────────────────────────────────────────────
stopped "the Cython rebuild"
bash "$R" build 2>&1 | grep -iE 'error|copying|FAILED' | tail -n 3 | show
ls ChessAI.*.so >/dev/null 2>&1 || { echo "[q217] ☠️ Cython build left no .so — restore /mnt/e/chess_data/so_backup_g72/"; exit 1; }
echo "[q217] Cython gen 7.3 shipped (want 249 / 37279382):    $(bash "$R" wac fp_g73 V2_PRESET=shipped 2>&1 | grep -E '^SOLVED|^NODES|^EBF' | tr '\n' ' ')"
echo "[q217] Cython SEE_PER_MOVE=0 (want 247 / 39861964):     $(bash "$R" wac fp_g73as72 V2_PRESET=shipped SEE_PER_MOVE=0 2>&1 | grep -E '^SOLVED|^NODES|^EBF' | tr '\n' ' ')"
echo "[q217] Cython v1 (want 250 / 35310778):                 $(bash "$R" wac fp_g73v1 2>&1 | grep -E '^SOLVED|^NODES|^EBF' | tr '\n' ' ')"

# ── 3 pinned bench ────────────────────────────────────────────────────────────────────────────────────────────────────
echo "[q217] $(date) waiting for a quiet box (no UI, no other job) before the pinned bench, until 16:25"
until ! busy || [ $(date +%s) -gt $BENCH_LATEST ]; do stopped "the pinned bench"; sleep 20; done
if [ $(date +%s) -gt $BENCH_LATEST ]; then
	echo "[q217] $(date) no quiet box before 16:25 — pinned bench skipped (run it later)"
else
	echo "[q217] $(date) == TIMED: UCI bench, gen 7.2 and gen 7.3 production binaries, pinned, alternating x6"
	for i in 1 2 3 4 5 6; do
		echo "[q217]   g72 $(printf 'bench\nquit\n' | taskset -c 5 "$OLD" 2>/dev/null | grep '^bench')"
		echo "[q217]   g73 $(printf 'bench\nquit\n' | taskset -c 5 "$NEW" 2>/dev/null | grep '^bench')"
	done
fi
echo "[q217] $(date) QUEUE 217 DONE"
