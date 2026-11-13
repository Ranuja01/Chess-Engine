#!/bin/bash
# 2026-11-13 ~11:40, queue #223: install the MOVEGEN_FAST speed build (owner 11-13: "It seems like pure speed so i think we can
# ship it"), then the lazy generation cost read. Starts when queue #222 has ended (its early pruning read is timed).
#  1 Install chess_engine_uci_g73fpgo (built and gated in #222 step A: bench 249 / 37,279,382 and the gen 7.2 / 7.1 / 6.7 / v1
#    fingerprints; pinned bench -2.1% time, nps +2.1%; #219 -3.52% time to depth 13) as production. Gen 7.3's first build
#    stays as chess_engine_uci_g73pgo. Gated again on the installed file: default 249 / 37,279,382, MOVEGEN_FAST=0 the same.
#  2 Cython module rebuild (the current .so backed up to /mnt/e/chess_data/so_backup_g73/) and its fingerprints; the toggles
#    line must show MOVEGEN_FAST=1 under V2_PRESET=shipped and 0 with MOVEGEN_FAST=0.
#  3 The new default is live in the installed binary: cachegrind instructions (load-insensitive), default vs MOVEGEN_FAST=0,
#    12 game positions at depth 8. REGISTERED: nodes equal and the default executes fewer instructions (equal instructions
#    would mean the default did not change).
#  4 Lazy generation cost read (owner 11-13: "if ... we only generated that, shouldn't it automatically save time?"):
#    build_uci/sv2lazy_pgo, SEARCH_ARM=1, SV2_PRUNE_EARLY=1, SV2_LAZY_GEN 0 / 2 / 1, the same 12 positions at depth 10,
#    cachegrind instructions plus the per-function report. Mode 2 visits the same nodes as 0, so 2 - 0 = the cost of the
#    captures-only pass plus generating the captures a second time, at every generated list; 1 - 0 per node = the net saving.
#    REGISTERED: 2 - 0 between +1% and +5% of instructions; 1 - 0 per node between -1% and -4%. No verdict rides on it: it
#    says where the saving goes and whether a phase 2 that generates only the quiet moves is worth building.
# Stop: touch /mnt/e/chess_data/q223.stop (checked between steps). Never edit this file, the binaries or the sources while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
OLD="$ND/build_uci/chess_engine_uci_g73pgo"
NEW="$ND/build_uci/chess_engine_uci_g73fpgo"
LAZY="$ND/build_uci/sv2lazy_pgo"
G=/mnt/e/chess_data/sv2_speed/game12.txt
O=/mnt/e/chess_data/q223
STOP=/mnt/e/chess_data/q223.stop
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"; mkdir -p "$O"
show() { sed "s/^/[q223]   /"; }
stopped() { [ -e "$STOP" ] && { echo "[q223] $(date) stop file present before $1 — QUEUE 223 DONE"; exit 0; }; }
k() { printf 'setoption name Knobs value %s\nbench\nquit\n' "$2" | "$1" 2>/dev/null | grep '^bench'; }
irefs() { grep -E 'I *refs' "$1" | grep -oE '[0-9,]+$' | tr -d ,; }
echo "[q223] $(date '+%Y-%m-%d %H:%M:%S') queued; waiting for queue #222 to end"
until grep -qE 'QUEUE 222 (DONE|ABORTED)' /mnt/e/chess_data/q222.log; do stopped "the start"; sleep 30; done
echo "[q223] $(date '+%Y-%m-%d %H:%M:%S') start"
cmp -s build_uci/chess_engine_uci "$OLD" && echo "[q223] backup ok: production == chess_engine_uci_g73pgo" \
	|| { echo "[q223] ☠️ production differs from chess_engine_uci_g73pgo — QUEUE 223 ABORTED"; exit 1; }

# ── 1 install ─────────────────────────────────────────────────────────────────────────────────────────────────────────
d=$(printf 'bench\nquit\n' | "$NEW" 2>/dev/null | grep '^bench')
d0=$(k "$NEW" MOVEGEN_FAST=0)
echo "[q223] candidate default (want 249 / 37279382):        $d"
echo "[q223] candidate MOVEGEN_FAST=0 (want the same):       $d0"
echo "[q223] id: $(printf 'uci\nquit\n' | "$NEW" 2>/dev/null | grep '^id name')"
echo "$d" | grep -q "solved 249/300 nodes 37279382 " && echo "$d0" | grep -q "solved 249/300 nodes 37279382 " \
	|| { echo "[q223] ☠️ fingerprints differ — production NOT replaced, QUEUE 223 ABORTED"; exit 1; }
cp "$NEW" build_uci/.chess_engine_uci.tmp && mv -f build_uci/.chess_engine_uci.tmp build_uci/chess_engine_uci \
	&& cmp -s build_uci/chess_engine_uci "$NEW" && echo "[q223] $(date) INSTALLED: build_uci/chess_engine_uci == chess_engine_uci_g73fpgo" \
	|| { echo "[q223] ☠️ install failed — QUEUE 223 ABORTED"; exit 1; }

# ── 2 Cython module ───────────────────────────────────────────────────────────────────────────────────────────────────
stopped "the Cython rebuild"
mkdir -p /mnt/e/chess_data/so_backup_g73 && cp ChessAI.*.so /mnt/e/chess_data/so_backup_g73/ \
	&& echo "[q223] .so backed up to /mnt/e/chess_data/so_backup_g73/" || { echo "[q223] ☠️ .so backup failed — QUEUE 223 ABORTED"; exit 1; }
bash "$R" build 2>&1 | grep -iE 'error|copying|FAILED' | tail -n 3 | show
ls ChessAI.*.so >/dev/null 2>&1 || { echo "[q223] ☠️ Cython build left no .so — restore /mnt/e/chess_data/so_backup_g73/"; exit 1; }
bash "$R" wac fp_g73f V2_PRESET=shipped > "$O/cy_default.txt" 2>&1
bash "$R" wac fp_g73f_mg0 V2_PRESET=shipped MOVEGEN_FAST=0 > "$O/cy_mg0.txt" 2>&1
bash "$R" wac fp_g73f_v1 > "$O/cy_v1.txt" 2>&1
echo "[q223] Cython shipped (want 249 / 37279382, MOVEGEN_FAST=1): $(grep -E '^SOLVED|^NODES|^EBF' "$O/cy_default.txt" | tr '\n' ' ')· $(grep -oE 'MOVEGEN_FAST=[0-9]+' "$O/cy_default.txt" | head -1)"
echo "[q223] Cython MOVEGEN_FAST=0 (want the same, 0):            $(grep -E '^SOLVED|^NODES|^EBF' "$O/cy_mg0.txt" | tr '\n' ' ')· $(grep -oE 'MOVEGEN_FAST=[0-9]+' "$O/cy_mg0.txt" | head -1)"
echo "[q223] Cython v1 (want 250 / 35310778):                     $(grep -E '^SOLVED|^NODES|^EBF' "$O/cy_v1.txt" | tr '\n' ' ')"

# ── 3 the default is live: instructions, default vs MOVEGEN_FAST=0 ────────────────────────────────────────────────────
stopped "the live check"
for m in 1 0; do
	printf "bench $G 8\nquit\n" | env MOVEGEN_FAST=$m valgrind --tool=cachegrind --cache-sim=no \
		--cachegrind-out-file="$O/cg_mg$m.out" build_uci/chess_engine_uci > "$O/bench_mg$m.txt" 2> "$O/cg_mg$m.log"
	echo "[q223] production MOVEGEN_FAST=$m: $(grep -oE 'solved [0-9]+/[0-9]+ nodes [0-9]+' "$O/bench_mg$m.txt") · Ir $(irefs "$O/cg_mg$m.log")"
done
echo "[q223] default - MOVEGEN_FAST=0 instructions: $(awk -v a="$(irefs "$O/cg_mg0.log")" -v b="$(irefs "$O/cg_mg1.log")" 'BEGIN { printf "%+.3f%%", 100 * (b - a) / a }')"

# ── 4 lazy generation cost read ───────────────────────────────────────────────────────────────────────────────────────
for m in 0 2 1; do
	stopped "lazy $m"
	printf "bench $G 10\nquit\n" | env SEARCH_ARM=1 SV2_PRUNE_EARLY=1 SV2_LAZY_GEN=$m valgrind --tool=cachegrind --cache-sim=no \
		--cachegrind-out-file="$O/cg_lazy$m.out" "$LAZY" > "$O/bench_lazy$m.txt" 2> "$O/cg_lazy$m.log"
	cg_annotate "$O/cg_lazy$m.out" > "$O/cg_lazy$m.txt" 2>&1
	echo "[q223] lazy $m: $(grep -oE 'solved [0-9]+/[0-9]+ nodes [0-9]+' "$O/bench_lazy$m.txt") · Ir $(irefs "$O/cg_lazy$m.log")"
done
n() { grep -oE 'nodes [0-9]+' "$O/bench_lazy$1.txt" | grep -oE '[0-9]+'; }
echo "[q223] lazy 2 - 0 instructions (same nodes): $(awk -v a="$(irefs "$O/cg_lazy0.log")" -v b="$(irefs "$O/cg_lazy2.log")" 'BEGIN { printf "%+.3f%%", 100 * (b - a) / a }')"
echo "[q223] lazy 1 - 0 instructions per node:     $(awk -v a="$(irefs "$O/cg_lazy0.log")" -v b="$(irefs "$O/cg_lazy1.log")" -v na="$(n 0)" -v nb="$(n 1)" 'BEGIN { printf "%+.3f%% (nodes %+.3f%%)", 100 * (b / nb - a / na) / (a / na), 100 * (nb - na) / na }')"
echo "[q223] $(date) QUEUE 223 DONE"
