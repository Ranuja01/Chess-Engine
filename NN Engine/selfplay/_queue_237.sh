#!/bin/bash
# 2026-11-14 ~12:15, queue #237: install SW5, search v2's interior as the default (owner 11-14: "if the play shows its positive
# we can ship and continue"; #225 + #236 pooled 2,000 g +15.8, 95% +3.0 ... +28.7, 0 forfeits). No timed work; done before the
# owner's 14:00 gaming.
#  1 Install chess_engine_uci_g73spgo (built and gated in #225: default 249 / 37,279,382, SEARCH_ARM=0 the same, gen 7.2 / 7.1 /
#    6.7 knobs; identity games 0 / 120 diverged) as production. The previous build stays as chess_engine_uci_g73fpgo. Gated
#    again on the installed file.
#  2 Cython module rebuilt from the committed sources (the preset sets SEARCH_ARM=1; current .so backed up to
#    /mnt/e/chess_data/so_backup_g73s/); fingerprints, and the toggles line must show SEARCH_ARM=1 shipped, 0 with SEARCH_ARM=0.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
OLD="$ND/build_uci/chess_engine_uci_g73fpgo"
NEW="$ND/build_uci/chess_engine_uci_g73spgo"
O=/mnt/e/chess_data/q237
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
mkdir -p "$O"
show() { sed "s/^/[q237]   /"; }
k() { printf 'setoption name Knobs value %s\nbench\nquit\n' "$2" | "$1" 2>/dev/null | grep '^bench'; }
echo "[q237] $(date '+%Y-%m-%d %H:%M:%S') start"
cmp -s build_uci/chess_engine_uci "$OLD" && echo "[q237] backup ok: production == chess_engine_uci_g73fpgo" \
	|| { echo "[q237] ☠️ production differs from chess_engine_uci_g73fpgo — QUEUE 237 ABORTED"; exit 1; }

# ── 1 install ─────────────────────────────────────────────────────────────────────────────────────────────────────────
d=$(printf 'bench\nquit\n' | "$NEW" 2>/dev/null | grep '^bench'); d0=$(k "$NEW" SEARCH_ARM=0)
echo "[q237] candidate default (want 249 / 37279382):  $d"
echo "[q237] candidate SEARCH_ARM=0 (want the same):   $d0"
echo "[q237] id: $(printf 'uci\nquit\n' | "$NEW" 2>/dev/null | grep '^id name')"
echo "$d" | grep -q "solved 249/300 nodes 37279382 " && echo "$d0" | grep -q "solved 249/300 nodes 37279382 " \
	|| { echo "[q237] ☠️ fingerprints differ — production NOT replaced, QUEUE 237 ABORTED"; exit 1; }
cp "$NEW" build_uci/.chess_engine_uci.tmp && mv -f build_uci/.chess_engine_uci.tmp build_uci/chess_engine_uci \
	&& cmp -s build_uci/chess_engine_uci "$NEW" && echo "[q237] $(date) INSTALLED: build_uci/chess_engine_uci == chess_engine_uci_g73spgo" \
	|| { echo "[q237] ☠️ install failed — QUEUE 237 ABORTED"; exit 1; }
echo "[q237] installed bench: $(printf 'bench\nquit\n' | build_uci/chess_engine_uci 2>/dev/null | grep '^bench')"

# ── 2 Cython module ───────────────────────────────────────────────────────────────────────────────────────────────────
mkdir -p /mnt/e/chess_data/so_backup_g73s && cp ChessAI.*.so /mnt/e/chess_data/so_backup_g73s/ \
	&& echo "[q237] .so backed up to /mnt/e/chess_data/so_backup_g73s/" || { echo "[q237] ☠️ .so backup failed — QUEUE 237 ABORTED"; exit 1; }
bash "$R" build 2>&1 | grep -iE 'error|copying|FAILED' | tail -n 3 | show
ls ChessAI.*.so >/dev/null 2>&1 || { echo "[q237] ☠️ Cython build left no .so — restoring"; cp /mnt/e/chess_data/so_backup_g73s/ChessAI.*.so .; exit 1; }
bash "$R" wac fp_g73s V2_PRESET=shipped > "$O/cy_default.txt" 2>&1
bash "$R" wac fp_g73s_a0 V2_PRESET=shipped SEARCH_ARM=0 > "$O/cy_a0.txt" 2>&1
bash "$R" wac fp_g73s_v1 > "$O/cy_v1.txt" 2>&1
echo "[q237] Cython shipped (want 249 / 37279382, SEARCH_ARM=1): $(grep -E '^SOLVED|^NODES|^EBF' "$O/cy_default.txt" | tr '\n' ' ')· $(grep -oE ' SEARCH_ARM=[0-9]+' "$O/cy_default.txt" | head -1)"
echo "[q237] Cython SEARCH_ARM=0 (want the same, 0):            $(grep -E '^SOLVED|^NODES|^EBF' "$O/cy_a0.txt" | tr '\n' ' ')· $(grep -oE ' SEARCH_ARM=[0-9]+' "$O/cy_a0.txt" | head -1)"
echo "[q237] Cython v1 (want 250 / 35310778):                   $(grep -E '^SOLVED|^NODES|^EBF' "$O/cy_v1.txt" | tr '\n' ' ')"
echo "[q237] $(date) QUEUE 237 DONE"
