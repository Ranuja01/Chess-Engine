#!/bin/bash
# 2026-10-09, queue #47 — verify the CONN plumbing (retune plumbing 2/n). (1) CLOSURE: the pawn_struct block (66-76 + CONN
# 235-336) must reproduce the engine's published pawn_struct under the SHIP (connected on) within its truncation budget;
# (2) TABLE MODE: CONN_V2=1 with no file starts every cell at the live term — totals must track the ship to within rounding
# (≤ ~1 mp per connected pawn); (3) symmetry under CONN_V2=1. ≤ 4 engines. No build.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
O=/mnt/e/chess_data/texel/revival
cd "$ND" || exit 1
echo "[q47] $(date) start"
env V2_PRESET=shipped LIMIT=3000 OUT=/mnt/e/chess_data/texel/conn_closure.npz bash "$R" pyrun diagnostics/_texel_feature_pass.py 2>&1 \
  | grep -v -E "^\[toggles\]|EVAL_ARM" | tail -14
env V2_PRESET=shipped CONN_V2=1 bash "$R" pyrun diagnostics/_revival_screen.py MODE=dump OUT=$O/conn_table.csv 2>&1 | grep -E "CONN_V2|REVIVAL DUMP|rror"
env V2_PRESET=shipped bash "$R" pyrun diagnostics/_revival_screen.py MODE=dump OUT=$O/off_now.csv 2>&1 | grep -E "REVIVAL DUMP|rror"
SYM=$(env V2_PRESET=shipped CONN_V2=1 bash "$R" pyrun diagnostics/_eval_symmetry.py N=4000 2>&1 | grep -E "violations")
echo "[q47] CONN_V2=1 symmetry: $(echo "$SYM" | tr '\n' ' ')"
echo "[q47] $(date) QUEUE 47 DONE"
