#!/bin/bash
# 2026-10-11 night, queue #54 — prepare the NEXT retune fits on the gen-2 base (owner OK 10-11: piece-value knobs).
# After #53 (the gen-2 d10 depth pass): REBUILD with the V2_VAL_{N,B,R,Q}_{MG,EG} knobs (all 0 ⇒ byte-identical), verify
# fingerprints (gen 2 256 / 47,473,313 · gen 1 254 / 50,622,239 · v1 250 / 35,310,778) + symmetry, then the feature/static
# export of the gen-2 ship (feat_gen2_c*.npz) — the base the next fits read. Fixed-depth / static only (safe any hour).
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
D=/mnt/e/chess_data/retune
until grep -q "QUEUE 53 DONE" /mnt/e/chess_data/q53_depth_gen2.log 2>/dev/null; do sleep 60; done
cd "$ND" || exit 1
echo "[q54] $(date) start"
bash "$R" build 2>&1 | tail -3
[ -n "$(ls ChessAI*.so 2>/dev/null)" ] || { echo "[q54] ☠️ BUILD FAILED — no .so"; exit 1; }
F2=$(env V2_PRESET=shipped bash "$R" wac fp_v2 2>&1 | grep -E "SOLVED|NODES" | tr '\n' ' ')
F1=$(env V2_PRESET=shipped V2_SHIP_GEN=1 bash "$R" wac fp_g1 2>&1 | grep -E "SOLVED|NODES" | tr '\n' ' ')
FV1=$(bash "$R" wac fp_v1 2>&1 | grep -E "SOLVED|NODES" | tr '\n' ' ')
echo "[q54] gen2: $F2 | gen1: $F1 | v1: $FV1"
echo "$F2" | grep -q "256/300.*47473313" || { echo "[q54] ☠️ gen-2 FINGERPRINT MISMATCH — aborting"; exit 1; }
echo "$F1" | grep -q "254/300.*50622239" || { echo "[q54] ☠️ gen-1 FINGERPRINT MISMATCH — aborting"; exit 1; }
echo "$FV1" | grep -q "250/300.*35310778" || { echo "[q54] ☠️ v1 FINGERPRINT MISMATCH — aborting"; exit 1; }
SYM=$(env V2_PRESET=shipped bash "$R" pyrun diagnostics/_eval_symmetry.py N=4000 2>&1 | grep -E "violations")
echo "[q54] gen2 symmetry: $(echo "$SYM" | tr '\n' ' ')"
[ "$(echo "$SYM" | grep -c 'violations 0 ')" = "2" ] || { echo "[q54] ☠️ symmetry FAILED — aborting"; exit 1; }
env V2_PRESET=shipped bash "$R" pyrun diagnostics/_px_export.py IN=$D/corpus_v1.csv OUT=$D/feat_gen2.npz 2>&1 | grep -E "CORPUS|chunk"
echo "[q54] $(date) QUEUE 54 DONE"
