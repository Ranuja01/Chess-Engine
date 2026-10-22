#!/bin/bash
# 2026-10-21, queue #90 (owner: go) — re-read the SF11 pawn-ending SEARCH rules on gen 5, the fire-count way:
# each rule alone vs gen 5, self-play OFF vs ON at 30k nodes from eg_critical + kp_dense starts (150 each), then an SF18 d16 move
# test at the first-divergence points. Node-limited ⇒ safe while the owner games. JOBS=3 (≤ 3 procs).
# The 10-15 reads (#69 d10 bias, #71 NULL_NPM_GUARD −1.4 ± 17.9 node-limited) ran with the false-draw bugs live.
# Registered: PE_PASSER_EXT the likeliest positive (fixed better > worse on kp; it alone cut MSE −2.1% on 10-15);
# NULL_NPM_GUARD / PE_PRUNE_GATE fire only in pawn-only positions ⇒ few divergences, mean Δ ≈ 0.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
B="$ND/build_uci/chess_engine_uci"
SF="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/stockfish_18_linux/stockfish-ubuntu-x86-64-avx2"
cd "$ND" || exit 1
echo "[q90] $(date) start"
FP=$(printf 'bench\nquit\n' | "$B" 2>/dev/null | grep "^bench")
echo "[q90] $FP"
echo "$FP" | grep -q "solved 258/300 nodes 47689586" || { echo "[q90] ☠️ BENCH MISMATCH (expected gen 5) — aborting"; exit 1; }
for RULE in PE_PASSER_EXT=1 NULL_NPM_GUARD=1 PE_PRUNE_GATE=1; do
  tag=$(echo "${RULE%%=*}" | tr 'A-Z' 'a-z')
  echo "[q90] $(date) $RULE"
  bash selfplay/overnight_runner.sh pyrun diagnostics/_fix_firecount.py BIN="$B" N=150 NODES=30000 PLIES=200 JOBS=3 SEED=21 \
    SETS=eg:selfplay/eg_critical_fens.txt,kp:selfplay/kp_dense_fens.txt ON="$RULE" \
    SF="$SF" JUDGE_DEPTH=16 OUT=/mnt/e/chess_data/pe_rules_$tag.csv > /mnt/e/chess_data/pe_rules_$tag.log 2>&1
  echo "[q90] $(date) $RULE done (rc $?): $(grep -E '^  (ALL|eg |kp )' /mnt/e/chess_data/pe_rules_$tag.log | tr -s ' ' | tr '\n' '|')"
done
echo "[q90] $(date) QUEUE 90 DONE"
