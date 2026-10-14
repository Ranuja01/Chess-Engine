#!/bin/bash
# 2026-10-14, queue #65 — the retune's last pieces while the owner is busy (interview day):
#  (1) RE-ANCHOR the external judge: ship gen 3 scores ~54.8% vs SF18 @1000 (4 seeds) — the judge drifts weaker with each
#      ship. Ship vs SF18 @1000 / 1500 / 2000 nodes, 500 g each, fresh seeds 230-232 ⇒ pick the node budget nearest 50% for
#      POT's gates (memory the-sf18-gauntlet-anchor-drifted-too-weak).
#  (2) SPSA on the scalars the linear fit cannot reach (spsa_spec_gen3_scalars.json: KS curve MAX / HALF / EG_PCT — ONSET kept
#      450, owner rule — and winnability BASE / SP / OCB), eval lane fixed d6, two COLD-START runs (seeds 1, 2), 150 iters × 48
#      paired games, a 2.0 c 0.45 (memory spsa-tuning-needs-replication-not-convergence-stats: only knobs that AGREE across
#      both runs count; ratify by games). Node-limited / fixed-depth only (safe any hour). ≤ 4 engines.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
SF="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/stockfish_18_linux/stockfish-ubuntu-x86-64-avx2"
cd "$ND" || exit 1
echo "[q65] $(date) start"
FP=$(env V2_PRESET=shipped bash "$R" wac fp_v2 2>&1 | grep -E "SOLVED|NODES" | tr '\n' ' ')
echo "$FP" | grep -q "259/300.*47869841" || { echo "[q65] ☠️ FINGERPRINT MISMATCH (expected gen 3) — aborting"; exit 1; }
s=230
for n in 1000 1500 2000; do
  bash "$R" gauntlet 500 $n 4 anchor_g3_sf${n}_s${s} $s V2_PRESET=shipped
  echo "[q65] $(date) anchor SF18 @$n done (seed $s)"
  s=$((s + 1))
done
for sd in 1 2; do
  bash "$R" pyrun selfplay/spsa.py --spec selfplay/spsa_spec_gen3_scalars.json --iters 150 --games 48 --concurrency 4 \
    --lane eval --depth 6 --base "V2_PRESET=shipped" --adj-sf "$SF" --a 2.0 --c 0.45 --seed $sd --tag spsa_g3_scalars_s$sd \
    > /mnt/e/chess_data/q65_spsa_s$sd.log 2>&1
  echo "[q65] $(date) SPSA seed $sd done: $(tail -3 /mnt/e/chess_data/q65_spsa_s$sd.log | tr '\n' ' ')"
done
echo "[q65] $(date) QUEUE 65 DONE"
