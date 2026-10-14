#!/bin/bash
# 2026-10-14 night, queue #66 — POT DATA: endgame enrichment while the machine is otherwise idle (after #65's SPSA).
# The corpus is thin exactly where POT will be fitted (pawn endings 1.6% of our endgames, KRKP 0.75%, KPKP 0.15%) and the depth
# target covers only slice A. (1) sample ~60k positions carrying a Stockfish endgame.cpp signature from the nested own-play
# games, ≤ 4,000 per signature (EG_SIG_QUOTA), disjoint from the whole corpus, with game ids; (2) SF18 d14 labels (4 shards);
# (3) our gen-3 d10 depth pass on them (the depth target for POT's nested fit). Fixed-depth only. ≤ 4 engines.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
D=/mnt/e/chess_data/retune
until grep -qE "QUEUE 65 DONE|☠️" /mnt/e/chess_data/q65_anchor_spsa.log 2>/dev/null; do sleep 120; done
cd "$ND" || exit 1
echo "[q66] $(date) start"
FP=$(env V2_PRESET=shipped bash "$R" wac fp_v2 2>&1 | grep -E "SOLVED|NODES" | tr '\n' ' ')
echo "$FP" | grep -q "259/300.*47869841" || { echo "[q66] ☠️ FINGERPRINT MISMATCH (expected gen 3) — aborting"; exit 1; }
EXC=$D/A_ownplay_fens.csv,$D/A2_fens.csv,$D/V1_fens.csv,$D/KP_fens.csv,$D/GAUNT_fens.csv,$D/VAR_sf18_s0.csv,$D/VAR_sf18_s1.csv,$D/VAR_sf18_s2.csv,$D/VAR_sf18_s3.csv
[ -s $D/EGX_fens.csv ] || bash "$R" pyrun diagnostics/_build_game_regret_set.py N=60000 EG_SIG_QUOTA=4000 MAX_GAMES=120000 PLY_STRIDE=2 \
    'GAMES_GLOB=*/*/game.jsonl' EXCLUDE_EXTRA=$EXC SAMPLE_ONLY=$D/EGX_fens.csv 2>&1 | grep -E "SAMPLE_ONLY|rror" | sed 's/^/[q66] /'
head -1 $D/EGX_fens.csv > /tmp/q66_hdr.csv
for i in 0 1 2 3; do cp /tmp/q66_hdr.csv $D/EGX_fens_s$i.csv; done
tail -n +2 $D/EGX_fens.csv | awk -v B="$D/EGX_fens" '{print >> (B "_s" (NR-1)%4 ".csv")}'
for i in 0 1 2 3; do
  bash "$R" pyrun diagnostics/_build_regret_set.py IN=$D/EGX_fens_s$i.csv K=1 SF_DEPTH=14 OUT=$D/EGX_fens_sf18_s$i.csv > $D/q66_label_s$i.log 2>&1 &
done
wait
{ head -1 $D/EGX_fens_sf18_s0.csv; for i in 0 1 2 3; do tail -n +2 $D/EGX_fens_sf18_s$i.csv; done; } > $D/EGX_sf18.csv
echo "[q66] $(date) EGX labelled: $(($(wc -l < $D/EGX_sf18.csv) - 1))"
for s in 0 1 2 3; do
  ( until env V2_PRESET=shipped PRESET=LONG_FORMAT MAX_DEPTH=10 USE_OPENING_BOOK=0 bash "$R" pyrun \
        diagnostics/_depth_residual_pass.py IN=$D/EGX_sf18.csv OUT=$D/EGX_ours1012_d10.csv SHARD=$s/4 CHUNK=800 \
        > $D/q66_d10_$s.log 2>&1; [ $? -ne 3 ]; do :; done ) &
done
wait
echo "[q66] $(date) EGX depth rows: $(cat $D/EGX_ours1012_d10_s*of4.csv 2>/dev/null | grep -vc '^fen,')"
echo "[q66] $(date) QUEUE 66 DONE"
