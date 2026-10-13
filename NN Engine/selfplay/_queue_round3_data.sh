#!/bin/bash
# 2026-10-13, queue #60 — RETUNE ROUND 3 data on the gen-3 base (owner away ~1.5 days; autonomous).
#  (1) WINNABILITY CAP dumps: the gen-3 static export with each material-class cap ON at its start value (10-09 SF15.1 forms:
#      QNOQ −27/+3 · OCBX −17/+1 · ROOKE −28 · LONEMINOR −64) — the fitter prices each as a multiplier on its Δ (CAPS=).
#  (2) NEW SLICE "GAUNT": positions from games vs STRONGER opponents (SF18 @1000 gauntlets + Mediocre) — the territory a
#      stronger engine steers us into; sampled with the game id (by-game split), disjoint from the whole corpus.
#  (3) SF18 d14 single-best labels for the new slice (4 shards, resumable).
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
D=/mnt/e/chess_data/retune
cd "$ND" || exit 1
echo "[q60] $(date) start"
FP=$(env V2_PRESET=shipped bash "$R" wac fp_v2 2>&1 | grep -E "SOLVED|NODES" | tr '\n' ' ')
echo "[q60] v2 shipped: $FP"
echo "$FP" | grep -q "259/300.*47869841" || { echo "[q60] ☠️ FINGERPRINT MISMATCH (expected gen 3) — aborting"; exit 1; }
for cap in "QNOQ:POT_V2_WIN_QNOQ=-27 POT_V2_WIN_QNOQ_MINOR=3" "OCBX:POT_V2_WIN_OCBX=-17 POT_V2_WIN_OCBX_PC=1" \
           "ROOKE:POT_V2_WIN_ROOKE=-28" "LONEMINOR:POT_V2_WIN_LONEMINOR=-64"; do
  name="${cap%%:*}"; knobs="${cap#*:}"
  env V2_PRESET=shipped $knobs bash "$R" pyrun diagnostics/_px_export.py IN=$D/corpus_v1.csv OUT=$D/feat_gen3cap_$name.npz 2>&1 | grep -E "CORPUS EXPORT DONE"
  echo "[q60] $(date) cap dump $name done"
done
EXC=$D/A_ownplay_fens.csv,$D/A2_fens.csv,$D/V1_fens.csv,$D/KP_fens.csv,$D/VAR_sf18_s0.csv,$D/VAR_sf18_s1.csv,$D/VAR_sf18_s2.csv,$D/VAR_sf18_s3.csv
[ -s $D/GS_fens.csv ] || bash "$R" pyrun diagnostics/_build_game_regret_set.py N=80000 EG_QUOTA=0.4 MAX_GAMES=30000 PLY_STRIDE=4 \
    'GAMES_GLOB=g1000_*/*.jsonl' EXCLUDE_EXTRA=$EXC SAMPLE_ONLY=$D/GS_fens.csv 2>&1 | grep -E "SAMPLE_ONLY|rror"
[ -s $D/GM_fens.csv ] || bash "$R" pyrun diagnostics/_build_game_regret_set.py N=20000 EG_QUOTA=0.4 MAX_GAMES=5000 PLY_STRIDE=4 \
    'GAMES_GLOB=mediocre_*/*.jsonl' EXCLUDE_EXTRA=$EXC,$D/GS_fens.csv SAMPLE_ONLY=$D/GM_fens.csv 2>&1 | grep -E "SAMPLE_ONLY|rror"
{ head -1 $D/GS_fens.csv; tail -n +2 $D/GS_fens.csv; tail -n +2 $D/GM_fens.csv; } > $D/GAUNT_fens.csv
echo "[q60] $(date) GAUNT sampled: $(($(wc -l < $D/GAUNT_fens.csv) - 1)) positions"
head -1 $D/GAUNT_fens.csv > /tmp/q60_hdr.csv
for i in 0 1 2 3; do cp /tmp/q60_hdr.csv $D/GAUNT_fens_s$i.csv; done
tail -n +2 $D/GAUNT_fens.csv | awk -v B="$D/GAUNT_fens" '{print >> (B "_s" (NR-1)%4 ".csv")}'
for i in 0 1 2 3; do
  bash "$R" pyrun diagnostics/_build_regret_set.py IN=$D/GAUNT_fens_s$i.csv K=1 SF_DEPTH=14 OUT=$D/GAUNT_fens_sf18_s$i.csv \
      > $D/q60_label_s$i.log 2>&1 &
done
wait
echo "[q60] $(date) GAUNT labelled: $(cat $D/GAUNT_fens_sf18_s*.csv 2>/dev/null | grep -vc '^fen,')"
echo "[q60] $(date) QUEUE 60 DONE"
