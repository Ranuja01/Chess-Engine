#!/bin/bash
# 2026-10-09, queue #48 — RETUNE CORPUS, slice A (owner: start now, pausable). 140,000 own-play positions (nested v2-era games,
# 50% endgame-leaning, disjoint from every depth/held-out/regret set) labelled with SF18 d14 SINGLE BEST (K=1), 4 shards in
# parallel (4 engines). RESUMABLE: _build_regret_set.py skips FENs already in its OUT, so to PAUSE kill this queue's process
# group and to RESUME just relaunch this script — at most the last ≤ 50 rows per shard are redone. Outputs on E: (not synced).
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
D=/mnt/e/chess_data/retune
cd "$ND" || exit 1
echo "[q48] $(date) start"
if [ ! -s $D/A_shard3.csv ]; then
  head -1 $D/A_ownplay_fens.csv > /tmp/q48_hdr.csv
  for i in 0 1 2 3; do cp /tmp/q48_hdr.csv $D/A_shard$i.csv; done
  tail -n +2 $D/A_ownplay_fens.csv | awk -v D=$D 'BEGIN{FS=","} {print >> (D "/A_shard" (NR-1)%4 ".csv")}'
  echo "[q48] sharded: $(wc -l < $D/A_shard0.csv) / $(wc -l < $D/A_shard1.csv) / $(wc -l < $D/A_shard2.csv) / $(wc -l < $D/A_shard3.csv) lines"
fi
for i in 0 1 2 3; do
  bash "$R" pyrun diagnostics/_build_regret_set.py IN=$D/A_shard$i.csv K=1 SF_DEPTH=14 OUT=$D/A_sf18_s$i.csv \
      > $D/q48_s$i.log 2>&1 &
done
wait
echo "[q48] $(date) rows: $(cat $D/A_sf18_s*.csv 2>/dev/null | grep -vc '^fen,')"
echo "[q48] $(date) QUEUE 48 DONE"
