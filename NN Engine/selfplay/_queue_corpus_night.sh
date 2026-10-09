#!/bin/bash
# 2026-10-09 night, queue #49 (owner: "a huge magnitude … diverse, even whacky positions"). After slice A (#48):
#  1. LABEL ~800k more diverse positions, SF18 d14 single-best (~420k/h on 4 engines), for the retune's STATIC component:
#     A2 own-play v2-era games at a denser stride · V1 the older v1-era games (different play, different positions; labels are
#     engine-independent) · KP 20k K+pawns stress (SEED 77, never 41 = the held-out) · VAR 100k "whacky" piece-swap/960 positions
#     (_build_variant_regret_set, castling off). All disjoint from A and from every depth/held-out/regret set.
#  2. OUR d10 DEPTH PASS on slice A (140k) — the depth-target rows (~5.5 rows/s ⇒ ~7 h).
# Every step is RESUMABLE (re-running skips done rows) ⇒ pausable on the owner's word: kill the process group, relaunch later.
# Fixed-depth only (safe in the owner's evening window). ≤ 4 engines at any time. Outputs on E:\chess_data\retune.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
D=/mnt/e/chess_data/retune
until grep -q "QUEUE 48 DONE" $D/q48.log 2>/dev/null; do sleep 60; done
cd "$ND" || exit 1
echo "[q49] $(date) start"
shard4() {   # $1 = csv with a fen column → $1_s{0..3}.csv
  head -1 "$1" > /tmp/q49_hdr.csv
  for i in 0 1 2 3; do cp /tmp/q49_hdr.csv "${1%.csv}_s$i.csv"; done
  tail -n +2 "$1" | awk -v B="${1%.csv}" '{print >> (B "_s" (NR-1)%4 ".csv")}'
}
label4() {   # $1 = base name: labels ${1}_s{i}.csv → ${1}_sf18_s{i}.csv in parallel
  for i in 0 1 2 3; do
    bash "$R" pyrun diagnostics/_build_regret_set.py IN=$D/${1}_s$i.csv K=1 SF_DEPTH=14 OUT=$D/${1}_sf18_s$i.csv \
        > $D/q49_${1}_s$i.log 2>&1 &
  done
  wait
  echo "[q49] $(date) $1 labelled: $(cat $D/${1}_sf18_s*.csv 2>/dev/null | grep -vc '^fen,')"
}
# ── 1a. A2: own-play v2-era, denser stride, disjoint from A ──
[ -s $D/A2_fens.csv ] || bash "$R" pyrun diagnostics/_build_game_regret_set.py N=400000 EG_QUOTA=0.4 MAX_GAMES=40000 PLY_STRIDE=3 \
    'GAMES_GLOB=*/*/game.jsonl' EXCLUDE_EXTRA=$D/A_ownplay_fens.csv SAMPLE_ONLY=$D/A2_fens.csv 2>&1 | grep -E "SAMPLE_ONLY|rror"
[ -s $D/A2_fens_s3.csv ] || shard4 $D/A2_fens.csv
label4 A2_fens
# ── 1b. V1: the older flat-layout games (v1 era), disjoint from A and A2 ──
[ -s $D/V1_fens.csv ] || bash "$R" pyrun diagnostics/_build_game_regret_set.py N=300000 EG_QUOTA=0.4 MAX_GAMES=60000 PLY_STRIDE=5 \
    'GAMES_GLOB=*/*.jsonl' EXCLUDE_EXTRA=$D/A_ownplay_fens.csv,$D/A2_fens.csv SAMPLE_ONLY=$D/V1_fens.csv 2>&1 | grep -E "SAMPLE_ONLY|rror"
[ -s $D/V1_fens_s3.csv ] || shard4 $D/V1_fens.csv
label4 V1_fens
# ── 1c. KP: K+pawns stress, SEED 77 (41 is the held-out stress set) ──
if [ ! -s $D/KP_fens.csv ]; then
  bash "$R" pyrun diagnostics/gen_kp_fens.py N=20000 SEED=77 MIX=pure,minor,rook,imbal,dense OUT=$D/KP_fens.txt 2>&1 | tail -1
  { echo fen; grep -v '^#' $D/KP_fens.txt | sed 's/ *#.*$//' | grep -v '^$'; } > $D/KP_fens.csv
fi
[ -s $D/KP_fens_s3.csv ] || shard4 $D/KP_fens.csv
label4 KP_fens
# ── 1d. VAR: whacky piece-swap / 960 positions (generator labels itself; SHARD-split, resumable) ──
for i in 0 1 2 3; do
  bash "$R" pyrun diagnostics/_build_variant_regret_set.py N=100000 K=1 SF_DEPTH=14 SHARD=$i/4 WALK_MAX=45 SEED=20261009 \
      OUT=$D/VAR_sf18_s$i.csv > $D/q49_VAR_s$i.log 2>&1 &
done
wait
echo "[q49] $(date) VAR labelled: $(cat $D/VAR_sf18_s*.csv 2>/dev/null | grep -vc '^fen,')"
# ── 2. our d10 depth pass on slice A ──
[ -s $D/A_sf18.csv ] || { head -1 $D/A_sf18_s0.csv > $D/A_sf18.csv; for i in 0 1 2 3; do tail -n +2 $D/A_sf18_s$i.csv >> $D/A_sf18.csv; done; }
FP=$(env V2_PRESET=shipped bash "$R" wac fp_v2 2>&1 | grep -E "SOLVED|NODES" | tr '\n' ' ')
echo "[q49] v2 shipped: $FP"
echo "$FP" | grep -q "254/300.*50622239" || { echo "[q49] ☠️ FINGERPRINT MISMATCH — aborting the depth pass"; exit 1; }
for s in 0 1 2 3; do
  ( until env V2_PRESET=shipped PRESET=LONG_FORMAT MAX_DEPTH=10 USE_OPENING_BOOK=0 bash "$R" pyrun \
        diagnostics/_depth_residual_pass.py IN=$D/A_sf18.csv OUT=$D/A_ours1004_d10.csv SHARD=$s/4 CHUNK=800 \
        > $D/q49_d10_$s.log 2>&1; [ $? -ne 3 ]; do :; done ) &
done
wait
echo "[q49] $(date) A depth rows: $(cat $D/A_ours1004_d10_s*of4.csv 2>/dev/null | grep -vc '^fen,')"
echo "[q49] $(date) QUEUE 49 DONE"
