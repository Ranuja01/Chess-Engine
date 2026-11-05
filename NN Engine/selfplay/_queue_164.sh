#!/bin/bash
# 2026-11-05 evening, queue #164: two cheap data jobs for the box time left after #162 / #163, each answering a question the
# retune will face (owner 11-05: more data is fine if it helps; the first fit's learning curve decides whether volume does).
#  1 LABEL NOISE: SF18 d18 single-best on 10,000 positions (2,500 per eval stage, the first rows of each stage block of
#    CRIT_fens.csv) vs their corpus d14 labels. If d14 is noisy, deeper labels (cheap) improve the target more than rows.
#  2 CUR slice: 40,000 positions from the 78k fastchess games on E: since 10-19 (gens 4-6.7, self-play + Mediocre; the
#    corpus own-play slices are from games before 10-13), by-game ids, EG_QUOTA 0.4, disjoint from every corpus slice;
#    SF18 d14 K=1 labels; gen 6.7's d10 pass (ours1105, MOVE=1). Fresh positions from the engine as it plays now.
# At most 3 processes. Hard stop 2026-11-06 09:30 (everything here is resumable). Stop: touch /mnt/e/chess_data/q164.stop.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
D=/mnt/e/chess_data/retune
LOGD=/mnt/e/chess_data/q164
STOP=/mnt/e/chess_data/q164.stop
CUTOFF=$(date -d '2026-11-06 09:30' +%s)
TAG=ours1105
PY=/home/ranuja/anaconda3/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"; mkdir -p "$LOGD"
echo "[q164] $(date) waiting for queues #162 and #163 (stop: $STOP)"
until grep -qE 'QUEUE 162 (DONE|ABORTED)' /mnt/e/chess_data/q162.log 2>/dev/null && grep -qE 'QUEUE 163 (DONE|ABORTED)' /mnt/e/chess_data/q163.log 2>/dev/null; do
  [ -e "$STOP" ] && { echo "[q164] stop file present — QUEUE 164 DONE (nothing run)"; exit 0; }
  sleep 60
done
sleep 20
pgrep -f "fastchess-linux-x86-64/fastchess|vs_sf.py|_depth_residual_pass|_build_regret_set" >/dev/null && { echo "[q164] another data job is running — QUEUE 164 ABORTED"; exit 1; }
echo "[q164] $(date '+%Y-%m-%d %H:%M:%S') start"
left() { echo $(( CUTOFF - $(date +%s) )); }
[ $(left) -gt 600 ] || { echo "[q164] past the cut-off — QUEUE 164 DONE (nothing run)"; exit 0; }

# ── 1 label noise: d18 on 10k (2 processes) while CUR is sampled (1 process) ────────────────────────────────────────────
if [ ! -s $D/D18_fens.csv ]; then
  "$PY" - "$D" <<'EOF'
import csv, os, sys
D = sys.argv[1]
rows = list(csv.DictReader(open(os.path.join(D, "CRIT_fens.csv"), newline="")))
out, n = [], {}
for r in rows:
    if n.get(r["stage"], 0) < 2500:
        out.append(r); n[r["stage"]] = n.get(r["stage"], 0) + 1
with open(os.path.join(D, "D18_fens.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["fen", "stage", "split", "group"]); w.writeheader(); w.writerows(out)
print("[q164] D18 sample: %d rows %s" % (len(out), n))
EOF
fi
for i in 0 1; do head -1 $D/D18_fens.csv > $D/D18_fens_s$i.csv; done
tail -n +2 $D/D18_fens.csv | awk -v B="$D/D18_fens" '{print >> (B "_s" (NR-1)%2 ".csv")}'
for i in 0 1; do
  ( timeout $(left) bash "$R" pyrun diagnostics/_build_regret_set.py IN=$D/D18_fens_s$i.csv K=1 SF_DEPTH=18 OUT=$D/D18_sf18d18_s$i.csv \
      > $LOGD/d18_$i.log 2>&1; echo "[q164] $(date) d18 s$i: rc $? · $(grep -vc '^fen,' $D/D18_sf18d18_s$i.csv 2>/dev/null) rows" ) &
done
EXC=$D/A_ownplay_fens.csv,$D/A2_fens.csv,$D/V1_fens.csv,$D/KP_fens.csv,$D/GAUNT_fens.csv,$D/VARG_fens.csv,$D/ODDS_fens.csv
[ -s $D/CUR_fens.csv ] || timeout $(left) bash "$R" pyrun diagnostics/_build_game_regret_set.py "GAMES_PGN=/mnt/e/chess_data/fastchess/*.pgn" \
    N=40000 PLY_STRIDE=8 EG_QUOTA=0.4 MAX_GAMES=5000 EXCLUDE_EXTRA=$EXC SAMPLE_ONLY=$D/CUR_fens.csv 2>&1 | grep -E "SAMPLE_ONLY|rror" | sed 's/^/[q164]   /'
wait
[ -s $D/CUR_fens.csv ] || { echo "[q164] ☠️ CUR sampling produced nothing — QUEUE 164 DONE"; exit 1; }

# ── 2 CUR labels (2 processes), then gen 6.7's d10 pass (3 processes) ─────────────────────────────────────────────────
for i in 0 1; do head -1 $D/CUR_fens.csv > $D/CUR_fens_s$i.csv; done
tail -n +2 $D/CUR_fens.csv | awk -v B="$D/CUR_fens" '{print >> (B "_s" (NR-1)%2 ".csv")}'
for i in 0 1; do
  ( timeout $(left) bash "$R" pyrun diagnostics/_build_regret_set.py IN=$D/CUR_fens_s$i.csv K=1 SF_DEPTH=14 OUT=$D/CUR_fens_sf18_s$i.csv \
      > $LOGD/cur_label_$i.log 2>&1; echo "[q164] $(date) CUR labels s$i: rc $? · $(grep -vc '^fen,' $D/CUR_fens_sf18_s$i.csv 2>/dev/null) rows" ) &
done
wait
[ -e "$STOP" ] && { echo "[q164] stop file present — QUEUE 164 DONE"; exit 0; }
for i in 0 1; do
  for k in 0 1 2; do
    [ $(jobs -rp | wc -l) -ge 3 ] && wait -n
    ( while :; do
        timeout 3600 env V2_PRESET=shipped PRESET=LONG_FORMAT MAX_DEPTH=10 USE_OPENING_BOOK=0 RESIGN_THRESHOLD=-99999999 \
          bash "$R" pyrun diagnostics/_depth_residual_pass.py IN=$D/CUR_fens_sf18_s$i.csv OUT=$D/CUR_${TAG}_d10_f$i.csv SHARD=$k/3 \
          CHUNK=800 MOVE=1 >> $LOGD/cur_d10_${i}_$k.log 2>&1
        rc=$?; [ $rc -ne 3 ] && break
        { [ -e "$STOP" ] || [ $(left) -le 0 ]; } && break
      done; echo "[q164] $(date) CUR d10 f$i k$k: rc $rc" ) &
  done
done
wait
echo "[q164] $(date) d18 $(cat $D/D18_sf18d18_s*.csv 2>/dev/null | grep -vc '^fen,') / 10000 · CUR labels $(cat $D/CUR_fens_sf18_s*.csv 2>/dev/null | grep -vc '^fen,') · CUR d10 $(cat $D/CUR_${TAG}_d10_f*.csv 2>/dev/null | grep -vc '^fen,')"
echo "[q164] $(date) QUEUE 164 DONE"
