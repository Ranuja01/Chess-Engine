#!/bin/bash
# 2026-11-05, queue #163 (relaunched 16:10): stage-2 (criticality) labels for the retune. The corpus labels are SF18 d14
# SINGLE best (K=1), which cannot tell a critical position (one move matters) from a quiet one. SF18 labels at ~22 rows/s
# per process (#161), so this no longer waits for the depth passes: it starts after #161 with 2 processes, beside #160 / #162.
#  CRIT  = 50,000 slice-A positions, 12,500 per eval stage (mg / mgl / egl / eg), picked by key hash (deterministic), first;
#  CRIT2 = the rest of slice A (train and val kept with their split, mates excluded), after.
# Labels: SF18 d14 MULTI-PV K=4 → best-vs-2nd gap per position. A SEPARATE set (corpus charter: never pool multi-PV with the
# single-best labels). Hard stop 2026-11-06 09:30 (the labeller is append-only and resumable). Stop: touch q163.stop.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
D=/mnt/e/chess_data/retune
LOGD=/mnt/e/chess_data/q163
STOP=/mnt/e/chess_data/q163.stop
CUTOFF=$(date -d '2026-11-06 09:30' +%s)
PY=/home/ranuja/anaconda3/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"; mkdir -p "$LOGD"
echo "[q163] $(date) waiting for queue #161 (stop: $STOP)"
until grep -qE 'QUEUE 161 (DONE|ABORTED)' /mnt/e/chess_data/q161.log 2>/dev/null; do
  [ -e "$STOP" ] && { echo "[q163] stop file present — QUEUE 163 DONE (nothing run)"; exit 0; }
  sleep 30
done
sleep 20
pgrep -f "fastchess-linux-x86-64/fastchess|vs_sf.py|_build_regret_set" >/dev/null && { echo "[q163] a match or labeller is running — QUEUE 163 ABORTED"; exit 1; }
echo "[q163] $(date '+%Y-%m-%d %H:%M:%S') start"

if [ ! -s $D/CRIT_fens.csv ] || [ ! -s $D/CRIT2_fens.csv ]; then
  "$PY" - "$D" <<'EOF'
import csv, hashlib, os, sys
from collections import defaultdict
D = sys.argv[1]
by = defaultdict(list)
for r in csv.DictReader(open(os.path.join(D, "corpus_v2.csv"), newline="")):
    if r["slice"] == "A" and r["split"] in ("train", "val") and r["mate"] == "0":
        by[r["stage"]].append((hashlib.sha1(r["key"].encode()).hexdigest(), r["fen"], r["stage"], r["split"], r["group"]))
first, rest = [], []
for st in ("mg", "mgl", "egl", "eg"):
    rows = sorted(by[st])
    first += rows[:12500]
    rest += rows[12500:]
for name, rows in (("CRIT_fens.csv", first), ("CRIT2_fens.csv", sorted(rest))):
    with open(os.path.join(D, name), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["fen", "stage", "split", "group"]); w.writerows(x[1:] for x in rows)
print("[q163] CRIT %d rows (%s) · CRIT2 %d rows" % (len(first), {st: min(12500, len(by[st])) for st in ("mg", "mgl", "egl", "eg")}, len(rest)))
EOF
fi
for s in CRIT CRIT2; do
  for i in 0 1; do head -1 $D/${s}_fens.csv > $D/${s}_fens_s$i.csv; done
  tail -n +2 $D/${s}_fens.csv | awk -v B="$D/${s}_fens" '{print >> (B "_s" (NR-1)%2 ".csv")}'
done

t0=$(date +%s)
for s in CRIT CRIT2; do
  for i in 0 1; do
    left=$(( CUTOFF - $(date +%s) )); [ $left -gt 60 ] && [ ! -e "$STOP" ] || break 2
    while [ $(jobs -rp | wc -l) -ge 2 ]; do wait -n; done
    left=$(( CUTOFF - $(date +%s) )); [ $left -gt 60 ] || break 2
    ( timeout $left bash "$R" pyrun diagnostics/_build_regret_set.py IN=$D/${s}_fens_s$i.csv K=4 SF_DEPTH=14 \
        OUT=$D/${s}_sf18k4_s$i.csv > $LOGD/label_${s}_$i.log 2>&1
      echo "[q163] $(date) $s s$i: rc $? · $(grep -vc '^fen,' $D/${s}_sf18k4_s$i.csv 2>/dev/null) rows" ) &
  done
done
wait
c1=$(cat $D/CRIT_sf18k4_s*.csv 2>/dev/null | grep -vc '^fen,'); c2=$(cat $D/CRIT2_sf18k4_s*.csv 2>/dev/null | grep -vc '^fen,')
echo "[q163] $(date) labelled CRIT $c1 / 50000 · CRIT2 $c2 / $(($(wc -l < $D/CRIT2_fens.csv) - 1)) in $(( ($(date +%s) - t0) / 60 )) min"
echo "[q163] $(date) QUEUE 163 DONE"
