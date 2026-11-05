#!/bin/bash
# 2026-11-05 afternoon, queue #163: stage-2 (criticality) labels for the retune, in the box time left after #162. The corpus
# labels are SF18 d14 SINGLE best (K=1), which cannot tell a critical position (one move matters) from a quiet one. Here:
# 50,000 slice-A positions (they carry gen 6.7's d10 score from #160), 12,500 per eval stage (mg / mgl / egl / eg), picked
# by key hash (deterministic, train and val kept with their split), labelled SF18 d14 MULTI-PV K=4 → best-vs-2nd gap per
# position. A SEPARATE set (corpus charter: never pool multi-PV with the single-best labels). 3 processes after #162; hard
# stop 2026-11-06 09:30 (the labeller is append-only and resumable, so a stop loses nothing). Stop: touch q163.stop.
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
echo "[q163] $(date) waiting for queue #162 (stop: $STOP)"
until grep -qE 'QUEUE 162 (DONE|ABORTED)' /mnt/e/chess_data/q162.log 2>/dev/null; do
  [ -e "$STOP" ] && { echo "[q163] stop file present — QUEUE 163 DONE (nothing run)"; exit 0; }
  sleep 60
done
sleep 20
pgrep -f "fastchess-linux-x86-64/fastchess|vs_sf.py|_depth_residual_pass|_build_regret_set" >/dev/null && { echo "[q163] a match, depth pass or labeller is running — QUEUE 163 ABORTED"; exit 1; }
echo "[q163] $(date '+%Y-%m-%d %H:%M:%S') start"
[ $(date +%s) -lt $CUTOFF ] || { echo "[q163] past the cut-off — QUEUE 163 DONE (nothing run)"; exit 0; }

if [ ! -s $D/CRIT_fens.csv ]; then
  "$PY" - "$D" <<'EOF'
import csv, hashlib, os, sys
from collections import defaultdict
D = sys.argv[1]
by = defaultdict(list)
for r in csv.DictReader(open(os.path.join(D, "corpus_v2.csv"), newline="")):
    if r["slice"] == "A" and r["split"] in ("train", "val") and r["mate"] in ("0", "", "False"):
        by[r["stage"]].append((hashlib.sha1(r["key"].encode()).hexdigest(), r["fen"], r["stage"], r["split"], r["group"]))
out = []
for st in ("mg", "mgl", "egl", "eg"):
    out += sorted(by[st])[:12500]
with open(os.path.join(D, "CRIT_fens.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["fen", "stage", "split", "group"]); w.writerows(x[1:] for x in out)
print("[q163] CRIT sample: %d rows %s" % (len(out), {st: min(12500, len(by[st])) for st in ("mg", "mgl", "egl", "eg")}))
EOF
fi
for i in 0 1 2; do head -1 $D/CRIT_fens.csv > $D/CRIT_fens_s$i.csv; done
tail -n +2 $D/CRIT_fens.csv | awk -v B="$D/CRIT_fens" '{print >> (B "_s" (NR-1)%3 ".csv")}'
echo "[q163] shards: $(for i in 0 1 2; do echo -n "$(($(wc -l < $D/CRIT_fens_s$i.csv) - 1)) "; done)"

t0=$(date +%s)
for i in 0 1 2; do
  ( timeout $(( CUTOFF - $(date +%s) )) bash "$R" pyrun diagnostics/_build_regret_set.py IN=$D/CRIT_fens_s$i.csv K=4 SF_DEPTH=14 \
      OUT=$D/CRIT_sf18k4_s$i.csv > $LOGD/label_$i.log 2>&1
    echo "[q163] $(date) shard $i: rc $? · $(grep -vc '^fen,' $D/CRIT_sf18k4_s$i.csv 2>/dev/null) rows" ) &
done
wait
n=$(cat $D/CRIT_sf18k4_s*.csv 2>/dev/null | grep -vc '^fen,')
echo "[q163] $(date) labelled $n / 50000 in $(( ($(date +%s) - t0) / 60 )) min ($( [ $n -lt 50000 ] && echo 'stopped at the cut-off: re-run to resume' || echo complete))"
echo "[q163] $(date) QUEUE 163 DONE"
