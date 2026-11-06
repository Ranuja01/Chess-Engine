#!/bin/bash
# 2026-11-06, queue #169 (owner 11-06: "at the end get a static bench report"): the eval bench report for retune arm S against gen
# 6.7, template #117 (the gen 6.4 report). A read, not a gate. Everything here is fixed-depth, fixed-node or static, so it can run
# beside #168's games and the A2 pass; it starts when #168's gates are done. The timed rows (WAC at LIGHTNING, Cython speed, pinned UCI
# bench) belong to a ship's build ledger, on the compiled binary, and are not here. Reused, deterministic: gen 6.7's STS / WAC rows from
# #157 (STS300 d10 2,018 · full STS d10 9,830 · STS300 @ 249k 1,928 · WAC @ 249k 250) and the SF rows of REFERENCE-BENCH-LADDER.
#  1 how much of each report corpus S was trained on: a static read on rows it trained on flatters it.
#  2 S: Cython fingerprint, STS300 d10, full STS d10, STS300 and WAC at 249,014 nodes; SF11 STS300 d10 as the harness check (want 1,983).
#  3 static ladder, _reference_ceiling.py, 3,000 rows, win% MSE vs SF18 d14: gen 6.7 with the references, then S. Gen 6.7's static is
#    gen 6.4's (only search changed since), so own-play 147.5 and diverse 107.7 should reproduce.
#  4 win% accuracy on the SF18 WDL label, 5,988 held-out quiet rows, static and d10, both arms, paired against gen 6.7. Gen 6.4's
#    static was 384.2 (SF11 334.9); full tables in bench1106/.
# Stop: touch /mnt/e/chess_data/q169.stop (checked between steps). Never edit this file while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
D=/mnt/e/chess_data/retune
W=/mnt/e/chess_data/wdlcal
B=/mnt/e/chess_data/bench1106
STOP=/mnt/e/chess_data/q169.stop
PY=/home/ranuja/anaconda3/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"; mkdir -p "$B"
show() { sed "s/^/[q169]   /"; }
stopped() { [ -e "$STOP" ] && { echo "[q169] $(date) stop file present before $1 — QUEUE 169 DONE"; exit 0; }; }
echo "[q169] $(date) waiting for #168's gates (stop: $STOP)"
until grep -qE 'arms for the games:|QUEUE 168 (DONE|ABORTED)' /mnt/e/chess_data/q168.log 2>/dev/null; do
  [ -e "$STOP" ] && { echo "[q169] stop file present — QUEUE 169 DONE (nothing run)"; exit 0; }
  sleep 60
done
echo "[q169] $(date) start"
ENVL=$(cat $D/arm_v3s/env.txt)
[ -n "$ENVL" ] || { echo "[q169] ☠️ arm S's tables are missing — QUEUE 169 ABORTED"; exit 1; }
G67="V2_PRESET=shipped"
S="V2_PRESET=shipped $ENVL"

# ── 1 overlap with S's training rows ─────────────────────────────────────────────────────────────────────────────────
$PY - <<EOF 2>&1 | show
import csv
split = {}
for r in csv.DictReader(open("$D/corpus_v3.csv", newline="")):
    split[r["key"]] = r["split"]
for name, path in (("own-play ladder", "diagnostics/ks_sets/playdist_ceiling.csv"), ("diverse ladder", "diagnostics/ks_sets/diverse_corpus_wide.csv"), ("WDL rows", "$W/cal_in.csv")):
    n = {}
    for r in csv.DictReader(open(path, newline="")):
        s = split.get(" ".join(r["fen"].split()[:4]), "not in corpus_v3")
        n[s] = n.get(s, 0) + 1
    t = sum(n.values())
    print("%-16s %6d rows: " % (name, t) + " · ".join("%s %.1f%%" % (k, 100.0 * v / t) for k, v in sorted(n.items())))
EOF

# ── 2 benches ────────────────────────────────────────────────────────────────────────────────────────────────────────
stopped "the benches"
echo "[q169] $(date) == fingerprint and benches, S (gen 6.7's are on record from #157)"
echo "[q169] S WAC d10 (want 246 / 36509830): $(bash "$R" wac fp_s169 $S | grep -E '^SOLVED|^NODES|^EBF' | tr '\n' ' ')"
echo "[q169] S STS300 d10: $(bash "$R" sts rec_s169 $S | tail -1)"
echo "[q169] S STS full d10: $(bash "$R" sts_full recf_s169 $S | grep 'STS score' | tail -1)"
echo "[q169] S STS300 @249k: $(bash "$R" sts sts_eqn_s169 $S MAX_DEPTH=64 NODE_LIMIT=249014 | tail -1)"
echo "[q169] S WAC @249k: $(bash "$R" wac wac_eqn_s169 $S MAX_DEPTH=64 NODE_LIMIT=249014 | grep -E '^SOLVED|^NODES' | tr '\n' ' ')"
echo "[q169] SF11 d10 STS300 reproduction (want 1983):"
bash "$R" pyrun diagnostics/_reference_ladder.py --depth 10 --engines sf11 --suites sts300.epd 2>&1 | grep -iE "sf11|fail" | show

# ── 3 static ladder ──────────────────────────────────────────────────────────────────────────────────────────────────
stopped "the static ladder"
echo "[q169] $(date) == STATIC LADDER (_reference_ceiling.py, 3,000 rows, win% MSE vs SF18 d14)"
for c in playdist_ceiling diverse_corpus_wide; do
  DP=$B/$c.csv; rm -f "$DP"
  rc() { env OMP_NUM_THREADS=1 PRESET=LONG_FORMAT USE_OPENING_BOOK=0 CORPUS=diagnostics/ks_sets/$c.csv N=3000 DUMP=$DP "$@" \
           bash "$R" pyrun diagnostics/_reference_ceiling.py 2>&1 | grep -E "^  [A-Za-z].* [0-9]+\.[0-9]+ .* [0-9]+$|skip|rror" | sed "s/^/[q169] $c /"; }
  rc REFS=1 DUMPLABEL="gen 6.7" $G67
  rc REFS=0 DUMPLABEL="arm S" $S
done

# ── 4 win% accuracy on the SF18 WDL label ────────────────────────────────────────────────────────────────────────────
stopped "the WDL accuracy"
echo "[q169] $(date) == WIN% ACCURACY ON THE SF18 WDL LABEL (5,988 rows; static + d10)"
env $G67 bash "$R" pyrun diagnostics/_v1_v2_accuracy.py MODE=static TAG=g67 2>&1 | tail -1 | show
env $S bash "$R" pyrun diagnostics/_v1_v2_accuracy.py MODE=static TAG=v3s 2>&1 | tail -1 | show
d10() {   # $1 tag, the rest the engine env
  local tag=$1; shift
  [ -s $W/d10_$tag.csv ] && return
  for k in 0 1; do
    ( until env OMP_NUM_THREADS=1 PRESET=LONG_FORMAT MAX_DEPTH=10 USE_OPENING_BOOK=0 "$@" nice -n 10 bash "$R" pyrun \
          diagnostics/_depth_residual_pass.py IN=$W/cal_d10_in.csv OUT=$W/d10_$tag.csv SHARD=$k/2 CHUNK=1000 > /dev/null 2>&1; \
          [ $? -ne 3 ]; do [ -e "$STOP" ] && break; done ) &
  done
  wait
  { head -1 $W/d10_${tag}_s0of2.csv; tail -q -n +2 $W/d10_${tag}_s0of2.csv $W/d10_${tag}_s1of2.csv; } > $W/d10_$tag.csv
  echo "[q169] d10 $tag rows $(($(wc -l < $W/d10_$tag.csv) - 1))"
}
d10 g67 $G67
stopped "S's d10 pass"
d10 v3s $S
for ref in static_g67 d10_g67; do
  env OMP_NUM_THREADS=1 bash "$R" pyrun diagnostics/_v1_v2_accuracy.py MODE=read REF=$ref > $B/accuracy_$ref.txt 2>&1
  echo "[q169] paired vs $ref -> $B/accuracy_$ref.txt ($(wc -l < $B/accuracy_$ref.txt) lines)"
done
grep -E "label|arm  |static_(g67|v3s|sf11|v1) |d10_(g67|v3s|v1) |calibrated|paired" $B/accuracy_static_g67.txt | cut -c1-120 | show
grep -A4 -E "paired vs" $B/accuracy_d10_g67.txt | grep -E "paired|d10_v3s" | cut -c1-120 | show
echo "[q169] $(date) QUEUE 169 DONE"
