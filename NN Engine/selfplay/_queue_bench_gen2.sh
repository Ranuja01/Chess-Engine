#!/bin/bash
# 2026-10-11 night, queue #55 (owner: "get numbers again on how much we are off now, and where") — the 10-07 BENCH LADDER
# (#29/#30) re-run for SHIP GEN 2: reference ladder (SF11 · SF15.1 classical · SF15.1 NNUE · SF18 static) + our arms
# (v2 gen 2 = shipped · v2 gen 1 = the 10-04 ship · v1) on the SAME corpora and loss, per-position dumps, _gap_strata.py
# (stage / structure / material class / |target|), _endgame_types.py, the type corpora (variant / odds / K+P), and STS300 at
# equal nodes. Waits for queue #54 (never beside a rebuild). Static, 1 engine. Dumps on E:\chess_data\bench1011.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
B=/mnt/e/chess_data/bench1011
mkdir -p "$B"
until grep -qE "QUEUE 54 DONE|☠️" /mnt/e/chess_data/q54_gen2_base.log 2>/dev/null; do sleep 120; done
grep -q "☠️" /mnt/e/chess_data/q54_gen2_base.log && { echo "[q55] ☠️ queue #54 aborted — not benching an unverified build"; exit 1; }
cd "$ND" || exit 1
echo "[q55] $(date) start"
for c in playdist_ceiling diverse_corpus_wide bench_variant bench_odds bench_kp; do
  [ -s diagnostics/ks_sets/$c.csv ] || { echo "[q55] ☠️ $c missing — skipped"; continue; }
  D=$B/$c.csv; rm -f "$D"
  rc() { env PRESET=LONG_FORMAT USE_OPENING_BOOK=0 CORPUS=diagnostics/ks_sets/$c.csv N=3000 DUMP=$D "$@" \
           bash "$R" pyrun diagnostics/_reference_ceiling.py 2>&1 | grep -E "^  [A-Za-z].* [0-9n].* [0-9]+$|skip|rror" | sed "s/^/[q55] $c /"; }
  rc REFS=1 DUMPLABEL="v2 gen2" V2_PRESET=shipped
  rc REFS=0 DUMPLABEL="v2 gen1" V2_PRESET=shipped V2_SHIP_GEN=1
  rc REFS=0 DUMPLABEL="v1" EVAL_ARM=0
  T=""; case $c in bench_*) T="TYPES=diagnostics/ks_sets/$c.csv";; esac
  bash "$R" pyrun diagnostics/_gap_strata.py DUMPS=$D OURS="v2 gen2" $T 2>&1 | sed "s/^/[q55] $c strata /"
  bash "$R" pyrun diagnostics/_endgame_types.py DUMPS=$D 2>&1 | sed "s/^/[q55] $c egtypes /"
done
for a in "v2_gen2|V2_PRESET=shipped" "v2_gen1|V2_PRESET=shipped V2_SHIP_GEN=1"; do
  name="${a%%|*}"; knobs="${a#*|}"
  echo "[q55] STS equal-nodes $name: $(bash "$R" sts sts_eqn_$name $knobs MAX_DEPTH=64 NODE_LIMIT=249014)"
done
echo "[q55] $(date) QUEUE 55 DONE"
