#!/bin/bash
# 2026-10-07 (owner 10-06: break the static ladder down by POSITION TYPE — 960/variant vs classic, odds, K+P stress — and by
# stage/structure; win% error throughout). Waits for queue #29. Builds the type corpora (`_bench_type_corpora.py`; odds
# starts labelled with SF18 d14), then the same reference ladder + our arms + `_gap_strata.py` per corpus.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
O=/mnt/e/chess_data/texel/revival
B=/mnt/e/chess_data/bench1007
Q29LOG="$ND/selfplay/games/q29_bench.log"
mkdir -p "$B"
cd "$ND" || exit 1
echo "[q30] $(date) waiting for queue #29"
until grep -qE "QUEUE 29 DONE" "$Q29LOG" 2>/dev/null; do sleep 300; done
OLD="KS_V2_W_N=31 KS_V2_W_B=31 KS_V2_W_R=47 KS_V2_W_Q=78 KS_V2_COORD=256 KS_V2_WEAK=64 KS_V2_ADJ=50 KS_V2_NO_QUEEN=402 KS_V2_CHK_Q=260 KS_V2_CHK_R=193 KS_V2_CHK_B=141 KS_V2_CHK_N=189 KS_V2_MAX=4000 KS_V2_HALF=646 KS_V2_EG_PCT=100 KS_V2_ADJ_INST=-12 KS_V2_UNSAFE=19 KS_V2_FLANK_ATT=11 KS_V2_FLANK_ATT2=-1 KS_V2_KNIGHT_DEF=15 KS_V2_CONTEST_EXCESS=14 KS_V2_CONTEST_SQ=30 KS_V2_CONTEST_SQ_Q=19 KS_V2_BLOCKERS=0 KSB_V2=0 KAUF_V2_MAG=0 PS_V2_CONN_MAG=0"
PAIR="KFL_V2=1 KFL_V2_FILE=$O/kfl_depth.txt PST_V2_FILE=$O/pst_depth.txt"
echo "[q30] $(date) start"
bash "$R" pyrun diagnostics/_bench_type_corpora.py N=3000 DEPTH=14 2>&1 | grep -E "BENCH CORPUS|rror" | sed "s/^/[q30] /"
for c in bench_variant bench_odds bench_kp; do
  [ -s diagnostics/ks_sets/$c.csv ] || { echo "[q30] ☠️ $c missing — skipped"; continue; }
  D=$B/$c.csv; rm -f "$D"
  rc() { env PRESET=LONG_FORMAT USE_OPENING_BOOK=0 CORPUS=diagnostics/ks_sets/$c.csv N=3000 DUMP=$D "$@" \
           bash "$R" pyrun diagnostics/_reference_ceiling.py 2>&1 | grep -E "^  [A-Za-z].* [0-9n].* [0-9]+$|skip|rror" | sed "s/^/[q30] $c /"; }
  rc REFS=1 DUMPLABEL="v2 shipped" V2_PRESET=shipped
  rc REFS=0 DUMPLABEL="v2 10-01" V2_PRESET=shipped $OLD
  rc REFS=0 DUMPLABEL="v2 +kfl+pst" V2_PRESET=shipped $PAIR
  rc REFS=0 DUMPLABEL="v1" EVAL_ARM=0
  bash "$R" pyrun diagnostics/_gap_strata.py DUMPS=$D OURS="v2 shipped" TYPES=diagnostics/ks_sets/$c.csv 2>&1 | sed "s/^/[q30] $c strata /"
done
echo "[q30] $(date) QUEUE 30 DONE"
