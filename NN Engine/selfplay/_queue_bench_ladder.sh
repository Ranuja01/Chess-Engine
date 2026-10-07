#!/bin/bash
# 2026-10-07 (owner, before the POT design): BENCH NUMBERS — how much closer are we, and is the remaining gap explained by
# a missing feature? Waits for queue #28. Static, on 1-2 engines.
#  (1) REFERENCE LADDER (SF11 · SF15.1 classical · SF15.1 NNUE · SF18 static) + OUR ARMS on the SAME 3,000 rows and loss as
#      the 09-26 ladder (EVAL-V2-INVENTORY §10: v2+FitA 170.17 / 126.93; SF11 151.41 / 95.26; v1 188.85 / 238.77), both
#      corpora, per-position DUMP for stratification. Arms: v2 shipped · v2 as of 10-01 (exact env reproduction) ·
#      v2 + KFL+PST (pending pair) · v1.
#  (2) _gap_strata.py: where the (ours − SF11) excess sits — phase / material class / |target|.
#  (3) STS300 at EQUAL NODES (249,014; the 09-22 instrument: v2 then 1689; SF11 2374) for the same arms.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
O=/mnt/e/chess_data/texel/revival
B=/mnt/e/chess_data/bench1007
Q28LOG="$ND/selfplay/games/q28_reanchor.log"
mkdir -p "$B"
cd "$ND" || exit 1
echo "[q29] $(date) waiting for queue #28"
until grep -qE "QUEUE 28 DONE|aborting" "$Q28LOG" 2>/dev/null; do sleep 300; done
while pgrep -f "vs_sf.py|tournament.py" >/dev/null; do sleep 60; done
OLD="KS_V2_W_N=31 KS_V2_W_B=31 KS_V2_W_R=47 KS_V2_W_Q=78 KS_V2_COORD=256 KS_V2_WEAK=64 KS_V2_ADJ=50 KS_V2_NO_QUEEN=402 KS_V2_CHK_Q=260 KS_V2_CHK_R=193 KS_V2_CHK_B=141 KS_V2_CHK_N=189 KS_V2_MAX=4000 KS_V2_HALF=646 KS_V2_EG_PCT=100 KS_V2_ADJ_INST=-12 KS_V2_UNSAFE=19 KS_V2_FLANK_ATT=11 KS_V2_FLANK_ATT2=-1 KS_V2_KNIGHT_DEF=15 KS_V2_CONTEST_EXCESS=14 KS_V2_CONTEST_SQ=30 KS_V2_CONTEST_SQ_Q=19 KS_V2_BLOCKERS=0 KSB_V2=0 KAUF_V2_MAG=0 PS_V2_CONN_MAG=0"
PAIR="KFL_V2=1 KFL_V2_FILE=$O/kfl_depth.txt PST_V2_FILE=$O/pst_depth.txt"
echo "[q29] $(date) start"
for c in playdist_ceiling diverse_corpus_wide; do
  D=$B/$c.csv; rm -f "$D"
  rc() { env PRESET=LONG_FORMAT USE_OPENING_BOOK=0 CORPUS=diagnostics/ks_sets/$c.csv N=3000 DUMP=$D "$@" \
           bash "$R" pyrun diagnostics/_reference_ceiling.py 2>&1 | grep -E "^  [A-Za-z].* [0-9]+\.[0-9]+ .* [0-9]+$|skip|rror" | sed "s/^/[q29] $c /"; }
  rc REFS=1 DUMPLABEL="v2 shipped" V2_PRESET=shipped
  rc REFS=0 DUMPLABEL="v2 10-01" V2_PRESET=shipped $OLD
  rc REFS=0 DUMPLABEL="v2 +kfl+pst" V2_PRESET=shipped $PAIR
  rc REFS=0 DUMPLABEL="v1" EVAL_ARM=0
  bash "$R" pyrun diagnostics/_gap_strata.py DUMPS=$D OURS="v2 shipped" 2>&1 | sed "s/^/[q29] $c strata /"
done
for a in "v2_shipped|V2_PRESET=shipped" "v2_1001|V2_PRESET=shipped $OLD" "v2_pair|V2_PRESET=shipped $PAIR" "v1|EVAL_ARM=0"; do
  name="${a%%|*}"; knobs="${a#*|}"
  echo "[q29] STS equal-nodes $name: $(bash "$R" sts sts_eqn_$name $knobs MAX_DEPTH=64 NODE_LIMIT=249014)"
done
echo "[q29] $(date) QUEUE 29 DONE"
