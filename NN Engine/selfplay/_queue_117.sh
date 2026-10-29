#!/bin/bash
# 2026-10-30, queue #117: the transition bench report (owner: "the bench runs, including the comparison table for all engines
# ... also the static eval tests"), for gen 6.3 and the ship candidate (threats mg 50 / eg 100 + the RFP margin #116's plateau
# picked). Chained after #116. ONE process at a time (timed benches need a quiet box). Never edit this file while it runs.
# Template: _queue_v2_record.sh (the 10-20 gen 4 record). Reused, not re-run (deterministic): the SF rows of
# REFERENCE-BENCH-LADDER, v1's benches (fingerprint unchanged), gen 4's d10 STS / full STS.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
UCI="$ND/build_uci/chess_engine_uci"
O=/mnt/e/chess_data/fastchess
W=/mnt/e/chess_data/wdlcal
B=/mnt/e/chess_data/bench1030
TH_ENV="THREAT_V2_PCT=100 THREAT_V2_HANGING=1 THREAT_V2_KING=1 THREAT_V2_PAWN_TARGETS=1 THREAT_V2_MG_PCT=50 THREAT_V2_EG_PCT=100"
TH_UCI=THREAT_V2_PCT=100,THREAT_V2_HANGING=1,THREAT_V2_KING=1,THREAT_V2_PAWN_TARGETS=1,THREAT_V2_MG_PCT=50,THREAT_V2_EG_PCT=100
mkdir -p "$B"
cd "$ND" || exit 1
echo "[q117] $(date) waiting for queue 116"
until grep -q 'QUEUE 116 DONE' /mnt/e/chess_data/q116.log 2>/dev/null; do sleep 60; done
while pgrep -f "fastchess|vs_sf.py" >/dev/null; do sleep 60; done
echo "[q117] $(date) start"

# The candidate's margin: 600 unless #116's plateau found a neighbour more than +10 better (the larger if both).
elo() { grep -E '^Elo:' "$O/$1.log" 2>/dev/null | tail -1 | awk '{print $2}'; }
e8=$(elo c_r800_10); e4=$(elo c_r400_10); RM=600
best=$(awk -v a="${e8:-0}" -v b="${e4:-0}" 'BEGIN{ if (a > 10 && a >= b) print 800; else if (b > 10) print 400; else print 600 }')
RM=$best
echo "[q117] plateau: RFP 800 ${e8:-?} · RFP 400 ${e4:-?} vs 600 ⇒ candidate margin $RM"
G63="V2_PRESET=shipped"
CAND="V2_PRESET=shipped $TH_ENV RFP_MARGIN=$RM"
G62="V2_PRESET=shipped THREAT_ORDER_W=0"
M50="V2_PRESET=shipped $TH_ENV"
G4="V2_PRESET=shipped V2_SHIP_GEN=4"
show() { sed "s/^/[q117]   /"; }

echo "[q117] == fingerprints (WAC d10)"
for a in "g63|$G63" "cand|$CAND"; do echo "[q117] $(echo ${a%%|*}): $(bash "$R" wac fp_${a%%|*} ${a#*|} | grep -E '^SOLVED|^NODES|^EBF' | tr '\n' ' ')"; done

echo "[q117] $(date) == TIMED: WAC at LIGHTNING (depth reached per move)"
for a in "g63|$G63" "cand|$CAND"; do echo "[q117] ${a%%|*}:"; bash "$R" wac_timed_depth rect_${a%%|*} ${a#*|} 2>&1 | grep -vE '^\s*$' | show; done
echo "[q117] $(date) == TIMED: WAC speed, pinned core, 3 reps (MAX_NPS = the measurement)"
for a in "g62|$G62" "g63|$G63" "m50_r1000|$M50" "cand|$CAND"; do
  echo "[q117] ${a%%|*}: $(bash "$R" wac_speed recs_${a%%|*} 3 ${a#*|} 2>&1 | grep -E 'NODES:|MAX_NPS|MEDIAN|SPREAD' | tr '\n' ' ')"
done
echo "[q117] $(date) == TIMED: native UCI bench (PGO production binary), alternating, 3 reps"
for i in 1 2 3; do
  echo "[q117] g63  $(printf 'bench\nquit\n' | "$UCI" 2>/dev/null | grep '^bench')"
  echo "[q117] cand $(printf "setoption name Knobs value $TH_UCI,RFP_MARGIN=$RM\nbench\nquit\n" | "$UCI" 2>/dev/null | grep '^bench')"
done

echo "[q117] $(date) == STS300 and full STS at d10"
for a in "g63|$G63" "cand|$CAND"; do
  echo "[q117] ${a%%|*} STS300 d10: $(bash "$R" sts rec_${a%%|*} ${a#*|} | tail -1)"
  echo "[q117] ${a%%|*} STS full d10: $(bash "$R" sts_full recf_${a%%|*} ${a#*|} | grep 'STS score' | tail -1)"
done

echo "[q117] $(date) == EQUAL NODES (249,014 per position; the REFERENCE-BENCH-LADDER instrument)"
for a in "g4|$G4" "g63|$G63" "cand|$CAND"; do
  echo "[q117] ${a%%|*} STS300 @249k: $(bash "$R" sts sts_eqn_${a%%|*} ${a#*|} MAX_DEPTH=64 NODE_LIMIT=249014 | tail -1)"
  echo "[q117] ${a%%|*} WAC @249k: $(bash "$R" wac wac_eqn_${a%%|*} ${a#*|} MAX_DEPTH=64 NODE_LIMIT=249014 | grep -E '^SOLVED|^NODES' | tr '\n' ' ')"
done
echo "[q117] SF11 d10 STS300 reproduction (want 1983):"
bash "$R" pyrun diagnostics/_reference_ladder.py --depth 10 --engines sf11 --suites sts300.epd 2>&1 | grep -iE "sf11|fail" | show

echo "[q117] $(date) == STATIC LADDER (_reference_ceiling.py, 3,000 rows, win% MSE vs SF18 d14; the 10-07 snapshot's corpora)"
for c in playdist_ceiling diverse_corpus_wide; do
  D=$B/$c.csv; rm -f "$D"
  rc() { env OMP_NUM_THREADS=1 PRESET=LONG_FORMAT USE_OPENING_BOOK=0 CORPUS=diagnostics/ks_sets/$c.csv N=3000 DUMP=$D "$@" \
           bash "$R" pyrun diagnostics/_reference_ceiling.py 2>&1 | grep -E "^  [A-Za-z].* [0-9]+\.[0-9]+ .* [0-9]+$|skip|rror" | sed "s/^/[q117] $c /"; }
  rc REFS=1 DUMPLABEL="gen 6.3" $G63
  rc REFS=0 DUMPLABEL="candidate" $CAND
done

echo "[q117] $(date) == WIN% ACCURACY ON THE SF18 WDL LABEL (5,988 rows; static + d10)"
if [ "$RM" != 600 ]; then   # d10 dump for a candidate margin other than the one already on disk
  tag=g62ord_m50e100_r$RM
  for k in 0 1; do
    ( until env OMP_NUM_THREADS=1 PRESET=LONG_FORMAT MAX_DEPTH=10 USE_OPENING_BOOK=0 $G63 $TH_ENV RFP_MARGIN=$RM nice -n 10 bash "$R" pyrun \
          diagnostics/_depth_residual_pass.py IN=$W/cal_d10_in.csv OUT=$W/d10_$tag.csv SHARD=$k/2 CHUNK=1000 > /dev/null 2>&1; \
          [ $? -ne 3 ]; do :; done ) &
  done
  wait
  { head -1 $W/d10_${tag}_s0of2.csv; tail -q -n +2 $W/d10_${tag}_s0of2.csv $W/d10_${tag}_s1of2.csv; } > $W/d10_$tag.csv
  echo "[q117] d10 $tag rows $(($(wc -l < $W/d10_$tag.csv) - 1))"
fi
env OMP_NUM_THREADS=1 bash "$R" pyrun diagnostics/_v1_v2_accuracy.py MODE=read REF=d10_g62ord > $B/accuracy_read.txt 2>&1
echo "[q117] accuracy table -> $B/accuracy_read.txt ($(wc -l < $B/accuracy_read.txt) lines)"
env OMP_NUM_THREADS=1 bash "$R" pyrun diagnostics/_wdl_gap_map.py $W > $B/gap_map.txt 2>&1
echo "[q117] gap map -> $B/gap_map.txt ($(wc -l < $B/gap_map.txt) lines)"

echo "[q117] $(date) == SF11 TERM COUNTERFACTUAL ON THE WDL LABEL (endgame rows; which SF11 term closes our gap)"
# SCALE_US = each arm's calibrated best scale on these rows (gen 6.3 2.82, the threats static 2.76; SF11 2.76)
env OMP_NUM_THREADS=1 $G63 bash "$R" pyrun diagnostics/_triangulate_sf11.py MODE=egshare WDL=$W SCALE_US=2.82 > $B/egshare_wdl_g63.txt 2>&1
env OMP_NUM_THREADS=1 $CAND bash "$R" pyrun diagnostics/_triangulate_sf11.py MODE=egshare WDL=$W SCALE_US=2.76 > $B/egshare_wdl_cand.txt 2>&1
for t in g63 cand; do echo "[q117] $t -> $B/egshare_wdl_$t.txt ($(wc -l < $B/egshare_wdl_$t.txt) lines)"; done
echo "[q117] $(date) QUEUE 117 DONE"
