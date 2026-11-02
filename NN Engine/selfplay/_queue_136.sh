#!/bin/bash
# 2026-11-02 afternoon, queue #136 (step 0 of dev_notes/active/SEARCH-CORRECTNESS-ITEMS-2026-11-02.md). Never edit while it runs.
# Fixed-node work only (no timed games). Waits for queue #135.
#  1 build the edited source twice: plain (must be result-identical to production) and with -DFIXCNT_BUILD (the counters);
#    both benches must equal production gen 6.4's 36,991,901, or the queue aborts
#  2 fire counts of the open correctness items: self-play at 30k nodes/move from eg_critical, kp_dense and UHO starts, 30 each
#    (_fix_firecount.py OFF_ONLY=1: counting only, no fix is switched on)
# Registered: mdp, prunes with the best mated, rule-50 TT cutoffs and 50-move mates each fire in < 5% of UHO games; killer /
# counter swaps displace a capture on >= 2% of cache hits; TT cutoffs with a PV window are a visible share (no sign predicted).
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
STOP=/mnt/e/chess_data/q136.stop
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"
echo "[q136] $(date '+%Y-%m-%d %H:%M:%S') waiting for queue 135"
while pgrep -f "fastchess|_queue_135\.sh" > /dev/null; do sleep 30; done
START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[q136] $START start (stop file $STOP)"

nodes() { printf "bench\nquit\n" | "$1" 2>/dev/null | grep '^bench' | sed -n 's/.* nodes \([0-9]*\).*/\1/p'; }
OUT=chess_engine_uci_src bash selfplay/build_uci.sh | sed 's/^/[q136]   /'
OUT=chess_engine_uci_cnt2 EXTRA_FLAGS=-DFIXCNT_BUILD bash selfplay/build_uci.sh | sed 's/^/[q136]   /'
bs=$(nodes build_uci/chess_engine_uci_src); bc=$(nodes build_uci/chess_engine_uci_cnt2)
echo "[q136] bench: plain build $bs · counter build $bc (want 36991901 both)"
[ "$bs" = 36991901 ] && [ "$bc" = 36991901 ] || { echo "[q136] ☠️ a build is not result-identical — QUEUE 136 ABORTED"; exit 1; }

[ -e "$STOP" ] && { echo "[q136] stop file — skip the fire counts"; exit 0; }
echo "[q136] $(date) fire counts (counter build, 30k nodes/move, 30 starts per set)"
timeout 5400 bash "$R" pyrun diagnostics/_fix_firecount.py BIN=build_uci/chess_engine_uci_cnt2 OFF_ONLY=1 N=30 NODES=30000 JOBS=2 2>&1 \
  | grep -vE "^\s*$" | sed 's/^/[q136]   /'
echo "[q136] $(date) QUEUE 136 DONE"
