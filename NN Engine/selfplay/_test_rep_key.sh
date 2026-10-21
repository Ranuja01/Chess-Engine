#!/bin/bash
# B2 FIX_REP_KEY unit test (2026-10-20). Run on a binary built with the B2 code: bash selfplay/_test_rep_key.sh [binary]
#
# FALSE DRAW: from 4k2r/8/8/8/8/8/8/Q3K3 w k (Black still has kingside rights) play 1.Qb1 Kf8 2.Qa1 — Black (down Q vs R)
# to move. ...Ke8 restores the START placement but WITHOUT castling rights. Placement-only keys count it as the 2nd
# occurrence ⇒ "draw" ⇒ the losing side grabs a false 0.
# TRUE REPEAT control: the same moves from a start with NO rights ('-'). ...Ke8 really repeats the start position (once
# before the root + once in the search).
#
# Pre-registered (before the first run):
#                     FIX_REP_KEY=0   =1 (key)     =2 (pre-root 3-fold)   =3
#   false draw        cp 0, Ke8       negative     negative               negative
#   true repeat       cp 0            cp 0         negative               negative
#   bench             258/47,689,413  ~identical (no game history in WAC; only in-tree rights-differing repeats change)
B=${1:-"$(dirname "$0")/../build_uci/chess_engine_uci_fix"}
run() {   # $1 knob, $2 fen
  printf 'isready\nposition fen %s moves a1b1 e8f8 b1a1\ngo depth 10\n' "$2" | { FIX_REP_KEY=$1 timeout 60 "$B"; } 2>/dev/null |
    awk '/^info depth/{s=$0} /^bestmove/{match(s,/score (cp|mate) -?[0-9]+/); print substr(s,RSTART,RLENGTH), $1, $2; exit}'
}
for K in 0 1 2 3; do
  echo "FIX_REP_KEY=$K  false-draw: $(run $K '4k2r/8/8/8/8/8/8/Q3K3 w k - 0 1')   true-repeat: $(run $K '4k2r/8/8/8/8/8/8/Q3K3 w - - 0 1')"
done
# Result 2026-10-21 (b2 build): predictions held on draw / no draw; the SCORE sign did not — the test position has ...Rh1+ skewering
# the king and winning Qa1, so Black is WINNING (+261 when searched). At =0 the engine stopped at DEPTH 3 after 45 nodes and took the
# "draw": iterative deepening breaks on `is_draw && score == 0` (B3, FIX_DRAW_EXIT). Bench =3: 251 / 47,905,411 (−7 solves).
for K in 0 1; do
  echo "FIX_REP_KEY=$K FIX_DRAW_EXIT=1  false-draw: $(FIX_DRAW_EXIT=1 run $K '4k2r/8/8/8/8/8/8/Q3K3 w k - 0 1')   true-repeat: $(FIX_DRAW_EXIT=1 run $K '4k2r/8/8/8/8/8/8/Q3K3 w - - 0 1')"
done
for K in 0 1 2 3; do
  echo "FIX_REP_KEY=$K  $(printf 'bench\nquit\n' | FIX_REP_KEY=$K "$B" 2>/dev/null | grep '^bench')"
done
echo "FIX_DRAW_EXIT=1  $(printf 'bench\nquit\n' | FIX_DRAW_EXIT=1 "$B" 2>/dev/null | grep '^bench')"
