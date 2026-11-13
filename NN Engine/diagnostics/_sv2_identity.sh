#!/bin/bash
# Search v2 identity check (dev_notes/active/SEARCH-V2-PLAN-2026-11-10.md §7e): SEARCH_ARM 0 vs 1 on each bench-format set,
# same UCI binary, fixed depth (bench clears the search tables per position, LONG_FORMAT limits). Every v2 change that is
# meant to keep the search's results (a port step, a refactor) must leave every set SAME.
#   bash diagnostics/_sv2_identity.sh <uci binary> <set dir> [depth=10]
#   A0= / A1= (env): the two arms' environment, default SEARCH_ARM=0 / SEARCH_ARM=1; e.g. a search v2 knob checked at
#   identity: A0="SEARCH_ARM=1 SV2_PRUNE_EARLY=1" A1="SEARCH_ARM=1 SV2_PRUNE_EARLY=1 SV2_LAZY_GEN=2".
# Sets: diagnostics/suites/wac_uci.txt plus the copies diagnostics/_sv2_sets.py writes (default /mnt/e/chess_data/sv2_id).
# RESULT (11-11, S1 step 3, chess_engine_uci_sv2): WAC 39,861,964 · WAC mirrored 38,059,595 · STS300 100,753,502 · STS300
# mirrored 98,825,166 · 48 bare-king endings 6,715,062 · perpetual probe 41,426, all SAME (the solved counts on the mirrored
# sets are not meaningful: their bm fields do not all convert).
B="$1"; D="$2"; DEPTH="${3:-10}"; A0="${A0:-SEARCH_ARM=0}"; A1="${A1:-SEARCH_ARM=1}"
cd "/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine" || exit 1
fails=0
for f in diagnostics/suites/wac_uci.txt "$D"/*.txt; do
	a0=$(printf "bench $f $DEPTH\nquit\n" | env $A0 "$B" 2>/dev/null | grep -oE 'solved [0-9]+/[0-9]+ nodes [0-9]+')
	a1=$(printf "bench $f $DEPTH\nquit\n" | env $A1 "$B" 2>/dev/null | grep -oE 'solved [0-9]+/[0-9]+ nodes [0-9]+')
	[ "$a0" = "$a1" ] && v=SAME || { v=DIFFERENT; fails=$((fails + 1)); }
	echo "$(basename "$f") d$DEPTH: arm0 [$a0] arm1 [$a1] $v"
done
echo "identity ($A0 vs $A1): $fails set(s) differ"
