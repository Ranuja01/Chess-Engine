#!/bin/bash
# 2026-11-21 ~18:50: S2-4 check 1 follow-up (dev_notes/active/S2-4-INVESTIGATION-2026-11-21.md §d). In the WAC d10 bench,
# 10 of 24 positions with no relabelled pre-search store still searched a different node count with PS_TT_FLAG_FIX=1
# (8 by 1-5 nodes, three by 548 / 922 / 7,778). pre_minimizer narrows only beta, so ps_exact_mislabel counts every store
# the knob relabels; the remaining channel is what the bench keeps between positions (the eval cache and the move-gen
# cache; clearSearchTables empties the TT and the q-cache). This runs each of those positions, plus two that matched, alone
# in a fresh process per arm (s6pgo, bench <one-line file> 10).
# REGISTERED: in isolation every pair is equal (arm 0 nodes == arm 1 nodes). Any unequal pair = a side path the audit
# missed: re-audit before any pairing run.
# Starts only in the Saturday evening block (20:52-23:59) with no timed game running (checked twice, 60 s apart); fixed
# depth, single process, about a minute.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
S6="$ND/build_uci/chess_engine_uci_s6pgo"
OUT=/mnt/e/chess_data/side_1121/iso
STOP=/mnt/e/chess_data/iso_s24.stop
POS="36 86 125 177 235 252 254 269 273 277 24 27"
mkdir -p "$OUT"; cd "$ND" || exit 1; rm -f "$STOP"
timed_running() { pgrep -f '[f]astchess.*tc=[0-9]' >/dev/null; }
in_window() { local t=$(date +%H%M); [ "$(date +%a)" = Sat ] && [ "$t" -ge 2052 ] && [ "$t" -le 2359 ]; }
echo "[iso] $(date '+%Y-%m-%d %H:%M:%S') queued; waiting for the evening block with no timed game"
until in_window && ! timed_running && sleep 60 && ! timed_running; do
	[ -e "$STOP" ] && { echo "[iso] stop file — nothing run"; exit 0; }
	[ "$(date +%a)" = Sun ] && { echo "[iso] the window passed — nothing run"; exit 0; }
	sleep 60
done
grep -vE '^#|^$' diagnostics/suites/wac_uci.txt > "$OUT/wac_lines.txt"
ok=0; bad=0
for i in $POS; do
	sed -n "$((i + 1))p" "$OUT/wac_lines.txt" > "$OUT/p$i.txt"
	r=()
	for a in 0 1; do
		printf 'setoption name Knobs value PS_TT_FLAG_FIX=%s\nbench %s 10\nquit\n' "$a" "$OUT/p$i.txt" | timeout 300 "$S6" > "$OUT/p${i}_$a.out" 2>&1
		r+=("$(grep -oE '^bench: solved [0-9]+/[0-9]+ nodes [0-9]+' "$OUT/p${i}_$a.out" | sed 's/bench: //')")
	done
	if [ -n "${r[0]}" ] && [ "${r[0]}" = "${r[1]}" ]; then ok=$((ok + 1)); v=equal; else bad=$((bad + 1)); v=DIFFER; fi
	echo "[iso] WAC position $i (0-based): arm 0 [${r[0]}] · arm 1 [${r[1]}] — $v"
done
echo "[iso] $(date '+%H:%M') $ok equal, $bad differing · ISO S24 DONE"
