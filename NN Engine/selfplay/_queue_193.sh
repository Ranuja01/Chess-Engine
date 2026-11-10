#!/bin/bash
# 2026-11-10 ~19:00, queue #193 (owner 11-10: the transition close-out order "ok"; "keep the long runs going" overnight).
# S0 diagnostics at FIXED DEPTH only (node counts do not depend on machine load, so it is safe beside anything):
#  1 POT test B data (search v2 plan §7c; POT-REREAD-ANALYSIS §5): our best move at depth 10 and at depth 12 on
#    ks_sets/game_regret_set.csv (15k game positions, SF18 multi-PV d14), v2 shipped through the Cython path
#    (V2_PRESET=shipped, PRESET=LONG_FORMAT, tables cleared per position), per-row regret written by the worker mode of
#    _ks_footprint_regret.py, 3 workers per depth → /mnt/e/chess_data/s0_testB/d{10,12}_s{0,1,2}.csv
#    (fen, our move, regret, corpus phase, engine phase score, criticality). The split by POT's U is done offline.
#    Caveat: the regret uses SF's cp through the fixed win% formula (material-scaled); both depths share each row's label.
#  2 owner-game triage re-run on gen 7.1 (17 cases incl. game 7 g7a / g7b), as on 10-18: MODE=sf (SF18 d18 line + our
#    static at its end), search at depth 10 normal and with pruning off, report. The 10-18 json files are kept as *_1018.
# Stop: touch /mnt/e/chess_data/q193.stop (checked between steps).
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
STOP=/mnt/e/chess_data/q193.stop
O=/mnt/e/chess_data/s0_testB
D=/mnt/e/chess_data
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export RESIGN_THRESHOLD=-99999999   # the smoke test resigned in a lost position and returned no move
cd "$ND" || exit 1
rm -f "$STOP"; mkdir -p "$O"
echo "[q193] $(date '+%Y-%m-%d %H:%M:%S') start"

regret_pass() {  # depth
	local dep=$1 i pids=""
	[ -e "$STOP" ] && return
	echo "[q193] $(date) test B pass, depth $dep, 3 workers"
	for i in 0 1 2; do
		env WORKER=1 SET=diagnostics/ks_sets/game_regret_set.csv DEPTH=$dep SLICE=$i/3 SPLIT=all OUT=$O/d${dep}_s$i.csv \
			V2_PRESET=shipped bash "$R" pyrun diagnostics/_ks_footprint_regret.py > $O/d${dep}_s$i.log 2>&1 &
		pids="$pids $!"
	done
	wait $pids
	echo "[q193] $(date) depth $dep rows: $(cat $O/d${dep}_s*.csv 2>/dev/null | wc -l) · mean regret $(cat $O/d${dep}_s*.csv | awk -F, '{ s += $3; n++ } END { if (n) printf "%.3f", s / n }')"
}
regret_pass 10
regret_pass 12

if [ ! -e "$STOP" ]; then
	echo "[q193] $(date) triage re-run on gen 7.1"
	for f in triage_sf triage_normal triage_noprune; do [ -e $D/$f.json ] && [ ! -e $D/${f}_1018.json ] && cp $D/$f.json $D/${f}_1018.json; done
	env V2_PRESET=shipped bash "$R" pyrun diagnostics/_case_triage.py MODE=sf > $D/triage_sf_1110.log 2>&1
	env V2_PRESET=shipped MAX_DEPTH=10 bash "$R" pyrun diagnostics/_case_triage.py MODE=search TAG=normal > $D/triage_normal_1110.log 2>&1
	env V2_PRESET=shipped MAX_DEPTH=10 ENABLE_LMR=0 ENABLE_FUTILITY=0 ENABLE_RAZORING=0 ENABLE_LMP=0 ENABLE_RFP=0 ENABLE_ROOT_RAZOR=0 \
		SEE_PRUNE_CAPTURES=0 bash "$R" pyrun diagnostics/_case_triage.py MODE=search TAG=noprune > $D/triage_noprune_1110.log 2>&1
	bash "$R" pyrun diagnostics/_case_triage.py MODE=report 2>&1 | sed 's/^/[q193]   /'
fi
echo "[q193] $(date) QUEUE 193 DONE"
