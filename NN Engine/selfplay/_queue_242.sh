#!/bin/bash
# 2026-11-14 ~22:00, queue #242: test build for the shallow-overrules-deep audit switches (all default 0), light (one compile,
# then single-threaded benches) while the owner games. Never touches production or the Cython module.
#   gates: default and SEARCH_ARM=0 == 249 / 37,279,382; SV2_VERIFY_DEEPER=1 == 250 / 37,320,382 (D7 unchanged);
#   each new switch moves the bench (proves it live), recorded for the screens.
# Never runs while q241.go exists (the overnight timed queue).
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
W=/mnt/e/chess_data/r10a
T="$ND/build_uci/chess_engine_uci_s6test"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
mkdir -p "$W"
[ -e /mnt/e/chess_data/q241.go ] && { echo "[q242] q241.go present — QUEUE 242 DONE"; exit 0; }
fp() { local k=""; [ -n "$1" ] && k="setoption name Knobs value $1"; printf '%s\nbench\nquit\n' "$k" | nice -n 10 "$T" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+'; }
echo "[q242] $(date '+%Y-%m-%d %H:%M:%S') start"
OUT=chess_engine_uci_s6test nice -n 10 bash selfplay/build_uci.sh > "$W/build.out" 2>&1 || { echo "[q242] ☠️ test build failed"; grep -iE 'error' "$W/build.out" | head -20 | sed 's/^/[q242]   /'; exit 1; }
echo "[q242] $(date '+%H:%M') built"
d=$(fp ""); a0=$(fp "SEARCH_ARM=0"); d7=$(fp "SV2_VERIFY_DEEPER=1")
echo "[q242] default [$d] · SEARCH_ARM=0 [$a0] · D7 [$d7]"
if [ "$d" != "solved 249/300 nodes 37279382" ] || [ "$a0" != "$d" ] || [ "$d7" != "solved 250/300 nodes 37320382" ]; then echo "[q242] ☠️ the new code moved a node"; exit 1; fi
for k in SV2_VERIFY_DEEPER=2 SV2_VERIFY_PROMOTE=1 SV2_VERIFY_DEEPER=1,SV2_VERIFY_PROMOTE=1 SV2_STATIC_CUT_STORE=1 SV2_HIST_ALLNODE_GATE=1 SV2_MALUS_SEARCHED_ONLY=1 ASP_KEEP_FAILHIGH=1 FIX_RESEARCH=1 SV2_VERIFY_DEEPER=1,FIX_RESEARCH=1; do
	echo "[q242]   $k: [$(fp "$k")]"
done
echo "[q242] $(date) QUEUE 242 DONE"
