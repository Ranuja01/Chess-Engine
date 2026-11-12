#!/bin/bash
# 2026-11-12 ~13:00, queue #214: size lazy generation before building it (S3 picker option B). Probe build (-DSV2_MOVE_PROBE,
# sv2_probe.h's new lazy-gen counters), search v2 (SEARCH_ARM=1), fixed depth: WAC d10 and game24 d12, the step-0 sets.
# On generated (cache-miss) interior lists: how many moves are quiet, how long the leading run of captures / promotions is,
# how often the node cuts inside that run (then no quiet generation was needed), and how many quiets those lists held.
# Fixed depth, so it runs beside #211's node-limited SF18 games without either affecting the other. A probe build must still
# bench 247 / 39,861,964. Test copy only (build_uci/sv2probe2). Stop: touch /mnt/e/chess_data/q214.stop.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
W=/mnt/e/chess_data/s3_lazy
STOP=/mnt/e/chess_data/q214.stop
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"; mkdir -p "$W"
echo "[q214] $(date '+%Y-%m-%d %H:%M:%S') start: probe build"
OUT=sv2probe2 EXTRA_FLAGS="-DSV2_MOVE_PROBE" bash selfplay/build_uci.sh > "$W/build.out" 2>&1 || { echo "[q214] ☠️ build failed — QUEUE 214 ABORTED"; tail -20 "$W/build.out" | sed 's/^/[q214]   /'; exit 1; }
[ -e "$STOP" ] && { echo "[q214] stopped"; exit 0; }
printf 'setoption name Knobs value SEARCH_ARM=1\nbench\nquit\n' | "$ND/build_uci/sv2probe2" 2>/dev/null | grep -E '^bench|sv2probe' > "$W/wac_d10.txt"
sed 's/^/[q214] WAC d10: /' "$W/wac_d10.txt"
[ -e "$STOP" ] && { echo "[q214] stopped"; exit 0; }
printf 'setoption name Knobs value SEARCH_ARM=1\nbench /mnt/e/chess_data/sv2_profile/game24.txt 12\nquit\n' | "$ND/build_uci/sv2probe2" 2>/dev/null | grep -E '^bench|sv2probe' > "$W/game24_d12.txt"
sed 's/^/[q214] game24 d12: /' "$W/game24_d12.txt"
echo "[q214] $(date) QUEUE 214 DONE"
