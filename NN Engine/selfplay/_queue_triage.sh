#!/bin/bash
# 2026-10-18 — EVAL vs DEPTH vs PRUNING triage of the owner's saved game positions (_case_triage.py), candidate base, d10.
# Single-threaded math (OMP etc. = 1). Sequential: SF pass, normal search, no-pruning search, report.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
cd "$ND" || exit 1
T="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 V2_PRESET=candidate"
env $T bash "$R" pyrun diagnostics/_case_triage.py MODE=sf > /dev/null 2>&1
env $T MAX_DEPTH=10 bash "$R" pyrun diagnostics/_case_triage.py MODE=search TAG=normal > /dev/null 2>&1
env $T MAX_DEPTH=10 ENABLE_LMR=0 ENABLE_FUTILITY=0 ENABLE_RAZORING=0 ENABLE_LMP=0 ENABLE_RFP=0 ENABLE_ROOT_RAZOR=0 SEE_PRUNE_CAPTURES=0 \
  bash "$R" pyrun diagnostics/_case_triage.py MODE=search TAG=noprune > /dev/null 2>&1
bash "$R" pyrun diagnostics/_case_triage.py MODE=report
