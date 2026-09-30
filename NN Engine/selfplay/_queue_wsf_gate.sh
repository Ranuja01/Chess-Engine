#!/bin/bash
# 2026-10-01: POT winnability REFERENCE FORM (WSF_V2 scale factor, 2-feature fit) vs a FRESH shipped baseline,
# SF18 @250k, seeds 42 + 43 (unused). Sequential: one engine job at a time.
R="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine/selfplay/overnight_runner.sh"
CAND="WSF_V2=1 WSF_V2_BASE=-37 WSF_V2_SP=34 WSF_V2_OCB=-80"
for s in 42 43; do
  bash "$R" gauntlet 500 400 4 gauntlet_ship4_s$s $s V2_PRESET=shipped
  bash "$R" gauntlet 500 400 4 gauntlet_wsf_s$s $s V2_PRESET=shipped $CAND
done
echo "WSF GATE QUEUE DONE"
