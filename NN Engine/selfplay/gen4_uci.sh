#!/bin/bash
# Gen 4 (2026-10-19, the ship the search transition started from) as a UCI engine: the gen 5 binary with V2_SHIP_GEN=4
# restores gen 4 exactly (bench 258 / 47,689,413 against gen 5's 258 / 47,689,586, checked 11-10). Gen 5 = gen 4 +
# FIX_DRAW_EXIT + FIX_QS_CHECK. Not a PGO build, and the pre-6.1 time manager: both are part of what the transition changed.
export V2_SHIP_GEN=4
exec "/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine/build_uci/chess_engine_uci_gen5" "$@"
