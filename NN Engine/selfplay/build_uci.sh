#!/bin/bash
# Build the native UCI binary (2026-10-19; dev_notes/UCI-BINARY-DESIGN-2026-10-19.md). Same compile flags as setupAI.py so the
# engine code generates the same results as the Cython module (the `bench` parity gate proves it from M3 on).
# Low priority, one compile job at a time. Output: NN Engine/build_uci/chess_engine_uci
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
cd "$ND" || exit 1
mkdir -p build_uci
FLAGS="-Ofast -march=native -flto -fopenmp -fno-rtti -std=c++20 -mpopcnt -mbmi2 -I. -I/usr/include/eigen3"
SRC="cpp_bitboard.cpp eval_v2.cpp ThreadPool.cpp search_engine.cpp position.cpp book.cpp uci_main.cpp"
nice -n 19 g++ $FLAGS $SRC -o build_uci/chess_engine_uci -pthread -flto=1 2> build_uci/build.log
rc=$?
grep -E "error" build_uci/build.log | head -20
[ $rc -eq 0 ] && echo "BUILD OK: build_uci/chess_engine_uci" || { echo "BUILD FAILED (rc $rc) — see build_uci/build.log"; tail -20 build_uci/build.log; }
exit $rc
