#!/bin/bash
# Build the native UCI binary (2026-10-19; dev_notes/current/UCI-BINARY-DESIGN-2026-10-19.md). Same compile flags as setupAI.py so the
# engine code generates the same results as the Cython module (the `bench` parity gate proves it from M3 on).
# Low priority, one compile job at a time. Output: NN Engine/build_uci/chess_engine_uci
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
cd "$ND" || exit 1
mkdir -p build_uci
FLAGS="-Ofast -march=native -flto -fopenmp -fno-rtti -std=c++20 -mpopcnt -mbmi2 -I. -I/usr/include/eigen3"
SRC="cpp_bitboard.cpp eval_v2.cpp ThreadPool.cpp search_engine.cpp position.cpp book.cpp uci_main.cpp"
# Profile-guided build in two passes (results are unchanged; verify with the bench): PGO=gen builds an instrumented
# binary that writes its profile into PGO_DIR when it exits; PGO=use rebuilds from that profile. Code the training run
# never reached keeps normal optimisation (-fprofile-partial-training). FLAGS is expanded unquoted, so PGO_DIR must not
# contain spaces (the repo path does: put the profile elsewhere, e.g. /mnt/e/chess_data/pgo_<tag>).
case "$PGO_DIR" in *" "*) echo "BUILD FAILED: PGO_DIR contains a space: $PGO_DIR"; exit 1 ;; esac
# EXTRA_FLAGS: appended as given, e.g. "-g -fno-omit-frame-pointer" for a copy perf can symbolise.
FLAGS="$FLAGS $EXTRA_FLAGS"
case "$PGO" in
  gen) FLAGS="$FLAGS -fprofile-generate=${PGO_DIR:?PGO_DIR needed}" ;;
  use) FLAGS="$FLAGS -fprofile-use=${PGO_DIR:?PGO_DIR needed} -fprofile-partial-training -Wno-missing-profile" ;;
esac
nice -n 19 g++ $FLAGS $SRC -o build_uci/${OUT:-chess_engine_uci} -pthread -flto=1 2> build_uci/build.log
rc=$?
grep -E "error" build_uci/build.log | head -20
[ $rc -eq 0 ] && echo "BUILD OK: build_uci/${OUT:-chess_engine_uci}" || { echo "BUILD FAILED (rc $rc) — see build_uci/build.log"; tail -20 build_uci/build.log; }
exit $rc
