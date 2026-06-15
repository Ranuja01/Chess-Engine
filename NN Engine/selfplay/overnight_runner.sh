#!/usr/bin/env bash
#
# Vetted dispatcher for the unattended overnight eval-speed run.
#
# This is the ONLY WSL command Claude is permitted to run autonomously overnight
# (see .claude/settings.local.json -> permissions.allow). Every subcommand below is
# a fixed, "regular" operation: a clean engine rebuild, a fixed-depth bench, or the
# self-play tournament. There is no eval/exec of arbitrary input -- the only free
# arguments are a result tag and a space-separated list of ENGINE env knobs, which
# are handed to `env` and consumed by the engine's own getenv allow-list. Read this
# file once and you know the full blast radius of the overnight run.
#
# Usage:
#   overnight_runner.sh build                         # clean rebuild (production)
#   overnight_runner.sh wac  <tag> [KNOB=v ...]       # WAC d10, prints solves + node sum
#   overnight_runner.sh sts  <tag> [KNOB=v ...]       # STS300 d10, prints score
#   overnight_runner.sh tournament <minutes> <p2cfg>  # base vs p2cfg, timed
#   overnight_runner.sh result                        # dump tournament.json
#   overnight_runner.sh clock                         # HH:MM (to size --max-minutes)
#   overnight_runner.sh ps                            # list any running bench/tournament procs

set -uo pipefail

ENGINE="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
PY="/home/ranuja/anaconda3/bin/python"
SF="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/stockfish/stockfish-windows-x86-64-avx2.exe"
cd "$ENGINE" || { echo "engine dir not found"; exit 2; }

cmd="${1:-}"; shift || true

case "$cmd" in
  build)
    # Force-clean production rebuild (no PROFILE_EVAL). Same clean step used all session.
    touch cpp_bitboard.cpp cpp_bitboard.h search_engine.cpp search_engine.h ChessAI.pyx
    rm -rf build ChessAI.cpp ChessAI.*.so
    "$PY" setupAI.py build_ext --inplace 2>&1 | tail -n 5
    ;;

  wac)
    # Fixed-depth tactical bench. Deterministic (book off). Prints solves + summed node count.
    tag="${1:?tag required}"; shift || true
    env "$@" MAX_DEPTH=10 USE_OPENING_BOOK=0 PRESET=LONG_FORMAT \
        "$PY" diagnostics/tactical_test.py wac.epd "$tag" > "/tmp/wac_${tag}.out" 2> "/tmp/wac_${tag}.err" || true
    echo -n "SOLVED: "; grep -hoE 'Solved [0-9]+/[0-9]+' "/tmp/wac_${tag}.out" || echo "?"
    echo -n "NODES: ";  grep -hoP '\(nodes=\K[0-9]+' "/tmp/wac_${tag}.err" | awk '{s+=$1} END{print s+0}'
    echo -n "HIST: ";   grep -hoE '\[cutoff_histogram\] .*' "/tmp/wac_${tag}.err" | tail -1 || echo "(none)"
    ;;

  wac_timed)
    # Same as wac but wrapped in /usr/bin/time -v -> reports user-seconds + nps (the speed gate).
    tag="${1:?tag required}"; shift || true
    /usr/bin/time -v env "$@" MAX_DEPTH=10 USE_OPENING_BOOK=0 PRESET=LONG_FORMAT \
        "$PY" diagnostics/tactical_test.py wac.epd "$tag" > "/tmp/wac_${tag}.out" 2> "/tmp/wac_${tag}.err" || true
    echo -n "SOLVED: "; grep -hoE 'Solved [0-9]+/[0-9]+' "/tmp/wac_${tag}.out" || echo "?"
    nodes=$(grep -hoP '\(nodes=\K[0-9]+' "/tmp/wac_${tag}.err" | awk '{s+=$1} END{print s+0}')
    secs=$(grep -hoP 'User time \(seconds\): \K[0-9.]+' "/tmp/wac_${tag}.err")
    echo "NODES: $nodes"
    echo -n "WALL: "; grep -hoE 'wall clock.*' "/tmp/wac_${tag}.err" || echo "?"
    echo "USER_SECONDS: ${secs:-?}"
    awk -v n="$nodes" -v s="${secs:-0}" 'BEGIN{ if (s+0>0) printf "NPS: %d\n", n/s }'
    ;;

  sts)
    # Fixed-depth positional bench. Prints "STS score: X/3000 (Y%)".
    tag="${1:?tag required}"; shift || true
    env "$@" MAX_DEPTH=10 USE_OPENING_BOOK=0 PRESET=LONG_FORMAT \
        "$PY" diagnostics/sts_test.py sts300.epd "$tag" 2>/dev/null | grep -E 'STS score' || echo "STS score: (none)"
    ;;

  tournament)
    # Timed self-play A/B: baseline (no knobs) vs the qualifying speed bundle (p2cfg).
    mins="${1:?minutes required}"; shift || true
    p2cfg="${1:-}"; shift || true
    export STOCKFISH_PATH="$SF"
    "$PY" selfplay/tournament.py \
        --p1-label base --p1-config "" \
        --p2-label fast --p2-config "$p2cfg" \
        --preset LIGHTNING --concurrency 6 --max-minutes "$mins" \
        --openings selfplay/openings_uho.txt --adjudicate-draw --quiet --tag overnight_speed
    ;;

  result)
    "$PY" -c "import json; d=json.load(open('selfplay/games/overnight_speed/tournament.json')); print(d)" 2>/dev/null \
        || echo "no tournament.json yet"
    ;;

  clock)
    date '+%H:%M (%s epoch)'
    ;;

  ps)
    ps -eo pid,etime,args | grep -E 'tactical_test|sts_test|tournament.py|setupAI' | grep -v grep || echo "none running"
    ;;

  *)
    echo "unknown subcommand: '$cmd'"; exit 2
    ;;
esac
