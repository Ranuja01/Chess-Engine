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
#   overnight_runner.sh movematch <tag> <themes|all> [KNOB=v ...]  # themed move-match scorecard
#   overnight_runner.sh movematch_diff <base_tag> <cand_tag>       # per-theme delta + changed moves
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
    echo -n "EBF: ";    grep -hoP 'ebf=\K[0-9.]+' "/tmp/wac_${tag}.err" | awk '{s+=$1;n++} END{if(n)printf "%.3f\n",s/n; else print "?"}'
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

  wac_timed_depth)
    # TIMED (lightning) WAC: let iterative deepening run to the clock instead of a fixed depth, so a
    # node-efficient config converts its savings into DEPTH. Reports solves + MEAN DEPTH on non-mate
    # positions (a found mate stops deepening early, so those are excluded) + the mate-found count.
    tag="${1:?tag required}"; shift || true
    env "$@" PRESET=LIGHTNING MAX_DEPTH=64 USE_OPENING_BOOK=0 \
        "$PY" diagnostics/tactical_test.py wac.epd "$tag" > "/tmp/wact_${tag}.out" 2> "/tmp/wact_${tag}.err" || true
    echo -n "SOLVED: "; grep -hoE 'Solved [0-9]+/[0-9]+' "/tmp/wact_${tag}.out" || echo "?"
    "$PY" - "diagnostics/results/tactical_results_${tag}.csv" <<'PYEOF'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
def fnum(x):
    try: return float(x)
    except Exception: return None
# Mate/decisive positions stop iterative deepening early -> exclude from the depth mean.
def is_mate(r):
    e = fnum(r.get('eval')); return e is not None and (abs(e) >= 9000000 or e <= -15000)
scored = [r for r in rows if r.get('result') in ('PASS', 'fail')]
have_d = [r for r in scored if fnum(r.get('depth')) is not None]
nonmate = [r for r in have_d if not is_mate(r)]
md = sum(fnum(r['depth']) for r in nonmate) / len(nonmate) if nonmate else 0.0
mdall = sum(fnum(r['depth']) for r in have_d) / len(have_d) if have_d else 0.0
print(f"MEAN_DEPTH_NONMATE: {md:.3f}  (n={len(nonmate)})")
print(f"MEAN_DEPTH_ALL: {mdall:.3f}  (n={len(have_d)})")
print(f"MATE_FOUND: {sum(1 for r in scored if is_mate(r))}")
PYEOF
    ;;

  sts_timed_depth)
    # TIMED (lightning) STS: positional move-choice at equal time + the same non-mate depth mean.
    tag="${1:?tag required}"; shift || true
    env "$@" PRESET=LIGHTNING MAX_DEPTH=64 USE_OPENING_BOOK=0 \
        "$PY" diagnostics/sts_test.py sts300.epd "$tag" > "/tmp/stst_${tag}.out" 2> "/tmp/stst_${tag}.err" || true
    grep -hE 'STS score' "/tmp/stst_${tag}.out" || echo "STS score: ?"
    "$PY" - "diagnostics/results/sts_results_${tag}.csv" <<'PYEOF'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
def fnum(x):
    try: return float(x)
    except Exception: return None
def is_mate(r):
    e = fnum(r.get('eval')); return e is not None and (abs(e) >= 9000000 or e <= -15000)
scored = [r for r in rows if r.get('score') not in (None, '')]   # drop book hits
have_d = [r for r in scored if fnum(r.get('depth')) is not None]
nonmate = [r for r in have_d if not is_mate(r)]
md = sum(fnum(r['depth']) for r in nonmate) / len(nonmate) if nonmate else 0.0
print(f"MEAN_DEPTH_NONMATE: {md:.3f}  (n={len(nonmate)})")
print(f"MATE_FOUND: {sum(1 for r in scored if is_mate(r))}")
PYEOF
    ;;

  movematch)
    # Themed move-match scorecard for an eval candidate. Args: <tag> <themes|all> [KNOB=v ...].
    # Defaults precede "$@" so a candidate can override MAX_DEPTH (the delta-depth audit) or themes.
    tag="${1:?tag required}"; shift || true
    themes="${1:-all}"; shift || true
    targ=(); [ "$themes" != "all" ] && targ=(--themes "$themes")
    env MAX_DEPTH=10 USE_OPENING_BOOK=0 PRESET=LONG_FORMAT "$@" \
        "$PY" diagnostics/movematch.py run "$tag" "${targ[@]}" 2>/dev/null \
        | grep -E 'movematch run|^TOTAL|results ->|\([0-9]+%\)' || echo "movematch: (none)"
    ;;

  movematch_diff)
    # Per-theme score delta + changed-move list between two movematch runs. Args: <base_tag> <cand_tag>.
    base="${1:?base tag required}"; cand="${2:?cand tag required}"
    "$PY" diagnostics/movematch.py diff "$base" "$cand" 2>/dev/null || echo "movematch_diff: (none)"
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
    ps -eo pid,etime,args | grep -E 'tactical_test|sts_test|movematch|tournament.py|setupAI' | grep -v grep || echo "none running"
    ;;

  *)
    echo "unknown subcommand: '$cmd'"; exit 2
    ;;
esac
