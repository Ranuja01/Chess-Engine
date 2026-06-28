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

  ks_static)
    # King-safety STATIC-gap analyzer under forwarded KS_* knobs (our_static vs SF_static on the
    # king-danger corpus). Deterministic, no SF (labels are pre-baked in the CSV). Args: <tag> [KNOB=v ...].
    tag="${1:?tag required}"; shift || true
    env "$@" OMP_NUM_THREADS=1 USE_OPENING_BOOK=0 \
        "$PY" diagnostics/_kingsafety_static.py > "/tmp/ksstat_${tag}.out" 2> "/tmp/ksstat_${tag}.err" || true
    grep -hE 'STATIC GAP|ALL |phase|att|opening|early_mid|late_mid|endgame' "/tmp/ksstat_${tag}.out" || cat "/tmp/ksstat_${tag}.err"
    ;;

  evalmode_sweep)
    # Sweep eval-mode combinations at the quiescent decision sites (QSTANDPAT_EVAL_MODE / FUTILITY_EVAL_MODE:
    # 0=full, 1=cheap material+PST, 2=light surrogate) on sts_timed_depth = the NET positional read at equal
    # time (a cheaper eval that buys depth shows as a HIGHER STS only if depth gain > eval coarsening).
    # Baseline (all full) STS@time ~1501. One allowlisted call runs the whole grid.
    declare -a grid=(
      "base:"
      "fut1:FUTILITY_EVAL_MODE=1"
      "fut2:FUTILITY_EVAL_MODE=2 KS_LIGHT_MAG=120"
      "fut2nks:FUTILITY_EVAL_MODE=2 KS_LIGHT_MAG=0"
      "sp1:QSTANDPAT_EVAL_MODE=1"
      "sp1fut1:QSTANDPAT_EVAL_MODE=1 FUTILITY_EVAL_MODE=1"
    )
    for entry in "${grid[@]}"; do
      label="${entry%%:*}"; knobs="${entry#*:}"
      env $knobs PRESET=LIGHTNING MAX_DEPTH=64 USE_OPENING_BOOK=0 OMP_NUM_THREADS=1 \
          "$PY" diagnostics/sts_test.py sts300.epd "em_${label}" > "/tmp/em_${label}.out" 2>/dev/null || true
      sc=$(grep -hoE 'STS score: [0-9]+/[0-9]+ +\([0-9.]+%\)' "/tmp/em_${label}.out")
      dp=$("$PY" - "diagnostics/results/sts_results_em_${label}.csv" 2>/dev/null <<'PYEOF'
import csv,sys
try: rows=list(csv.DictReader(open(sys.argv[1])))
except Exception: print("d?"); sys.exit()
def f(x):
    try: return float(x)
    except: return None
sc=[r for r in rows if r.get('score') not in (None,'')]
hd=[r for r in sc if f(r.get('depth')) is not None]
nm=[r for r in hd if not (f(r.get('eval')) is not None and (abs(f(r['eval']))>=9000000 or f(r['eval'])<=-15000))]
print("d%.2f"%(sum(f(r['depth']) for r in nm)/len(nm)) if nm else "d?")
PYEOF
)
      echo "${label}	${sc:-?}	${dp}	[$knobs]"
    done
    ;;

  ks_sweep)
    # King-safety static-gap GRID sweep (loops inside the dispatcher so one allowlisted call runs the
    # whole sweep). Each line: <label> then the ALL + key midgame buckets. Edit the grid below to retune.
    declare -a grid=(
      "off:KING_SAFETY_MAG=0"
      "base700:KING_SAFETY_MAG=700"
      "knee8:KING_SAFETY_MAG=700 KS_KNEE=8"
      "def4:KING_SAFETY_MAG=700 KS_DEFENDER=4"
      "def4knee8:KING_SAFETY_MAG=700 KS_DEFENDER=4 KS_KNEE=8"
      "strong:KING_SAFETY_MAG=1500 KS_DEFENDER=3 KS_KNEE=10"
      "gentle:KING_SAFETY_MAG=1200 KS_DEFENDER=4 KS_KNEE=8 KS_ATTACK_COUNT=0"
    )
    for entry in "${grid[@]}"; do
      label="${entry%%:*}"; knobs="${entry#*:}"
      env $knobs OMP_NUM_THREADS=1 USE_OPENING_BOOK=0 \
          "$PY" diagnostics/_kingsafety_static.py > "/tmp/kssw_${label}.out" 2>/dev/null || true
      echo "### $label  ($knobs)"
      grep -hE 'ALL |early_mid\|att3|early_mid\|att5|late_mid\|att4' "/tmp/kssw_${label}.out" || echo "  (no output)"
    done
    ;;

  sts_sweep)
    # STS GRID sweep in king_safety REPLACEMENT mode (latent_threat off) — loops inside the dispatcher so
    # one allowlisted call runs the whole sweep. Target: match/beat baseline 1503 (= latent_threat's +92).
    # OMP pinned for determinism. Each line: <label> STS score.
    # Collapse-campaign Phase 1: screen the parked correctness/collapse toggles individually on STS (the
    # positional no-regression guard) vs baseline 1503. Survivors (STS held) get folded into a Gap-T bundle.
    declare -a grid=(
      "baseline:"
      "gapt:VERIFY_MARGIN=16000"
      "b_tr:VERIFY_MARGIN=16000 ENABLE_ROOK_DBLCOUNT_FIX=1 ENABLE_ROOK_DBLCOUNT_SYM_UP=1"
      "b_trq:VERIFY_MARGIN=16000 ENABLE_ROOK_DBLCOUNT_FIX=1 ENABLE_ROOK_DBLCOUNT_SYM_UP=1 ENABLE_QPREC_PHASE_GATE=1"
      "b_trqk:VERIFY_MARGIN=16000 ENABLE_ROOK_DBLCOUNT_FIX=1 ENABLE_ROOK_DBLCOUNT_SYM_UP=1 ENABLE_QPREC_PHASE_GATE=1 ENABLE_KNIGHT_MOB_FIX=1"
      "b_rq:ENABLE_ROOK_DBLCOUNT_FIX=1 ENABLE_ROOK_DBLCOUNT_SYM_UP=1 ENABLE_QPREC_PHASE_GATE=1"
    )
    for entry in "${grid[@]}"; do
      label="${entry%%:*}"; knobs="${entry#*:}"
      sc=$(env $knobs OMP_NUM_THREADS=1 MAX_DEPTH=10 USE_OPENING_BOOK=0 PRESET=LONG_FORMAT \
            "$PY" diagnostics/sts_test.py sts300.epd "stssw_${label}" 2>/dev/null | grep -hoE 'STS score: [0-9]+/[0-9]+ +\([0-9.]+%\)')
      echo "${label}	${sc:-(none)}	[$knobs]"
    done
    ;;

  ourmove)
    # Our engine's move on explicit FENs, NO Stockfish (for when WSL SF-exec is flaky). Args: [KEY=VAL ...] '<fen>'...
    envs=(); fens=()
    for a in "$@"; do
      if [[ "$a" == *=* ]]; then envs+=("$a"); else fens+=("$a"); fi
    done
    env PRESET=LONG_FORMAT MAX_DEPTH=12 USE_OPENING_BOOK=0 OMP_NUM_THREADS=1 "${envs[@]}" \
        "$PY" -c $'import sys\nsys.path.insert(0,"diagnostics")\nfrom tactical_test import run_one\nfor f in sys.argv[1:]:\n r=run_one(f,set())\n print("mv=%-6s ev=%-8s d=%-3s %s"%(r["uci"],r["eval"],r["depth"],f))' "${fens[@]}"
    ;;

  fenvs)
    # fen_vs_sf on EXPLICIT FENs with forwarded engine knobs (position-fix check for the collapse campaign).
    # Args: [KEY=VAL ...] '<fen>' ['<fen>' ...]. Prints our move/eval vs SF best/cp + match flag.
    export STOCKFISH_PATH="$SF"
    envs=(); fens=()
    for a in "$@"; do
      if [[ "$a" == *=* ]]; then envs+=("$a"); else fens+=("$a"); fi
    done
    env PRESET=LONG_FORMAT MAX_DEPTH=12 USE_OPENING_BOOK=0 OMP_NUM_THREADS=1 "${envs[@]}" \
        "$PY" selfplay/fen_vs_sf.py "${fens[@]}"
    ;;

  ks_explain)
    # King-safety visualizer for a FEN (board + zone attack-heat + per-component readout). Args: '<fen>' [--side ...].
    "$PY" diagnostics/ks_explain.py "$@"
    ;;

  movematch)
    # Themed move-match scorecard for an eval candidate. Args: <tag> <themes|all> [KNOB=v ...].
    # Defaults precede "$@" so a candidate can override MAX_DEPTH (the delta-depth audit) or themes.
    tag="${1:?tag required}"; shift || true
    themes="${1:-all}"; shift || true
    targ=(); [ "$themes" != "all" ] && targ=(--themes "$themes")
    # Pin EVERY thread pool to 1 (OpenMP + the BLAS/TF pools keras pulls in). Without this, each process
    # spawns stray threads and a few concurrent runs oversubscribe the physical cores -> the "fixed-depth"
    # search drifts (non-deterministic). Fully pinned, each run is single-core and concurrent runs are
    # byte-identical to single-process (verified: 6-wide == single-proc, +0). Bound concurrency to <= the
    # PHYSICAL core count (nproc is hyperthreaded). Pins precede "$@" so a candidate can still override.
    env MAX_DEPTH=10 USE_OPENING_BOOK=0 PRESET=LONG_FORMAT \
        OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
        VECLIB_MAXIMUM_THREADS=1 TF_NUM_INTEROP_THREADS=1 TF_NUM_INTRAOP_THREADS=1 "$@" \
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
    conc="${1:-6}"; shift || true          # optional: concurrency (default 6; lower when fewer cores are free)
    ttag="${1:-overnight_speed}"; shift || true   # optional: output tag (default overnight_speed; set a fresh one to avoid clobbering)
    export STOCKFISH_PATH="$SF"
    # Pin each engine to one thread (eval has OpenMP regions); at concurrency 6 default-OMP would
    # oversubscribe the physical cores. Symmetric for both sides, so the A/B stays fair, and it matches
    # the OMP=1 regime move-match candidates are measured under.
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
           VECLIB_MAXIMUM_THREADS=1 TF_NUM_INTEROP_THREADS=1 TF_NUM_INTRAOP_THREADS=1
    "$PY" selfplay/tournament.py \
        --p1-label base --p1-config "" \
        --p2-label fast --p2-config "$p2cfg" \
        --preset LIGHTNING --concurrency "$conc" --max-minutes "$mins" \
        --openings selfplay/openings_uho.txt --adjudicate-draw --quiet --tag "$ttag"
    ;;

  annotate)
    # SF-annotate every game under games/<tag>/ -> per-ply SF cp + our eval_breakdown (analysis.csv).
    # The long parallel SF batch that feeds the term-attribution diagnostic (eval_breakdown --tag).
    tag="${1:?tag required}"; shift || true
    conc="${1:-4}"; shift || true              # SF workers (default 4)
    export STOCKFISH_PATH="$SF"
    "$PY" selfplay/annotate.py --tag "$tag" --concurrency "$conc" --sf-depth 12 "$@"
    ;;

  pattern_diag)
    # Offline miseval-pattern miner over an ANNOTATED tag: offender clusters / fixable-bias-vs-scatter /
    # transform leaks, by material config. No SF/engine. Args: <tag> [extra pattern_diag.py flags...].
    tag="${1:?tag required}"; shift || true
    "$PY" selfplay/pattern_diag.py --tag "$tag" "$@"
    ;;

  breakdown_tag)
    # Term-attribution catalog: rank plies of an ANNOTATED tag by our-vs-SF divergence, attribute the worst
    # over-reads to eval terms. Reads recorded sf_cp (no live SF). Args: <tag> [top=50] [min-div=2.0].
    tag="${1:?tag required}"; shift || true
    top="${1:-50}"; shift || true
    mindiv="${1:-2.0}"; shift || true
    export STOCKFISH_PATH="$SF"
    "$PY" diagnostics/eval_breakdown.py --tag "$tag" --top "$top" --min-div "$mindiv" "$@"
    ;;

  breakdown_fen)
    # Per-FEN static-eval term attribution + live SF static/search. Args: [KEY=VAL ...] '<fen>' ['<fen>' ...].
    # KEY=VAL args are forwarded as engine env knobs (e.g. PASSER_BLOCK_ADV=50); the rest are FENs.
    export STOCKFISH_PATH="$SF"
    envs=(); fens=()
    for a in "$@"; do
      if [[ "$a" == *=* ]]; then envs+=("$a"); else fens+=("$a"); fi
    done
    env "${envs[@]}" "$PY" diagnostics/eval_breakdown.py --fen "${fens[@]}"
    ;;

  tournament_seeded)
    # Timed self-play A/B from a CUSTOM openings file (seeded collapse-zone playouts / seeded SPRT).
    # Args: <minutes> "<p2 knobs>" <openings_file> [conc=4] [tag=seeded_play]
    mins="${1:?minutes required}"; shift || true
    p2cfg="${1:-}"; shift || true
    openings="${1:?openings file required}"; shift || true
    conc="${1:-4}"; shift || true
    ttag="${1:-seeded_play}"; shift || true
    export STOCKFISH_PATH="$SF"
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
           VECLIB_MAXIMUM_THREADS=1 TF_NUM_INTEROP_THREADS=1 TF_NUM_INTRAOP_THREADS=1
    "$PY" selfplay/tournament.py \
        --p1-label base --p1-config "" \
        --p2-label cand --p2-config "$p2cfg" \
        --preset LIGHTNING --concurrency "$conc" --max-minutes "$mins" \
        --openings "$openings" --adjudicate-draw --quiet --tag "$ttag"
    ;;

  flips)
    # Collapse-rate KPI: mine one arm's flips from a played/annotated tag. Args: <tag> <arm> [drop=3.0].
    # Prints games / <arm>_losses / flips_found  (=> flips per game = the floor metric).
    tag="${1:?tag required}"; shift || true
    arm="${1:?arm required}"; shift || true
    drop="${1:-3.0}"; shift || true
    "$PY" selfplay/flip_extract.py "$tag" "$drop" "$arm" 2>&1 | head -2
    ;;

  move_proxy)
    # Move-decisive proxy: fen_vs_sf base vs candidate over a FEN csv; prints SF-best match + over-read delta.
    # Args: <csv> <n> "<cand knobs>"   (cand knobs forwarded as env to the candidate pass only).
    csv="${1:?csv required}"; shift || true
    n="${1:?n required}"; shift || true
    cand="${1:-}"; shift || true
    export STOCKFISH_PATH="$SF"
    env OMP_NUM_THREADS=1 PRESET=LIGHTNING USE_OPENING_BOOK=0 "$PY" selfplay/fen_vs_sf.py --csv "$csv" "$n" > /tmp/mp_base.txt 2>/dev/null
    env OMP_NUM_THREADS=1 PRESET=LIGHTNING USE_OPENING_BOOK=0 $cand "$PY" selfplay/fen_vs_sf.py --csv "$csv" "$n" > /tmp/mp_cand.txt 2>/dev/null
    "$PY" - <<'PYEOF'
import re
def parse(p):
    d={}
    for ln in open(p):
        m=re.search(r'ev=\s*(-?\d+)\s+mv=(\S+)\s+d=(\d+).*cp=\s*(-?\d+)\s+best=(\S+).*?(game_\d+)',ln)
        if m:
            ev,mv,dp,cp,best,g=m.groups(); d[g]=dict(ev=int(ev),mv=mv,cp=int(cp),best=best)
    return d
b=parse('/tmp/mp_base.txt'); c=parse('/tmp/mp_cand.txt')
keys=[k for k in b if k in c]
if not keys:
    print('move_proxy: no parsed rows'); raise SystemExit
mm=lambda d: sum(1 for k in keys if d[k]['mv']==d[k]['best'])
err=lambda d: sum(abs(d[k]['ev']/10.0-d[k]['cp']) for k in keys)/len(keys)
plugged=sum(1 for k in keys if b[k]['mv']!=b[k]['best'] and c[k]['mv']==c[k]['best'])
broke=sum(1 for k in keys if b[k]['mv']==b[k]['best'] and c[k]['mv']!=c[k]['best'])
print(f'N={len(keys)}  SF-best: base {mm(b)} cand {mm(c)}  (plugged {plugged} / broke {broke})')
print(f'over-read |our_cp-sf_cp|: base {err(b):.0f} cand {err(c):.0f} delta {err(c)-err(b):+.0f}')
PYEOF
    ;;

  depth_probe)
    # How many collapse positions does OUR engine SOLVE (match SF-best) as fixed depth increases?
    # Distinguishes within-depth (tunable now) vs deeper-depth (depth lever) vs never (eval/future-sight).
    # Args: <csv> <n>  [optional cand knobs forwarded to the engine].
    csv="${1:?csv required}"; shift || true
    n="${1:?n required}"; shift || true
    export STOCKFISH_PATH="$SF"
    for d in 10 16 22; do
      env OMP_NUM_THREADS=1 PRESET=LONG_FORMAT MAX_DEPTH=$d USE_OPENING_BOOK=0 "$@" \
        "$PY" selfplay/fen_vs_sf.py --csv "$csv" "$n" 2>/dev/null \
        | awk -v D=$d '/ours:/{t++; if(index($0," OK "))m++} END{printf "depth %s: our-move==SF-best  %d/%d\n", D, m, t+0}'
    done
    ;;

  cploss_probe)
    # ACPL proxy (overall move-quality inner loop): mean centipawn-loss of OUR move vs SF over general
    # midgame positions. Lower = better general play. Args: <tag> <n> [KEY=VAL knobs...].
    tag="${1:?tag required}"; shift || true
    n="${1:-150}"; shift || true
    export STOCKFISH_PATH="$SF"
    env OMP_NUM_THREADS=1 PRESET=LONG_FORMAT MAX_DEPTH=10 USE_OPENING_BOOK=0 SF_MOVETIME=0.25 "$@" \
        "$PY" selfplay/cploss_probe.py "$tag" "$n" 2>/dev/null | grep '^cploss:'
    ;;

  runup_probe)
    # Fast run-up collapse probe (PACE inner loop): play OUR engine vs SF a few plies from each seed, cut off
    # on collapse/survive, report collapse-rate. Args: <seed_csv> <n> <plies> [KEY=VAL knobs...].
    csv="${1:?csv required}"; shift || true
    n="${1:-50}"; shift || true
    plies="${1:-8}"; shift || true
    export STOCKFISH_PATH="$SF"
    env OMP_NUM_THREADS=1 PRESET=LONG_FORMAT MAX_DEPTH=10 USE_OPENING_BOOK=0 SF_MOVETIME=0.15 "$@" \
        "$PY" selfplay/runup_probe.py "$csv" "$n" "$plies" 2>/dev/null | grep '^runup:'
    ;;

  vs_sf)
    # Collapse-mining: OUR engine vs a strength-targeted Stockfish (UCI_Elo), alternating colors + draw
    # adjudication (rule-compliant, fast tails); flags games where our eval peaked winning then we failed to
    # win and dumps their run-up FENs (the corpus seed for the diagnose->fix->ship loop). For the overnight
    # hole-mining at full strength use elo=0 (unlimited) or 3000. Needs WSL->SF interop up. Args:
    #   <elo> <games> [preset=LIGHTNING] [win_thresh=2000] [KEY=VAL our-engine knobs...]
    elo="${1:-2400}"; shift || true
    games="${1:-20}"; shift || true
    preset="${1:-LIGHTNING}"; shift || true
    wt="${1:-2000}"; shift || true
    export STOCKFISH_PATH="$SF"
    env OMP_NUM_THREADS=1 "$@" \
        "$PY" selfplay/vs_sf.py --sf-elo "$elo" --games "$games" --preset "$preset" \
        --win-threshold "$wt" --openings selfplay/openings_uho.txt --adjudicate-draw --quiet \
        --tag "vssf_${elo}"
    ;;

  triage)
    # Classify the collapse points in games/<tag>/collapses.csv as EVAL (our move/eval still wrong at
    # deep depth = an eval hole) vs HORIZON (deeper search avoids it). Re-searches each decision FEN at
    # MAX_DEPTH (deep) + SF compare. Args: <games_dir> [sf_time=2] [MAX_DEPTH=18].
    gd="${1:?games_dir required}"; shift || true
    sft="${1:-2}"; shift || true
    md="${1:-18}"; shift || true
    export STOCKFISH_PATH="$SF"
    env OMP_NUM_THREADS=1 PRESET=LONG_FORMAT MAX_DEPTH="$md" USE_OPENING_BOOK=0 "$@" \
        "$PY" diagnostics/triage_collapses.py "$gd" "$sft"
    ;;

  pyrun)
    # Run a project python helper with the anaconda interpreter (analysis scripts). Args: <script.py> [args...].
    export STOCKFISH_PATH="$SF"
    "$PY" "$@"
    ;;

  bias_sweep)
    # Cheap PACE perturbation sweep: reeval an annotated tag under each MOD_* setting and print the key
    # signed-bias buckets (+1B+2P, +1R+2P) and the equal-material control. Args: <tag>. One process per
    # setting (knobs are read once at engine init). The agent reads the pattern to pick directions.
    tag="${1:?tag required}"; shift || true
    for setting in \
      'default:' \
      'matpawns+:MOD_MAT_PAWNS=300' \
      'matpawns-:MOD_MAT_PAWNS=-300' \
      'matoppb-:MOD_MAT_OPPB=-600' \
      'ltback+:MOD_LT_BACKING=400' \
      'ltback-:MOD_LT_BACKING=-400' \
      'pairopen+:MOD_PAIR_OPEN=400' \
      'pairopen-:MOD_PAIR_OPEN=-400' \
      'combo:MOD_MAT_PAWNS=300 MOD_MAT_OPPB=-400 MOD_LT_BACKING=400'; do
      label="${setting%%:*}"; knobs="${setting#*:}"
      env $knobs "$PY" selfplay/annotate.py --tag "$tag" --reeval >/dev/null 2>&1
      echo -n "$label	"
      "$PY" selfplay/pattern_diag.py --tag "$tag" 2>/dev/null \
        | awk '/\+1B\+2P /{b=$3} /\+1R\+2P /{r=$3} /even \(d=0\)/{e=$4} END{printf "1B2P=%s 1R2P=%s even=%s\n", b, r, e}'
    done
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
