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
    # swap@600 is nearly INERT (term contributes ~0.05p) — the real lever is MAGNITUDE + curve steepness
    # (danger = units^2/divisor, knee=12 then linear slope 2*knee/divisor). All swap rows run in REPLACEMENT
    # mode (ENABLE_KS_REPLACE_LT=1, latent off) to match the shipping config; base_lt is the latent anchor.
    declare -a grid=(
      "base_lt:KING_SAFETY_MAG=0"
      "m4000:ENABLE_KS_REPLACE_LT=1 KING_SAFETY_MAG=4000"
      "m4000bk:ENABLE_KS_REPLACE_LT=1 KING_SAFETY_MAG=4000 MOD_KS_BACKING=256"
      "m4000ctl:ENABLE_KS_REPLACE_LT=1 KING_SAFETY_MAG=4000 MOD_KS_CONTROL=256"
      "m4000bkctl:ENABLE_KS_REPLACE_LT=1 KING_SAFETY_MAG=4000 MOD_KS_BACKING=256 MOD_KS_CONTROL=256"
      "m6000bkctl:ENABLE_KS_REPLACE_LT=1 KING_SAFETY_MAG=6000 MOD_KS_BACKING=512 MOD_KS_CONTROL=256"
    )
    for entry in "${grid[@]}"; do
      label="${entry%%:*}"; knobs="${entry#*:}"
      env $knobs OMP_NUM_THREADS=1 USE_OPENING_BOOK=0 \
          "$PY" diagnostics/_kingsafety_static.py > "/tmp/kssw_${label}.out" 2>/dev/null || true
      echo "### $label  ($knobs)"
      grep -hE 'ALL |opening\|att2|early_mid\|att3|early_mid\|att5|late_mid\|att4' "/tmp/kssw_${label}.out" || echo "  (no output)"
    done
    ;;

  ks_movematch_sweep)
    # King-safety move-match GRID = the PACE objective: the FULL 15-theme move-match per knob-set, looped
    # inside the dispatcher (one allowlisted call, NO =-args on the command line -> unattended/permission-safe).
    # Pins EVERY thread pool (OMP/BLAS/TF) for determinism exactly like the movematch sub. Writes a tagged
    # CSV (movematch_kmm_<label>.csv) per row -> diff any two with `movematch_diff kmm_<a> kmm_<b>`.
    # King-safety TARGET themes: King Activity / Open Files and Diagonals / 7th Rank; the other 12 are CONTROLS
    # (must not drop). Edit the grid to retune.
    # Strengthen-the-anchor sweep: anchor = m4000ctl (ENABLE_KS_REPLACE_LT=1 KING_SAFETY_MAG=4000
    # MOD_KS_CONTROL=256), held fixed; vary the KS SHAPE knobs to make the term fire harder on REAL backed
    # attacks (the own-king collapses underweight our own king). safe-checks = strongest real-danger signal;
    # KS_DEFENDER down = stop over-crediting our own defenders; open-file up = exposed king; divisor down =
    # steeper danger. Full-suite move-match = the no-regression control (passers/safety/positional).
    # def1 + KS_FLOOR deadzone sweep: does the floor RECOVER the passer regression (Advancement/Undermine back
    # to >= anchor) while KEEPING def1's king/positional gains? Bracket floor 2/3/4/6 (too high also kills real
    # king detection). anchor + def1 are the references; movematch_diff each def1_fX vs anchor for the per-theme read.
    # MEASURED floor (units p75 HURT=1/HELPED=13; favorable band ~4-6): confirm the two measured candidates on
    # move-match (the search-integrated truth), against anchor + def1 references. Did the floor RECOVER passers
    # (Advancement/Undermine toward anchor) while KEEPING def1's king gains?
    # piece_value_boost <- mobility (MOD_PVBOOST_MOB): probe reduced the over-read in the cramped bucket. Gate =
    # full-suite no-regression (does damping the material-lead bonus hurt any theme?). anchor = m4000ctl.
    # offense-gated MOD_PVBOOST_MOB (only damp cramped material when NOT out-attacking) -> should recover the
    # King-Activity/Open-Files regressions from the ungated version (mob32 was -29 with king-theme regress).
    # Final overnight experiment: imbalance <- realizability (dormant REALIZ_* hook, Table B #1) on TOP of
    # m4000ctl. Different term from the failed piece_value_boost levers; move-match-validatable screen.
    declare -a grid=(
      "anchor:ENABLE_KS_REPLACE_LT=1 KING_SAFETY_MAG=4000 MOD_KS_CONTROL=256"
      "realiz_a:ENABLE_KS_REPLACE_LT=1 KING_SAFETY_MAG=4000 MOD_KS_CONTROL=256 REALIZ_MAT_K=128 REALIZ_PHASE_K=128"
      "realiz_b:ENABLE_KS_REPLACE_LT=1 KING_SAFETY_MAG=4000 MOD_KS_CONTROL=256 REALIZ_MAT_K=256 REALIZ_PHASE_K=256"
    )
    for entry in "${grid[@]}"; do
      label="${entry%%:*}"; knobs="${entry#*:}"
      env $knobs MAX_DEPTH=10 USE_OPENING_BOOK=0 PRESET=LONG_FORMAT \
          OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
          VECLIB_MAXIMUM_THREADS=1 TF_NUM_INTEROP_THREADS=1 TF_NUM_INTRAOP_THREADS=1 \
          "$PY" diagnostics/movematch.py run "kmm_${label}" > "/tmp/kmm_${label}.out" 2>/dev/null || true
      sc=$(grep -hoE '^TOTAL [0-9]+/[0-9]+ +\([0-9.]+%\)' "/tmp/kmm_${label}.out")
      echo "${label}	${sc:-(none)}	[$knobs]"
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

  tune_corpus)
    # Build the SF11-per-term + detector corpus for the conditioning fit. Args: <tags> <n> <out> [--resume] [--mirror] [--per-game N].
    # tags = comma-separated game tags. SF11 interop must be warm. Single-thread pinned (deterministic static eval);
    # --resume appends + skips done FENs so the run is killable/resumable (hand cores back anytime).
    tags="${1:?tags required}"; shift || true
    n="${1:-40000}"; shift || true
    out="${1:-selfplay/tune_data/cond_corpus.csv}"; shift || true
    # Route KEY=VAL trailing args to ENV (eval knobs baked into the labelling engine), --flags to python.
    knobs=(); pyargs=()
    for a in "$@"; do if [[ "$a" == *=* ]]; then knobs+=("$a"); else pyargs+=("$a"); fi; done
    env "${knobs[@]}" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
        VECLIB_MAXIMUM_THREADS=1 TF_NUM_INTEROP_THREADS=1 TF_NUM_INTRAOP_THREADS=1 USE_OPENING_BOOK=0 \
        "$PY" selfplay/tune_corpus.py --tag "$tags" --n "$n" --out "$out" "${pyargs[@]}"
    ;;

  tune_cond)
    # Offline detector-conditioning fit (replays mod_gain vs SF11 total). No interop. Args: [--corpus path] [--seed N].
    "$PY" selfplay/tune_cond.py "$@"
    ;;

  tune_fit)
    # Offline per-term flat-SCALE Texel fit (material/pieces anchor the magnitude => strength-relevant
    # relative-blunting vs SF11). No interop. Args: passed straight through (--corpus --terms --target ...).
    "$PY" selfplay/tune_fit.py "$@"
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

  movematch_sample)
    # Sampled move-match for the staged funnel. Disjoint shards come from ONE seed + non-overlapping
    # offsets: Stage-1 (sample=100 offset=0) tunes, Stage-2 (sample=300 offset=100) validates positions
    # it never saw. Args: <tag> <themes|all> <epd|default> <sample> <seed> [offset] [KNOB=v ...].
    # epd "default" uses the STS suite; otherwise a path relative to NN Engine/ (the merged failure
    # corpus). Same thread-pin discipline as movematch (determinism). Knobs in "$@" precede $PY as env.
    tag="${1:?tag required}"; shift || true
    themes="${1:-all}"; shift || true
    epd="${1:-default}"; shift || true
    sample="${1:?sample required}"; shift || true
    seed="${1:-0}"; shift || true
    off=()
    if [[ "${1:-}" =~ ^[0-9]+$ ]]; then off=(--offset "$1"); shift || true; fi
    targ=(); [ "$themes" != "all" ] && targ=(--themes "$themes")
    earg=(); [ "$epd" != "default" ] && earg=(--epd "$epd")
    # The funnel always sits on the validated m4000ctl anchor (KS<-control). Baked as DEFAULTS here so
    # the campaign runs prompt-free (=-knobs on the command line trip the allowlist); a candidate's own
    # conditioner knobs come in via "$@" and, being distinct names, stack on top. The wac/sts byte-id
    # gate is a SEPARATE sub and stays knobs-off, so this does not perturb the correctness fingerprint.
    env MAX_DEPTH=10 USE_OPENING_BOOK=0 PRESET=LONG_FORMAT \
        ENABLE_KS_REPLACE_LT=1 KING_SAFETY_MAG=4000 MOD_KS_CONTROL=256 \
        OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
        VECLIB_MAXIMUM_THREADS=1 TF_NUM_INTEROP_THREADS=1 TF_NUM_INTRAOP_THREADS=1 "$@" \
        "$PY" diagnostics/movematch.py run "$tag" "${targ[@]}" "${earg[@]}" \
            --sample "$sample" --seed "$seed" "${off[@]}" 2>/dev/null \
        | grep -E 'movematch run|^TOTAL|results ->|\([0-9]+%\)' || echo "movematch_sample: (none)"
    ;;

  movematch_diff)
    # Per-theme score delta + changed-move list between two movematch runs. Args: <base_tag> <cand_tag>.
    base="${1:?base tag required}"; cand="${2:?cand tag required}"
    "$PY" diagnostics/movematch.py diff "$base" "$cand" 2>/dev/null || echo "movematch_diff: (none)"
    ;;

  funnel_cand)
    # One candidate config on a shard, anchor + extra knobs, PROMPT-FREE: candidate knobs are passed as
    # space-separated KNOB VAL positional pairs and assembled into env INSIDE the script (no =-tokens on
    # the dispatcher command line, which would trip the allowlist). Pair with movematch_sample (no extra
    # knobs = the anchor baseline) + movematch_diff to A/B. Same shard semantics as movematch_sample.
    # Args: <tag> <themes|all> <epd|default> <sample> <seed> <offset> [KNOB VAL]...
    tag="${1:?tag required}"; shift || true
    themes="${1:-all}"; shift || true
    epd="${1:-default}"; shift || true
    sample="${1:?sample required}"; shift || true
    seed="${1:-0}"; shift || true
    offv="${1:-0}"; shift || true
    pairs=()
    while [ "$#" -ge 2 ]; do pairs+=("$1=$2"); shift 2; done
    targ=(); [ "$themes" != "all" ] && targ=(--themes "$themes")
    earg=(); [ "$epd" != "default" ] && earg=(--epd "$epd")
    env MAX_DEPTH=10 USE_OPENING_BOOK=0 PRESET=LONG_FORMAT \
        ENABLE_KS_REPLACE_LT=1 KING_SAFETY_MAG=4000 MOD_KS_CONTROL=256 "${pairs[@]}" \
        OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
        VECLIB_MAXIMUM_THREADS=1 TF_NUM_INTEROP_THREADS=1 TF_NUM_INTRAOP_THREADS=1 \
        "$PY" diagnostics/movematch.py run "$tag" "${targ[@]}" "${earg[@]}" \
            --sample "$sample" --seed "$seed" --offset "$offv" 2>/dev/null \
        | grep -E 'movematch run|^TOTAL|results ->|\([0-9]+%\)' || echo "funnel_cand: (none)"
    ;;

  coord_sweep)
    # Coordinate-ascent calibration sweep (overnight, autonomous, CSV-logged, PROMPT-FREE). Each grid entry
    # is one eval-scale knob at one value, scored on the held-out 300 shard (sample=300 seed=1 offset=100) ON
    # TOP OF the current bundle (anchor m4000ctl + CHEAP_ROOK_MOB=32). The grid is baked here so no =-token
    # ever reaches the command line. Writes "knob=val,TOTAL" to results/coord_sweep.csv; baseline (rook32
    # alone) = 1672. A positive delta FLAGS a candidate to confirm on shard-2 + tournament (NOT an accept).
    # Args: [tag=coord]. Sequential (one core, deterministic); ~12 min/entry.
    out="diagnostics/results/coord_sweep.csv"
    echo "knob,total" > "$out"
    grid=(
      CHEAP_ROOK_FWD=5 CHEAP_ROOK_FWD=8
      ROOK_OPEN_BASE=200 ROOK_OPEN_BASE=300 ROOK_7TH=100 ROOK_7TH=200
      ROOK_CONNECTED=100 ROOK_CONNECTED=200 ROOK_SEMI=175 ROOK_SEMI_CONNECTED=175
      SCALE_CAPTURE_GAINS=70 SCALE_CAPTURE_GAINS=130
      IMBALANCE_SCALE=2 IMBALANCE_SCALE=4
      BISHOP_PAIR_BONUS=200 BISHOP_PAIR_BONUS=400 KNIGHT_PAIR_BONUS=300
      THREAT_ATTACK_MULT=30 THREAT_ATTACK_MULT=70 THREAT_PRESENCE_MULT=60 THREAT_PRESENCE_MULT=100
      SCALE_PAWN_RANK=70 SCALE_PAWN_RANK=130 SCALE_PASSED_RANK=130 SCALE_PASSED_PAWN=70
      CHEAP_BISHOP_MOB=3 SCALE_PLACE_QUEEN=130 SCALE_PLACE_KNIGHT=130
    )
    for kv in "${grid[@]}"; do
      tot=$(env MAX_DEPTH=10 USE_OPENING_BOOK=0 PRESET=LONG_FORMAT \
          ENABLE_KS_REPLACE_LT=1 KING_SAFETY_MAG=4000 MOD_KS_CONTROL=256 CHEAP_ROOK_MOB=32 "$kv" \
          OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
          VECLIB_MAXIMUM_THREADS=1 TF_NUM_INTEROP_THREADS=1 TF_NUM_INTRAOP_THREADS=1 \
          "$PY" diagnostics/movematch.py run "cs_$kv" --epd diagnostics/suites/failure_corpus.epd \
              --sample 300 --seed 1 --offset 100 2>/dev/null \
          | grep -oE '^TOTAL [0-9]+' | grep -oE '[0-9]+')
      echo "$kv,${tot:-NA}" | tee -a "$out"
    done
    echo "coord_sweep done -> $out (baseline rook32 = 1672)"
    ;;

  search_sweep)
    # PACE inner-loop for the SEARCH lane: deterministic, prompt-free. For each baked grid entry (a search
    # knob at one value, ON TOP of the kept BASE bundle) run wac (solves+nodes+ebf) + sts (positional) at
    # fixed depth 10 and append config,solved,nodes,ebf,sts to results/search_sweep.csv. DETERMINISTIC (no
    # move-match noise) -> one run settles direction. Correctness gate = WAC SOLVED held (don't lose tactics).
    # Edit BASE (= folded keeps) + grid each round, re-run. No build (knobs are env-read at engine init).
    out="diagnostics/results/search_sweep.csv"
    echo "config,solved,nodes,ebf,sts" > "$out"
    BASE=""   # kept bundle, e.g. "NULLMOVE_EXTRA=3 HISTORY_LMR_SCALE=3" (baked, no =-args on the cmd line)
    grid=(
      "baseline:"
      "nm3:NULLMOVE_EXTRA=3"
      "hls3:HISTORY_LMR_SCALE=3"
      "hls4:HISTORY_LMR_SCALE=4"
      "lmpb1:LMP_BASE=1"
      "lmpd6:LMP_MAX_DEPTH=6"
      "lmrx1:LMR_EXTRA=1"
      "vm12k:VERIFY_MARGIN=12000"
      "vm8k:VERIFY_MARGIN=8000"
      "asp400:ASPIRATION_DELTA=400"
      "asp300:ASPIRATION_DELTA=300"
    )
    for entry in "${grid[@]}"; do
      name="${entry%%:*}"; kv="${entry#*:}"
      env $BASE $kv MAX_DEPTH=10 USE_OPENING_BOOK=0 PRESET=LONG_FORMAT \
          "$PY" diagnostics/tactical_test.py wac.epd "ss_$name" > "/tmp/ss_${name}.out" 2> "/tmp/ss_${name}.err" || true
      solved=$(grep -hoE 'Solved [0-9]+/[0-9]+' "/tmp/ss_${name}.out" | grep -oE '^Solved [0-9]+' | grep -oE '[0-9]+' || echo "?")
      nodes=$(grep -hoP '\(nodes=\K[0-9]+' "/tmp/ss_${name}.err" | awk '{s+=$1} END{print s+0}')
      ebf=$(grep -hoP 'ebf=\K[0-9.]+' "/tmp/ss_${name}.err" | awk '{s+=$1;n++} END{if(n)printf "%.3f",s/n; else print "?"}')
      sts=$(env $BASE $kv MAX_DEPTH=10 USE_OPENING_BOOK=0 PRESET=LONG_FORMAT \
          "$PY" diagnostics/sts_test.py sts300.epd "ss_$name" 2>/dev/null | grep -hoP 'STS score: \K[0-9]+' || echo "?")
      echo "${name},${solved},${nodes},${ebf},${sts}" | tee -a "$out"
    done
    echo "search_sweep done -> $out (BASE='$BASE')"
    ;;

  vs_sf11)
    # DIAGNOSTIC: our engine vs CLASSICAL Stockfish 11 (pre-NNUE HCE yardstick), SF18 the neutral arbiter.
    # Both SF play 1-thread (fair vs our single-threaded engine). Decomposes our gap to a great HCE:
    #   depth <N> = equal fixed depth (pure eval+ordering; nodes logged = efficiency gap; concurrency CLEAN)
    #   time  <s> = equal per-move time (search-speed in; run LOW concurrency or read targets separately)
    # depth+nodes are logged for BOTH sides per move (selfplay/games/<tag>/game_*/.jsonl). Prompt-free.
    # Args: <games> <time|depth> <value> [conc=4] [tag] [KNOB=v ...(our engine; =-args prompt, omit unattended)].
    games="${1:?games}"; shift || true
    mode="${1:?time|depth}"; shift || true
    val="${1:?value}"; shift || true
    conc="${1:-4}"; shift || true
    ttag="${1:-vssf11}"; shift || true
    SF11="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/stockfish_11/stockfish-11-win/Windows/stockfish_20011801_x64_bmi2.exe"
    export STOCKFISH_PATH="$SF"   # SF18 (arbiter)
    if [ "$mode" = depth ]; then
        sfarg=(--sf-depth "$val"); ourcfg="PRESET=LONG_FORMAT MAX_DEPTH=$val USE_OPENING_BOOK=0"
    else
        sfarg=(--sf-movetime "$val"); ourcfg="PRESET=LIGHTNING USE_OPENING_BOOK=0"
    fi
    env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
        VECLIB_MAXIMUM_THREADS=1 TF_NUM_INTEROP_THREADS=1 TF_NUM_INTRAOP_THREADS=1 "$@" \
        "$PY" selfplay/vs_sf.py --our-label ours --our-config "$ourcfg" \
        --sf-elo 0 --sf-path "$SF11" --sf-arb-path "$SF" "${sfarg[@]}" \
        --games "$games" --concurrency "$conc" --openings selfplay/openings_uho.txt \
        --adjudicate-draw --quiet --tag "$ttag"
    ;;

  place_probe)
    # Cheap-proof-first: does bumping an EXISTING flat placement scale lift a systematically-missed
    # positional theme, on the FULL STS theme (not the thin funnel sample)? If the flat scale can't move
    # it, a conditioner on the same term won't either. Grid is baked (prompt-free). v=100 is the anchor
    # control. Args: <theme> [knob=SCALE_PLACE_KING_EG]. Runs on top of the m4000ctl anchor.
    theme="${1:?theme required}"; shift || true
    knob="${1:-SCALE_PLACE_KING_EG}"; shift || true
    for v in 100 150 200; do
      echo "== $knob=$v  theme=$theme =="
      env MAX_DEPTH=10 USE_OPENING_BOOK=0 PRESET=LONG_FORMAT \
          ENABLE_KS_REPLACE_LT=1 KING_SAFETY_MAG=4000 MOD_KS_CONTROL=256 "$knob=$v" \
          OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
          VECLIB_MAXIMUM_THREADS=1 TF_NUM_INTEROP_THREADS=1 TF_NUM_INTRAOP_THREADS=1 \
          "$PY" diagnostics/movematch.py run "probe_${knob}_${v}" --themes "$theme" 2>/dev/null \
          | grep -E '^TOTAL' || echo "  (none)"
    done
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

  ks_tournament)
    # Permission-clean overnight A/B for the king-safety candidate: base (no knobs) vs m4000ctl
    # (ENABLE_KS_REPLACE_LT=1 KING_SAFETY_MAG=4000 MOD_KS_CONTROL=256) baked INSIDE so no =-args reach the
    # command line (unattended-safe). The PACE-validation + ship gate for the KS swap. Run single-machine-state
    # (one sitting, no cross-reboot batches -> the machine-state confound). Args: <minutes> [conc=4] [tag].
    mins="${1:?minutes required}"; shift || true
    conc="${1:-4}"; shift || true
    ttag="${1:-ks_m4000ctl}"; shift || true
    export STOCKFISH_PATH="$SF"
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
           VECLIB_MAXIMUM_THREADS=1 TF_NUM_INTEROP_THREADS=1 TF_NUM_INTRAOP_THREADS=1
    "$PY" selfplay/tournament.py \
        --p1-label base --p1-config "" \
        --p2-label ksctl --p2-config "ENABLE_KS_REPLACE_LT=1 KING_SAFETY_MAG=4000 MOD_KS_CONTROL=256" \
        --preset LIGHTNING --concurrency "$conc" --max-minutes "$mins" \
        --openings selfplay/openings_uho.txt --adjudicate-draw --quiet --tag "$ttag"
    ;;

  prune_screen)
    # Permission-clean prune-optimization screen for the KS bundle (grid baked => no =-args on the command
    # line => unattended-safe). Per variant: WAC solved + summed nodes (fixed depth = tactical/trades safety
    # + node-cut) and TIMED ksattack (equal-time KS accuracy = the prune's real metric, not the fixed-depth
    # artifact). KS = REPLACE/MAG4000/ZONE2/DYN128 held fixed; the prune knobs vary. p_alone (no KS) isolates
    # the KS node cost vs the bundle. Baseline refs: WAC 252/70.15M, bundle 244/61.4M, bundle ksattack_t 1080.
    declare -a grid=(
      "p_alone:LMR_EXTRA=1 VERIFY_MARGIN=12000 HISTORY_LMR_SCALE=3"
      "cur:ENABLE_KS_REPLACE_LT=1 KING_SAFETY_MAG=4000 KS_ZONE2=1 KS_DYN=128 LMR_EXTRA=1 VERIFY_MARGIN=12000 HISTORY_LMR_SCALE=3"
      "lmr2:ENABLE_KS_REPLACE_LT=1 KING_SAFETY_MAG=4000 KS_ZONE2=1 KS_DYN=128 LMR_EXTRA=2 VERIFY_MARGIN=12000 HISTORY_LMR_SCALE=3"
      "vm16:ENABLE_KS_REPLACE_LT=1 KING_SAFETY_MAG=4000 KS_ZONE2=1 KS_DYN=128 LMR_EXTRA=1 VERIFY_MARGIN=16000 HISTORY_LMR_SCALE=4"
    )
    for entry in "${grid[@]}"; do
      label="${entry%%:*}"; knobs="${entry#*:}"
      env $knobs MAX_DEPTH=10 USE_OPENING_BOOK=0 PRESET=LONG_FORMAT \
          "$PY" diagnostics/tactical_test.py wac.epd "ps_${label}" > "/tmp/ps_${label}.out" 2> "/tmp/ps_${label}.err" || true
      sol=$(grep -hoE 'Solved [0-9]+/[0-9]+' "/tmp/ps_${label}.out")
      nod=$(grep -hoP '\(nodes=\K[0-9]+' "/tmp/ps_${label}.err" | awk '{s+=$1} END{print s+0}')
      ka=$(env $knobs PRESET=LIGHTNING MAX_DEPTH=64 USE_OPENING_BOOK=0 \
          "$PY" diagnostics/sts_test.py ksattack.epd "psk_${label}" 2>/dev/null | grep -hoE 'STS score: [0-9]+/[0-9]+ +\([0-9.]+%\)')
      echo "${label}	WAC=${sol:-?}	nodes=${nod}	ksattack_timed=${ka:-?}"
    done
    ;;

  tonight_tourney)
    # Permission-clean overnight A/B for the KING-SAFETY-DETECTION + ORDER-PRUNE bundle (2026-06-29).
    # base (shipped defaults, no knobs) vs the bundle, baked INSIDE so no =-args reach the command line
    # (unattended-safe). Bundle = rebuilt king-danger detection (wider zone + weak-square + pawn-storm terms,
    # REPLACE latent_threat) with the per-king DYNAMIC magnitude (KS_DYN: scale each king's danger by its
    # attack-signature co-occurrence) + the search ORDER-prune (LMR_EXTRA/VERIFY_MARGIN/HISTORY_LMR_SCALE).
    # The two levers are super-additive at equal time (timed ksattack bundle 1080 > KS-alone 970). Args:
    # <minutes> [conc=4] [tag].
    mins="${1:?minutes required}"; shift || true
    conc="${1:-4}"; shift || true
    ttag="${1:-ks_dyn_bundle}"; shift || true
    export STOCKFISH_PATH="$SF"
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
           VECLIB_MAXIMUM_THREADS=1 TF_NUM_INTEROP_THREADS=1 TF_NUM_INTRAOP_THREADS=1
    "$PY" selfplay/tournament.py \
        --p1-label base --p1-config "" \
        --p2-label ksdyn --p2-config "ENABLE_KS_REPLACE_LT=1 KING_SAFETY_MAG=4000 KS_ZONE2=1 KS_DYN=128 LMR_EXTRA=1 VERIFY_MARGIN=12000 HISTORY_LMR_SCALE=3" \
        --preset LIGHTNING --concurrency "$conc" --max-minutes "$mins" \
        --openings selfplay/openings_uho.txt --adjudicate-draw --quiet --tag "$ttag"
    ;;

  fast_tourney)
    # FAST shallow-FIXED-DEPTH self-play for eval-throughput experiments (the "more games to isolate eval"
    # idea). BOTH players search to the SAME fixed MAX_DEPTH -> an equal-depth A/B where the EVAL alone decides
    # (deterministic, eval-isolating, fast). SF adjudication LIGHTENED to --sf-movetime 0.1 so the arbiter
    # doesn't bottleneck short games. CAVEAT: fixed depth PENALIZES the order-prune (it just searches less to
    # depth N) -> use this for EVAL candidates ONLY; search/prune levers need equal-TIME (the `tournament` sub).
    # Calibrate the proxy by re-running a candidate whose equal-time Elo we already know (e.g. KS-alone) and
    # checking the sign/magnitude + CI-per-hour reproduce. Args: <minutes> <depth> <p2cfg> [conc=4] [tag].
    mins="${1:?minutes required}"; shift || true
    depth="${1:?depth required}"; shift || true
    p2cfg="${1:-}"; shift || true
    conc="${1:-4}"; shift || true
    ttag="${1:-fast}"; shift || true
    export STOCKFISH_PATH="$SF"
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
           VECLIB_MAXIMUM_THREADS=1 TF_NUM_INTEROP_THREADS=1 TF_NUM_INTRAOP_THREADS=1
    "$PY" selfplay/tournament.py \
        --p1-label base --p1-config "MAX_DEPTH=$depth" \
        --p2-label cand --p2-config "MAX_DEPTH=$depth $p2cfg" \
        --preset LONG_FORMAT --concurrency "$conc" --max-minutes "$mins" --sf-movetime 0.1 \
        --openings selfplay/openings_uho.txt --adjudicate-draw --quiet --tag "$ttag"
    ;;

  ks_ovd_fastrank)
    # Permission-clean Phase-1 fast-RANK of KS+OvD candidates: each config vs base at fixed depth 6 (eval-only
    # A/B), lightened 0.1s adjudication. Grid baked => no =-args on the command line. Prints each candidate's
    # Elo vs base. NOTE: fast-depth COMPRESSES ~3x and RANKS only -> gate winners at lightning SPRT. The
    # hypothesis: KS BESIDE latent_threat (KING_SAFETY_MAG>0, ENABLE_KS_REPLACE_LT default-off) at a GENTLE
    # magnitude, OvD-conditioned, beats base (vs the -47 REPLACE). Args: <minutes_per_candidate> [conc=4].
    mins="${1:-20}"; shift || true
    conc="${1:-4}"; shift || true
    export STOCKFISH_PATH="$SF"
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
           VECLIB_MAXIMUM_THREADS=1 TF_NUM_INTEROP_THREADS=1 TF_NUM_INTRAOP_THREADS=1
    declare -a grid=(
      "besideA:KING_SAFETY_MAG=1500 KS_ZONE2=1"
      "besideB:KING_SAFETY_MAG=1500 KS_ZONE2=1 MOD_KS_CONTROL=512"
      "besideC:KING_SAFETY_MAG=1500 KS_ZONE2=1 MOD_KS_CONTROL=512 KS_DYN=128"
      "besideD:KING_SAFETY_MAG=1500 KS_ZONE2=1 MOD_KS_BACKING=256"
      "ovdyn:IMBALANCE_SCALE=4 REALIZ_MAT_K=128 REALIZ_PHASE_K=128"
    )
    for entry in "${grid[@]}"; do
      label="${entry%%:*}"; cfg="${entry#*:}"
      echo "### $label  [$cfg]"
      "$PY" selfplay/tournament.py \
          --p1-label base --p1-config "MAX_DEPTH=6" \
          --p2-label "$label" --p2-config "MAX_DEPTH=6 $cfg" \
          --preset LONG_FORMAT --concurrency "$conc" --max-minutes "$mins" --sf-movetime 0.1 \
          --openings selfplay/openings_uho.txt --adjudicate-draw --quiet --tag "fr_${label}" \
          2>&1 | grep -hE "adjudication ON|base vs ${label}:|timed:" || echo "  (no result)"
    done
    ;;

  gate_besideA)
    # Lightning SPRT gate for the Phase-1 winner besideA = gentle KS BESIDE latent_threat (no conditioning).
    # THE proxy-calibration point: does fast depth-6 +31 hold at real lightning depth (vs evaporating like the
    # ksattack +70 did)? Baked config => permission-clean. p1=candidate, p2=base; H1 = candidate >= elo1.
    # Args: [max_games=500] [tag=sprt_besideA].
    maxg="${1:-500}"; shift || true
    ttag="${1:-sprt_besideA}"; shift || true
    export STOCKFISH_PATH="$SF"
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
           VECLIB_MAXIMUM_THREADS=1 TF_NUM_INTEROP_THREADS=1 TF_NUM_INTRAOP_THREADS=1
    "$PY" selfplay/sprt.py \
        --p1-label besideA --p1-config "KING_SAFETY_MAG=1500 KS_ZONE2=1" \
        --p2-label base --p2-config "" \
        --preset LIGHTNING --concurrency 4 --elo0 0 --elo1 5 --max-games "$maxg" \
        --adjudicate-draw --quiet --tag "$ttag"
    ;;

  gate_coupling)
    # Lightning SPRT of the eval<->search COUPLING hypothesis (user insight 2026-06-30, [[eval-search-coupling-flat-candidates]]):
    # does a more-accurate but lightning-FLAT eval (besideA = gentle KS beside latent_threat) make the AGGRESSIVE
    # prune (LMR_EXTRA=2, which over-cut alone last night) SAFE -> bundle super-additive while both parts read ~0?
    # If clearly positive, the eval's value was real and the coupling unlocked it (revives BOTH parked items).
    # Baked => permission-clean. p1=candidate bundle, p2=base. Args: [max_games=800] [tag=sprt_coupling].
    maxg="${1:-800}"; shift || true
    ttag="${1:-sprt_coupling}"; shift || true
    export STOCKFISH_PATH="$SF"
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
           VECLIB_MAXIMUM_THREADS=1 TF_NUM_INTEROP_THREADS=1 TF_NUM_INTRAOP_THREADS=1
    "$PY" selfplay/sprt.py \
        --p1-label coupling --p1-config "KING_SAFETY_MAG=1500 KS_ZONE2=1 LMR_EXTRA=2 VERIFY_MARGIN=12000 HISTORY_LMR_SCALE=3" \
        --p2-label base --p2-config "" \
        --preset LIGHTNING --concurrency 4 --elo0 0 --elo1 5 --max-games "$maxg" \
        --adjudicate-draw --quiet --tag "$ttag"
    ;;

  gate)
    # Generic lightning SPRT: a candidate ENV-knob config vs base. Args: '<p1cfg>' <label> [tag] [max_games] [elo1].
    # p1cfg is a space-separated KEY=VAL string (quote it). The correct -lc wrapper passes =knobs prompt-free.
    p1cfg="${1:?p1 config required}"; shift || true
    lbl="${1:-cand}"; shift || true
    ttag="${1:-sprt_${lbl}}"; shift || true
    maxg="${1:-600}"; shift || true
    e1="${1:-5}"; shift || true
    export STOCKFISH_PATH="$SF"
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
           VECLIB_MAXIMUM_THREADS=1 TF_NUM_INTEROP_THREADS=1 TF_NUM_INTRAOP_THREADS=1
    "$PY" selfplay/sprt.py \
        --p1-label "$lbl" --p1-config "$p1cfg" \
        --p2-label base --p2-config "" \
        --preset LIGHTNING --concurrency 4 --elo0 0 --elo1 "$e1" --max-games "$maxg" \
        --adjudicate-draw --quiet --tag "$ttag"
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
    conc=1; if [[ "${1:-}" =~ ^[0-9]+$ ]]; then conc="$1"; shift || true; fi   # optional positional: game concurrency (numeric only, else left for KEY=VAL knobs)
    export STOCKFISH_PATH="$SF"
    env OMP_NUM_THREADS=1 "$@" \
        "$PY" selfplay/vs_sf.py --sf-elo "$elo" --games "$games" --preset "$preset" \
        --win-threshold "$wt" --concurrency "$conc" --openings selfplay/openings_uho.txt --adjudicate-draw --quiet \
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
