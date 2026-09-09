# -*- coding: utf-8 -*-
"""Does a config's SEARCH pick the same move when an UNRELATED knob is perturbed?

`_move_change_arms.py` answers a different question: it is a ONE-PLY STATIC proxy comparing the argmax of
statically-scored children, which is the right instrument for an EVAL knob and blind to search behaviour.
This one runs the actual search.

A sound search should pick the same move regardless of the ASPIRATION window width -- the window is a
search-economy device, not a decision input. If widening it flips the chosen move, the position's result
is not being determined by the evidence but by which attempt happened to resolve. Measured as the
move-change rate between two ASPIRATION_DELTA values, per config, over the same positions.

🚨 The reading is COMPARATIVE. Baseline's flip rate is the CONTROL: some positions are genuinely
near-tied and any engine will wobble on them. What matters is whether a candidate flips MORE than
baseline on the same positions. A candidate at baseline's rate is stable; one at several times baseline's
rate is deciding by accident.

⚠️ Knobs latch at engine init, so every arm is a separate process (one per config x delta).
⚠️ Judge on the FLIP RATE, not on which move is "right" -- this measures stability, not correctness.

  pyrun diagnostics/_search_stability.py [N=30] [DELTAS=500,2000] [PRESET=STANDARD] [MAXD=64]
                                         [ARM='KEY=VAL KEY=VAL']   (repeatable; baseline always included)
"""
import os, sys, csv, json, subprocess

OPTS = {}
ARMS = []
for a in sys.argv[1:]:
    if a.startswith("ARM="):
        ARMS.append(a[4:])
    elif "=" in a:
        k, v = a.split("=", 1)
        OPTS[k] = v

N = int(OPTS.get("N", 30))
DELTAS = [d.strip() for d in OPTS.get("DELTAS", "500,2000").split(",")]
PRESET = OPTS.get("PRESET", "STANDARD")
MAXD = OPTS.get("MAXD", "64")

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(THIS_DIR)
PY = sys.executable

corpus = os.path.join(ENGINE, "selfplay", "tune_data", "cploss_corpus.csv")
fens = []
with open(corpus, newline="") as fh:
    for row in csv.DictReader(fh):
        f = row.get("fen") or row.get("FEN")
        if f:
            fens.append(f.strip())
        if len(fens) >= N:
            break
if not fens:
    print("!! no FENs read from", corpus)
    sys.exit(1)

# One process per arm: the child searches every FEN and prints the chosen move per line.
CHILD = (
    "import sys,os,json\n"
    "sys.path.insert(0,'diagnostics')\n"
    "from tactical_test import run_one\n"
    "out=[]\n"
    "for f in sys.argv[1:]:\n"
    "    try: out.append(run_one(f,set()).get('uci'))\n"
    "    except Exception: out.append(None)\n"
    "print('MOVES='+json.dumps(out))\n"
)


def arm_moves(knobs, delta):
    env = dict(os.environ)
    env.update({"PRESET": PRESET, "MAX_DEPTH": MAXD, "USE_OPENING_BOOK": "0",
                "OMP_NUM_THREADS": "1", "TF_CPP_MIN_LOG_LEVEL": "3",
                "ASPIRATION_DELTA": str(delta)})
    for kv in knobs.split():
        if "=" in kv:
            k, v = kv.split("=", 1)
            env[k] = v
    r = subprocess.run([PY, "-c", CHILD] + fens, cwd=ENGINE, env=env,
                       capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if line.startswith("MOVES="):
            return json.loads(line[6:])
    # An arm that produced no MOVES line FAILED. Returning a list of None makes every position
    # unscoreable and prints "0/0 (0.0%)", which reads like a perfectly stable arm -- the worst
    # possible failure mode for a comparative instrument. Surface it instead.
    tail = (r.stderr or "").strip().splitlines()
    print(f"  !! ARM FAILED (rc={r.returncode}) knobs={knobs or '(defaults)'} delta={delta}")
    for ln in tail[-3:]:
        print(f"     {ln}")
    return None


# ---------------------------------------------------------------------------------------------------
# VS=1 -- CROSS-ARM mode: diff baseline against each ARM at ONE delta, and DUMP the positions that flipped.
#
# The default mode above diffs a config against ITSELF across aspiration widths (a stability question).
# This mode asks a different one: WHERE DOES A DIFFERENT EVAL CHOOSE A DIFFERENT MOVE? Point it at
# ENABLE_ORACLE_EVAL=1 and the dump is an empirical inventory of the positions where our eval's deficiency
# actually costs us a decision -- as opposed to merely differing numerically.
#
# ⚠️ That distinction is the whole point. Ranking positions by raw eval difference has failed here before:
# 52% of sampled positions had a bad eval but a FINE move ([[most-eval-error-is-move-neutral]]), and the
# static-eval "top culprits" turned out to be search-absorbed. Filtering on a CHANGED MOVE removes that
# entire class.
# ⚠️ The baseline flip rate under a pure aspiration-width change is ~20.8% on quiet positions, so a raw
# flip count means little on its own -- score the dumped positions (win% loss vs an SF judge) before
# drawing conclusions, and rank by WIN%, never centipawns ([[rank-by-winpct-not-cp-it-inverts-conclusions]]).
if OPTS.get("VS") == "1":
    delta = DELTAS[0]
    dump = OPTS.get("DUMP", "")
    base = arm_moves("", delta)
    if base is None:
        sys.exit("baseline arm failed")
    rows = []
    print(f"positions={len(fens)}  delta={delta}  preset={PRESET}  MAXD={MAXD}  (cross-arm mode)")
    for i, a in enumerate(ARMS):
        cand = arm_moves(a, delta)
        if cand is None:
            print(f"  arm{i+1:<5} SKIPPED (failed)   {a}")
            continue
        flips = scored = 0
        for j, f in enumerate(fens):
            if base[j] is None or cand[j] is None:
                continue
            scored += 1
            if base[j] != cand[j]:
                flips += 1
                rows.append({"arm": f"arm{i+1}", "fen": f, "base_move": base[j], "cand_move": cand[j]})
        pct = (100.0 * flips / scored) if scored else 0.0
        print(f"  arm{i+1:<5} flips={flips:>3}/{scored:<3} ({pct:5.1f}%)   {a}")
    if dump and rows:
        with open(dump, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["arm", "fen", "base_move", "cand_move"])
            w.writeheader()
            w.writerows(rows)
        print(f"dumped {len(rows)} flipped positions -> {dump}")
    sys.exit(0)

configs = [("baseline", "")] + [(f"arm{i+1}", a) for i, a in enumerate(ARMS)]
print(f"positions={len(fens)}  deltas={DELTAS}  preset={PRESET}  (control = baseline flip rate)")
for name, knobs in configs:
    per_delta = [arm_moves(knobs, d) for d in DELTAS]
    if any(pd is None for pd in per_delta):
        print(f"  {name:<9} SKIPPED (an arm failed)          {knobs or '(defaults)'}")
        continue
    flips = 0
    scored = 0
    for i in range(len(fens)):
        vals = [pd[i] for pd in per_delta]
        if any(v is None for v in vals):
            continue
        scored += 1
        if len(set(vals)) > 1:
            flips += 1
    pct = (100.0 * flips / scored) if scored else 0.0
    print(f"  {name:<9} flips={flips:>3}/{scored:<3} ({pct:5.1f}%)   {knobs or '(defaults)'}")
print("READ: a candidate flipping at several times the baseline rate is deciding by accident, not evidence.")
