# -*- coding: utf-8 -*-
"""Footprint-filtered D7 regret: the SENSITIVITY instrument for a subsystem change the general set is too
blunt to resolve. On the game-representative multi-PV set, run our fixed-depth move for a BASE arm and one
or more CANDIDATE arms; then compare SF18 win%-regret ONLY on the positions where the candidate CHANGES our
move vs base. A KS change is move-neutral on most positions (memory most-eval-error-is-move-neutral), so its
aggregate held-regret delta drowns in noise; but on the positions it actually moves, the mean regret of the
new move vs the old is directly resolvable and is the honest "when it changes our move, is it better?" test.

Not overfit: it is just a filter over real game positions, adjudicated by SF18 ground-truth labels.

  pyrun diagnostics/_ks_footprint_regret.py [SET=ks_sets/game_regret_set.csv] [DEPTH=7] [JOBS=4]
        [SPLIT=all] [SPLIT_FRAC=0.7] [MAXN=0]

⚠️ FIXED depth (deterministic), a PROXY for game depth. A positive footprint delta that replicates on the
v2 cross-set is a real (usually small) structural gain; games still decide the Elo.
"""
import os, sys, csv, math, subprocess

for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v


def _winpct(cp):
    return 50.0 + 50.0 * (2.0 / (1.0 + math.exp(-0.00368208 * cp)) - 1.0)


# ---------------- WORKER: one config, one slice -> per-position (fen, our_move, regret) ----------------
if os.environ.get("WORKER") == "1":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['PRESET'] = 'LONG_FORMAT'
    os.environ['MAX_DEPTH'] = os.environ.get('DEPTH', '7')
    os.environ['USE_OPENING_BOOK'] = '0'
    THIS = os.path.dirname(os.path.abspath(__file__)); ENGINE = os.path.dirname(THIS)
    sys.path.insert(0, ENGINE); sys.path.insert(0, THIS); sys.path.insert(0, os.path.join(ENGINE, "selfplay"))
    import chess
    from tactical_test import run_one
    SET = os.environ["SET"]; MISS = float(os.environ.get("MISS", "30"))
    si, sn = (int(x) for x in os.environ["SLICE"].split("/"))
    frac = float(os.environ.get("SPLIT_FRAC", "0.7"))
    which = os.environ.get("SPLIT", "all")
    MAXN = int(os.environ.get("MAXN", "0"))
    rows = list(csv.DictReader(open(SET, newline="")))
    import random as _r
    _r.Random(1234).shuffle(rows)
    if MAXN:
        rows = rows[:MAXN]
    cut = int(frac * len(rows))
    rows = rows[:cut] if which == "tune" else (rows[cut:] if which == "held" else rows)
    rows = rows[si::sn]
    out = open(os.environ["OUT"], "w", newline="")
    w = csv.writer(out)
    for r in rows:
        fen = r.get("fen")
        try:
            best_cp = float(r["best_cp"])
            mm = {}
            for pair in (r.get("moves") or "").split(";"):
                if ":" in pair:
                    u, c = pair.rsplit(":", 1); mm[u] = float(c)
        except Exception:
            continue
        if not mm:
            continue
        try:
            our = run_one(fen, set())["uci"]
        except Exception:
            continue
        stm_white = (fen.split()[1] == 'w')
        our_cp = mm.get(our)
        if our_cp is None:
            worst = min(mm.values()) if stm_white else max(mm.values())
            our_cp = (worst - MISS) if stm_white else (worst + MISS)
        bw = _winpct(best_cp) if stm_white else (100.0 - _winpct(best_cp))
        ow = _winpct(our_cp) if stm_white else (100.0 - _winpct(our_cp))
        reg = max(0.0, bw - ow)
        w.writerow([fen, our, "%.6f" % reg])
    out.close()
    sys.exit(0)

# ---------------- DRIVER ----------------
THIS = os.path.dirname(os.path.abspath(__file__)); ENGINE = os.path.dirname(THIS)
PY = sys.executable
SET = os.environ.get("SET", "ks_sets/game_regret_set.csv")
if not os.path.isabs(SET):
    SET = os.path.join(THIS, SET)
DEPTH = os.environ.get("DEPTH", "7")
JOBS = int(os.environ.get("JOBS", "4"))
SPLIT = os.environ.get("SPLIT", "all")
SPLIT_FRAC = os.environ.get("SPLIT_FRAC", "0.7")
MAXN = int(os.environ.get("MAXN", "0"))

BUNDLE = {"OVD_BOUNDED_MODE": 2, "OVD_CAP": 300, "OVD_KNEE": 40,
          "CENTRAL_BOUNDED_MODE": 1, "CENTRAL_CAP": 150, "CENTRAL_KNEE": 200}


def cfg(**kw):
    d = dict(BUNDLE); d.update(kw); return d


BASE = ("bundle+defaware1 (the floor)", cfg(KS_DEFAWARE_MODE=1))
CANDS = [
    # KS redesign step 1: value-aware square_control contest (least-valuable-attacker + pawn-exclusion +
    # attackedBy2) feeding defaware's contested_zone, vs the raw popcount contest. The honest-inputs upgrade.
    ("+SQC (value-aware contest)", cfg(KS_DEFAWARE_MODE=1, KS_SQC_MODE=1)),
]


def collect(config, tag):
    """Run JOBS slice-workers; return {fen: (our_move, regret)}."""
    env = dict(os.environ, SET=SET, DEPTH=DEPTH, SPLIT=SPLIT, SPLIT_FRAC=SPLIT_FRAC, MAXN=str(MAXN))
    knob_args = ["%s=%s" % (k, v) for k, v in config.items()]
    procs = []
    for i in range(JOBS):
        outp = "/tmp/_fp_%s_%d.csv" % (tag, i)
        e = dict(env, WORKER="1", SLICE="%d/%d" % (i, JOBS), OUT=outp)
        procs.append((subprocess.Popen([PY, "-u", os.path.abspath(__file__)] + knob_args,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                       text=True, cwd=ENGINE, env=e), outp))
    d = {}
    for p, outp in procs:
        p.communicate()
        try:
            for row in csv.reader(open(outp, newline="")):
                if len(row) == 3:
                    d[row[0]] = (row[1], float(row[2]))
        except Exception:
            pass
    return d


base = collect(BASE[1], "base")
print("SET=%s DEPTH=%s SPLIT=%s  base=%s  positions=%d\n" % (os.path.basename(SET), DEPTH, SPLIT, BASE[0], len(base)), flush=True)
print("  %-38s %7s %8s   %10s %10s %9s" % ("candidate", "changed", "%chg", "reg_base", "reg_cand", "delta"), flush=True)
for name, c in CANDS:
    cand = collect(c, "cand")
    shared = [f for f in base if f in cand]
    changed = [f for f in shared if base[f][0] != cand[f][0]]
    if not changed:
        print("  %-38s %7d %8s   %10s %10s %9s" % (name, 0, "-", "-", "-", "-")); continue
    rb = sum(base[f][1] for f in changed) / len(changed)
    rc = sum(cand[f][1] for f in changed) / len(changed)
    print("  %-38s %7d %7.1f%%   %10.4f %10.4f %+9.4f"
          % (name, len(changed), 100.0 * len(changed) / len(shared), rb, rc, rc - rb), flush=True)
print("\n  read: on the positions the candidate CHANGES our move, reg_cand < reg_base (negative delta) means\n"
      "  the new move is genuinely better per SF18 -> a real structural gain where the unit acts. Positive =\n"
      "  the change makes our move worse. Confirm the sign replicates on the v2 cross-set before folding in.", flush=True)
