# -*- coding: utf-8 -*-
"""REGRET-based low-depth tuner: coordinate-descend the bounded knobs to MINIMISE mean SF-regret of our
fixed-depth chosen move -- a GRADED, move-based, Elo-aligned objective (unlike the binary top-1 match, which
has no gradient, and unlike static corpus fit, which is anti-correlated with Elo).

  regret(pos) = SF_eval(SF_best) - SF_eval(our_move)   (White-POV cp; clamped to [0, CAP]; our move not in
                SF's cached top-K -> treated as MISS cp below the worst listed). A shrink that doesn't change
                our move can't change regret => not gameable the way scalar fit is.

Needs the multi-PV cache from `_build_regret_set.py`. Each candidate re-runs ONLY our fixed-depth search.

  pyrun diagnostics/_regret_tune.py [SET=ks_sets/regret_set.csv] [DEPTH=7] [PASSES=2] [CAP=400] [MISS=30]

⚠️ FIXED depth (deterministic). Validate the winner on move-match + STS, then SPRT. Low depth is a PROXY.
"""
import os, sys, csv, subprocess

for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v

# ---------------- WORKER: one knob-config -> mean regret over the multi-PV cache at fixed depth ----------------
if os.environ.get("WORKER") == "1":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['PRESET'] = 'LONG_FORMAT'
    os.environ['MAX_DEPTH'] = os.environ.get('DEPTH', '7')
    os.environ['USE_OPENING_BOOK'] = '0'
    THIS = os.path.dirname(os.path.abspath(__file__)); ENGINE = os.path.dirname(THIS)
    sys.path.insert(0, ENGINE); sys.path.insert(0, THIS); sys.path.insert(0, os.path.join(ENGINE, "selfplay"))
    import math
    import chess
    from tactical_test import run_one
    SET = os.environ["SET"]; NLIM = int(os.environ.get("N", "0"))
    CAP = float(os.environ.get("CAP", "400")); MISS = float(os.environ.get("MISS", "30"))
    # REGRET_MODE=winpct (default, outcome-aligned: win% LOST by our move -- steep near 0, flat when already
    # decided, so a slip in a balanced position outweighs one in a won position) | cp (raw centipawn gap).
    MODE = os.environ.get("REGRET_MODE", "winpct").lower()

    def _winpct(cp):
        return 50.0 + 50.0 * (2.0 / (1.0 + math.exp(-0.00368208 * cp)) - 1.0)
    rows = list(csv.DictReader(open(SET, newline="")))
    if NLIM:
        rows = rows[:NLIM]
    tot_reg = 0.0; n = 0; match = 0; miss = 0
    for r in rows:
        fen = r.get("fen"); best_uci = r.get("best_uci")
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
            res = run_one(fen, set())
        except Exception:
            continue
        our = res["uci"]
        # POV: cached cp is WHITE-pov; regret must be from the SIDE-TO-MOVE's view or Black-to-move rows
        # invert (best = lowest white-cp) and the naive winpct(best)-winpct(our) goes negative -> clamps to
        # 0, silently dropping every black-to-move position. Convert to side-to-move win% first.
        stm_white = (fen.split()[1] == 'w')
        our_cp = mm.get(our)
        if our_cp is None:
            vals = list(mm.values())
            worst = min(vals) if stm_white else max(vals)   # worst listed FOR THE MOVER
            our_cp = (worst - MISS) if stm_white else (worst + MISS)
            miss += 1                             # our move fell outside SF's cached top-K
        if MODE == "cp":
            reg = (best_cp - our_cp) if stm_white else (our_cp - best_cp)
            reg = 0.0 if reg < 0 else (CAP if reg > CAP else reg)
        else:                                     # win% points lost, side-to-move POV, bounded [0,100]
            bw = _winpct(best_cp) if stm_white else (100.0 - _winpct(best_cp))
            ow = _winpct(our_cp) if stm_white else (100.0 - _winpct(our_cp))
            reg = bw - ow
            if reg < 0:
                reg = 0.0
        tot_reg += reg; n += 1
        if our == best_uci:
            match += 1
    print("REGRET mean=%.3f match=%d miss=%d total=%d" % (tot_reg / max(1, n), match, miss, n))
    sys.exit(0)

# ---------------- DRIVER: coordinate descent to MINIMISE mean regret, seeded from static-best ----------------
THIS = os.path.dirname(os.path.abspath(__file__)); ENGINE = os.path.dirname(THIS)
PY = sys.executable
SET = os.environ.get("SET", "ks_sets/regret_set.csv")
if not os.path.isabs(SET):
    SET = os.path.join(THIS, SET)
DEPTH = os.environ.get("DEPTH", "7")
PASSES = int(os.environ.get("PASSES", "2"))
N = os.environ.get("N", "0")

SEED = {"OVD_BOUNDED_MODE": 2, "OVD_CAP": 300, "OVD_KNEE": 40,
        "CENTRAL_BOUNDED_MODE": 1, "CENTRAL_CAP": 150, "CENTRAL_KNEE": 200}
GRID = {"OVD_CAP": [200, 250, 300, 400], "OVD_KNEE": [20, 40, 80, 120],
        "CENTRAL_CAP": [125, 150, 175, 200, 250], "CENTRAL_KNEE": [100, 150, 200, 300, 400]}
_only = [x.strip() for x in os.environ.get("GRID_ONLY", "").split(",") if x.strip()]
if _only:
    GRID = {k: GRID[k] for k in _only if k in GRID}


def evaluate(cfg):
    env = dict(os.environ, WORKER="1", SET=SET, DEPTH=DEPTH, N=N)
    args = [PY, "-u", os.path.abspath(__file__)] + ["%s=%s" % (k, v) for k, v in cfg.items()]
    out = subprocess.run(args, capture_output=True, text=True, cwd=ENGINE, env=env).stdout
    for line in out.splitlines():
        if line.startswith("REGRET"):
            d = dict(tok.split("=") for tok in line.split()[1:])
            return float(d["mean"]), int(d["match"]), int(d["miss"]), int(d["total"])
    return float("inf"), -1, 0, 0


def fmt(cfg):
    return " ".join("%s=%s" % (k, v) for k, v in sorted(cfg.items()))


default_reg, default_m, default_miss, tot = evaluate({"OVD_BOUNDED_MODE": 0, "CENTRAL_BOUNDED_MODE": 0})
cur = dict(SEED)
seed_reg, seed_m, seed_miss, tot = evaluate(cur)
best_reg = seed_reg
print("SET=%s DEPTH=%s N=%s   (lower regret = better)" % (os.path.basename(SET), DEPTH, tot), flush=True)
print("  our-move-not-in-SF-top-K (miss): default %d/%d (%.1f%%), seed %d/%d (%.1f%%)  <- if high, raise K"
      % (default_miss, tot, 100.0*default_miss/max(1, tot), seed_miss, tot, 100.0*seed_miss/max(1, tot)), flush=True)
print("default(off)  regret=%.3f match=%d/%d   static-seed regret=%.3f match=%d/%d"
      % (default_reg, default_m, tot, seed_reg, seed_m, tot), flush=True)

for p in range(PASSES):
    moved = 0
    for knob, vals in GRID.items():
        for v in vals:
            if cur.get(knob) == v:
                continue
            trial = dict(cur); trial[knob] = v
            reg, m, _, _ = evaluate(trial)
            if reg < best_reg - 1e-6:
                best_reg = reg; cur[knob] = v; moved += 1
                print("  p%d %-14s -> %-5s regret=%.3f match=%d" % (p, knob, v, best_reg, m), flush=True)
    print("[pass %d] %d moves, regret=%.3f" % (p, moved, best_reg), flush=True)
    if not moved:
        print("converged", flush=True)
        break

final_reg, final_m, final_miss, tot = evaluate(cur)
print("\nBEST CONFIG: %s" % fmt(cur), flush=True)
print("regret=%.3f match=%d/%d   [static-seed regret=%.3f match=%d, default regret=%.3f match=%d]"
      % (final_reg, final_m, tot, seed_reg, seed_m, default_reg, default_m), flush=True)
