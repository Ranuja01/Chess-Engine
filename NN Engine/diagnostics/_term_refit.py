# -*- coding: utf-8 -*-
"""CONSTRAINED SUBSYSTEM REFIT — maximise NET positions moved, never distance to a reference.

WHY THIS EXISTS (2026-09-20). `_term_separability.py` showed that a weighting fitted to fix our move
failures against SF11 **fixed 11 and broke 28** — 12.3% of the positions we already got right, and
**32.2% of the cell where we are right and SF11 is WRONG**. Net −17. That is
`corpus-fit-is-anti-correlated-with-elo` caught in the act, with the mechanism visible for the first time:
a reference-distance objective is rewarded for the positions it targets and blind to everything it breaks,
and what it breaks includes every position where WE are better than the reference.

Owner's framing, and it is the objective implemented here:
  *"if we could maintain our small advantage and tune the others towards SF11, that would be ideal"*
  *"if our eval is just correct to SF18 search over the SF11 eval then that's a win for us to maintain"*

⇒ **maximise `fix`, subject to keeping `keep` and `adv`** — not "minimise error".

Input: the `VEC_OUT` constraint dumps. Each row is `pos_id, class, d_1..d_k` where `d = x_target − x_sibling`
and a position is SATISFIED iff every one of its rows has `d·w > 0`.
  class `fix`  our failure on a position SF11's eval gets right   -> the objective
  class `keep` we already get it right, SF11 too                  -> must not break
  class `adv`  ⭐ we get it right and SF11 does NOT                -> must not break (the advantage)

☠️ THE INCUMBENT SCORES NET 0 BY CONSTRUCTION (w = 1 fixes none, breaks none). **A candidate is only
interesting if its NET is positive**, and the number to quote is the HELD-OUT net, not the fitted one --
every prior corpus fit in this project looked good in-sample.

  pyrun diagnostics/_term_refit.py FIT=ks_sets/vec_a.csv,ks_sets/vec_b.csv HELD=ks_sets/vec_c.csv
        [W_LO=0.25] [W_HI=4.0] [LAMBDA=3.0] [RESTARTS=40] [SEED=0]
"""
import os, sys, csv, math, random
from collections import defaultdict

for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v

import numpy as np
from scipy.optimize import minimize

THIS = os.path.dirname(os.path.abspath(__file__))
W_LO = float(os.environ.get("W_LO", "0.25"))
W_HI = float(os.environ.get("W_HI", "4.0"))
LAMBDA = float(os.environ.get("LAMBDA", "3.0"))   # price of breaking a position we already have
RESTARTS = int(os.environ.get("RESTARTS", "40"))
SEED = int(os.environ.get("SEED", "0"))


def load(paths):
    """-> {class: [ (pos_id, D) ]}, plus the term names."""
    out, terms = defaultdict(list), None
    for p in paths:
        p = p.strip()
        if not p:
            continue
        if not os.path.isabs(p):
            p = os.path.join(THIS, p)
        rows = defaultdict(list)
        cls_of = {}
        with open(p, newline="") as fh:
            rd = csv.reader(fh)
            hdr = next(rd)
            t = [h[2:] for h in hdr[2:]]
            if terms is None:
                terms = t
            elif terms != t:
                sys.exit("term set differs between %s and the first file -- refusing to pool" % p)
            for r in rd:
                key = (p, r[0])
                cls_of[key] = r[1]
                rows[key].append([float(x) for x in r[2:]])
        for key, mat in rows.items():
            out[cls_of[key]].append((key, np.array(mat, dtype=float)))
    return out, terms


def margins(mats, w):
    """Per position: the WORST row. A position is satisfied iff this is > 0."""
    return np.array([float((D @ w).min()) for _, D in mats]) if mats else np.zeros(0)


def score(data, w, base=None):
    """Scored RELATIVE TO THE INCUMBENT, which is the only meaningful definition of collateral damage.

    ☠️ An absolute `margin > 0` test mis-scores TIES: a position where two children share the top score
    has margin exactly 0, so it reads as "broken" even at w = 1, and the incumbent scored −4 instead of
    the 0 it must score by construction. Broken now means: it WAS satisfied and now is not.
    """
    ones = np.ones(len(w))
    out = []
    for cls in ("fix", "keep", "adv"):
        mats = data.get(cls, [])
        m_new = margins(mats, w)
        m_inc = margins(mats, ones) if base is None else base[cls]
        if cls == "fix":
            out.append(int(((m_new > 0) & (m_inc <= 0)).sum()))
        else:
            out.append(int(((m_new <= 0) & (m_inc > 0)).sum()))
    fixed, kept_broken, adv_broken = out
    return fixed, kept_broken, adv_broken, fixed - kept_broken - adv_broken


def main():
    fit_paths = os.environ.get("FIT", "ks_sets/vec_a.csv,ks_sets/vec_b.csv").split(",")
    held_paths = os.environ.get("HELD", "ks_sets/vec_c.csv").split(",")
    fit, terms = load(fit_paths)
    held, _ = load(held_paths)
    k = len(terms)

    print("CONSTRAINED SUBSYSTEM REFIT — objective = NET positions moved (fix − broken)")
    print("  terms: %s" % ",".join(terms))
    for nm, d in (("FIT", fit), ("HELD", held)):
        print("  %-4s positions: fix=%d  keep=%d  adv=%d"
              % (nm, len(d.get("fix", [])), len(d.get("keep", [])), len(d.get("adv", []))))

    ones = np.ones(k)
    print("\n  INCUMBENT (w = 1, the shipped config): fixed=%d broken_keep=%d broken_adv=%d  NET=%+d"
          % score(fit, ones))
    print("    ⇒ net 0 by construction. A candidate must beat this, on HELD-OUT data.")

    # Smooth surrogate for a count objective: reward fix-margins, price broken keep/adv at LAMBDA.
    # ⚠️ A heuristic, not an exact MIP -- reported counts below are always the EXACT counts under the
    # returned w, so the surrogate only steers the search, it never flatters the result.
    def _sig(m):
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, m))))

    def obj(w):
        val = 0.0
        # REWARD: a `fix` position whose worst row is positive.
        for _, D in fit.get("fix", []):
            val += _sig(float((D @ w).min()) / 100.0)
        # PENALTY: a `keep` or `adv` position whose worst row has gone negative. LAMBDA prices it, and
        # `adv` is priced the same as `keep` -- breaking a position where we beat SF11 is not cheaper
        # than breaking any other, it is simply rarer and harder to win back.
        for cls in ("keep", "adv"):
            for _, D in fit.get(cls, []):
                val -= LAMBDA * (1.0 - _sig(float((D @ w).min()) / 100.0))
        return -val

    rng = random.Random(SEED)
    best, best_w = None, None
    for i in range(RESTARTS):
        w0 = np.array([rng.uniform(W_LO, W_HI) for _ in range(k)]) if i else np.ones(k)
        res = minimize(obj, w0, method="L-BFGS-B", bounds=[(W_LO, W_HI)] * k,
                       options=dict(maxiter=300))
        w = res.x
        sc = score(fit, w)
        if best is None or sc[3] > best[3]:
            best, best_w = sc, w

    print("\n  FITTED (%d restarts, LAMBDA=%.1f):" % (RESTARTS, LAMBDA))
    print("    w = " + "  ".join("%s=%.2f" % (t, v) for t, v in zip(terms, best_w)))
    print("    on FIT : fixed=%d broken_keep=%d broken_adv=%d  NET=%+d" % best)
    hs = score(held, best_w)
    print("    on HELD: fixed=%d broken_keep=%d broken_adv=%d  NET=%+d" % hs)
    print("\n  ⇒ %s" % ("HELD-OUT NET IS POSITIVE — a coarse retune has real headroom; take it to the d7"
                        " regret gate, then games." if hs[3] > 0 else
                        "HELD-OUT NET IS NOT POSITIVE — coarse subsystem reweighting does not pay. The"
                        " lever is inside the terms (shapes, gates, phase curves) or in new signal;"
                        " do NOT spend a games night on a global rescale."))
    print("  ⚠️ Static one-ply proxy on QUIET, criticality-filtered positions. Games still decide, and"
          " every prior corpus-fitted config lost in games.")


if __name__ == "__main__":
    main()
