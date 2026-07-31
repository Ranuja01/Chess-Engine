# -*- coding: utf-8 -*-
"""THE decisive test: can our cheap detectors separate a FANTASY winning-advantage from a REAL one?

The prior "collapse vs control" fit found material-lead — but that is SHARED with real wins, so damping it kills
wins (l1a = -11%). The RIGHT contrast is fantasy-win vs real-win, both "we think we're winning":
  FANTASY (label 1): our_static >= +150cp but SF18 <= +50cp  (we over-read a win that isn't there)
  REAL    (label 0): our_static >= +150cp and SF18 >= +150cp (we AND SF agree we're winning)
Fit P(fantasy | detectors). Reads corpus_ks.csv (our_static = over_read_cp + sf18_static_cp).

  AUC > 0.7 + sane detector => a TARGETED damp exists (fires on fantasy, spares real wins) => build it.
  AUC ~ 0.5                 => our cheap detectors CANNOT see it => we are MISSING an SF11-style feature.

  python diagnostics/fit_fantasy.py diagnostics/corpus_ks.csv
"""
import sys, csv, random
import numpy as np

DET = ["kdu_us", "kdu_opp", "off_us", "def_us", "off_opp", "def_opp", "mob_us", "mob_opp",
       "npedge", "pawn_lead", "totpawns", "oppb", "is_endgame"]


def auc(y, s):
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    order = np.argsort(s); r = np.empty(len(s)); r[order] = np.arange(1, len(s) + 1)
    return (r[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "diagnostics/corpus_ks.csv"
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        g = lambda k: float(r.get(k, 0) or 0)
        r["_ourstat"] = g("over_read_cp") + g("sf18_static_cp")
        r["_sf"] = g("sf18_static_cp")
        r["counter_pressure"] = g("off_opp") - g("def_us")
        r["our_overreach"] = g("off_us") - g("def_opp")
        r["unbacked"] = -g("npedge")
    feats = DET + ["counter_pressure", "our_overreach", "unbacked"]

    fant = [r for r in rows if r["is_collapse"] == "1" and r["_ourstat"] >= 150 and r["_sf"] <= 50]
    real = [r for r in rows if r["is_collapse"] == "0" and r["_ourstat"] >= 150 and r["_sf"] >= 150]
    print(f"[fantasy-vs-real] FANTASY(win-we-think, SF~0)={len(fant)}   REAL(win, SF-agrees)={len(real)}")
    if len(fant) < 10 or len(real) < 10:
        print("  ** one class too small from the corpus -> need to harvest won-game peaks (see handoff). **")
        # still attempt if marginally small
        if len(fant) < 5 or len(real) < 5:
            return
    data = [(r, 1) for r in fant] + [(r, 0) for r in real]
    y = np.array([d[1] for d in data], dtype=float)
    X = np.array([[float(d[0].get(f, 0) or 0) for f in feats] for d in data], dtype=float)
    mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1.0; Xs = (X - mu) / sd
    idx = list(range(len(y))); random.Random(1234).shuffle(idx)
    cut = int(0.7 * len(y)); tr, ho = np.array(idx[:cut]), np.array(idx[cut:])
    w = np.zeros(len(feats)); b = 0.0
    for _ in range(5000):
        p = 1 / (1 + np.exp(-(Xs[tr] @ w + b))); e = p - y[tr]
        w -= 0.1 * (Xs[tr].T @ e / len(tr) + w / len(tr)); b -= 0.1 * e.mean()
    print(f"  AUC train={auc(y[tr], Xs[tr]@w+b):.3f}  HOLDOUT={auc(y[ho], Xs[ho]@w+b):.3f}")
    for f, wi in sorted(zip(feats, w), key=lambda t: -abs(t[1]))[:8]:
        fm = np.mean([float(r.get(f, 0) or 0) for r in fant]); rm = np.mean([float(r.get(f, 0) or 0) for r in real])
        print(f"    {f:>18} w={wi:+.3f}  fantasy={fm:+.0f}  real={rm:+.0f}  (Δ={fm-rm:+.0f})")
    print("\n  READ: HOLDOUT>0.7 + sane detector => a TARGETED realizability exists (spares real wins) => build it.")
    print("        ~0.5 => our cheap detectors can't separate fantasy from real => MISSING SF11-style feature.")


if __name__ == "__main__":
    main()
