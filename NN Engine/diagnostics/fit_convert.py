# -*- coding: utf-8 -*-
"""Fit P(collapse | detectors) on the convertibility corpus (verify_triage_static.py --dump output).

Tests whether cheap classical detectors SEPARATE the collapse (unconvertible) class from control (real edges).
Runs the fit on ALL rows AND on the MIDGAME subset (is_endgame==0) — the endgame collapses are trivially
separable by material (pawn_lead/is_endgame), so the midgame-only fit is the real test of whether the
compensation detectors (king-danger-to-us, counter-pressure) carry the point-costing collapses.

  python diagnostics/fit_convert.py <corpus.csv>
"""
import sys, csv, random
import numpy as np

BASE = ["kdu_us", "kdu_opp", "off_us", "def_us", "off_opp", "def_opp", "mob_us", "mob_opp",
        "npedge", "pawn_lead", "totpawns", "oppb", "is_endgame"]


def auc(y, s):
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(s); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    return (ranks[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def fit_report(rows, feats, tag):
    y = np.array([int(r["is_collapse"]) for r in rows], dtype=float)
    if y.sum() < 5 or (len(y) - y.sum()) < 5:
        print(f"\n=== {tag}: too few rows ({int(y.sum())} collapse / {int(len(y)-y.sum())} control) ==="); return
    X = np.array([[float(r.get(f, 0) or 0) for f in feats] for r in rows], dtype=float)
    mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1.0
    Xs = (X - mu) / sd
    idx = list(range(len(y))); random.Random(1234).shuffle(idx)
    cut = int(0.7 * len(y)); tr, ho = np.array(idx[:cut]), np.array(idx[cut:])
    w = np.zeros(len(feats)); b = 0.0
    for _ in range(4000):
        p = 1 / (1 + np.exp(-(Xs[tr] @ w + b))); err = p - y[tr]
        w -= 0.1 * (Xs[tr].T @ err / len(tr) + w / len(tr)); b -= 0.1 * err.mean()
    print(f"\n=== {tag}  ({int(y.sum())} collapse / {int(len(y)-y.sum())} control) ===")
    print(f"    AUC train={auc(y[tr], Xs[tr]@w+b):.3f}  HOLDOUT={auc(y[ho], Xs[ho]@w+b):.3f}   (>=0.75 separates)")
    for f, wi in sorted(zip(feats, w), key=lambda t: -abs(t[1]))[:8]:
        cm = np.mean([float(r.get(f, 0) or 0) for r in rows if r["is_collapse"] == "1"])
        km = np.mean([float(r.get(f, 0) or 0) for r in rows if r["is_collapse"] == "0"])
        print(f"    {f:>18}  w={wi:+.3f}   collapse={cm:+.0f}  control={km:+.0f}  (Δ={cm-km:+.0f})")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "corpus.csv"
    rows = list(csv.DictReader(open(path)))
    if not rows:
        print("[fit] empty corpus"); return
    for r in rows:
        g = lambda k: float(r.get(k, 0) or 0)
        r["counter_pressure"] = g("off_opp") - g("def_us")
        r["our_overreach"] = g("off_us") - g("def_opp")
        r["unbacked"] = -g("npedge")
    feats = BASE + ["counter_pressure", "our_overreach", "unbacked"]
    print(f"[fit] {len(rows)} rows, {len(feats)} detectors")
    fit_report(rows, feats, "ALL")
    fit_report([r for r in rows if r.get("is_endgame") == "0"], [f for f in feats if f != "is_endgame"], "MIDGAME-ONLY")
    print("\n[fit] READ: MIDGAME-ONLY AUC>=0.75 with a sane-signed king-danger / counter_pressure detector")
    print("      (kdu_us↑, counter_pressure↑ => collapse) => build the convertibility factor on it.")


if __name__ == "__main__":
    main()
