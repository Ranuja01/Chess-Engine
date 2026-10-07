# -*- coding: utf-8 -*-
"""ENDGAME-LEG INSPECTION (owner, 2026-10-07). The bench (C3 §21) found our static eval UNDER-CONFIDENT in endgames: the side
ahead is rated ~6 win% points less winning than SF18's search (SF11: −3.4), worst in pawn endings (−11.4), queen endings
(−9.1) and rook+minor (−8.6), on par with SF11 in pure rook / pure minor endings. Question: is it plain MAGNITUDE (the eg
share is too small overall) or specific terms' endgame values — before any POT/winnability design?

Data: the DEPTH target on endgame-phase rows only (v2 phase256 < 128; mg + eg labelled sets; our d10 search of the 10-04 ship
`fitC_*_ours1004_d10` vs SF18 d14), by-GAME val split, STM + global-SCALE nuisances (`_revival_screen.fitter`).
ARMS (each fitted alone):  A  EG-STRETCH — one parameter s: ours' += s · base · (256 − ph)/256 (scale the endgame share)
                           B  eg legs of all 235 v2 columns     C  eg legs of the file-tied PST     D  A + B + C
Per arm: val change, and the side-ahead BIAS by endgame type (win% points; + over-, − under-rating) BEFORE → AFTER.
Also prints the bias of our d10 SEARCH itself (base), to see whether search inherits the static under-confidence.
  pyrun diagnostics/_eg_leg_inspect.py [LAMBDA=1e-2]
"""
import os, sys
import numpy as np
THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
import _revival_screen as RS
from _joint_depth_preview import pst_kauf
from _endgame_types import classify

KV = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)


def main():
    fens, D, ph, sf, base, val, stm = RS.load_rows()
    m = ph < 128
    fens = [f for f, k in zip(fens, m) if k]
    D, ph, sf, base, val, stm = D[m], ph[m], sf[m], base[m], val[m], stm[m]
    tgt, tr = RS.wp(sf), ~val
    egw = ((256.0 - ph) / 256.0)[:, None]
    PST = np.array([pst_kauf(f)[0] for f in fens])
    X_stretch = -10.0 * base[:, None] * egw          # −X·δ/10 = δ · base · egw  ⇒ δ is the endgame stretch fraction
    blocks = {"A EG-STRETCH (1 param)": X_stretch, "B eg legs, v2 cols": D * egw, "C eg legs, PST": PST * egw}
    blocks["D A+B+C"] = np.concatenate(list(blocks.values()), 1)
    fit, loss_of, vb = RS.fitter(base, tgt, stm, tr, val)
    lam = float(KV.get("LAMBDA", 1e-2))
    types = [classify(f) for f in fens]
    tnames = sorted({t[0] for t in types if t})

    def bias_table(cp):
        out = {}
        for tn in tnames + ["ALL"]:
            sel = [i for i, t in enumerate(types) if val[i] and t and (tn == "ALL" or t[0] == tn) and abs(sf[i]) > 25]
            if len(sel) < 10:
                continue
            out[tn] = (len(sel), float(np.mean([(RS.wp(cp[i]) - tgt[i]) * np.sign(sf[i]) for i in sel])))
        return out

    print("ENDGAME-LEG INSPECTION  rows %d (val %d by game) · phase256 < 128 · base = our d10 search (%s)" % (len(sf), val.sum(), RS.OURS))
    b0 = bias_table(base)
    print("  our d10 SEARCH bias vs SF18 (side ahead, win%% pts): " + " · ".join("%s %+.1f (n %d)" % (k, v[1], v[0]) for k, v in b0.items()))
    for name, X in blocks.items():
        p = fit(X, 0.0 if name.startswith("A") else lam)
        cp = base * (1.0 + p[-1]) - (X @ p[:-2]) / 10.0 + p[-2] * stm
        extra = ("  stretch s %+.3f (endgame share ×%.3f)" % (p[0], 1 + p[0])) if name.startswith("A") else ""
        if name.startswith("D"):
            extra = "  stretch s %+.3f" % p[0]
        print("\n  %-24s val %+6.2f%% · global scale %+.3f%s" % (name, 100 * (loss_of(X, p, val) / vb - 1), p[-1], extra))
        b1 = bias_table(cp)
        print("    bias after: " + " · ".join("%s %+.1f" % (k, v[1]) for k, v in b1.items()))


if __name__ == "__main__":
    main()
