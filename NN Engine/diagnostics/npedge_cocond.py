# -*- coding: utf-8 -*-
"""Design the co-condition: within the LOW-npedge subset (the ramp danger zone), what separates the
REAL wins (convert, must be spared) from the FANTASY wins (over-read, must be damped)?

npedge alone clips ~30% of real wins (they have npedge<250 too). The damp needs a SECOND detector that,
among low-npedge positions, is high on fantasy and low on real. This fits fantasy-vs-real ON the
low-npedge subset and ranks every detector by marginal separation (fantasy mean vs real mean + AUC).

  python diagnostics/npedge_cocond.py diagnostics/corpus_ks.csv [npedge_max=250]
"""
import sys, csv
import numpy as np

DET = ["kdu_us", "kdu_opp", "off_us", "def_us", "off_opp", "def_opp", "mob_us", "mob_opp",
       "npedge", "pawn_lead", "totpawns", "oppb", "is_endgame"]


def auc1(fant, real, key):
    f = np.array([float(r.get(key, 0) or 0) for r in fant])
    r = np.array([float(r.get(key, 0) or 0) for r in real])
    if len(f) == 0 or len(r) == 0: return float("nan")
    s = np.concatenate([f, r]); y = np.concatenate([np.ones(len(f)), np.zeros(len(r))])
    order = np.argsort(s); rank = np.empty(len(s)); rank[order] = np.arange(1, len(s) + 1)
    return (rank[y == 1].sum() - len(f) * (len(f) + 1) / 2) / (len(f) * len(r))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "diagnostics/corpus_ks.csv"
    npmax = float(sys.argv[2]) if len(sys.argv) > 2 else 250.0
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        g = lambda k: float(r.get(k, 0) or 0)
        r["_ourstat"] = g("over_read_cp") + g("sf18_static_cp")
        r["_sf"] = g("sf18_static_cp")
        r["counter_pressure"] = g("off_opp") - g("def_us")
        r["our_overreach"] = g("off_us") - g("def_opp")
        r["off_ratio"] = g("off_opp") - g("off_us")   # opp attacking more than us
        r["net_def"] = g("def_us") - g("off_opp")      # our defense clears opp offense (KPK => ~0)
    fant = [r for r in rows if r["is_collapse"] == "1" and r["_ourstat"] >= 150 and r["_sf"] <= 50]
    real = [r for r in rows if r["is_collapse"] == "0" and r["_ourstat"] >= 150 and r["_sf"] >= 150]
    fl = [r for r in fant if float(r.get("npedge", 0) or 0) < npmax]
    rl = [r for r in real if float(r.get("npedge", 0) or 0) < npmax]
    print(f"[co-cond] npedge<{npmax:.0f}:  FANTASY {len(fl)}/{len(fant)}   REAL {len(rl)}/{len(real)}")
    print(f"  (these {len(rl)} real wins are the ones an npedge-only damp would wrongly hit)\n")
    feats = DET + ["counter_pressure", "our_overreach", "off_ratio", "net_def"]
    scored = []
    for f in feats:
        a = auc1(fl, rl, f)
        fm = np.mean([float(r.get(f, 0) or 0) for r in fl]) if fl else 0
        rm = np.mean([float(r.get(f, 0) or 0) for r in rl]) if rl else 0
        scored.append((abs(a - 0.5), a, f, fm, rm))
    print("  detector separation WITHIN low-npedge (AUC away from 0.5 = discriminating; want fantasy!=real):")
    for _, a, f, fm, rm in sorted(scored, reverse=True):
        print(f"    {f:>16}  AUC={a:.2f}  fantasy={fm:+8.0f}  real={rm:+8.0f}  (Δ={fm-rm:+.0f})")
    print("\n  READ: the top detector with fantasy!=real is the co-condition. AUC>0.7 => a clean second gate")
    print("        exists to spare the low-npedge real wins; ~0.5 across the board => no cheap co-cond,")
    print("        reconsider (endgame split? the low-npedge real wins may be mostly endgames that convert).")
    # is the low-npedge real subset mostly endgames (=> handle via phase/counterplay gate)?
    en = sum(1 for r in rl if float(r.get("is_endgame", 0) or 0) >= 0.5)
    print(f"\n  low-npedge REAL: {en}/{len(rl)} are endgames (is_endgame=1)")
    en_f = sum(1 for r in fl if float(r.get("is_endgame", 0) or 0) >= 0.5)
    print(f"  low-npedge FANTASY: {en_f}/{len(fl)} are endgames")


if __name__ == "__main__":
    main()
