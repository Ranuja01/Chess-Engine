# -*- coding: utf-8 -*-
"""Characterize our king-safety eval vs SF11's, to answer: MISSING DETECTOR vs TUNING.

Reads the verify_triage_static --dump corpus (needs our_ks_cp + sf11_ks_cp, both mover-POV net cp; negative =
king-safety favours the OPPONENT = our king in danger). Reports, for collapse and control separately:
  - Pearson correlation(our_ks, sf11_ks) and OLS slope (our_ks ~ sf11_ks).
      high corr + slope<1  => we DETECT the danger but UNDER-WEIGHT it        => TUNING / realizability.
      near-zero corr       => our eval does NOT track SF's king danger at all => MISSING DETECTOR.
  - "SF sees danger we miss": rows where sf11_ks <= -100 (SF: our king in danger) but our_ks >= -20 (we shrug).

  python diagnostics/ks_gap.py diagnostics/corpus_ks.csv
"""
import sys, csv
import numpy as np


def stats(rows, tag):
    o = np.array([float(r["our_ks_cp"]) for r in rows if r.get("sf11_ks_cp") not in ("", None)])
    s = np.array([float(r["sf11_ks_cp"]) for r in rows if r.get("sf11_ks_cp") not in ("", None)])
    if len(o) < 5:
        print(f"[{tag}] n={len(o)} too few"); return
    corr = np.corrcoef(o, s)[0, 1]
    slope = np.polyfit(s, o, 1)[0]          # our = slope*sf + b
    miss = int(np.sum((s <= -100) & (o >= -20)))          # SF sees danger, we don't
    danger = int(np.sum(s <= -100))
    print(f"\n[{tag}] n={len(o)}")
    print(f"  mean our_ks={o.mean():+.0f}cp   mean sf11_ks={s.mean():+.0f}cp   (both mover-POV; -ve = our king in danger)")
    print(f"  corr(our,sf11)={corr:+.2f}   OLS slope our~sf11 = {slope:+.2f}")
    print(f"  SF sees danger (sf11_ks<=-100): {danger} positions; of those we MISS (our_ks>=-20): {miss}"
          f"  ({100.0*miss/danger:.0f}%)" if danger else "  (no SF-danger positions)")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "diagnostics/corpus_ks.csv"
    rows = list(csv.DictReader(open(path)))
    coll = [r for r in rows if r["is_collapse"] == "1"]
    ctrl = [r for r in rows if r["is_collapse"] == "0"]
    stats(coll, "COLLAPSE")
    stats([r for r in coll if r["is_endgame"] == "0"], "COLLAPSE-MIDGAME")
    stats(ctrl, "CONTROL")
    print("\nREAD: high corr + slope<1 => TUNING (detect, under-weight).  near-0 corr => MISSING DETECTOR.")
    print("      high 'MISS %' => our eval is blind to the king danger SF sees on exactly the collapse class.")


if __name__ == "__main__":
    main()
