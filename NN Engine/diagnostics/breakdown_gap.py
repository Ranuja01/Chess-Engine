# -*- coding: utf-8 -*-
"""Mass-compare OUR per-term eval breakdown to SF11's, feature by feature, to MAP + CLASSIFY the gaps.

Tuning re-weights features we already have; it cannot add a dimension we don't represent. So when a retune goes
flat, the next question is: on which SF11 features are we systematically LOW, and WHY. This reads an SF11-labelled
corpus (tune_corpus without --no-sf11) and, per SF11 feature, reports:
  mean_sf   - SF11's average contribution (White-POV pawns)
  mean_ours - our mapped term(s)' average (White-POV pawns; our terms are Black-positive milli-pawns -> -v/1000)
  gap       - mean_sf - mean_ours  (+ = SF reads MORE here = we under-read)
  corr      - correlation of our mapped value with SF's, position by position (do we TRACK its variation?)
  zero%     - fraction of positions where |ours| ~ 0 while |SF| is meaningful (do we even FIRE here?)
and a CLASSIFICATION:
  MISSING   - we compute ~0 where SF is non-trivial (no representation / gated off)  -> BUILD / un-gate
  MALFORMED - we compute something but corr is low (our function tracks the wrong thing)  -> FIX the function
  MIS-SCALED- we track it (corr high) but the magnitude is off  -> RETUNE (the cheap fix)
  OK        - track + magnitude both close
NOT covered here: "slow" (an NPS/profiler question, not a static-value one) -> the eval profiler, separately.

Usage:  python diagnostics/breakdown_gap.py [corpus.csv]  (default cond_corpus_v2.csv)
"""
import csv
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, "selfplay", "tune_data", "cond_corpus_v2.csv")

# SF11 feature -> the OUR term(s) that are supposed to cover it (our_eval_reference.md correspondence map).
# Sum when several of our terms jointly express one SF bundle. SF11 cols are already White-POV pawns; our terms
# are Black-positive milli-pawns, converted to White-POV pawns as -v/1000 (done below).
MAP = {
    "sf11_material":   ["material"],
    "sf11_space":      ["central"],
    "sf11_kingsafety": ["king_safety", "latent_threat"],   # our king danger + king-directed latent threats
    "sf11_mobility":   ["mobility"],                        # gated OFF by default -> expect ~0 = MISSING
    "sf11_threats":    ["capture_gains", "latent_threat"],
    "sf11_passed":     ["passed_pawn_support"],
    "sf11_pawns":      ["pawn_struct"],                     # gated OFF by default -> expect ~0
}


def col(rows, name):
    out = []
    for r in rows:
        v = r.get(name, "")
        out.append(float(v) if v not in ("", "----", None) else 0.0)
    return np.array(out)


def report(rows, have, label):
    n = len(rows)
    if n == 0:
        print("\n[%s]  n=0 (empty stratum)" % label)
        return
    print("\n[%s]  n=%d" % (label, n))
    print("%-16s %8s %9s %8s %7s %7s   %s" % ("SF11 feature", "mean_sf", "mean_ours", "gap", "corr", "zero%", "class"))
    for sfcol, ours_terms in MAP.items():
        if sfcol not in have:
            continue
        sf = col(rows, sfcol)
        ours = np.zeros(n)
        for t in ours_terms:
            if t in have:
                ours += -col(rows, t) / 1000.0   # Black-positive milli-pawns -> White-POV pawns
        mean_sf, mean_ours = float(np.mean(sf)), float(np.mean(ours))
        gap = mean_sf - mean_ours
        corr = float(np.corrcoef(ours, sf)[0, 1]) if np.std(ours) > 1e-9 and np.std(sf) > 1e-9 else 0.0
        sf_meaningful = np.abs(sf) > 0.10                       # SF says something here
        zero_frac = float(np.mean(np.abs(ours[sf_meaningful]) < 0.02)) if sf_meaningful.any() else 0.0
        if zero_frac > 0.7 or (abs(mean_ours) < 0.02 and abs(mean_sf) > 0.05):
            cls = "MISSING (build/un-gate)"
        elif abs(corr) < 0.3:
            cls = "MALFORMED (fix function)"
        elif abs(gap) > 0.15:
            cls = "MIS-SCALED (retune)"
        else:
            cls = "OK"
        print("%-16s %8.3f %9.3f %+8.3f %7.2f %6.0f%%   %s"
              % (sfcol.replace("sf11_", ""), mean_sf, mean_ours, gap, corr, 100 * zero_frac, cls))


def main():
    rows = list(csv.DictReader(open(CORPUS)))
    have = set(rows[0].keys()) if rows else set()
    print("corpus %s  n=%d" % (os.path.basename(CORPUS), len(rows)))
    report(rows, have, "ALL")
    # Sharpness / decision-tail strata: the collapse-relevant error lives in balanced MIDGAME positions (where
    # eval precision decides), not decided or endgame ones. If a feature is malformed OVERALL but WORSE here,
    # that's the strength-relevant gap; if it's flat across strata, the average is not hiding a tail.
    def sub(pred):
        return [r for r in rows if pred(r)]
    endgame = lambda r: r.get("is_endgame") in ("1", 1)
    near = lambda r: r.get("status") == "near_equal"
    report(sub(lambda r: (not endgame(r)) and near(r)), have, "midgame near-equal (decision tail)")
    report(sub(lambda r: (not endgame(r)) and not near(r)), have, "midgame decided")
    report(sub(endgame), have, "endgame")


if __name__ == "__main__":
    main()
