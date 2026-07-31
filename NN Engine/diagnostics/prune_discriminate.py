# -*- coding: utf-8 -*-
"""
Prune discriminator (Step 4) — does a cheap signal separate RFP's WRONG fires from its CORRECT ones?

Reads the labeled CSVs from prune_verify.py (columns incl: id, rd, a, b, seval, cheap, eval_instab, wrong)
and reports, per candidate detector, the AUC against the WRONG label + wrong-rate by rd. This is the
make-or-break of the prune-verification plan:
  - a detector with real AUC (>~0.6) => a conditional RFP gate is buildable and we know its form.
  - nothing separates => RFP's over-prunes aren't cheaply distinguishable => a gate can't work (learned
    offline, before building) = the interpretable "genuinely hard, not a bug" answer.

Detectors computed from the CSV fields (no extra engine calls):
  eval_instab   = |seval - cheap|   (full-vs-cheap gap = tactical loadedness; the primary hypothesis)
  margin_slack  = how far seval cleared the window (RFP_MIN: a - seval ; RFP_MAX: seval - b) — small slack
                  = marginal fire = the risky ones
  abs_seval     = |seval|
  rd            = remaining depth

Usage (via runner pyrun):  prune_discriminate.py /tmp/pv_rd*.csv
"""
import sys
import csv
import glob

import numpy as np


def auc(score, label):
    """AUC = P(score higher on positives than negatives). Rank-based, ties averaged."""
    label = np.asarray(label)
    score = np.asarray(score, dtype=float)
    npos = label.sum(); nneg = len(label) - npos
    if npos == 0 or nneg == 0:
        return float('nan')
    order = score.argsort()
    ranks = np.empty(len(score), float)
    ranks[order] = np.arange(1, len(score) + 1)
    # average ties
    _, inv, cnt = np.unique(score, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt)
    start = csum - cnt
    avg = (start + csum + 1) / 2.0
    ranks = avg[inv]
    return (ranks[label == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)


def main():
    paths = []
    for a in sys.argv[1:]:
        paths.extend(glob.glob(a))
    rows = []
    for p in paths:
        rows.extend(list(csv.DictReader(open(p))))
    if not rows:
        print("no rows"); return
    for r in rows:
        for k in ('a', 'b', 'rd', 'seval', 'cheap', 'eval_instab', 'wrong'):
            r[k] = int(r[k])
        r['margin_slack'] = (r['a'] - r['seval']) if r['id'] == 'RFP_MIN' else (r['seval'] - r['b'])
        r['abs_seval'] = abs(r['seval'])

    wrong = np.array([r['wrong'] for r in rows])
    n = len(rows)
    print("total labeled=%d  WRONG=%d (%.1f%%)" % (n, wrong.sum(), 100.0 * wrong.mean()))

    print("\nwrong-rate by rd:")
    for rd in sorted(set(r['rd'] for r in rows)):
        sub = [r for r in rows if r['rd'] == rd]
        w = np.mean([r['wrong'] for r in sub])
        print("  rd=%d  n=%-6d wrong=%.1f%%" % (rd, len(sub), 100 * w))
    print("\nwrong-rate by site:")
    for sid in sorted(set(r['id'] for r in rows)):
        sub = [r for r in rows if r['id'] == sid]
        print("  %-8s n=%-6d wrong=%.1f%%" % (sid, len(sub), 100 * np.mean([r['wrong'] for r in sub])))

    print("\ndetector AUC vs WRONG (0.5=no signal; >0.6 = usable; <0.4 = inverted-usable):")
    for det in ('eval_instab', 'margin_slack', 'abs_seval', 'rd'):
        vals = np.array([r[det] for r in rows], float)
        a = auc(vals, wrong)
        # also mean on wrong vs ok for interpretability
        mw = vals[wrong == 1].mean() if wrong.sum() else float('nan')
        mo = vals[wrong == 0].mean() if (wrong == 0).any() else float('nan')
        print("  %-14s AUC=%.3f   mean(wrong)=%.0f  mean(ok)=%.0f" % (det, a, mw, mo))


if __name__ == '__main__':
    main()
