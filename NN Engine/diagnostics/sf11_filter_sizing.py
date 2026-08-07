# -*- coding: utf-8 -*-
"""How big are the retargeting sets? — sizing the SF11-agreement filter before building anything.

The plan: stop fitting to SF18-SEARCH (unreachable by construction -- SF18's OWN static eval misses it
by 68.9, and least-squares answers an unreachable target by shrinking toward the mean, which is the
flattening that cost 86 Elo). Fit instead to SF11-CLASSICAL, a hand-written static function that a
hand-written static function can actually match.

Two sets, and they play DIFFERENT roles:

  TRAIN   rows where SF11-static agrees with the SF18-SEARCH target. Agreement is the evidence that the
          position's truth is findable STATICALLY -- where they disagree, the truth is tactical and no
          leaf score can encode it, so those rows are the unrepresentable gradient we want gone.

  GUARD   rows where WE agree with SF18-search and SF11 does NOT. Our positional edge over SF11.
          ⚠️ HELD OUT, never trained on: these rows are selected for us already being right, so fitting
          toward our own output there is self-confirming and regression-to-the-mean guarantees a
          flattering number. As a "must not regress" constraint it is sound.

All agreement is measured in WIN%, not centipawns -- the same 750 removals invert between the two
metrics with only 33-53% worst-decile overlap.

  pyrun diagnostics/sf11_filter_sizing.py [LABELS=ks_sets/sf11_static_labels.csv]
                                          [CORPUS=ks_sets/diverse_corpus_wide.csv] [TOL=10]
"""
import os, sys, csv, math

for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v

THIS = os.path.dirname(os.path.abspath(__file__))
LABELS = os.environ.get("LABELS", "ks_sets/sf11_static_labels.csv")
CORPUS = os.environ.get("CORPUS", "ks_sets/diverse_corpus_wide.csv")
for _n in ("LABELS", "CORPUS"):
    pass
if not os.path.isabs(LABELS):
    LABELS = os.path.join(THIS, LABELS)
if not os.path.isabs(CORPUS):
    CORPUS = os.path.join(THIS, CORPUS)
TOL = float(os.environ.get("TOL", "10"))     # win% points; |a-b| <= TOL counts as agreement


def winpct(cp):
    return 50.0 + 50.0 * (2.0 / (1.0 + math.exp(-0.00368208 * cp)) - 1.0)


def main():
    lab = {}
    for r in csv.DictReader(open(LABELS, newline="")):
        lab[r["fen"]] = r
    rows = list(csv.DictReader(open(CORPUS, newline="")))
    print("corpus %d   labelled %d   agreement tolerance %.0f win%% points\n" % (len(rows), len(lab), TOL))

    n = both = sf11_only = ours_only = neither = 0
    sse_sf11 = sse_ours = 0.0
    for r in rows:
        f = r.get("fen")
        L = lab.get(f)
        if not L or not L.get("sf11_static"):
            continue
        try:
            tgt = winpct(float(r["target_total"]) * 100.0)
            ours = winpct(float(r["our_total_base"]) * 100.0) if r.get("our_total_base") else None
            s11 = winpct(float(L["sf11_static"]) * 100.0)
        except Exception:
            continue
        if ours is None:
            continue
        n += 1
        a11 = abs(s11 - tgt) <= TOL
        aou = abs(ours - tgt) <= TOL
        sse_sf11 += (s11 - tgt) ** 2
        sse_ours += (ours - tgt) ** 2
        if a11 and aou:
            both += 1
        elif a11:
            sf11_only += 1
        elif aou:
            ours_only += 1
        else:
            neither += 1

    if not n:
        print("no usable rows -- check the corpus column names (need fen/target_total/our_total_base)")
        return

    print("  %-34s %8s %8s" % ("bucket", "rows", "share"))
    for lbl, v in [("SF11 agrees AND we agree", both),
                   ("SF11 agrees, we DON'T", sf11_only),
                   ("we agree, SF11 DOESN'T  (GUARD)", ours_only),
                   ("neither agrees (tactical?)", neither)]:
        print("  %-34s %8d %7.1f%%" % (lbl, v, 100.0 * v / n))

    train = both + sf11_only
    print("\n  TRAIN set (SF11 agrees)          %8d %7.1f%%" % (train, 100.0 * train / n))
    print("  GUARD set (ours>SF11)            %8d %7.1f%%" % (ours_only, 100.0 * ours_only / n))
    print("  DISCARDED (neither)              %8d %7.1f%%" % (neither, 100.0 * neither / n))

    print("\n  mean win%%^2 error vs SF18-search:  SF11 %.1f   OURS %.1f" % (sse_sf11 / n, sse_ours / n))
    print("\nREADING IT")
    print("  TRAIN is the learnable corpus -- retarget the fit onto SF11's static value on those rows.")
    print("  GUARD is our edge over SF11; hold it out and require no regression. It is NOT training data:")
    print("  it was selected for us already being right, so fitting to it is self-confirming.")
    print("  DISCARDED is where neither static eval reaches the search verdict -- the component that")
    print("  flattened us. Removing it from the gradient is the entire point of the retarget.")


if __name__ == "__main__":
    main()
