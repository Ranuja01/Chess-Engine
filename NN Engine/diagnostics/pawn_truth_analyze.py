# -*- coding: utf-8 -*-
"""Answer the pawn-redesign design questions from `pawn_truth.csv` (see pawn_truth_generator.py).

The brief's questions, restated as things this script computes:
  Q1  Is a WEAK pawn on the 6th worth more than a STRONG pawn on the 5th?
        -> compare marginal value(weak, r) against value(strong, r-1) for every r.
  Q2  Does that one-rank difference change depending on WHICH ranks?
        -> the per-rank advancement gradient, value(r) - value(r-1), reported rank by rank.
  Q3  Do the two axes roughly cancel?
        -> the strength premium value(strong, r) - value(weak, r) against the rank gradient at r.
  Q4  What else decides it?
        -> the same contrasts split by enemy obstruction, by file, and by backdrop (pieces vs K+P).

Every figure carries n and a 95% CI (normal approx). A contrast whose CI spans 0 is NOT an answer, and is
printed as such rather than being read as a small effect -- the whole point of the generator's randomized
backdrops is that we get to say which of these we actually know.

  pyrun diagnostics/pawn_truth_analyze.py [IN=diagnostics/ks_sets/pawn_truth.csv]
"""
import os, sys, csv, math
from collections import defaultdict

for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v

THIS = os.path.dirname(os.path.abspath(__file__))
IN = os.environ.get("IN", os.path.join(THIS, "ks_sets", "pawn_truth.csv"))

# "Strong" and "weak" as the brief uses them, kept explicit here so the definition is auditable.
# ⚠️ `unsupported` is deliberately NOT in WEAK. It places friendly pawns two ranks ahead on the adjacent
# files, and the test pawn becomes their supporter as soon as it advances one square -- validation showed it
# scoring ABOVE `supported` in most cells, which is a latent-chain effect, not weakness. Lumping it into
# WEAK contaminated the contrast in the first two runs. `isolated` is the only clean weak anchor we have.
STRONG = ("supported", "phalanx")
WEAK = ("isolated",)


def mean_ci(vals):
    n = len(vals)
    if n == 0:
        return (float("nan"), float("nan"), 0)
    m = sum(vals) / n
    if n < 2:
        return (m, float("nan"), n)
    var = sum((v - m) ** 2 for v in vals) / (n - 1)
    return (m, 1.96 * math.sqrt(var / n), n)


def median(vals):
    if not vals:
        return float("nan")
    s = sorted(vals)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def fmt(vals):
    """Mean +-CI with the MEDIAN alongside. The outcome filter that used to hide the tail is gone, so a
    mean and median that disagree sharply is the signal that a few decisive samples carry the cell."""
    m, ci, n = mean_ci(vals)
    if n == 0:
        return "      --          "
    if n < 2:
        return "%+7.0f      (n=1) " % m
    return "%+7.0f +-%3.0f med%+5.0f n=%-4d" % (m, ci, median(vals), n)


def verdict(vals_a, vals_b):
    """Signed difference of two independent means with a CI; says plainly when it resolves nothing."""
    ma, ca, na = mean_ci(vals_a)
    mb, cb, nb = mean_ci(vals_b)
    if na < 2 or nb < 2:
        return "insufficient data"
    d = ma - mb
    ci = math.sqrt(ca ** 2 + cb ** 2)
    if abs(d) < ci:
        return "%+.0f +-%.0f  NOT RESOLVED (CI spans 0)" % (d, ci)
    return "%+.0f +-%.0f  %s" % (d, ci, "FIRST is worth more" if d > 0 else "SECOND is worth more")


def main():
    if not os.path.exists(IN):
        sys.exit("missing %s -- run pawn_truth_generator.py first" % IN)
    rows = []
    with open(IN, newline="") as fh:
        for r in csv.DictReader(fh):
            r["rank"] = int(r["rank"]); r["file"] = int(r["file"]); r["value"] = float(r["value"])
            rows.append(r)
    if not rows:
        sys.exit("%s is empty" % IN)
    print("pawn ground truth: %d samples\n" % len(rows))

    def sel(**kw):
        out = rows
        for k, v in kw.items():
            out = [r for r in out if (r[k] in v if isinstance(v, tuple) else r[k] == v)]
        return [r["value"] for r in out]

    ranks = sorted({r["rank"] for r in rows})

    # ---- the value table: what a pawn is worth, by rank x strength ----------------------------------
    print("MARGINAL VALUE OF THE PAWN (cp, SF18 search, White POV)")
    print("  %-6s %-22s %-22s %-22s" % ("rank", "STRONG (supp/phalanx)", "WEAK (isol/unsupp)", "premium strong-weak"))
    for r in ranks:
        s, w = sel(rank=r, friendly=STRONG), sel(rank=r, friendly=WEAK)
        print("  %-6d %-22s %-22s %s" % (r, fmt(s), fmt(w), verdict(s, w)))

    # ---- Q2: is the advancement gradient rank-dependent? ---------------------------------------------
    print("\nQ2  ADVANCEMENT GRADIENT -- value(r) - value(r-1), all contexts pooled")
    for r in ranks[1:]:
        print("  %d->%d  %s" % (r - 1, r, verdict(sel(rank=r), sel(rank=r - 1))))

    # ---- Q1/Q3: weak-on-r vs strong-on-(r-1) ---------------------------------------------------------
    print("\nQ1/Q3  WEAK on rank r  vs  STRONG on rank r-1  (does one rank of advancement beat structure?)")
    for r in ranks[1:]:
        if r - 1 not in ranks:
            continue
        print("  weak@%d vs strong@%d   %s"
              % (r, r - 1, verdict(sel(rank=r, friendly=WEAK), sel(rank=r - 1, friendly=STRONG))))

    # ---- Q4: what else moves it ----------------------------------------------------------------------
    print("\nQ4a  BY ENEMY OBSTRUCTION (all ranks pooled)")
    for e in sorted({r["enemy"] for r in rows}):
        print("  %-10s %s" % (e, fmt(sel(enemy=e))))
    print("\nQ4b  STRENGTH PREMIUM WITHIN EACH OBSTRUCTION  (does structure matter more when blocked?)")
    for e in sorted({r["enemy"] for r in rows}):
        print("  %-10s %s" % (e, verdict(sel(enemy=e, friendly=STRONG), sel(enemy=e, friendly=WEAK))))
    print("\nQ4c  BY FILE")
    for f in sorted({r["file"] for r in rows}):
        print("  file %-4s %s" % ("abcdefgh"[f], fmt(sel(file=f))))
    print("\nQ4d  BY BACKDROP  (phase: pieces on the board vs kings-and-pawns only)")
    for b in sorted({r["backdrop"] for r in rows}):
        print("  %-8s %s" % (b, fmt(sel(backdrop=b))))
    print("\nQ4e  ADVANCEMENT GRADIENT PER BACKDROP  (does phase change the shape of the rank curve?)")
    for b in sorted({r["backdrop"] for r in rows}):
        grad = []
        for r in ranks[1:]:
            ma, _, na = mean_ci(sel(backdrop=b, rank=r))
            mb, _, nb = mean_ci(sel(backdrop=b, rank=r - 1))
            grad.append("%d->%d %s" % (r - 1, r, ("%+.0f" % (ma - mb)) if na and nb else "--"))
        print("  %-8s %s" % (b, "  ".join(grad)))

    # ---- the per-context table the redesign would actually be built from -----------------------------
    print("\nFULL CELL TABLE (cp) -- rows are friendly context, columns are rank, split by obstruction")
    for e in sorted({r["enemy"] for r in rows}):
        print("\n  obstruction = %s" % e)
        print("    %-13s %s" % ("", "".join("%9d" % r for r in ranks)))
        for fr in sorted({r["friendly"] for r in rows}):
            cells = []
            for r in ranks:
                m, _, n = mean_ci(sel(enemy=e, friendly=fr, rank=r))
                cells.append("%9s" % ("--" if not n else "%+.0f" % m))
            print("    %-13s %s" % (fr, "".join(cells)))


if __name__ == "__main__":
    main()
