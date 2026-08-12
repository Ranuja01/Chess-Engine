# -*- coding: utf-8 -*-
"""Generate section 4 (the case book) of dev_notes/PAWN_MODEL.md directly from the truth CSVs.

The case book is the part of the model doc that answers "is a weak pawn on the 6th better than a strong one
on the 5th?" and its many siblings. Written by hand it would drift from the data the first time anything is
re-measured, and a canonical doc that has quietly drifted is worse than no doc. So it is GENERATED, in
place, between markers, and regenerating is the only supported way to change it.

Every contrast carries n and a 95% CI, and any contrast whose CI spans zero prints as **UNRESOLVED** rather
than as a small number -- this project has repeatedly mistaken an underpowered null for a finding.

Provenance is printed per source file, because manufactured-position magnitudes have already been shown
wrong by 2.6x against real positions (see PAWN_MODEL.md section 8).

  pyrun diagnostics/pawn_truth_casebook.py [WRITE=1]      # WRITE=1 injects into PAWN_MODEL.md
"""
import os, sys, csv, math, glob
from collections import defaultdict

for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v

THIS = os.path.dirname(os.path.abspath(__file__))
ENG = os.path.dirname(THIS)
DOC = os.path.join(ENG, "dev_notes", "PAWN_MODEL.md")
BEGIN, END = "<!-- CASEBOOK:BEGIN -->", "<!-- CASEBOOK:END -->"
WRITE = os.environ.get("WRITE", "0") == "1"

STRONG = ("supported", "phalanx")
WEAK = ("isolated",)          # `unsupported` is NOT weak -- the test pawn latently supports it

# ☠️ QUARANTINE. These CSVs were produced BEFORE the generator defects were fixed and must never appear as
# evidence, however tempting their sample counts are:
#   pawn_truth.csv       -- all three original defects (context pawns on r+1 were DEFENDED BY the test pawn;
#                           |value|>400 filtered on the DEPENDENT VARIABLE; sparse backdrops made an extra
#                           pawn decisive rather than marginal). Its rank-7 passer read +153 cp.
#   pawn_truth_valid.csv -- scratch file, overwritten repeatedly during instrument validation; its contents
#                           correspond to no single consistent generator version.
QUARANTINE = {"pawn_truth.csv", "pawn_truth_valid.csv"}


def mean_ci(v):
    n = len(v)
    if n < 2:
        return (float("nan"), float("nan"), n)
    m = sum(v) / n
    var = sum((x - m) ** 2 for x in v) / (n - 1)
    return (m, 1.96 * math.sqrt(var / n), n)


def median(v):
    if not v:
        return float("nan")
    s = sorted(v); n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def verdict(a, b, la, lb):
    """Signed difference of two independent means. Says UNRESOLVED when the CI spans zero."""
    ma, ca, na = mean_ci(a)
    mb, cb, nb = mean_ci(b)
    if na < 2 or nb < 2:
        return "insufficient data (n=%d/%d)" % (na, nb)
    d, ci = ma - mb, math.sqrt(ca ** 2 + cb ** 2)
    if abs(d) < ci:
        return "%+.0f ±%.0f — **UNRESOLVED**" % (d, ci)
    return "%+.0f ±%.0f — **%s** wins" % (d, ci, la if d > 0 else lb)


def load(path):
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                r["rank"] = int(r["rank"]); r["value"] = float(r["value"])
            except (KeyError, ValueError):
                continue
            rows.append(r)
    return rows


def section(rows, label):
    out = ["", "### Source: `%s` — %d samples" % (label, len(rows)), ""]

    def sel(**kw):
        o = rows
        for k, v in kw.items():
            o = [r for r in o if (r[k] in v if isinstance(v, tuple) else r[k] == v)]
        return [r["value"] for r in o]

    ranks = sorted({r["rank"] for r in rows})

    out += ["**Marginal value by rank** (cp; mean ±CI, median):", "",
            "| rank | " + " | ".join(str(r) for r in ranks) + " |",
            "|---|" + "---|" * len(ranks)]
    cells = []
    for r in ranks:
        m, c, n = mean_ci(sel(rank=r))
        cells.append("%+.0f ±%.0f<br>med %+.0f<br>n=%d" % (m, c, median(sel(rank=r)), n) if n >= 2 else "—")
    out += ["| all | " + " | ".join(cells) + " |", ""]

    out += ["**Q2 — does one rank of advancement change value, and by how much?**", ""]
    for r in ranks[1:]:
        if r - 1 in ranks:
            out.append("- rank %d vs %d: %s" % (r, r - 1, verdict(sel(rank=r), sel(rank=r - 1),
                                                                  "rank %d" % r, "rank %d" % (r - 1))))
    out.append("")

    out += ["**Q1/Q3 — weak pawn on rank r vs STRONG pawn on rank r−1**", ""]
    any_case = False
    for r in ranks[1:]:
        if r - 1 not in ranks:
            continue
        w, s = sel(rank=r, friendly=WEAK), sel(rank=r - 1, friendly=STRONG)
        if len(w) >= 2 and len(s) >= 2:
            any_case = True
            out.append("- weak@%d vs strong@%d: %s" % (r, r - 1, verdict(w, s, "weak@%d" % r,
                                                                         "strong@%d" % (r - 1))))
    if not any_case:
        out.append("- (no rank pair has both cells populated in this set)")
    out.append("")

    obs = sorted({r["enemy"] for r in rows}) if rows and "enemy" in rows[0] else []
    if obs:
        out += ["**Q4a — obstruction ordering** (what is IN FRONT of the pawn):", ""]
        for e in obs:
            m, c, n = mean_ci(sel(enemy=e))
            if n >= 2:
                out.append("- `%s`: %+.0f ±%.0f (med %+.0f, n=%d)" % (e, m, c, median(sel(enemy=e)), n))
        out.append("")
        out += ["**Q4b — does structure matter more when the pawn is stuck?**", ""]
        for e in obs:
            out.append("- `%s`: %s" % (e, verdict(sel(enemy=e, friendly=STRONG), sel(enemy=e, friendly=WEAK),
                                                  "strong", "weak")))
        out.append("")

    files = sorted({r["file"] for r in rows}) if rows and "file" in rows[0] else []
    if len(files) > 1:
        out += ["**Q4c — file** (centre vs edge):", ""]
        for f in files:
            vals = [r["value"] for r in rows if r["file"] == f]
            m, c, n = mean_ci(vals)
            if n >= 2:
                out.append("- file `%s`: %+.0f ±%.0f (med %+.0f, n=%d)" % ("abcdefgh"[int(f)], m, c,
                                                                           median(vals), n))
        out.append("")
    return out


def main():
    sets = sorted(glob.glob(os.path.join(THIS, "ks_sets", "pawn_truth*.csv")))
    if not sets:
        sys.exit("no pawn_truth*.csv found -- run pawn_truth_generator.py first")

    body = [BEGIN,
            "<!-- GENERATED by diagnostics/pawn_truth_casebook.py -- DO NOT EDIT BY HAND. Regenerate with:",
            "     pyrun diagnostics/pawn_truth_casebook.py WRITE=1 -->",
            "",
            "⚠️ **All figures below come from MANUFACTURED positions unless a source says otherwise.**",
            "Manufactured magnitudes have been shown wrong by 2.6× against real positions (§8), so treat the",
            "ORDERING as informative and the MAGNITUDES as provisional. A contrast marked **UNRESOLVED** is",
            "not a small effect — it is an absence of evidence.", ""]
    used, skipped = 0, []
    for p in sets:
        name = os.path.basename(p)
        if name in QUARANTINE:
            skipped.append(name)
            continue
        rows = load(p)
        if len(rows) >= 20:
            body += section(rows, name)
            used += 1
    if skipped:
        body += ["", "☠️ **Quarantined and excluded** (produced before the generator defects were fixed): "
                 + ", ".join("`%s`" % s for s in skipped) + ". See §8.", ""]
    body.append(END)
    text = "\n".join(body)

    if not WRITE:
        print(text)
        print("\n(dry run — pass WRITE=1 to inject into PAWN_MODEL.md)", file=sys.stderr)
        return

    doc = open(DOC, encoding="utf-8").read()
    if BEGIN in doc and END in doc:
        pre, rest = doc.split(BEGIN, 1)
        _, post = rest.split(END, 1)
        doc = pre + text + post
    else:
        sys.exit("markers not found in %s -- add %s / %s around section 4" % (DOC, BEGIN, END))
    open(DOC, "w", encoding="utf-8").write(doc)
    print("injected case book into %s (%d sets used, %d quarantined)" % (DOC, used, len(skipped)))


if __name__ == "__main__":
    main()
