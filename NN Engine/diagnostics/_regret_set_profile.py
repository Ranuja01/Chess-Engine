# -*- coding: utf-8 -*-
"""Profile and COMPARE regret sets to understand WHY they disagree — pure CSV analysis (no engine, no contamination).
Side-by-side distributions of: phase bucket, non-pawn material, criticality (SF best-vs-2nd win% gap), and
eval-state (|best_cp|). If the sets are composed differently, that explains config-effect disagreements; if the
compositions match, the disagreement is sampling variance in the sparse critical bands (=> need more critical data).

  pyrun diagnostics/_regret_set_profile.py SETS=ks_sets/game_regret_set.csv,ks_sets/game_regret_set_v2.csv
"""
import os, sys, csv, math
for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1); os.environ[_k] = _v
THIS = os.path.dirname(os.path.abspath(__file__))


def _winpct(cp):
    return 50.0 + 50.0 * (2.0 / (1.0 + math.exp(-0.00368208 * cp)) - 1.0)


_NPV = {'n': 3, 'b': 3, 'r': 5, 'q': 9}
MB = [(0, 6, "npm 0-6"), (7, 12, "npm 7-12"), (13, 19, "npm 13-19"), (20, 27, "npm 20-27"),
      (28, 40, "npm 28-40"), (41, 999, "npm 41+")]
CB = [(0, 3, "benign  <3%"), (3, 8, "minor 3-8%"), (8, 20, "moder 8-20%"), (20, 1e9, "CRIT >20%")]
EB = [(0, 50, "equal <=50"), (50, 150, "slight 50-150"), (150, 400, "clear 150-400"), (400, 1e9, "won >400")]


def profile(path):
    rows = list(csv.DictReader(open(path, newline="")))
    n = 0
    ph, mat, crit, est = {}, {}, {}, {}
    for r in rows:
        fen = r.get("fen", "")
        try:
            bcp = float(r["best_cp"])
        except Exception:
            continue
        n += 1
        ph[r.get("phase_bucket", "?")] = ph.get(r.get("phase_bucket", "?"), 0) + 1
        board = fen.split()[0] if fen else ""
        npm = sum(_NPV.get(c.lower(), 0) for c in board if c.lower() in _NPV)
        for lo, hi, lab in MB:
            if lo <= npm <= hi:
                mat[lab] = mat.get(lab, 0) + 1; break
        mm = {}
        for pair in (r.get("moves") or "").split(";"):
            if ":" in pair:
                u, c = pair.rsplit(":", 1)
                try:
                    mm[u] = float(c)
                except Exception:
                    pass
        stm_white = (fen.split()[1] == 'w') if len(fen.split()) > 1 else True
        wps = sorted(((_winpct(c) if stm_white else 100.0 - _winpct(c)) for c in mm.values()), reverse=True)
        cr = (wps[0] - wps[1]) if len(wps) >= 2 else 0.0
        for lo, hi, lab in CB:
            if lo <= cr < hi:
                crit[lab] = crit.get(lab, 0) + 1; break
        acp = abs(bcp)
        for lo, hi, lab in EB:
            if lo <= acp < hi:
                est[lab] = est.get(lab, 0) + 1; break
    return n, ph, mat, crit, est


sets = [s for s in os.environ.get("SETS", "").split(",") if s]
profs = []
for s in sets:
    p = s if os.path.isabs(s) else os.path.join(THIS, s)
    profs.append((os.path.basename(s), profile(p)))


def show(title, order, key):
    print("\n%s" % title)
    hdr = "  %-16s" % "" + "".join("%18s" % nm for nm, _ in profs)
    print(hdr)
    for lab in order:
        line = "  %-16s" % lab
        for _, (n, ph, mat, crit, est) in profs:
            d = {0: ph, 1: mat, 2: crit, 3: est}[key]
            c = d.get(lab, 0)
            line += "%10d (%4.1f%%)" % (c, 100.0 * c / max(1, n))
        print(line)


print("REGRET-SET PROFILE COMPARISON (pure CSV; %s)" % " vs ".join(nm for nm, _ in profs))
print("  %-16s" % "TOTAL" + "".join("%18d" % p[0] for _, p in profs))
show("PHASE bucket:", ["opening", "midgame", "endgame"], 0)
show("NON-PAWN MATERIAL:", [lab for _, _, lab in MB], 1)
show("CRITICALITY (SF best-vs-2nd):", [lab for _, _, lab in CB], 2)
show("EVAL-STATE (|best_cp|):", [lab for _, _, lab in EB], 3)
print("\n  read: if the columns MATCH, the sets sample the same distribution -> disagreement is sampling variance in\n"
      "  the sparse critical band (need more critical data). If they DIFFER, the sets measure different populations.", flush=True)
