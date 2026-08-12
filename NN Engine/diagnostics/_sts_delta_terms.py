# -*- coding: utf-8 -*-
"""Which EVAL TERM is responsible for the STS points a change wins and loses?

A balanced-STS delta is one scalar, and "the cost is diffuse" was asserted here from the ABSENCE of a
clamp-saturation signal -- which is not evidence of diffuseness, only absence of that one mechanism.
The STS harness already writes per-position results (`sts_results_<tag>.csv`: fen, our move, score), so
the delta is directly decomposable into the positions that changed and the terms that moved in them.

⚠️ Ordering only. Colour asymmetry is a defect whatever this reports -- eval(mirror(b)) != -eval(b) is
wrong on its face and corrupts every constant fitted over it. This ranks which fish to fry, it does NOT
decide whether a correctness fix is worth keeping.

Three steps, because engine knobs latch at init (ONE PROCESS PER SETTING):

  1. list     which positions changed score, and by how much
     pyrun diagnostics/_sts_delta_terms.py MODE=list BASE=symbase CAND=fix4 OUT=/tmp/chg.csv

  2. dump     per-term breakdown for those positions, once per setting
     pyrun diagnostics/_sts_delta_terms.py MODE=dump IN=/tmp/chg.csv OUT=/tmp/t_base.csv
     pyrun diagnostics/_sts_delta_terms.py MODE=dump IN=/tmp/chg.csv OUT=/tmp/t_cand.csv <KNOBS>

  3. compare  aggregate term movement over LOSERS vs GAINERS
     pyrun diagnostics/_sts_delta_terms.py MODE=compare IN=/tmp/chg.csv A=/tmp/t_base.csv B=/tmp/t_cand.csv

★ The GAINERS are the control and must be carried: the bundle is a NET, so it has winners too. A term
that moves the same way on both sides is not the story -- it is just the term that moved.
"""
import os, sys, csv, math

for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)

MODE = os.environ.get("MODE", "list")
RESULTS = os.path.join(THIS, "results")


def winpct(cp):
    return 50.0 + 50.0 * (2.0 / (1.0 + math.exp(-0.00368208 * cp)) - 1.0)


def mode_list():
    base = {r["fen"]: r for r in csv.DictReader(open(
        os.path.join(RESULTS, "sts_results_%s.csv" % os.environ["BASE"]), newline=""))}
    cand = {r["fen"]: r for r in csv.DictReader(open(
        os.path.join(RESULTS, "sts_results_%s.csv" % os.environ["CAND"]), newline=""))}
    rows, up, dn = [], 0, 0
    for fen, c in cand.items():
        b = base.get(fen)
        if not b or b["score"] == "" or c["score"] == "":
            continue
        d = int(c["score"]) - int(b["score"])
        if d == 0:
            continue
        rows.append({"fen": fen, "delta": d, "base_move": b["engine"], "cand_move": c["engine"],
                     "theme": c.get("theme", ""), "max": c.get("max", "")})
        up += d > 0
        dn += d < 0
    with open(os.environ["OUT"], "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["fen", "delta", "base_move", "cand_move", "theme", "max"])
        w.writeheader(); w.writerows(rows)
    tot = sum(r["delta"] for r in rows)
    print("changed positions %d of %d   gainers %d   losers %d   net %+d"
          % (len(rows), len(cand), up, dn, tot))
    print("-> %s" % os.environ["OUT"])


def mode_dump():
    import chess
    from ChessAI import ChessAI
    ai = ChessAI(None, None, chess.Board(), True)
    rows = list(csv.DictReader(open(os.environ["IN"], newline="")))
    out, keys = [], set()
    for r in rows:
        try:
            d = ai.ev_breakdown(chess.Board(r["fen"]))
        except Exception:
            continue
        d = {k: v for k, v in d.items() if isinstance(v, int)}
        d["fen"] = r["fen"]
        keys |= set(d)
        out.append(d)
    keys = ["fen"] + sorted(k for k in keys if k != "fen")
    with open(os.environ["OUT"], "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for d in out:
            w.writerow({k: d.get(k, "") for k in keys})
    print("dumped %d positions -> %s" % (len(out), os.environ["OUT"]))


def mode_compare():
    chg = {r["fen"]: int(r["delta"]) for r in csv.DictReader(open(os.environ["IN"], newline=""))}
    A = {r["fen"]: r for r in csv.DictReader(open(os.environ["A"], newline=""))}
    B = {r["fen"]: r for r in csv.DictReader(open(os.environ["B"], newline=""))}

    terms = sorted(set(next(iter(A.values())).keys()) - {"fen"})
    # Separate accumulators: a term that moves the SAME way on winners and losers is not the culprit,
    # it is merely the term the change touches. Only a term that separates them explains the net.
    acc = {t: {"lose": 0.0, "win": 0.0, "ln": 0, "wn": 0} for t in terms}
    for fen, d in chg.items():
        a, b = A.get(fen), B.get(fen)
        if not a or not b:
            continue
        side = "lose" if d < 0 else "win"
        for t in terms:
            try:
                delta = int(b[t]) - int(a[t])
            except Exception:
                continue
            if delta:
                acc[t][side] += abs(delta)
                acc[t]["ln" if side == "lose" else "wn"] += 1

    print("TERM MOVEMENT ON CHANGED STS POSITIONS  (|cand - base| summed, millipawns)\n")
    print("  %-26s %12s %6s %12s %6s %10s" % ("term", "on LOSERS", "n", "on GAINERS", "n", "L-per-pos"))
    ranked = sorted(terms, key=lambda t: -acc[t]["lose"])
    for t in ranked[:18]:
        v = acc[t]
        if not v["lose"] and not v["win"]:
            continue
        print("  %-26s %12.0f %6d %12.0f %6d %10.1f"
              % (t, v["lose"], v["ln"], v["win"], v["wn"],
                 v["lose"] / v["ln"] if v["ln"] else 0.0))
    print("\nREADING IT")
    print("  A term high on LOSERS and low on GAINERS is the candidate culprit.")
    print("  A term high on BOTH is just what the change touches -- not an explanation.")
    print("  If no term separates them, the cost really is spread across the eval and only games decide.")


{"list": mode_list, "dump": mode_dump, "compare": mode_compare}[MODE]()
