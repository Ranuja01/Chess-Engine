# -*- coding: utf-8 -*-
"""Per-position win%-error PROFILE of one knob arm vs the defaults.

`fit_bench_guarded` reports a single corpus MSE, which cannot distinguish "helps everything a little" from
"fixes a few disasters and breaks other things". That distinction decides whether an arm is worth targeting
or worth killing: a term that repairs the big errors and costs a little on the small ones is a CANDIDATE
(gate the small cases), while one that shaves everything uniformly is just a rescale.

Reports, over the same corpus the fit uses: how many positions improve vs worsen, the size of each, and the
split by tier / phase bucket / BASELINE-ERROR DECILE (the last answers "is it fixing the disasters?").

Knobs latch at extension init, so each config gets its own process; the parent re-invokes itself.

  pyrun diagnostics/_arm_error_profile.py CORPUS=diverse_corpus_wide.csv ENABLE_THREATS=1 ... \
        [ARMLABEL=so75_cap800]
Every KEY=VAL that is not CORPUS/ARMLABEL/CHILD is applied to the CANDIDATE arm only.
"""
import os, sys, csv, math, subprocess

RESERVED = ("CORPUS", "ARMLABEL", "CHILD")
ARM = {}
for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v
        if _k not in RESERVED:
            ARM[_k] = _v

THIS = os.path.dirname(os.path.abspath(__file__))
_c = os.environ.get("CORPUS", "diverse_corpus_wide.csv")
CORPUS = _c if ("/" in _c or os.sep in _c) else os.path.join(THIS, "ks_sets", _c)
LABEL = os.environ.get("ARMLABEL", "candidate")


def winpct(cp):
    return 50.0 + 50.0 * (2.0 / (1.0 + math.exp(-0.00368208 * cp)) - 1.0)


def child():
    sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
    import chess
    from ChessAI import ChessAI
    ai = ChessAI(None, None, chess.Board(), True)
    for i, r in enumerate(csv.DictReader(open(CORPUS))):
        try:
            bd = ai.ev_breakdown(chess.Board(r["fen"]))
            print("%d,%.6f" % (i, -bd.get("total", 0.0) / 1000.0))
        except Exception:
            continue


def run(knobs):
    # The parent folded every KEY=VAL into its OWN environment while parsing argv, so a child inheriting
    # os.environ verbatim would run the ARM even when asked for the baseline -- yielding a zero delta on
    # every row. Strip the arm keys first and re-add only what this call asks for.
    env = {k: v for k, v in os.environ.items() if k not in ARM}
    env["CHILD"] = "1"
    env.update(knobs)
    args = [sys.executable, os.path.abspath(__file__), "CHILD=1", "CORPUS=" + CORPUS] + \
           ["%s=%s" % (k, v) for k, v in knobs.items()]
    out = subprocess.run(args, capture_output=True, text=True, env=env).stdout
    d = {}
    for line in out.splitlines():
        p = line.strip().split(",")
        if len(p) == 2 and p[0].isdigit():
            d[int(p[0])] = float(p[1])
    return d


def report(title, groups):
    print("\n%s" % title)
    print("  %-22s %6s %8s %8s %10s" % ("bucket", "n", "better", "worse", "mean dErr"))
    for k in sorted(groups):
        g = groups[k]
        n = len(g)
        if not n:
            continue
        better = sum(1 for d in g if d < -1e-9)
        worse = sum(1 for d in g if d > 1e-9)
        print("  %-22s %6d %8d %8d %+10.2f" % (k, n, better, worse, sum(g) / n))


def main():
    if os.environ.get("CHILD"):
        child()
        return
    if not ARM:
        sys.exit("no candidate knobs given (everything except CORPUS/ARMLABEL is the arm)")
    rows = list(csv.DictReader(open(CORPUS)))
    print("corpus=%s  rows=%d  arm=%s  knobs=%s"
          % (os.path.basename(CORPUS), len(rows), LABEL,
             " ".join("%s=%s" % kv for kv in sorted(ARM.items()))))

    base = run({})
    cand = run(ARM)
    keys = sorted(set(base) & set(cand))
    if not keys:
        sys.exit("no positions scored")

    per, by_tier, by_phase, by_dec = [], {}, {}, {}
    pairs = []
    for k in keys:
        r = rows[k]
        try:
            tgt = float(r["target_total"])
        except (KeyError, ValueError):
            continue
        t = winpct(tgt * 100.0)
        eb = (winpct(base[k] * 100.0) - t) ** 2
        ec = (winpct(cand[k] * 100.0) - t) ** 2
        pairs.append((eb, ec - eb, r.get("tier", "?"), r.get("phase_bucket", "?")))

    pairs.sort(key=lambda x: -x[0])           # worst baseline error first
    n = len(pairs)
    for i, (eb, d, tier, ph) in enumerate(pairs):
        per.append(d)
        by_tier.setdefault(tier, []).append(d)
        by_phase.setdefault(ph, []).append(d)
        dec = "D%d %s" % (i * 10 // n, "(worst)" if i * 10 // n == 0 else "")
        by_dec.setdefault(dec, []).append(d)

    better = sum(1 for d in per if d < -1e-9)
    worse = sum(1 for d in per if d > 1e-9)
    print("\nOVERALL  n=%d   better=%d (%.0f%%)   worse=%d (%.0f%%)   mean dErr=%+.2f"
          % (n, better, 100.0 * better / n, worse, 100.0 * worse / n, sum(per) / n))
    print("  (dErr < 0 = the arm REDUCES win%%-squared error on that position)")
    print("  total error removed from improved positions: %+.0f" % sum(d for d in per if d < 0))
    print("  total error added  by  worsened positions:  %+.0f" % sum(d for d in per if d > 0))

    report("BY BASELINE-ERROR DECILE (D0 = positions we get MOST wrong today)", by_dec)
    report("BY TIER", by_tier)
    report("BY PHASE BUCKET", by_phase)
    print("\nReading it: concentrated gains in D0-D2 with small losses elsewhere = a TARGETABLE fix (gate the")
    print("cases it hurts). Uniform small gains everywhere = a rescale that STS will not reward.")


if __name__ == '__main__':
    main()
