# -*- coding: utf-8 -*-
"""Does a candidate reduce error WHERE THE DEFECT IS without raising it EVERYWHERE ELSE?

🚨 The regression detector we never had. Every KS candidate that "fixed collapses" and then lost Elo did
the same thing: it improved the ~79-position TARGET set and quietly degraded the general distribution the
games are actually played on. Validating on the target set alone cannot see that, and neither can the
signed mean, the corpus MSE, or STS -- all three were blind to the bidirectional error that
[[every-eval-term-error-is-bidirectional]] found.

So: measure mean |gap| vs SF11-static on TWO sets, baseline vs arm.
  TARGET  = positional collapse decision-FENs (where the defect lives)
  HOLDOUT = general positions from the wide corpus (where the regressions hide)

  target |gap| DOWN and holdout |gap| flat-or-down  -> real improvement
  target |gap| DOWN and holdout |gap| UP            -> the historical failure, caught for free

⚠️ This is a CHECK, never an OBJECTIVE. Fitting to a proxy is what produced −85.6 Elo; measuring against
one is fine. Do not put this in a descent loop.
⚠️ |gap| is orientation-free, so no collapsing-side sign convention to get wrong.
⚠️ SF11 cannot evaluate a position whose side to move is IN CHECK (returns None) -> skipped, and counted.
⚠️ Knobs latch at engine init, so the arm runs in its OWN process (one process per setting).

  pyrun diagnostics/_gap_two_set.py ARM=KS_DYN=64 [NT=80] [NH=200] [DEPTH=12]
"""
import os, sys, csv, subprocess

for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
THIS = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(THIS)
sys.path.insert(0, ENGINE); sys.path.insert(0, THIS)
sys.path.insert(0, os.path.join(ENGINE, "selfplay"))

NT = int(os.environ.get("NT", "80"))
NH = int(os.environ.get("NH", "200"))
DEPTH = int(os.environ.get("DEPTH", "12"))
FAMILY = os.environ.get("FAMILY", "vssf_2400")
CLASS = os.environ.get("CLASS", "positional")

PAIRS = [
    ("KingSafety", "king_safety",         "King safety"),
    ("Material",   "material",            "Material"),
    ("Threats",    "threats",             "Threats"),
    ("Passed",     "passed_pawn_support", "Passed"),
    ("Space",      "central",             "Space"),
    ("Imbalance",  "kaufman_imbalance",   "Imbalance"),
]


def load_sets():
    cls = os.path.join(THIS, "ks_sets", "collapse_dataset_classified.csv")
    seen, target = set(), []
    for r in csv.DictReader(open(cls)):
        if r.get("family") != FAMILY or r.get("ks_class") != CLASS:
            continue
        f = (r.get("decision_fen") or "").strip()
        if f and f not in seen:
            seen.add(f); target.append(f)
    wide = os.path.join(THIS, "ks_sets", "diverse_corpus_wide.csv")
    hold = []
    for r in csv.DictReader(open(wide)):
        f = (r.get("fen") or "").strip()
        # Never let a target position leak into the holdout -- that would mask exactly the trade we are
        # trying to detect.
        if f and f not in seen:
            hold.append(f)
    return target[:NT], hold[:NH]


def run_worker(out_path):
    import chess, chess.engine
    from ChessAI import ChessAI
    from eval_vs_sf11 import SF11Eval, SF11
    from arbiter import find_stockfish

    ai = ChessAI(None, None, chess.Board(), True)
    sf11 = SF11Eval(SF11)
    sf18 = chess.engine.SimpleEngine.popen_uci(find_stockfish())
    target, hold = load_sets()

    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["set", "fen", "term", "gap"])
        for setname, fens in (("target", target), ("holdout", hold)):
            for i, fen in enumerate(fens, 1):
                if i % 25 == 0:
                    print("    %s %d/%d" % (setname, i, len(fens)), flush=True)
                try:
                    b = chess.Board(fen)
                    bd = ai.ev_breakdown(b)
                    sf_total, sf_terms = sf11.eval(fen)
                    if sf_total is None:
                        continue
                    sc = sf18.analyse(b, chess.engine.Limit(depth=DEPTH))["score"].white()
                    s18 = 99.0 if (sc.is_mate() and sc.mate() > 0) else (-99.0 if sc.is_mate() else sc.score() / 100.0)
                except Exception:
                    continue
                if (sf_total > 0) != (s18 > 0):
                    continue
                our_total = -bd.get("total", 0) / 1000.0
                w.writerow([setname, fen, "TOTAL", our_total - sf_total])
                for lbl, ourk, sfk in PAIRS:
                    w.writerow([setname, fen, lbl, (-bd.get(ourk, 0) / 1000.0) - sf_terms.get(sfk, 0.0)])
    sf18.quit(); sf11.close()


if os.environ.get("WORKER") == "1":
    run_worker(os.environ["OUT"])
    sys.exit(0)

ARM = os.environ.get("ARM", "")
if not ARM:
    sys.exit("ARM=<KNOB=VAL[,KNOB=VAL...]> is required")

# Written into the workspace, not /tmp: /tmp is wiped between dispatcher calls.
base_out = os.path.join(THIS, "_gap_base.csv")
arm_out = os.path.join(THIS, "_gap_arm.csv")
for out_path, knobs in ((base_out, []), (arm_out, ARM.split(","))):
    print("\n  === %s ===" % ("BASELINE" if not knobs else ARM), flush=True)
    cmd = [sys.executable, "-u", os.path.abspath(__file__), "WORKER=1", "OUT=" + out_path,
           "NT=%d" % NT, "NH=%d" % NH, "DEPTH=%d" % DEPTH,
           "FAMILY=" + FAMILY, "CLASS=" + CLASS] + knobs
    subprocess.run(cmd, cwd=ENGINE, check=True)


def load(path):
    d = {}
    for r in csv.DictReader(open(path, newline="")):
        d.setdefault((r["set"], r["term"]), []).append(abs(float(r["gap"])))
    return d


base, arm = load(base_out), load(arm_out)
print("\n  arm: %s   target=%s/%s  holdout=general  (mean |gap| vs SF11, pawns)\n" % (ARM, FAMILY, CLASS))
print("  %-12s %22s %22s" % ("", "TARGET (defect lives)", "HOLDOUT (regressions hide)"))
print("  %-12s %8s %8s %6s %8s %8s %6s" % ("term", "base", "arm", "d", "base", "arm", "d"))
for term in ["TOTAL"] + [p[0] for p in PAIRS]:
    tb, ta = base.get(("target", term), []), arm.get(("target", term), [])
    hb, ha = base.get(("holdout", term), []), arm.get(("holdout", term), [])
    if not tb or not hb:
        continue
    f = lambda v: sum(v) / len(v)
    print("  %-12s %8.3f %8.3f %+6.3f %8.3f %8.3f %+6.3f"
          % (term, f(tb), f(ta), f(ta) - f(tb), f(hb), f(ha), f(ha) - f(hb)))
print("\n  n: target %d, holdout %d (positions surviving the SF11-direction gate)"
      % (len(base.get(("target", "TOTAL"), [])), len(base.get(("holdout", "TOTAL"), []))))
print("  ✅ WANT: TOTAL target d NEGATIVE and holdout d ~0 or negative.")
print("  ☠️ THE HISTORICAL FAILURE: target d negative, holdout d POSITIVE.\n")
