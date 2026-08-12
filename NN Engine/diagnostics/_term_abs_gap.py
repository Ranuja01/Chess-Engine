# -*- coding: utf-8 -*-
"""Per-term gap vs SF11-static reported as MEAN ABSOLUTE error, not mean signed error.

🚨 Why this exists. The signed mean is the wrong statistic for a term whose error changes SIGN. On the
2026-08-09 corpus the KingSafety signed gap was +0.192 -- near zero, ranked third -- while individual
positions ran -4.12 (we are SILENT when the enemy king is genuinely under attack) and +1.63 (we are LOUD
when it is not). A bidirectional error CANCELS in the mean and the term looks healthy.

That distinction decides what kind of fix is possible:
  large |gap|, small signed gap  -> DISCRIMINATION failure. A magnitude knob cannot fix it; scaling up
                                    helps one half of the corpus and hurts the other, netting to noise.
                                    This is the signature of every failed KS magnitude candidate.
  large |gap| ~= large signed gap -> CALIBRATION failure. A scale CAN fix it.

Reports both, plus the over/under split, so the two are never confused again.

⚠️ Term names are not 1:1 across engines (SF folds PSQT into Material; our capture_gains has no analogue).
Only the pairs below are clean enough to compare.

  pyrun diagnostics/_term_abs_gap.py [FAMILY=vssf_2400] [CLASS=positional] [N=200] [DEPTH=18]
"""
import os, sys, csv, time

for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
THIS = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(THIS)
sys.path.insert(0, ENGINE); sys.path.insert(0, THIS)
sys.path.insert(0, os.path.join(ENGINE, "selfplay"))

import chess, chess.engine
from ChessAI import ChessAI
from eval_vs_sf11 import SF11Eval, SF11
from arbiter import find_stockfish

FAMILY = os.environ.get("FAMILY", "vssf_2400")
CLASS = os.environ.get("CLASS", "positional")
NMAX = int(os.environ.get("N", "200"))
DEPTH = int(os.environ.get("DEPTH", "18"))

# (label, our breakdown key, SF11 term label)
PAIRS = [
    ("KingSafety", "king_safety",         "King safety"),
    ("Material",   "material",            "Material"),
    ("Threats",    "threats",             "Threats"),
    ("Passed",     "passed_pawn_support", "Passed"),
    ("Space",      "central",             "Space"),
    ("Imbalance",  "kaufman_imbalance",   "Imbalance"),
]

CSVP = os.path.join(THIS, "ks_sets", "collapse_dataset_classified.csv")
seen, fens = set(), []
for r in csv.DictReader(open(CSVP)):
    if r.get("family") != FAMILY or r.get("ks_class") != CLASS:
        continue
    f = (r.get("decision_fen") or "").strip()
    if f and f not in seen:
        seen.add(f); fens.append(f)
fens = fens[:NMAX]

ai = ChessAI(None, None, chess.Board(), True)
sf11 = SF11Eval(SF11)
sf18 = chess.engine.SimpleEngine.popen_uci(find_stockfish())

acc = {lbl: [] for lbl, _, _ in PAIRS}
nvalid = 0
nocheck = 0
print("  %d FENs to probe (SF11 first, then SF18 d%d)\n" % (len(fens), DEPTH), flush=True)
for idx, fen in enumerate(fens, 1):
    b = chess.Board(fen)
    # Per-stage progress, flushed. Without this a stall is indistinguishable from slowness -- and the
    # SF11 reader below blocks on readline() with NO timeout, so a binfmt hiccup hangs the run forever.
    t0 = time.time()
    print("  [%3d/%3d] sf11..." % (idx, len(fens)), end="", flush=True)
    try:
        bd = ai.ev_breakdown(b)
        sf_total, sf_terms = sf11.eval(fen)
        print(" %4.1fs sf18..." % (time.time() - t0), end="", flush=True)
        t1 = time.time()
        sc = sf18.analyse(b, chess.engine.Limit(depth=DEPTH))["score"].white()
        s18 = 99.0 if (sc.is_mate() and sc.mate() > 0) else (-99.0 if sc.is_mate() else sc.score() / 100.0)
        print(" %4.1fs" % (time.time() - t1), flush=True)
    except Exception as e:
        print("  SKIP (%s)" % type(e).__name__, flush=True)
        continue
    # SF11's `eval` refuses to score a position whose side to move is IN CHECK -- it prints
    # "Total evaluation: none (in check)" and the parser yields None. Those positions are simply
    # invisible to any SF11 comparison, which is worth remembering: our reference ladder has a blind
    # spot on exactly the sharpest positions.
    if sf_total is None:
        nocheck += 1
        continue
    # Validity gate, same as the dossier: only trust positions where SF11-static already agrees with
    # SF18-search on DIRECTION. Where SF11 is also wrong, the gap is not a portable lesson.
    if (sf_total > 0) != (s18 > 0):
        continue
    nvalid += 1
    for lbl, ourk, sfk in PAIRS:
        ours = -bd.get(ourk, 0) / 1000.0          # our millipawns (Black-positive) -> White-POV pawns
        acc[lbl].append(ours - sf_terms.get(sfk, 0.0))

sf18.quit(); sf11.close()

print("\n  family=%s class=%s   %d FENs, %d valid (SF11 dir == SF18 dir), %d unscorable by SF11 (in check)\n"
      % (FAMILY, CLASS, len(fens), nvalid, nocheck))
print("  %-12s %10s %10s %8s %8s   %s" % ("term", "mean|gap|", "mean gap", "over", "under", "verdict"))
rows = []
for lbl, _, _ in PAIRS:
    v = acc[lbl]
    if not v:
        continue
    mabs = sum(abs(x) for x in v) / len(v)
    msig = sum(v) / len(v)
    over = sum(1 for x in v if x > 0.25)
    under = sum(1 for x in v if x < -0.25)
    rows.append((mabs, lbl, msig, over, under))
rows.sort(reverse=True)
for mabs, lbl, msig, over, under in rows:
    # A signed mean far below the absolute mean means the error flips sign across the corpus.
    verdict = "DISCRIMINATION (sign flips)" if abs(msig) < 0.5 * mabs else "calibration (one-signed)"
    print("  %-12s %10.3f %10.3f %8d %8d   %s" % (lbl, mabs, msig, over, under, verdict))
print()
