# -*- coding: utf-8 -*-
"""Where does SF11-static TRACK SF18-search while WE miss it, and is the gap KING SAFETY?

The owner's method, made into an instrument. SF18-search is the truth. SF11/SF15 static are not truth --
they are PROOF that a swing is STATICALLY ENCODABLE: when a handcrafted static eval lands near SF18 and we
do not, that swing is reachable without search, and SF11's breakdown shows HOW. They get close far more
often than we do, so they are the teacher.

My earlier case-set asked the wrong question ("does our KS term match SF11's KS term") and answered it
against a weak reference, in absolute value, with endgames mixed in. This asks:

  LEARNABLE   |SF11_total - SF18| <= NEAR  (SF11 nailed it)  AND  |ours - SF18| >= MISS  (we blew it)
  KS-LIVE     a queen is on the board (king-attack territory; endgames where KS is tapered are excluded)
  KS-CARRIED  the King-safety term is the largest single ours-vs-SF11 term gap on that position

For the KS-CARRIED learnable set we print: our KS term vs SF11's, our per-king units (det_ks_units_w/b),
and SF11's KS value -- so we can read WHICH KING each engine charges and what detector we lack. Signed,
not absolute: a wrong-king error (we credit our attack while SF sees danger to our own king) shows as an
opposite sign, which absolute value hid.

⚠️ SF11 cannot evaluate an in-check position (returns None) -> skipped, counted.
⚠️ All three evals are cheap/static-or-shallow, so this runs in ~1 min (unlike our-engine-at-d18).

  pyrun diagnostics/_ks_learnable.py [N=120] [NEAR=1.25] [MISS=2.0] [DEPTH=16]
"""
import os, sys, csv

for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
THIS = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(THIS)
sys.path.insert(0, ENGINE); sys.path.insert(0, THIS)
sys.path.insert(0, os.path.join(ENGINE, "selfplay"))

N = int(os.environ.get("N", "120"))
NEAR = float(os.environ.get("NEAR", "1.25"))
MISS = float(os.environ.get("MISS", "2.0"))
DEPTH = int(os.environ.get("DEPTH", "16"))
FAMILY = os.environ.get("FAMILY", "vssf_2400")

import chess, chess.engine
from ChessAI import ChessAI
from eval_vs_sf11 import SF11Eval, SF11
from arbiter import find_stockfish

# Clean-comparable term pairs (SF folds PSQT into Material; our capture_gains has no analogue).
PAIRS = [
    ("KingSafety", "king_safety",         "King safety"),
    ("Material",   "material",            "Material"),
    ("Threats",    "threats",             "Threats"),
    ("Passed",     "passed_pawn_support", "Passed"),
    ("Space",      "central",             "Space"),
    ("Imbalance",  "kaufman_imbalance",   "Imbalance"),
]

seen, fens = set(), []
for r in csv.DictReader(open(os.path.join(THIS, "ks_sets", "collapse_dataset_classified.csv"))):
    if r.get("family") != FAMILY or r.get("ks_class") != "positional":
        continue
    f = (r.get("decision_fen") or "").strip()
    if f and f not in seen:
        seen.add(f); fens.append(f)
fens = fens[:N]

ai = ChessAI(None, None, chess.Board(), True)
sf11 = SF11Eval(SF11)
sf18 = chess.engine.SimpleEngine.popen_uci(find_stockfish())

learnable_ks, learnable_other, checked, n_live, n_total = [], [], 0, 0, 0
for i, fen in enumerate(fens, 1):
    b = chess.Board(fen)
    try:
        bd = ai.ev_breakdown(b)
        sf_total, sf_terms = sf11.eval(fen)
        sc = sf18.analyse(b, chess.engine.Limit(depth=DEPTH))["score"].white()
        s18 = 99.0 if (sc.is_mate() and sc.mate() > 0) else (-99.0 if sc.is_mate() else sc.score() / 100.0)
    except Exception:
        continue
    if i % 25 == 0:
        print("    %d/%d" % (i, len(fens)), flush=True)
    n_total += 1
    if sf_total is None:
        checked += 1
        continue
    ours = -bd.get("total", 0) / 1000.0
    # LEARNABLE: SF11 tracks the truth, we do not.
    if abs(sf_total - s18) > NEAR or abs(ours - s18) < MISS:
        continue
    live = bool(b.pieces(chess.QUEEN, chess.WHITE) or b.pieces(chess.QUEEN, chess.BLACK))
    if live:
        n_live += 1
    # Which term carries the ours-vs-SF11 gap?
    gaps = [(lbl, (-bd.get(ok, 0) / 1000.0) - sf_terms.get(sk, 0.0)) for lbl, ok, sk in PAIRS]
    gaps.sort(key=lambda kv: abs(kv[1]), reverse=True)
    top_lbl, top_gap = gaps[0]
    rec = dict(fen=fen, ours=ours, sf11=sf_total, s18=s18, live=live, top=top_lbl, topgap=top_gap,
               our_ks=-bd.get("king_safety", 0) / 1000.0, sf_ks=sf_terms.get("King safety", 0.0),
               ksw=bd.get("det_ks_units_w", 0), ksb=bd.get("det_ks_units_b", 0), gaps=gaps)
    (learnable_ks if (live and top_lbl == "KingSafety") else learnable_other).append(rec)

sf18.quit(); sf11.close()

print("\n" + "=" * 100)
print("  LEARNABLE SET  (|SF11-SF18| <= %.2f  AND  |ours-SF18| >= %.2f)   from %d scored, %d in-check skipped"
      % (NEAR, MISS, n_total, checked))
print("  KS-CARRIED (queens on, King-safety is the top term-gap):  %d" % len(learnable_ks))
print("  other-carried learnable:                                  %d" % len(learnable_other))
print("=" * 100)

print("\n  === KS-CARRIED LEARNABLE (study SF11's KS mechanism on these) ===")
print("  %8s %8s %8s | %7s %7s | %5s %5s | %s" % ("ours", "sf11", "sf18", "ourKS", "sf11KS", "uW", "uB", "read"))
for r in sorted(learnable_ks, key=lambda r: -abs(r["ours"] - r["s18"])):
    # Signed: opposite sign = we charge the WRONG king vs SF11.
    if r["our_ks"] * r["sf_ks"] < 0 and abs(r["sf_ks"]) > 0.3:
        read = "WRONG KING (opposite sign to SF11)"
    elif abs(r["sf_ks"]) - abs(r["our_ks"]) > 0.5:
        read = "UNDER (SF11 sees more danger)"
    elif abs(r["our_ks"]) - abs(r["sf_ks"]) > 0.5:
        read = "OVER"
    else:
        read = "close on KS; gap is elsewhere in total"
    print("  %+8.2f %+8.2f %+8.2f | %+7.2f %+7.2f | %5d %5d | %s"
          % (r["ours"], r["sf11"], r["s18"], r["our_ks"], r["sf_ks"], r["ksw"], r["ksb"], read))
    print("       %s" % r["fen"])

print("\n  === other-carried learnable (the gap is NOT king safety) — top term per position ===")
from collections import Counter
c = Counter(r["top"] for r in learnable_other)
for lbl, n in c.most_common():
    print("    %-12s %d" % (lbl, n))
print()
