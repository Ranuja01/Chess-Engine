# -*- coding: utf-8 -*-
"""WHERE is our 'material' over-read, in OUR terms, relative to SF11 — pin it down instead of guessing.

We have chased 'material' for weeks without locating it, because the term boundaries do not line up:
SF folds piece placement (PSQT) INTO its Material/piece terms, while WE split the same fact across
`material` (raw piece values), `pt_*` (placement layers), `piece_value_boost` (a ratio amplifier SF has
NO analogue for), and `kaufman_imbalance`. So 'our Material vs SF Material' was never apples-to-apples and
probably mislabels the real driver.

On the LEARNABLE set (SF11 tracks SF18, we do not), this dumps OUR full material-family decomposition and
SF11's full breakdown side by side, then attributes the ours-vs-SF11 TOTAL gap across OUR sub-terms so we
can read which one actually carries it. Raw piece values are validated and expected to be ~fine; the
hypothesis is placement + piece_value_boost.

  pyrun diagnostics/_material_decompose.py [N=120] [NEAR=1.25] [MISS=2.0] [DEPTH=16]
"""
import os, sys, csv
from collections import defaultdict

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

import chess, chess.engine
from ChessAI import ChessAI
from eval_vs_sf11 import SF11Eval, SF11
from arbiter import find_stockfish

# Our material-and-placement family (White-POV pawns after -v/1000). These are the terms that, together,
# correspond to what SF prices inside Material + Imbalance + the per-piece placement/mobility terms.
OUR_FAMILY = ["material", "pt_pawns", "pt_knights", "pt_bishops", "pt_rooks", "pt_queens",
              "piece_value_boost", "kaufman_imbalance", "pieces", "capture_gains", "central"]

seen, fens = set(), []
for r in csv.DictReader(open(os.path.join(THIS, "ks_sets", "collapse_dataset_classified.csv"))):
    if r.get("family") != "vssf_2400" or r.get("ks_class") != "positional":
        continue
    f = (r.get("decision_fen") or "").strip()
    if f and f not in seen:
        seen.add(f); fens.append(f)
fens = fens[:N]

ai = ChessAI(None, None, chess.Board(), True)
sf11 = SF11Eval(SF11)
sf18 = chess.engine.SimpleEngine.popen_uci(find_stockfish())

contrib = defaultdict(float)     # sum of |per-term ours-vs-SF11 residual share| across the learnable set
signed = defaultdict(float)
n_learn = 0
rows = []
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
    if sf_total is None:
        continue
    ours = -bd.get("total", 0) / 1000.0
    if abs(sf_total - s18) > NEAR or abs(ours - s18) < MISS:
        continue
    n_learn += 1
    # Each of our family terms, White-POV pawns. The over-read = how much of (ours - sf11_total) each carries.
    fam = {k: -bd.get(k, 0) / 1000.0 for k in OUR_FAMILY}
    for k, v in fam.items():
        signed[k] += v
    rows.append((fen, ours, sf_total, s18, fam, sf_terms))

sf18.quit(); sf11.close()

print("\n" + "=" * 92)
print("  MATERIAL DECOMPOSITION over %d LEARNABLE positions (SF11 tracks SF18, we miss by >= %.1f)"
      % (n_learn, MISS))
print("=" * 92)
print("\n  Mean of each OUR term across the learnable set (White-POV pawns, + = credits White):")
for k in sorted(OUR_FAMILY, key=lambda k: -abs(signed[k])):
    print("    %-20s %+7.3f" % (k, signed[k] / max(1, n_learn)))

# SF11 side means for reference
sf_keys = ["Material", "Imbalance", "Pawns", "Knights", "Bishops", "Rooks", "Queens", "Space", "Mobility"]
sf_mean = defaultdict(float)
for (_, _, _, _, _, sf_terms) in rows:
    for k in sf_keys:
        sf_mean[k] += sf_terms.get(k, 0.0)
print("\n  Mean of each SF11 term for the same positions:")
for k in sf_keys:
    print("    %-20s %+7.3f" % (k, sf_mean[k] / max(1, n_learn)))

print("\n  === the 6 worst-over-read learnable positions, our family vs SF11 ===")
for (fen, ours, sf11t, s18, fam, sf_terms) in sorted(rows, key=lambda r: -abs(r[1] - r[3]))[:6]:
    print("\n  ours %+.2f  sf11 %+.2f  sf18 %+.2f   (over-read %+.2f)" % (ours, sf11t, s18, ours - s18))
    print("    %s" % fen)
    big = sorted(fam.items(), key=lambda kv: -abs(kv[1]))
    print("    ours: " + "  ".join("%s %+.2f" % (k, v) for k, v in big if abs(v) >= 0.15))
    sfb = sorted(((k, sf_terms.get(k, 0.0)) for k in sf_keys), key=lambda kv: -abs(kv[1]))
    print("    sf11: " + "  ".join("%s %+.2f" % (k, v) for k, v in sfb if abs(v) >= 0.15))
print()
