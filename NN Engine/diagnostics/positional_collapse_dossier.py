# -*- coding: utf-8 -*-
"""Diagnose the POSITIONAL collapse class (now the dominant collapse mode). For each base positional collapse
decision-FEN: our static eval + per-term breakdown, SF11-static (validity-gated: keep only where SF11 agrees
with SF18-SEARCH direction), SF18 truth. Orient to the COLLAPSING side and measure the eval OVER/UNDER-read
(our vs SF11 and vs SF18), aggregate which TERMS drive it, and print a small REPRESENTATIVE set for human
interpretation. CAVEAT: term names overlap but aren't 1:1 (SF folds PSQT into Material; our pt_* = pure
placement) -> only Material/KingSafety/Threats/Passed/Space/Imbalance/Mobility are clean.

  MAX not needed; DEPTH=18 pyrun diagnostics/positional_collapse_dossier.py [N=80]
"""
import os, sys, csv
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'; os.environ.setdefault('KS_SAFE_CHECK_DEF', '5')
DEPTH = int(os.environ.get('DEPTH', '18'))
NMAX = int(os.environ.get('N', '80'))
THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
sys.path.insert(0, os.path.join(os.path.dirname(THIS), "selfplay"))
import chess, chess.engine
from collections import defaultdict
from ChessAI import ChessAI
from eval_vs_sf11 import SF11Eval, SF11
from arbiter import find_stockfish
ai = ChessAI(None, None, chess.Board(), True)
sf11 = SF11Eval(SF11)
sf18 = chess.engine.SimpleEngine.popen_uci(find_stockfish())


def sgn(x):
    return (x > 0) - (x < 0)


def s18(fen):
    s = sf18.analyse(chess.Board(fen), chess.engine.Limit(depth=DEPTH))["score"].white()
    return 99.0 if (s.is_mate() and s.mate() > 0) else (-99.0 if s.is_mate() else s.score() / 100.0)


# clean-comparable term pairs (ours-keys, SF11-label)
PAIRS = [("Material", ["material", "br_kaufman"], "Material"), ("KingSafety", ["king_safety"], "King safety"),
         ("Threats", ["latent_threat"], "Threats"), ("Passed", ["passed_pawn_support"], "Passed"),
         ("Space", ["central", "det_central"], "Space"), ("Imbalance", ["imbalance_white", "imbalance_black"], "Imbalance")]

CSVP = os.path.join(THIS, "ks_sets", "collapse_dataset_classified.csv")
# Target family/seeds for the residual-collapse mining (default: the shipped V3 baseline, all 3 night seeds).
FAM = os.environ.get("DOSSIER_FAMILY", "v3")
SEEDS = set((os.environ.get("DOSSIER_SEEDS", "0,1,2")).split(","))
rows = [r for r in csv.DictReader(open(CSVP))
        if r.get("family") == FAM and r.get("ks_class") == "positional"
        and r.get("seed") in SEEDS and r.get("decision_fen", "")]
seen = set(); items = []
for r in rows:
    f = r["decision_fen"]
    if f not in seen:
        seen.add(f); items.append((f, r.get("our_color", "white")))
items = items[:NMAX]


def povmul(color):
    return 1.0 if color == "white" else -1.0


agg = defaultdict(float); n_valid = 0
dossier = []
for fen, color in items:
    try:
        bd = ai.ev_breakdown(chess.Board(fen))
        if bd.get("checkmate"):
            continue
        our_tot = -bd.get("total", 0.0) / 1000.0
        sf_tot, sf_terms = sf11.eval(fen)
        sf18v = s18(fen)
    except Exception:
        continue
    if sf_tot is None or abs(sf18v) >= 98:
        continue
    if sgn(sf_tot) != sgn(sf18v):      # VALIDITY: SF11-static must agree with SF18-search direction
        continue
    n_valid += 1
    m = povmul(color)
    over_sf11 = m * (our_tot - sf_tot)          # + => we over-read the collapsing side's position vs SF11
    over_sf18 = m * (our_tot - sf18v)           # + => optimistic vs truth
    terms = {}
    for label, ks, sflabel in PAIRS:
        o = -sum(bd.get(k, 0.0) for k in ks) / 1000.0
        s = sf_terms.get(sflabel, 0.0)
        g = m * (o - s)                          # collapsing-side-POV gap
        terms[label] = g
        agg[label] += g
    agg["_OVER_sf11"] += over_sf11; agg["_OVER_sf18"] += over_sf18
    dossier.append((over_sf18, over_sf11, color, our_tot, sf_tot, sf18v, terms, fen))

print("POSITIONAL collapse dossier — %d valid FENs (SF11 dir agrees SF18)\n" % n_valid)
print("MEAN over-read (collapsing-side POV, + = we're too optimistic):")
print("  vs SF11-static: %+.2f    vs SF18-search: %+.2f  (pawns)" % (agg["_OVER_sf11"] / n_valid, agg["_OVER_sf18"] / n_valid))
print("\nMEAN per-term gap ours-vs-SF11 (collapsing-side POV, + = we assign MORE than SF11):")
for label, _, _ in PAIRS:
    print("  %-11s %+.3f" % (label, agg[label] / n_valid))

print("\n=== REPRESENTATIVE FENs (largest over-read vs SF18 truth) ===")
print("drivers = OUR OWN term contributions to the COLLAPSING side (pawns), the terms WE credit it:")
# our clean sub-terms (all ours; no SF apples-to-oranges)
OUR = [("mat", ["material"]), ("kauf", ["br_kaufman"]), ("place", ["pt_pawns", "pt_knights", "pt_bishops", "pt_rooks", "pt_queens"]),
       ("capg", ["capture_gains"]), ("pvb", ["piece_value_boost"]), ("space", ["central"]),
       ("threat", ["latent_threat"]), ("ks", ["king_safety"]), ("passed", ["passed_pawn_support"]),
       ("imb", ["imbalance_white", "imbalance_black"]), ("pair", ["pair_bonus"])]
dossier.sort(key=lambda t: -t[0])
for over18, over11, color, ot, st, s18v, terms, fen in dossier[:8]:
    bd = ai.ev_breakdown(chess.Board(fen)); m = povmul(color)
    parts = []
    for lab, ks in OUR:
        v = m * (-sum(bd.get(k, 0.0) for k in ks) / 1000.0)
        if abs(v) >= 0.3:
            parts.append("%s%+.1f" % (lab, v))
    print("\n[%s] our%+.2f  SF11%+.2f  SF18%+.2f   OVER-READ%+.2f" % (color, ot, st, s18v, over18))
    print("   we credit %s: %s" % (color, " ".join(parts) or "(diffuse)"))
    print("   %s" % fen)

# SPACE-DOMINATED representatives: the FENs where our Space/central term specifically most over-credits vs SF11.
print("\n=== TOP by SPACE over-read (our central vs SF11 Space, collapsing-side POV) ===")
dossier.sort(key=lambda t: -t[6].get("Space", 0.0))
for over18, over11, color, ot, st, s18v, terms, fen in dossier[:6]:
    bd = ai.ev_breakdown(chess.Board(fen)); m = povmul(color)
    parts = []
    for lab, ks in OUR:
        v = m * (-sum(bd.get(k, 0.0) for k in ks) / 1000.0)
        if abs(v) >= 0.3:
            parts.append("%s%+.1f" % (lab, v))
    print("\n[%s] our%+.2f  SF11%+.2f  SF18%+.2f | Space gap %+.2f | total OVER%+.2f" % (
        color, ot, st, s18v, terms.get("Space", 0.0), over18))
    print("   we credit %s: %s" % (color, " ".join(parts) or "(diffuse)"))
    print("   %s" % fen)

# KS-DRIVEN over-reads: genuine over-reads vs SF18 truth where OUR king-safety term is a big driver, with the
# raw attack-units (the internal that over-fires). This is the curated KS-attack corpus seed.
print("\n=== TOP KS-DRIVEN over-reads (our king_safety over-crediting the collapsing side; + raw attack-units) ===")
kslist = []
for over18, over11, color, ot, st, s18v, terms, fen in dossier:
    bd = ai.ev_breakdown(chess.Board(fen)); m = povmul(color)
    ks_pov = m * (-bd.get("king_safety", 0.0) / 1000.0)   # our KS oriented to the collapsing side (+ = we credit its attack)
    kslist.append((ks_pov, over18, color, ot, st, s18v, bd.get("det_ks_units_w", 0), bd.get("det_ks_units_b", 0), fen))
kslist.sort(key=lambda t: -t[0])
for ks_pov, over18, color, ot, st, s18v, uw, ub, fen in kslist[:10]:
    if over18 < 0.5:   # only genuine over-reads vs truth
        continue
    print("\n[%s] our%+.2f  SF11%+.2f  SF18%+.2f | our_KS(collapsing) %+.2f | units W%d/B%d | OVER%+.2f" % (
        color, ot, st, s18v, ks_pov, uw, ub, over18))
    print("   %s" % fen)
sf18.quit()
