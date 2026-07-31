# -*- coding: utf-8 -*-
"""Classify ledger collapses by whether PASSERS are involved, so the two-sided ledger verdict can be read:
does the target (passer) class shrink, and is any rise concentrated elsewhere?

Totals of ~60 carry Poisson error ~+-8, so the total-count comparison alone cannot decide V3. Splitting by
class is what makes a 200g run informative (cf. KS v1: flat TOTAL but target class -53% => shipped).

For each collapse decision FEN: does either side have a passed pawn, and is there an ADVANCED passer
(rank >= 5 for White / <= 3 for Black) -- the case V3 is actually about (advancement vs count).

Run: pyrun diagnostics/_collapse_class.py
"""
import os, csv
import chess

THIS = os.path.dirname(os.path.abspath(__file__))
GAMES = os.path.join(os.path.dirname(THIS), "selfplay", "games")
ARMS = [("ledger_base_s0", "BASE s0"), ("ledger_v3_s0", "V3   s0"),
        ("ledger_base_s1", "BASE s1"), ("ledger_v3_s1", "V3   s1")]

def passers(b):
    """Return (any_passer, advanced_passer) for the position."""
    anyp = adv = False
    for color in (chess.WHITE, chess.BLACK):
        them = not color
        for sq in b.pieces(chess.PAWN, color):
            f, r = chess.square_file(sq), chess.square_rank(sq)
            blocked = False
            for df in (-1, 0, 1):
                nf = f + df
                if not 0 <= nf <= 7:
                    continue
                for esq in b.pieces(chess.PAWN, them):
                    if chess.square_file(esq) != nf:
                        continue
                    er = chess.square_rank(esq)
                    if (er > r) if color == chess.WHITE else (er < r):
                        blocked = True
                        break
                if blocked:
                    break
            if not blocked:
                anyp = True
                if (r >= 4) if color == chess.WHITE else (r <= 3):
                    adv = True
    return anyp, adv

print(f"{'arm':<10}{'total':>7}{'passer':>9}{'advanced':>10}{'no-passer':>11}")
print("-" * 47)
rows = {}
for tag, label in ARMS:
    p = os.path.join(GAMES, tag, "collapses.csv")
    if not os.path.isfile(p):
        print(f"{label:<10} (missing)")
        continue
    n = np = na = 0
    for r in csv.DictReader(open(p)):
        fen = r.get("decision_fen") or r.get("drop_fen")
        if not fen:
            continue
        try:
            b = chess.Board(fen)
        except Exception:
            continue
        n += 1
        a, ad = passers(b)
        np += 1 if a else 0
        na += 1 if ad else 0
    rows[tag] = (n, np, na)
    print(f"{label:<10}{n:>7}{np:>9}{na:>10}{n-np:>11}")

print()
for s in ("s0", "s1"):
    b, v = rows.get(f"ledger_base_{s}"), rows.get(f"ledger_v3_{s}")
    if b and v:
        print(f"seed {s[1]}:  total {b[0]}->{v[0]} ({v[0]-b[0]:+d})   "
              f"PASSER {b[1]}->{v[1]} ({v[1]-b[1]:+d})   "
              f"ADVANCED {b[2]}->{v[2]} ({v[2]-b[2]:+d})   "
              f"NON-passer {b[0]-b[1]}->{v[0]-v[1]} ({(v[0]-v[1])-(b[0]-b[1]):+d})")
bt = [rows[k] for k in ("ledger_base_s0", "ledger_base_s1") if k in rows]
vt = [rows[k] for k in ("ledger_v3_s0", "ledger_v3_s1") if k in rows]
if bt and vt:
    B = [sum(x[i] for x in bt) for i in range(3)]
    V = [sum(x[i] for x in vt) for i in range(3)]
    print(f"\nPOOLED  total {B[0]}->{V[0]} ({V[0]-B[0]:+d})   PASSER {B[1]}->{V[1]} ({V[1]-B[1]:+d})   "
          f"ADVANCED {B[2]}->{V[2]} ({V[2]-B[2]:+d})   NON-passer {B[0]-B[1]}->{V[0]-V[1]} "
          f"({(V[0]-V[1])-(B[0]-B[1]):+d})")
    print("\nLedger WIN pattern = target class DOWN even if total is flat/up (KS v1 precedent).")
