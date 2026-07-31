# -*- coding: utf-8 -*-
"""Classify each collapse in collapse_dataset.csv as a KS-ATTACK collapse (we lost THROUGH our own king) vs OTHER
(material blunder / positional drift / endgame technique), and support cross-run VANISH-ATTRIBUTION so we can
judge a lever by the CATEGORICAL, PROPORTIONAL rule (did the KS-attack class it targets shrink, without waking
other classes?), not by absolute total collapse count.

Lightweight + engine-free: pure python-chess geometry on the drop_fen (the position where the game fell apart).
KS-attack signals, from OUR side's POV (our_color):
  - our king in check at the drop, OR
  - heavy enemy pressure on our king ring (>= KZ_ATTACKERS distinct enemy attackers on king + adjacent squares), AND
  - it was NOT primarily a material give-away (|material swing decision->drop| below MAT_BLUNDER pawns) -- a pure
    hang-a-piece loss is an OTHER collapse even if near the king.
A collapse with big enemy king-ring pressure but also a big material loss is tagged 'ks_and_material' (mixed).
This is a heuristic screen for the night; upgrade to SF18-labeling later (ledger TODO).

Attribution (deterministic games -> same (family, seed, game, our_color) is the SAME game across runs):
  pyrun diagnostics/classify_collapses.py                         # classify all, print per-family class counts
  pyrun diagnostics/classify_collapses.py --vanish ab_base ab_kauf --seed 0
        # for one seed: which collapses are in A(base) but NOT B(kauf) [FIXED], B-not-A [NEW], by class
Writes ks_sets/collapse_dataset_classified.csv (adds ks_class + features).
"""
import os, sys, csv
import chess

THIS = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(THIS, "ks_sets", "collapse_dataset.csv")
OUT = os.path.join(THIS, "ks_sets", "collapse_dataset_classified.csv")

KZ_ATTACKERS = int(os.environ.get("KZ_ATTACKERS", "2"))   # distinct enemy attackers on the king ring -> "heavy"
MAT_BLUNDER = float(os.environ.get("MAT_BLUNDER", "3.0"))  # pawns of material swing that marks a material give-away
PIECE_VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}

def material(board, color):
    return sum(PIECE_VAL.get(p.piece_type, 0) for sq, p in board.piece_map().items() if p.color == color)

def king_ring(sq):
    ring = {sq}
    kr, kf = chess.square_rank(sq), chess.square_file(sq)
    for dr in (-1, 0, 1):
        for df in (-1, 0, 1):
            r, f = kr + dr, kf + df
            if 0 <= r < 8 and 0 <= f < 8:
                ring.add(chess.square(f, r))
    return ring

def king_ring_pressure(board, our_color):
    """# of distinct enemy pieces that attack at least one square of our king ring."""
    enemy = not our_color
    ksq = board.king(our_color)
    if ksq is None:
        return 0
    ring = king_ring(ksq)
    attackers = set()
    for sq in ring:
        for a in board.attackers(enemy, sq):
            attackers.add(a)
    return len(attackers)

def classify(row):
    our_white = row["our_color"].startswith("w")
    our_color = chess.WHITE if our_white else chess.BLACK
    feats = {"in_check": 0, "kring_pressure": 0, "mat_swing": 0.0, "ks_class": "other"}
    try:
        db = chess.Board(row["drop_fen"])
    except Exception:
        feats["ks_class"] = "unparsed"
        return feats
    # king pressure is evaluated with the enemy to move conceptually; use the board as given for geometry
    feats["kring_pressure"] = king_ring_pressure(db, our_color)
    # in-check: our king attacked (independent of side to move)
    feats["in_check"] = 1 if db.attackers(not our_color, db.king(our_color)) else 0
    # material swing decision->drop from OUR POV (positive = we lost material)
    try:
        dd = chess.Board(row["decision_fen"])
        dec_bal = material(dd, our_color) - material(dd, not our_color)
        drop_bal = material(db, our_color) - material(db, not our_color)
        feats["mat_swing"] = dec_bal - drop_bal   # >0 => our relative material fell
    except Exception:
        feats["mat_swing"] = 0.0

    heavy_king = feats["in_check"] or feats["kring_pressure"] >= KZ_ATTACKERS
    material_blunder = feats["mat_swing"] >= MAT_BLUNDER
    if heavy_king and material_blunder:
        feats["ks_class"] = "ks_and_material"
    elif heavy_king:
        feats["ks_class"] = "ks_attack"
    elif material_blunder:
        feats["ks_class"] = "material"
    else:
        feats["ks_class"] = "positional"
    return feats

def load_rows():
    if not os.path.exists(DATA):
        sys.exit("missing %s -- run collect_collapses.py first" % DATA)
    return list(csv.DictReader(open(DATA)))

def classify_all():
    rows = load_rows()
    for r in rows:
        r.update(classify(r))
    cols = list(rows[0].keys()) if rows else []
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(rows)
    return rows

def key(r):
    return (r["seed"], r["game"], r["our_color"])

def cmd_summary():
    rows = classify_all()
    from collections import Counter, defaultdict
    print("classified %d collapses -> %s" % (len(rows), OUT))
    fams = defaultdict(Counter)
    for r in rows:
        fams[r["family"]][r["ks_class"]] += 1
    classes = ["ks_attack", "ks_and_material", "material", "positional", "unparsed"]
    print("\n%-22s %s" % ("family", "  ".join("%-14s" % c for c in classes)))
    for fam in sorted(fams, key=lambda f: -sum(fams[f].values())):
        c = fams[fam]
        print("%-22s %s   (tot %d)" % (fam, "  ".join("%-14d" % c.get(k, 0) for k in classes), sum(c.values())))

def cmd_vanish(famA, famB, seed):
    rows = classify_all()
    seed = str(seed)
    A = {key(r): r for r in rows if r["family"] == famA and str(r["seed"]) == seed}
    B = {key(r): r for r in rows if r["family"] == famB and str(r["seed"]) == seed}
    fixed = [A[k] for k in A if k not in B]      # collapsed in A, not in B
    new = [B[k] for k in B if k not in A]        # collapsed in B, not in A
    from collections import Counter
    print("VANISH-ATTRIBUTION  %s vs %s  seed=%s   (A=%d collapses, B=%d)" % (famA, famB, seed, len(A), len(B)))
    print("  FIXED by B (in %s, gone in %s): %d   by class: %s" % (
        famA, famB, len(fixed), dict(Counter(r["ks_class"] for r in fixed))))
    print("  NEW  in B (regressions):         %d   by class: %s" % (
        len(new), dict(Counter(r["ks_class"] for r in new))))
    print("\n  -- FIXED KS-attack collapses (the target class we WANT gone) --")
    for r in [x for x in fixed if x["ks_class"] in ("ks_attack", "ks_and_material")][:20]:
        print("    game %-4s %-5s swing=%s  %s" % (r["game"], r["our_color"], r["swing"], r["drop_fen"]))
    print("\n  -- NEW KS-attack collapses (the lever must NOT create these) --")
    for r in [x for x in new if x["ks_class"] in ("ks_attack", "ks_and_material")][:20]:
        print("    game %-4s %-5s swing=%s  %s" % (r["game"], r["our_color"], r["swing"], r["drop_fen"]))

if __name__ == "__main__":
    if "--vanish" in sys.argv:
        i = sys.argv.index("--vanish")
        famA, famB = sys.argv[i + 1], sys.argv[i + 2]
        seed = sys.argv[sys.argv.index("--seed") + 1] if "--seed" in sys.argv else "0"
        cmd_vanish(famA, famB, seed)
    else:
        cmd_summary()
