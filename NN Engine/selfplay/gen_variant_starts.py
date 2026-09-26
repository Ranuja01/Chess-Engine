# -*- coding: utf-8 -*-
"""Generate a VARIANT-START opening book: legal but unfamiliar starting positions, so games test how much chess
the engine knows rather than how well it fits standard opening structure (owner's design, 2026-09-26;
EVAL-V2-INVENTORY-2026-09-25.md §7). Output lines use tournament.py's FEN form: `<fen> [| moves] ; <tag>`.

Families (castling is always off: shuffled kings would need Chess960 castling, and a king that cannot castle
also puts king safety under real test):
  array_*   symmetric back ranks -- both sides own the SAME rank, so material is equal by construction:
            960-style shuffles of the standard set, and replaced sets (all N->B, all B->N, N->R, B->R, no queen,
            queen->minor/rook, knights+queens only, bishops+queens only, rooks+queens only, double queen, ...)
  mixed_*   different but near-equal material per side (e.g. White's knights are bishops, Black's bishops are
            knights; Q vs R+B; two rooks vs queen+pawn-free), placed on randomised back ranks
  pawns_*   endgame starts: kings plus many pawns each, optionally a piece or two (K+P, K+R+P, K+minor+P,
            K+Q+P), with random but legal pawn skeletons of equal count
Every base position then gets a short random walk (WALK_MIN..WALK_MAX plies, drawn per start) to diversify it;
starts that end in check, game over, or lopsided by more than MAT_TOL centipawns are rejected. Colour balance
is handled by the tournament, which plays every start from both sides.

  python selfplay/gen_variant_starts.py [N=2400] [OUT=selfplay/openings_variant.txt] [SEED=2026]
                                        [WALK_MIN=0] [WALK_MAX=6] [MAT_TOL=150]
"""
import os, sys, random
import chess

KV = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
THIS = os.path.dirname(os.path.abspath(__file__))
N = int(KV.get("N", 2400))
OUT = KV.get("OUT", os.path.join(THIS, "openings_variant.txt"))
SEED = int(KV.get("SEED", 2026))
WALK_MIN = int(KV.get("WALK_MIN", 0))
WALK_MAX = int(KV.get("WALK_MAX", 6))
MAT_TOL = int(KV.get("MAT_TOL", 150))
rng = random.Random(SEED)
VAL = {"p": 100, "n": 320, "b": 330, "r": 500, "q": 900, "k": 0}

# Back-rank piece multisets (7 non-king pieces + K). Each is shuffled per start.
ARRAY_SETS = {
    "std":        "RNBQKBNR",
    "all_NtoB":   "RBBQKBBR",
    "all_BtoN":   "RNNQKNNR",
    "NtoR":       "RRBQKBRR",
    "BtoR":       "RNRQKRNR",
    "noQ":        "RNB.KBNR",
    "QtoN":       "RNBNKBNR",
    "QtoB":       "RNBBKBNR",
    "QtoR":       "RNBRKBNR",
    "NQ_only":    "NNNQKNNN",
    "BQ_only":    "BBBQKBBB",
    "RQ_only":    "RRRQKRRR",
    "doubleQ":    "RNQQKBNR",
    "minors":     "NNB.KBBN",
    "rooks_minors": "RNB.KBNR",
}

# Mixed material: (white multiset, black multiset); both near-equal by VAL.
MIXED_SETS = {
    "NvB":        ("RBBQKBBR", "RNNQKNNR"),    # 4 bishops vs 4 knights            (+40 cp)
    "QvRB":       ("RN.QKBNR", "RNBRKBNR"),    # queen vs rook + bishop            (+70)
    "RRvQ":       ("RNB.KBNR", "QNB.KBN."),    # two rooks vs queen                (+100)
    "BBvNN":      ("RBBQKNNR", "RNNQKBBR"),    # same set, pairs placed differently (0)
    "QminorsvRRR": ("RNBQKBNR", "RRBRKRNR"),   # queen + two minors vs three rooks (+50)
}

# Endgame starts: (white extras, black extras) beyond king + pawns.
PAWN_SETS = {
    "KP":    ("", ""),
    "KRP":   ("R", "R"),
    "KNP":   ("N", "N"),
    "KBP":   ("B", "B"),
    "KNvB":  ("N", "B"),
    "KQP":   ("Q", "Q"),
    "KRNvRB": ("RN", "RB"),
}


def back_rank(multiset):
    cells = list(multiset)
    rng.shuffle(cells)
    return cells


def board_from_ranks(wrank, brank, pawns=True):
    b = chess.Board(None)
    for f, ch in enumerate(wrank):
        if ch != ".":
            b.set_piece_at(chess.square(f, 0), chess.Piece.from_symbol(ch.upper()))
    for f, ch in enumerate(brank):
        if ch != ".":
            b.set_piece_at(chess.square(f, 7), chess.Piece.from_symbol(ch.lower()))
    if pawns:
        for f in range(8):
            b.set_piece_at(chess.square(f, 1), chess.Piece(chess.PAWN, chess.WHITE))
            b.set_piece_at(chess.square(f, 6), chess.Piece(chess.PAWN, chess.BLACK))
    b.turn = chess.WHITE
    b.castling_rights = 0
    return b


def material(b):
    s = 0
    for p in b.piece_map().values():
        v = VAL[p.symbol().lower()]
        s += v if p.color == chess.WHITE else -v
    return s


def make_array(name):
    rank = back_rank(ARRAY_SETS[name])
    return board_from_ranks(rank, rank)          # identical ranks: material equal by construction


def make_mixed(name):
    w, bl = MIXED_SETS[name]
    return board_from_ranks(back_rank(w), back_rank(bl))


def make_pawn_endgame(name):
    """Kings + an equal number (5..8) of pawns per side on random files/ranks, plus the family's pieces."""
    we, be = PAWN_SETS[name]
    for _ in range(200):
        b = chess.Board(None)
        npawns = rng.randint(5, 8)
        wfiles = rng.sample(range(8), npawns)
        bfiles = rng.sample(range(8), npawns)
        ok = True
        for f in wfiles:
            sq = chess.square(f, rng.randint(1, 4))
            if b.piece_at(sq):
                ok = False
            b.set_piece_at(sq, chess.Piece(chess.PAWN, chess.WHITE))
        for f in bfiles:
            sq = chess.square(f, rng.randint(3, 6))
            if b.piece_at(sq):
                ok = False
                break
            b.set_piece_at(sq, chess.Piece(chess.PAWN, chess.BLACK))
        if not ok:
            continue
        empties_w = [s for s in chess.SQUARES if not b.piece_at(s) and chess.square_rank(s) <= 2]
        empties_b = [s for s in chess.SQUARES if not b.piece_at(s) and chess.square_rank(s) >= 5]
        rng.shuffle(empties_w)
        rng.shuffle(empties_b)
        b.set_piece_at(empties_w.pop(), chess.Piece(chess.KING, chess.WHITE))
        b.set_piece_at(empties_b.pop(), chess.Piece(chess.KING, chess.BLACK))
        for ch in we:
            b.set_piece_at(empties_w.pop(), chess.Piece.from_symbol(ch.upper()))
        for ch in be:
            b.set_piece_at(empties_b.pop(), chess.Piece.from_symbol(ch.lower()))
        b.turn = chess.WHITE
        b.castling_rights = 0
        if b.is_valid() and not b.is_check() and not b.is_game_over():
            return b
    return None


def random_walk(b, plies):
    for _ in range(plies):
        moves = list(b.legal_moves)
        if not moves:
            return None
        b.push(rng.choice(moves))
    return b


def main():
    families = [("array", n) for n in ARRAY_SETS] + [("mixed", n) for n in MIXED_SETS] + \
               [("pawns", n) for n in PAWN_SETS]
    out, seen, rejected = [], set(), 0
    per_family = max(1, N // len(families))
    for fam, name in families:
        made, tries = 0, 0
        while made < per_family and tries < per_family * 50:
            tries += 1
            b = {"array": make_array, "mixed": make_mixed, "pawns": make_pawn_endgame}[fam](name)
            if b is None or not b.is_valid():
                rejected += 1
                continue
            base_mat = material(b)
            walk = random_walk(b.copy(), rng.randint(WALK_MIN, WALK_MAX))
            if walk is None or walk.is_check() or walk.is_game_over() or not walk.is_valid():
                rejected += 1
                continue
            # Keep starts near their family's intended balance: the walk may not win material outright.
            if abs(material(walk) - base_mat) > MAT_TOL:
                rejected += 1
                continue
            walk.clear_stack()
            fen = walk.fen()
            key = walk.board_fen() + (" w" if walk.turn else " b")
            if key in seen:
                continue
            seen.add(key)
            out.append("%s ; %s_%s" % (fen, fam, name.replace(" ", "")))
            made += 1
    rng.shuffle(out)
    with open(OUT, "w") as f:
        f.write("# variant-start book: %d starts, %d families, seed %d, walk %d..%d plies, mat_tol %d cp\n"
                % (len(out), len(families), SEED, WALK_MIN, WALK_MAX, MAT_TOL))
        f.write("\n".join(out) + "\n")
    print("wrote %d starts (%d families, %d rejected) -> %s" % (len(out), len(families), rejected, OUT))


if __name__ == "__main__":
    main()
