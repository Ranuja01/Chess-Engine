# -*- coding: utf-8 -*-
"""Generate an ODDS-START opening book: the standard position with material removed from one side.

Owner's point (2026-09-26): odds games show how an engine plays WITH and AGAINST a material edge -- converting
an extra piece and defending a deficit -- which balanced openings never exercise. Stockfish at a short time
control beats our lightning setting even without a knight, so something fundamental is missing there. The
tournament plays every start from both colours, so each arm gets the extra material once and the deficit once;
read the result split by which side held the edge, not just the total.

Families (the handicapped side is White in the FEN; the pairing gives the handicap to both arms in turn):
  pawn (f2), knight (b1), bishop (c1), exchange (a1 rook -> knight), rook (a1), queen (d1),
  two_minors (b1 knight + c1 bishop), rook_for_pawns (a1 rook, Black gives up three pawns)
A short random walk (WALK_MIN..WALK_MAX plies) diversifies each start; castling rights follow the pieces.

  python selfplay/gen_odds_starts.py [PER=60] [OUT=selfplay/openings_odds.txt] [SEED=2027] [WALK_MIN=2] [WALK_MAX=8]
"""
import os, sys, random
import chess

KV = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
THIS = os.path.dirname(os.path.abspath(__file__))
PER = int(KV.get("PER", 60))
OUT = KV.get("OUT", os.path.join(THIS, "openings_odds.txt"))
SEED = int(KV.get("SEED", 2027))
WALK_MIN = int(KV.get("WALK_MIN", 2))
WALK_MAX = int(KV.get("WALK_MAX", 8))
rng = random.Random(SEED)
VAL = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330, chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}

# family -> (White squares emptied, White square -> replacement piece, Black squares emptied)
ODDS = {
    "pawn":           (["f2"], {}, []),
    "knight":         (["b1"], {}, []),
    "bishop":         (["c1"], {}, []),
    "exchange":       ([], {"a1": "N"}, []),
    "rook":           (["a1"], {}, []),
    "queen":          (["d1"], {}, []),
    "two_minors":     (["b1", "c1"], {}, []),
    "rook_for_pawns": (["a1"], {}, ["a7", "b7", "c7"]),
}


def material(b):
    return sum((VAL[p.piece_type] if p.color else -VAL[p.piece_type]) for p in b.piece_map().values())


def make(family):
    w_empty, w_replace, b_empty = ODDS[family]
    b = chess.Board()
    for s in w_empty + b_empty:
        b.remove_piece_at(chess.parse_square(s))
    for s, sym in w_replace.items():
        b.set_piece_at(chess.parse_square(s), chess.Piece.from_symbol(sym))
    b.castling_rights = b.clean_castling_rights()
    return b


def main():
    out, seen = [], set()
    for family in ODDS:
        made, tries = 0, 0
        while made < PER and tries < PER * 50:
            tries += 1
            b = make(family)
            base = material(b)
            for _ in range(rng.randint(WALK_MIN, WALK_MAX)):
                moves = list(b.legal_moves)
                if not moves:
                    break
                b.push(rng.choice(moves))
            # The walk must not change the material balance: the family defines the handicap.
            if b.is_check() or b.is_game_over() or material(b) != base:
                continue
            key = b.board_fen() + (" w" if b.turn else " b")
            if key in seen:
                continue
            seen.add(key)
            fen = b.fen()
            out.append("%s ; odds_%s" % (fen, family))
            made += 1
    rng.shuffle(out)
    with open(OUT, "w") as f:
        f.write("# odds-start book: %d starts, White holds the handicap in the FEN; pairing gives it to both arms\n"
                % len(out))
        f.write("\n".join(out) + "\n")
    print("wrote %d odds starts -> %s" % (len(out), OUT))


if __name__ == "__main__":
    main()
