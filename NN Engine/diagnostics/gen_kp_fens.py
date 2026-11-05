# -*- coding: utf-8 -*-
"""Generate randomized PAWN-DOMINATED start positions to isolate the pawn/passer evaluation.

Why: pawn-structure and passer terms are diluted in normal play -- a change worth real Elo in
pawn-dominated positions can read as neutral over a full-game corpus where piece play dominates. Playing
from positions that are mostly kings and pawns puts the subsystem we are tuning on the critical path, so
`isolated / backward / passed / realizability` differences actually decide games.

⚠️ This is a DIAGNOSTIC venue, not a strength proxy. The collapse leverage map puts endgame FENs at only
5-9% of forfeited points, so winning here does not imply Elo -- it tells us whether a pawn term WORKS,
which is a different question from whether it MATTERS.

Modes (MIX controls the blend):
  pure   - kings + pawns only (3-6 a side)
  minor  - kings + pawns + one minor each
  rook   - kings + pawns + one rook each
  imbal  - as above but one side gets an extra pawn or a minor-for-pawns imbalance
  dense  - STRESS ZONE: 7-8 pawns a side, no pieces. Unreachable in a real game, but both engines must
           still evaluate it, and with no pieces the pawn terms ARE the eval, so a disagreement with SF18
           cannot be blamed on piece play.
  nopawn - strip-down stress: kings + a small piece set each (R v N, Q v R, R+B v R, B+N v -, ...), no pawns
  heavy  - strip-down stress: rook / queen sets with 1-7 pawns a side
  mixed  - strip-down stress: 1-3 random pieces a side (often imbalanced) with 1-7 pawns a side
  (the three strip-down modes are never in the default MIX, so existing seeds reproduce exactly)

Positions are validated with python-chess (legal, not already over, side to move not giving check) and are
deduped. Deterministic given SEED.

  pyrun diagnostics/gen_kp_fens.py [N=300] [SEED=7] [OUT=selfplay/kp_fens.txt] [MIX=pure,minor,rook,imbal]
"""
import os, sys, random

for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS))
import chess

N = int(os.environ.get("N", "300"))
SEED = int(os.environ.get("SEED", "7"))
OUT = os.environ.get("OUT", os.path.join(os.path.dirname(THIS), "selfplay", "kp_fens.txt"))
MIX = [m.strip() for m in os.environ.get("MIX", "pure,minor,rook,imbal").split(",") if m.strip()]

rng = random.Random(SEED)
PAWN_SQUARES = [s for s in chess.SQUARES if 8 <= s <= 55]      # ranks 2..7 only


def place(board, piece_type, colour, taken):
    """Put one piece on a free square; pawns restricted to ranks 2-7. Returns the square or None."""
    pool = PAWN_SQUARES if piece_type == chess.PAWN else list(chess.SQUARES)
    rng.shuffle(pool)
    for sq in pool:
        if sq in taken:
            continue
        board.set_piece_at(sq, chess.Piece(piece_type, colour))
        taken.add(sq)
        return sq
    return None


def make_position(mode):
    board = chess.Board(None)
    taken = set()

    # Kings first, kept apart so the position is not instantly illegal.
    wk = place(board, chess.KING, chess.WHITE, taken)
    for _ in range(64):
        bk = place(board, chess.KING, chess.BLACK, taken)
        if bk is None:
            return None
        if chess.square_distance(wk, bk) >= 2:
            break
        board.remove_piece_at(bk); taken.discard(bk)
    else:
        return None

    if mode in STRIP_MODES:
        stripdown(board, taken, mode)
        return finish(board)

    # `dense` = the stress zone: 7-8 pawns a side and NO pieces. Unreachable from a real game, but both
    # engines must still evaluate it, and with no pieces on the board the pawn terms ARE the evaluation --
    # so any disagreement with SF18 is unambiguously a pawn-eval defect rather than a piece-play confound.
    # Random placement produces heavy doubling/tripling, which is exactly the structural extreme we want.
    npawns = rng.randint(7, 8) if mode == "dense" else rng.randint(3, 6)
    extra_w = extra_b = 0
    if mode == "imbal":
        # a clean extra pawn, or minor-for-two-pawns -- the imbalances passer eval must price
        if rng.random() < 0.5:
            extra_w = 1
        else:
            extra_b = 1
    for _ in range(npawns + extra_w):
        place(board, chess.PAWN, chess.WHITE, taken)
    for _ in range(npawns + extra_b):
        place(board, chess.PAWN, chess.BLACK, taken)

    if mode in ("minor", "imbal"):
        pt = rng.choice([chess.KNIGHT, chess.BISHOP])
        place(board, pt, chess.WHITE, taken)
        place(board, rng.choice([chess.KNIGHT, chess.BISHOP]), chess.BLACK, taken)
    if mode == "rook":
        place(board, chess.ROOK, chess.WHITE, taken)
        place(board, chess.ROOK, chess.BLACK, taken)
    return finish(board)


# Strip-down stress modes (2026-11-05, retune corpus): strip the board to one subsystem so a term that only fits
# opening-derived structures is exposed. nopawn = piece endings with no pawns (the material and king terms alone);
# heavy = rook / queen endings with pawns; mixed = random small piece sets with pawns, often imbalanced.
STRIP_MODES = ("nopawn", "heavy", "mixed")
PIECE = {"N": chess.KNIGHT, "B": chess.BISHOP, "R": chess.ROOK, "Q": chess.QUEEN}
NOPAWN_SETS = [("R", "N"), ("R", "B"), ("Q", "R"), ("RB", "R"), ("RN", "R"), ("Q", "RB"), ("Q", "RN"), ("Q", "BB"),
               ("Q", "RR"), ("BN", ""), ("BB", "N"), ("RR", "Q"), ("RB", "RN"), ("QR", "Q"), ("R", "R"), ("N", "N"),
               ("B", "N"), ("BB", "R"), ("Q", "NN"), ("RR", "R")]
HEAVY_SETS = [("R", "R"), ("Q", "Q"), ("RR", "RR"), ("QR", "QR"), ("Q", "RR"), ("RB", "RN"), ("RN", "R"), ("Q", "R"),
              ("R", "B"), ("R", "N"), ("QB", "Q"), ("RR", "RB")]


def stripdown(board, taken, mode):
    if mode == "nopawn":
        w, b = rng.choice(NOPAWN_SETS)
        nw = nb = 0
    else:
        if mode == "heavy":
            w, b = rng.choice(HEAVY_SETS)
        else:
            w = "".join(rng.choice("NNBBRRQ") for _ in range(rng.randint(1, 3)))
            b = "".join(rng.choice("NNBBRRQ") for _ in range(rng.randint(1, 3)))
        nw = rng.randint(1, 6)
        nb = max(1, min(7, nw + rng.choice([-1, 0, 0, 1])))
    if rng.random() < 0.5:
        w, b, nw, nb = b, w, nb, nw
    for c in w:
        place(board, PIECE[c], chess.WHITE, taken)
    for c in b:
        place(board, PIECE[c], chess.BLACK, taken)
    for _ in range(nw):
        place(board, chess.PAWN, chess.WHITE, taken)
    for _ in range(nb):
        place(board, chess.PAWN, chess.BLACK, taken)


def finish(board):
    board.turn = rng.choice([chess.WHITE, chess.BLACK])
    board.castling_rights = 0
    board.halfmove_clock = 0
    board.fullmove_number = 1

    # Legality: valid, not already finished, and the side NOT to move must not be in check.
    if not board.is_valid():
        return None
    if board.is_game_over(claim_draw=False):
        return None
    if board.is_check():
        return None          # keep starts quiet; a check at ply 0 skews the opening
    return board.fen()


def main():
    seen, out = set(), []
    tries = 0
    while len(out) < N and tries < N * 400:
        tries += 1
        fen = make_position(rng.choice(MIX))
        if not fen or fen in seen:
            continue
        seen.add(fen)
        out.append(fen)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write("# randomized pawn-dominated starts — DIAGNOSTIC venue for the pawn/passer subsystem\n")
        fh.write("# N=%d SEED=%d MIX=%s\n" % (len(out), SEED, ",".join(MIX)))
        for fen in out:
            fh.write(fen + "\n")
    print("wrote %d positions (%d attempts) -> %s" % (len(out), tries, OUT))


if __name__ == '__main__':
    main()
