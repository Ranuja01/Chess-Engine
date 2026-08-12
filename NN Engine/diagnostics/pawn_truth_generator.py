# -*- coding: utf-8 -*-
"""GROUND TRUTH for what a pawn is WORTH, by SF18 search, as a function of rank x file x context.

The pawn redesign needs a statable definition of good and bad BEFORE any table is built (owner's brief).
This produces one: for each randomized backdrop we score the position WITH and WITHOUT a single test pawn
placed in a specified structural context. The difference is that pawn's MARGINAL VALUE in that context.

Why marginal value against a per-cell baseline, rather than comparing whole positions:
  - contexts differ in how many friendly pawns they need ('supported' carries an extra pawn, 'isolated' does
    not), so comparing raw evals across contexts would compare different material. Each cell is measured
    against its OWN backdrop-minus-the-test-pawn, so every number is "what this one pawn adds here" and all
    cells are directly comparable.
  - it cancels the backdrop. Whatever the random pieces are worth appears in both terms and drops out.

Why many randomized backdrops rather than one hand-built position:
  - a manufactured position can be tactically special, and SF18 SEARCH prices tactics. Averaging the delta
    over many independent backdrops turns that into noise with a reported CI instead of a hidden bias, and
    it is the only way an answer can be shown to be a property of the STRUCTURE and not of one setup.
  - positions are deliberately allowed to be unrealistic but must be LEGAL and QUIET; see the filters.

Contexts are two orthogonal axes, so "strong vs weak" and "how blocked" never get conflated:
  friendly (strength):  isolated | unsupported | supported | phalanx | doubled
  enemy    (obstruction): passed | contested | opposed | blocked

  pyrun diagnostics/pawn_truth_generator.py [OUT=... DEPTH=16 BACKDROPS=24 FILES=a,c,e RANKS=2,3,4,5,6,7]
                                            [BACKDROP=pieces,kp] [SEED=0] [MAX_ABS_CP=400]
"""
import os, sys, csv, random, itertools

for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
import chess
import chess.engine

SF18 = os.environ["STOCKFISH_PATH"]
DEPTH = int(os.environ.get("DEPTH", "16"))
BACKDROPS = int(os.environ.get("BACKDROPS", "24"))
SEED = int(os.environ.get("SEED", "0"))
# ⚠️ OUTCOME FILTER -- default OFF, and it should normally stay off.
# The first version defaulted this to 400cp "to remove tactics". That filters on the DEPENDENT VARIABLE:
# it deletes precisely the samples where the pawn was worth a lot, which is the signal. It reported a
# rank-7 passer at +153cp (SF prices those near 7 pawns) and squashed every cell toward zero. Tactical
# noise is handled where it belongs -- by rejecting non-quiet POSITIONS before scoring, and by averaging
# over independent backdrops. Set it only to inspect how much the tail is carrying.
MAX_ABS_CP = int(os.environ.get("MAX_ABS_CP", "0")) or None
# A forced mate is still a real consequence of the pawn, so it is CLAMPED rather than dropped -- dropping
# it would be outcome filtering by another name.
MATE_CP = int(os.environ.get("MATE_CP", "2000"))
# Attempts to find a quiet backdrop before giving up on a sample. Retrying keeps the piece backdrop from
# being silently under-represented: the first run dropped 71% of 'pieces' samples and only 31% of 'kp'.
BACKDROP_TRIES = int(os.environ.get("BACKDROP_TRIES", "12"))
# ✅ BASELINE-BALANCE condition. Require the position WITHOUT the test pawn to be roughly level.
# This is legitimate where MAX_ABS_CP was not: `cp_without` is a property of the backdrop, fixed BEFORE the
# pawn is added, so conditioning on it cannot select on the treatment effect. It matters because eval is a
# saturating function of advantage -- in an already-won or already-drawn position an extra pawn changes the
# number by an amount that reflects DECISIVENESS, not the pawn's positional worth. Validation showed exactly
# that: in kings-and-pawns backdrops a rank-2 pawn read up to +613cp because it converted draws into wins.
BASE_ABS_CP = int(os.environ.get("BASE_ABS_CP", "200"))
OUT = os.environ.get("OUT", os.path.join(THIS, "ks_sets", "pawn_truth.csv"))

FILES = [ord(c) - ord('a') for c in os.environ.get("FILES", "a,c,e").split(",")]
RANKS = [int(v) - 1 for v in os.environ.get("RANKS", "2,3,4,5,6,7").split(",")]   # store 0-indexed
BACKDROP_KINDS = os.environ.get("BACKDROP", "pieces,kp").split(",")

FRIENDLY = ["isolated", "unsupported", "supported", "phalanx", "doubled"]
# `blocked` is blocked BY A PAWN and therefore cannot exist on the 7th (the blocker would be a pawn on the
# 8th). `piece_blocked` -- an enemy minor sitting on the stop square -- is the case that matters most and was
# missing entirely: it is representable at EVERY rank including the 7th, it is what a real blockade looks
# like, and it is the case our BLOCK[] docking was written for. Without it the 7th rank is sampled only as a
# free runner, which is why the first tables made a 7th-rank pawn look unconditionally enormous.
ENEMY = ["passed", "contested", "opposed", "blocked", "piece_blocked"]
BLOCKADERS = [chess.KNIGHT, chess.BISHOP]


def sq(f, r):
    return chess.square(f, r)


def place_context(board, f, r, friendly, enemy, rng=None):
    """Add the CONTEXT pawns around (f, r) for White's test pawn. Returns False if the context is
    geometrically impossible here (e.g. 'supported' on rank 2 needs a friendly pawn on rank 1)."""
    adj = [x for x in (f - 1, f + 1) if 0 <= x <= 7]
    if not adj:
        return False

    if friendly == "isolated":
        pass                                   # no friendly pawn on either adjacent file
    elif friendly == "unsupported":
        # Adjacent friendly pawns exist but all are MORE ADVANCED, so none can ever support this pawn.
        # They must sit on r+2, NOT r+1: a white pawn on (f,r) ATTACKS (f±1, r+1), so neighbours placed one
        # rank ahead are DEFENDED BY the test pawn, which makes it a chain base -- the strongest shape, not
        # the weakest. The first version of this function did exactly that and inverted the whole axis.
        if r + 2 > 6:
            return False
        for x in adj:
            board.set_piece_at(sq(x, r + 2), chess.Piece(chess.PAWN, chess.WHITE))
    elif friendly == "supported":
        if r - 1 < 1:
            return False
        board.set_piece_at(sq(adj[0], r - 1), chess.Piece(chess.PAWN, chess.WHITE))
    elif friendly == "phalanx":
        board.set_piece_at(sq(adj[0], r), chess.Piece(chess.PAWN, chess.WHITE))
    elif friendly == "doubled":
        if r - 1 < 1:
            return False
        board.set_piece_at(sq(f, r - 1), chess.Piece(chess.PAWN, chess.WHITE))
    else:
        return False

    if enemy == "passed":
        pass                                   # nothing ahead on this or adjacent files
    elif enemy == "contested":
        if r + 1 > 6:
            return False
        board.set_piece_at(sq(adj[-1], r + 1), chess.Piece(chess.PAWN, chess.BLACK))
    elif enemy == "opposed":
        if r + 2 > 6:
            return False
        board.set_piece_at(sq(f, r + 2), chess.Piece(chess.PAWN, chess.BLACK))
    elif enemy == "blocked":
        if r + 1 > 6:
            return False
        board.set_piece_at(sq(f, r + 1), chess.Piece(chess.PAWN, chess.BLACK))
    elif enemy == "piece_blocked":
        # A minor ON the stop square: the true blockade, and the only obstruction representable on the 7th.
        # The test pawn attacks (f±1, r+1), never (f, r+1), so it can never remove the blockader itself.
        if r + 1 > 7:
            return False
        pt = BLOCKADERS[0] if rng is None else rng.choice(BLOCKADERS)
        board.set_piece_at(sq(f, r + 1), chess.Piece(pt, chess.BLACK))
    else:
        return False
    return True


def build_backdrop(rng, kind, test_file):
    """Kings plus, for 'pieces', a balanced random piece set and filler pawns -- all kept AWAY from the test
    file so the backdrop cannot itself define the test pawn's structure. Returns a board or None."""
    board = chess.Board(None)
    far = [x for x in range(8) if abs(x - test_file) >= 2]
    if not far:
        return None

    # Kings: legal separation, off the test file and its neighbours.
    for _ in range(60):
        wk = sq(rng.choice(far), rng.randint(0, 7))
        bk = sq(rng.choice(far), rng.randint(0, 7))
        if chess.square_distance(wk, bk) >= 2:
            board.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
            board.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
            break
    else:
        return None

    def free(files, lo, hi):
        opts = [sq(x, y) for x in files for y in range(lo, hi + 1) if board.piece_at(sq(x, y)) is None]
        return rng.choice(opts) if opts else None

    # DENSITY MATTERS, and it is not cosmetic. The marginal value of a pawn depends on how much else is on
    # the board: in a 6-piece position an extra pawn is often simply decisive, so it prices at 300-450cp and
    # swamps every structural difference we are trying to read. Validation caught this -- a rank-2 pawn must
    # come out near a pawn, and it only does once the backdrop is dense enough for it to be marginal.
    if kind == "pieces":
        # Balanced: both sides get the same piece types, so the backdrop carries no material bias.
        pool = [chess.ROOK, chess.ROOK, chess.BISHOP, chess.BISHOP, chess.KNIGHT, chess.KNIGHT, chess.QUEEN]
        for pt in rng.sample(pool, rng.randint(3, 5)):
            for colour in (chess.WHITE, chess.BLACK):
                s = free(far, 0, 7)
                if s is not None:
                    board.set_piece_at(s, chess.Piece(pt, colour))
    # Filler pawns on far files, equal counts, ranks 2-6 only (never rank 1 or 8).
    for _ in range(rng.randint(3, 5) if kind == "pieces" else rng.randint(4, 6)):
        for colour in (chess.WHITE, chess.BLACK):
            s = free(far, 1, 6)
            if s is not None:
                board.set_piece_at(s, chess.Piece(chess.PAWN, colour))
    return board


def quiet_and_legal(board):
    """Reject anything whose eval would be about tactics or terminality rather than structure."""
    if board.status() != chess.STATUS_VALID:
        return False
    if board.is_check() or board.is_game_over(claim_draw=False):
        return False
    # No side to move should have a capture of a hanging piece available -- crude, but it removes the
    # positions where SF's number is a tactic. Checks are also excluded above.
    for mv in board.legal_moves:
        if board.is_capture(mv) and board.piece_type_at(mv.to_square) not in (chess.PAWN, None):
            if not board.is_attacked_by(not board.turn, mv.to_square):
                return False
    return True


def main():
    rng = random.Random(SEED)
    engine = chess.engine.SimpleEngine.popen_uci(SF18)
    engine.configure({"Threads": 1, "Hash": 128})
    limit = chess.engine.Limit(depth=DEPTH)

    def score_cp(board):
        # White-POV centipawns. Mates are clamped, not dropped -- see MATE_CP.
        return engine.analyse(board, limit)["score"].white().score(mate_score=MATE_CP)

    rows, kept = [], 0
    drops = {"no_backdrop": 0, "occupied": 0, "not_quiet": 0, "unbalanced_base": 0, "outcome_filter": 0}
    cells = list(itertools.product(BACKDROP_KINDS, FILES, RANKS, FRIENDLY, ENEMY))
    print("cells=%d backdrops=%d depth=%d -> up to %d searches"
          % (len(cells), BACKDROPS, DEPTH, 2 * len(cells) * BACKDROPS))

    for ci, (kind, f, r, friendly, enemy) in enumerate(cells):
        impossible = False
        for b in range(BACKDROPS):
            pair = None
            # Retry the backdrop until a QUIET one is found. Rejecting a whole sample on the first
            # unquiet draw silently under-sampled the piece backdrop 2:1 against kings-and-pawns.
            for t in range(BACKDROP_TRIES):
                # Seeded by (sample, try, file) ONLY -- deliberately NOT by the context. Every context cell
                # sees the SAME backdrop sequence, so context-to-context differences are paired and the
                # backdrop variance cancels out of exactly the contrasts the analysis reports.
                base = build_backdrop(random.Random(SEED * 100003 + b * 97 + t * 7919 + f), kind, f)
                if base is None:
                    drops["no_backdrop"] += 1
                    continue
                with_pawn = base.copy()
                if with_pawn.piece_at(sq(f, r)) is not None:
                    drops["occupied"] += 1
                    continue
                with_pawn.set_piece_at(sq(f, r), chess.Piece(chess.PAWN, chess.WHITE))
                if not place_context(with_pawn, f, r, friendly, enemy,
                                     random.Random(SEED * 7 + b * 13 + t)):
                    impossible = True       # geometry, not luck -- no retry will help
                    break
                without = with_pawn.copy()
                without.remove_piece_at(sq(f, r))
                # Synthetic boards must not inherit castling rights or an en-passant square for pieces that
                # were never there -- SF would reject or misread the FEN and the pair would not compare.
                for bd in (with_pawn, without):
                    bd.turn = chess.WHITE
                    bd.castling_rights = 0
                    bd.ep_square = None
                    bd.halfmove_clock = 0
                    bd.fullmove_number = 1
                if not (quiet_and_legal(with_pawn) and quiet_and_legal(without)):
                    drops["not_quiet"] += 1
                    continue
                # Score the baseline inside the retry loop so an already-decided backdrop is REPLACED
                # rather than dropped -- dropping it would bias which backdrops survive per context.
                a = score_cp(without)
                if abs(a) > BASE_ABS_CP:
                    drops["unbalanced_base"] += 1
                    continue
                pair = (with_pawn, without, a)
                break
            if impossible or pair is None:
                if impossible:
                    break
                continue
            with_pawn, without, a = pair
            c = score_cp(with_pawn)
            if MAX_ABS_CP is not None and abs(c - a) > MAX_ABS_CP:
                drops["outcome_filter"] += 1
                continue
            rows.append({"backdrop": kind, "file": f, "rank": r + 1, "friendly": friendly,
                         "enemy": enemy, "sample": b, "cp_without": a, "cp_with": c, "value": c - a,
                         "fen": with_pawn.fen()})
            kept += 1
        if (ci + 1) % 50 == 0:
            print("  cell %d/%d  kept=%d  drops=%s" % (ci + 1, len(cells), kept, drops))

    engine.quit()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["backdrop", "file", "rank", "friendly", "enemy", "sample",
                                           "cp_without", "cp_with", "value", "fen"])
        w.writeheader()
        w.writerows(rows)
    print("wrote %s  (kept %d)  drops=%s" % (OUT, kept, drops))
    if drops["outcome_filter"]:
        print("  ⚠️ %d samples cut by MAX_ABS_CP -- that is filtering on the dependent variable; "
              "the reported means are biased toward zero." % drops["outcome_filter"])


if __name__ == "__main__":
    main()
