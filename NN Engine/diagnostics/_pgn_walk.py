# -*- coding: utf-8 -*-
"""Walk a pasted game, SF-evaluate every ply, and report where a side's position actually collapsed.

`annotate` works on games/<tag>/ directories produced by the harness; this is for a game pasted in by hand
(a UI game, a bot game). It prints the SF eval trajectory in White-POV centipawns and ranks the plies by how
much the SIDE OF INTEREST lost on its own move -- i.e. the blunder list, not the eval curve.

Also dumps our own static eval at the worst plies so the two can be compared directly, which is the first
step of the standard triangulation (ours vs SF-static vs SF-search).

Run: bash <runner> pyrun diagnostics/_pgn_walk.py SIDE=black DEPTH=12 [KEY=VAL engine knobs] PGN=<file>
     (PGN file = movetext, SAN, headers optional. Defaults to the built-in sample if absent.)
"""
import os, sys
OPTS = {}
for a in sys.argv[1:]:
    if '=' in a:
        k, v = a.split('=', 1)
        if k in ('SIDE', 'DEPTH', 'PGN', 'TOP'):
            OPTS[k] = v
        else:
            os.environ[k] = v            # engine knob, before ChessAI init
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import io, chess, chess.pgn, chess.engine
THIS_DIR = os.path.dirname(os.path.abspath(__file__)); ENGINE_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, ENGINE_DIR); sys.path.insert(0, THIS_DIR)
from ChessAI import ChessAI
seed = chess.Board(); ai = ChessAI(None, None, seed, seed.turn)

SIDE = OPTS.get('SIDE', 'black').lower()
DEPTH = int(OPTS.get('DEPTH', 12))
TOP = int(OPTS.get('TOP', 8))
SF = os.environ.get('STOCKFISH_PATH')

pgn_path = OPTS.get('PGN')
if pgn_path and os.path.exists(pgn_path):
    text = open(pgn_path).read()
else:
    print("!! no PGN file given/found; pass PGN=<path>")
    sys.exit(1)

game = chess.pgn.read_game(io.StringIO(text))
if game is None:
    print("!! could not parse PGN")
    sys.exit(1)

eng = chess.engine.SimpleEngine.popen_uci(SF)
eng.configure({"Threads": 1})

board = game.board()
rows = []   # (ply, movenum, side, san, fen_before, sf_before, sf_after)
prev = None
for mv in game.mainline_moves():
    fen_before = board.fen()
    info = eng.analyse(board, chess.engine.Limit(depth=DEPTH))
    sf_before = info["score"].white().score(mate_score=100000)
    san = board.san(mv)
    mover = board.turn                              # True = white to move
    board.push(mv)
    info2 = eng.analyse(board, chess.engine.Limit(depth=DEPTH))
    sf_after = info2["score"].white().score(mate_score=100000)
    rows.append((len(rows) + 1, board.fullmove_number, mover, san, fen_before, sf_before, sf_after))
eng.quit()

want_white = (SIDE == 'white')
print("SF eval trajectory (White-POV cp, depth %d).  side of interest = %s" % (DEPTH, SIDE.upper()))
print("%-5s %-6s %-8s %9s %9s %8s" % ("ply", "move", "san", "before", "after", "delta"))
losses = []
for ply, mn, mover, san, fen, b, a in rows:
    if b is None or a is None:
        continue
    # loss on one's OWN move, signed so positive = damage done to the side of interest
    delta = (a - b) if not want_white else (b - a)
    tag = ""
    if mover == want_white:
        losses.append((delta, ply, mn, san, fen, b, a))
        tag = "  <- ours" if delta > 50 else ""
    print("%-5d %-6s %-8s %9s %9s %8s%s" % (ply, "%d%s" % (mn, "." if mover else "..."), san,
                                            b, a, delta if mover == want_white else "", tag))

print()
print("WORST %d moves by %s (SF cp lost on our own move):" % (TOP, SIDE.upper()))
losses.sort(reverse=True)
for delta, ply, mn, san, fen, b, a in losses[:TOP]:
    print("  ply %-3d  %d%s %-8s  %+6d -> %+6d   LOST %d cp" % (ply, mn, "." if not want_white else ".", san, b, a, delta))
    print("      fen_before: %s" % fen)
    try:
        bd = chess.Board(fen)
        ev = ai.ev_breakdown(bd)
        ks = ev.get('king_safety', 0)
        print("      ours static total=%+.2f  king_safety=%+.2f (Black-positive)" % (ev.get('total', 0) / 1000.0, ks / 1000.0))
    except Exception as e:
        print("      (our eval failed: %s)" % e)
