# -*- coding: utf-8 -*-
"""Side-by-side FULL term breakdown -- ours vs SF11-static -- for a hand-picked FEN list.

The dossier prints only the terms that map cleanly between engines. For the owner's chess-logic pass we
need EVERYTHING both engines say, because the question is "which term is SF11 using that we are not",
and a term we never considered will not appear in a curated subset.

FENs are read from a FILE, never argv: the dispatcher splits arguments on spaces, so a FEN passed on the
command line arrives shredded into eight pieces.

⚠️ Term names are NOT 1:1 across engines. SF folds PSQT into Material; our pt_* are pure placement; our
capture_gains has no SF analogue at all. Read the SHAPE (which side each engine credits, and by how
much), never a term-name equality.

  pyrun diagnostics/_logic_pass_breakdown.py [FENS=diagnostics/_logic_pass.fens] [DEPTH=18]
"""
import os, sys

for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
THIS = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(THIS)
sys.path.insert(0, ENGINE); sys.path.insert(0, THIS)
sys.path.insert(0, os.path.join(ENGINE, "selfplay"))

import chess, chess.engine
from ChessAI import ChessAI
from eval_vs_sf11 import SF11Eval, SF11
from arbiter import find_stockfish

DEPTH = int(os.environ.get("DEPTH", "18"))
FENS = os.environ.get("FENS", os.path.join(THIS, "_logic_pass.fens"))
if not os.path.isabs(FENS):
    FENS = os.path.join(ENGINE, FENS)

ai = ChessAI(None, None, chess.Board(), True)
sf11 = SF11Eval(SF11)
sf18 = chess.engine.SimpleEngine.popen_uci(find_stockfish())

fens = [ln.strip() for ln in open(FENS) if ln.strip() and not ln.startswith("#")]

for i, fen in enumerate(fens, 1):
    b = chess.Board(fen)
    # Our eval is absolute Black-positive millipawns; White-POV pawns = -v/1000.
    bd = ai.ev_breakdown(b)
    our_total = -bd.get("total", 0) / 1000.0
    sf_total, sf_terms = sf11.eval(fen)
    info = sf18.analyse(b, chess.engine.Limit(depth=DEPTH))["score"].white()
    s18 = 99.0 if (info.is_mate() and info.mate() > 0) else (-99.0 if info.is_mate() else info.score() / 100.0)

    print("\n" + "=" * 78)
    print("[%d] %s" % (i, fen))
    print("    ours %+.2f   SF11-static %+.2f   SF18-search(d%d) %+.2f   |  our error vs SF18 %+.2f"
          % (our_total, sf_total, DEPTH, s18, our_total - s18))
    print("-" * 78)

    ours = [(k, -v / 1000.0) for k, v in bd.items()
            if k != "total" and isinstance(v, (int, float)) and abs(v) >= 1]
    ours.sort(key=lambda kv: abs(kv[1]), reverse=True)
    sfs = sorted(sf_terms.items(), key=lambda kv: abs(kv[1]), reverse=True)

    print("  %-34s | %s" % ("OURS (White-POV pawns)", "SF11-STATIC (White-POV pawns)"))
    for r in range(max(len(ours), len(sfs))):
        lhs = "%-24s %+7.2f" % ours[r] if r < len(ours) else ""
        rhs = "%-24s %+7.2f" % sfs[r] if r < len(sfs) else ""
        print("  %-34s | %s" % (lhs, rhs))

sf18.quit()
sf11.close()
print()
