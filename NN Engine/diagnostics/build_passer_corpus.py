# -*- coding: utf-8 -*-
"""Round 0 of the passer redesign: build a dedicated PASSER validation corpus, phase-labeled, SF18-anchored,
with explicit BLOW-UP GUARDS. Sources:
  A) mined passer positions from position_bank.csv (rows already SF18-labeled) where a passed pawn exists;
  B) the positional-collapse dossier failure FENs (our known passer misreads);
  C) hand-built endgame guards (drawn doubled R+P + its single-pawn control; blockaded/連 passers vs a piece)
     — SF18 labels them so the TRUTH tier is correct regardless of construction.
Tiers (by our_total vs SF18, oriented to the passer owner): under_fire (we UNDER-credit a passer),
blowup_guard (we OVER-credit the pawn side), control (small gap). Phase-bucketed.

Output: ks_sets/passer_corpus.csv (fit_corpus-compatible schema + a `phase`/`tier`/`has_passer` set).
  pyrun diagnostics/build_passer_corpus.py [DEPTH=20]
"""
import os, sys, csv
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'; os.environ.setdefault('KS_SAFE_CHECK_DEF', '5')
DEPTH = int(os.environ.get('DEPTH', '20'))
THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
sys.path.insert(0, os.path.join(os.path.dirname(THIS), "selfplay"))
import chess, chess.engine
from collections import Counter
from ChessAI import ChessAI
from arbiter import find_stockfish
ai = ChessAI(None, None, chess.Board(), True)
BANK = os.path.join(THIS, "ks_sets", "position_bank.csv")
OUT = os.path.join(THIS, "ks_sets", "passer_corpus.csv")


def passers(board):
    """Return (n_white_passed, n_black_passed) via textbook mask (no enemy pawn on file/adjacent ahead)."""
    wp = bp = 0
    for color in (chess.WHITE, chess.BLACK):
        them = not color
        for sq in board.pieces(chess.PAWN, color):
            f, r = chess.square_file(sq), chess.square_rank(sq)
            span = 0
            for df in (-1, 0, 1):
                ff = f + df
                if 0 <= ff <= 7:
                    for rr in (range(r + 1, 8) if color == chess.WHITE else range(0, r)):
                        span |= chess.BB_SQUARES[chess.square(ff, rr)]
            if not (board.pieces(chess.PAWN, them) & chess.SquareSet(span)):
                if color == chess.WHITE:
                    wp += 1
                else:
                    bp += 1
    return wp, bp


def phase_bucket(ps):
    ps = float(ps)
    return "opening" if ps < 24 else "midgame" if ps < 64 else "endgame" if ps < 104 else "adveg"


DOSSIER = [  # our known passer misreads (SF18 will be measured)
    "8/2r4k/8/6PK/6P1/8/1R6/8 w - - 7 59",                    # drawn doubled R+P (#1) — over-value guard
    "rr4k1/5pbp/3N1np1/3P4/5N2/pp3PPB/1n1B3P/4RK1R w - - 0 26",  # black a3/b3 connected passers (#4) — under-fire
    "4k3/3qb1p1/2np3P/p3p3/Q3P3/p3B3/5P1K/5B2 b - - 0 32",    # #5
    "4B3/8/P3k3/2p5/2P1pp1P/1P2P3/3r4/1K6 w - - 0 60",        # e4 promotes vs a6 (#6)
]
GUARDS = [  # hand-built; SF18 labels the truth (all validated below)
    "8/7k/8/6P1/6K1/8/1R6/2r5 w - - 0 1",                     # single g-pawn control (vs the doubled draw)
    "6k1/8/8/8/2ppp3/8/6B1/6K1 b - - 0 1",                    # 3 connected passers vs bishop (both kings present)
    "8/8/3k4/2ppp3/8/5B2/8/6K1 b - - 0 1",                    # connected passers vs bishop, kings closer
]

rows_bank = [r for r in csv.DictReader(open(BANK)) if r.get("sf18", "") not in ("", None)]
sf18 = chess.engine.SimpleEngine.popen_uci(find_stockfish())


def s18(fen):
    s = sf18.analyse(chess.Board(fen), chess.engine.Limit(depth=DEPTH))["score"].white()
    return 99.0 if (s.is_mate() and s.mate() > 0) else (-99.0 if s.is_mate() else s.score() / 100.0)


def sgn(x):
    return (x > 0) - (x < 0)


out = []


def add(fen, sf18v, src):
    b = chess.Board(fen)
    if sf18v is None or not b.is_valid():
        print("  skip invalid: %s" % fen); return
    bd = ai.ev_breakdown(b)
    if bd.get("checkmate"):
        return
    wp, bp = passers(b)
    if wp + bp == 0 and src == "bank":
        return                                   # bank rows: keep only real passer positions
    our = -bd.get("total", 0.0) / 1000.0
    ps = bd.get("phase_score", 64)
    gap = our - sf18v                            # white-POV: + = we over-read white
    # orient to the side that owns the (net) passer
    owner = 1 if wp >= bp else -1                # +1 white owns more passers
    over = owner * gap                            # + = we over-credit the passer owner
    if abs(sf18v) >= 98:
        return
    if over >= 1.5:
        tier = "blowup_guard"
    elif over <= -1.0:
        tier = "under_fire"
    else:
        tier = "control"
    out.append({"fen": fen, "sf18": round(sf18v, 2), "our_total": round(our, 3),
                "wp": wp, "bp": bp, "phase": phase_bucket(ps), "tier": tier, "src": src,
                "over_read": round(over, 3)})


for r in rows_bank:
    try:
        add(r["fen"], float(r["sf18"]), "bank")
    except Exception:
        continue
for fen in DOSSIER + GUARDS:
    src = "dossier" if fen in DOSSIER else "guard"
    if not chess.Board(fen).is_valid():
        print("  skip invalid %s: %s" % (src, fen)); continue
    add(fen, s18(fen), src)
sf18.quit()

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["fen", "sf18", "our_total", "wp", "bp", "phase", "tier", "src", "over_read"])
    w.writeheader(); w.writerows(out)
print("passer corpus: %d rows -> %s" % (len(out), OUT))
print("tiers:", dict(Counter(r["tier"] for r in out)))
print("phases:", dict(Counter(r["phase"] for r in out)))
print("srcs:", dict(Counter(r["src"] for r in out)))
