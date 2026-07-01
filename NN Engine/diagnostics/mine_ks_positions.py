# -*- coding: utf-8 -*-
"""Mine a KING-ATTACK move-match set from real self-play games (needs SF; one-time interop step).

KS_INTERACT's benefit is invisible to general move-match (king_safety ~ 0 on positional themes), so it must
be tuned where the RIGHT move is a king attack/defense. This builds that set from OUR real games: it ranks
annotated plies by our-eval-vs-SF divergence (our misreads), then keeps the ones where SF's best move is a
king attack — a CHECK or a move landing in/next to the enemy king's zone — with a decisive SF eval. Those are
positions where we actually misjudged king safety, and where a stronger KS term should now pick SF's move.

Emits STS-schema EPD (same shape as label_collapses / load_sts_epd consumes):
    <epd4> bm <SAN>; id "ksatk(v1) KSAtk.NNN"; c8 "10"; c9 "<best_uci>";
-> diagnostics/suites/ksattack.epd, plus merged ksattack_corpus.epd (= STS + king-attack) for held-out shards.

Run via the dispatcher (STOCKFISH_PATH exported by `pyrun`); after an interop restore run it FIRST:
    bash overnight_runner.sh pyrun diagnostics/mine_ks_positions.py [tag ...]
Default tags = the annotated corpora. Tunables via env: KS_MINE_DIV (min |our-SF| pawns, default 1.5),
KS_MINE_CP (min |SF cp| decisive, default 150), KS_MINE_TOP (max plies to SF-probe, default 600).
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import glob
import json

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.path.dirname(THIS_DIR)
SELFPLAY_DIR = os.path.join(ENGINE_DIR, "selfplay")
SUITES_DIR = os.path.join(THIS_DIR, "suites")
GAMES_DIR = os.path.join(SELFPLAY_DIR, "games")
STS = os.path.join(SUITES_DIR, "STS1-STS15_LAN_v3.epd")
OUT = os.path.join(SUITES_DIR, "ksattack.epd")
MERGED = os.path.join(SUITES_DIR, "ksattack_corpus.epd")
sys.path.insert(0, ENGINE_DIR)
sys.path.insert(0, SELFPLAY_DIR)

import chess  # noqa: E402
from arbiter import Arbiter, find_stockfish  # noqa: E402

DEFAULT_TAGS = ["away_lightning", "hlmr_more1_lightning", "away_blitz", "nmp_lightning",
                "nmp_standard", "jun6_blitz", "jun6_standard", "away_standard"]
MIN_DIV = float(os.environ.get("KS_MINE_DIV", "1.5"))   # our-vs-SF divergence in pawns
MIN_CP = int(os.environ.get("KS_MINE_CP", "150"))       # decisive SF eval (cp)
TOP = int(os.environ.get("KS_MINE_TOP", "600"))         # cap on SF probes
SF_DEPTH = int(os.environ.get("KS_MINE_SF_DEPTH", "18"))


def gather(tags):
    """Annotated plies across tags, ranked by |our_eval - sf_cp| (our misreads), de-duped by FEN."""
    seen, items = set(), []
    for tag in tags:
        for p in sorted(glob.glob(os.path.join(GAMES_DIR, tag, "game_*", "game.annotated.jsonl"))):
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    if r.get("type") != "move" or r.get("opening"):
                        continue
                    ev, cp, fen = r.get("eval_white_pov"), r.get("sf_cp"), r.get("fen")
                    if not (isinstance(ev, int) and isinstance(cp, int) and fen) or fen in seen:
                        continue
                    seen.add(fen)
                    div = abs(ev / 1000.0 - cp / 100.0)   # both White-POV pawns
                    if div >= MIN_DIV:
                        items.append((div, fen))
    items.sort(reverse=True)
    return [fen for _, fen in items[:TOP]]


def is_king_attack(board, move):
    """SF's best move is a king attack: it gives check, or lands within 2 squares of the enemy king."""
    if board.gives_check(move):
        return True
    ek = board.king(not board.turn)
    return ek is not None and chess.square_distance(move.to_square, ek) <= 2


def main():
    tags = sys.argv[1:] or DEFAULT_TAGS
    sf = find_stockfish()
    if not sf:
        print("ERROR: Stockfish not found (set STOCKFISH_PATH).")
        return 1
    fens = gather(tags)
    print("gathered %d misread plies (div>=%.1f) across %d tags; SF-probing for king-attacks..."
          % (len(fens), MIN_DIV, len(tags)))
    arb = Arbiter(sf, depth=SF_DEPTH)
    lines = []
    try:
        for fen in fens:
            try:
                board = chess.Board(fen)
            except Exception:
                continue
            if board.is_game_over():
                continue
            cp, best, _ = arb.evaluate(board)
            if not best or cp is None or abs(cp) < MIN_CP:
                continue
            mv = chess.Move.from_uci(best)
            if not is_king_attack(board, mv):
                continue
            epd4 = " ".join(fen.split()[:4])
            lines.append('%s bm %s; id "ksatk(v1) KSAtk.%03d"; c8 "10"; c9 "%s";'
                         % (epd4, board.san(mv), len(lines), best))
    finally:
        arb.close()

    if not lines:
        print("no king-attack positions found.")
        return 1
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    sts = [ln.rstrip("\n") for ln in open(STS)] if os.path.exists(STS) else []
    with open(MERGED, "w") as f:
        f.write("\n".join(sts + lines) + "\n")
    # self-verify the EPD round-trips through the move-match loader
    from sts_test import load_sts_epd
    parsed = load_sts_epd(MERGED)
    nk = sum(1 for p in parsed if p[4].startswith("ksatk"))
    print("king-attack positions: %d -> %s" % (len(lines), OUT))
    print("merged: %d STS + %d KSAtk -> %s (verify: loader parsed %d king-attack)"
          % (len(sts), len(lines), MERGED, nk))
    return 0 if nk == len(lines) else 1


if __name__ == "__main__":
    sys.exit(main())
