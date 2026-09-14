# -*- coding: utf-8 -*-
"""Find DRAW-FLAGGED, TABLEBASE-WON positions with DTM >= MIN_DTM plies, for the interior-pruning test.

WHY. The short-mate demonstration so far used a mate-in-1 (6nk/8/6K1/4N3/8/8/8/8 w), which is played at the
ROOT -- and the root is never pruned. The worry the DTM-weighted gate must answer is RFP/futility/null-move
cutting a mating line INSIDE the tree, which only a mate >= 3 plies deep exercises.
Hand geometry is unreliable (even the mate-in-1 needed the tablebase), and random sampling found no SHORT
false positive at all in 240 corner-biased samples. So: construct the geometry these mates need -- a
defending king in a CORNER, boxed by its OWN minor on an adjacent square, attacking king close -- and let the
tablebase decide which are wins and how deep.

Reuses v2_case / tb_lookup / cheb from diagnostics/_draw_oracle.py (and its cache, so repeats are free).
⚠️ Run only when no other tablebase crawl is in flight -- two concurrent crawlers is impolite to the API.

  FOUT=<file> MIN_DTM=3 BUDGET=300 WANT=4 SEED=7 DELAY=0.3 [ONLY_SHIPPED=1]  pyrun diagnostics/_draw_deep_mate_finder.py

RESULT 2026-09-13: 0 wins with DTM >= 3 in 300 queries (844 candidates) over KNvKN/KBvKB; across the day's whole
tablebase cache every forced win in 1,592 minor-piece positions was DTM 1. An exit of 1 ("0 found") here is the
expected, informative outcome for these endings, not a failure of the tool.
"""
import os, sys, random

THIS   = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(THIS)                         # diagnostics/ -> NN Engine/
sys.path.insert(0, THIS)
os.chdir(ENGINE)

import chess
import _draw_oracle as O          # module-level only parses env + loads the cache; main() is not run

MIN_DTM = int(os.environ.get("MIN_DTM", "3"))
BUDGET  = int(os.environ.get("BUDGET", "300"))      # max NEW tablebase queries
WANT    = int(os.environ.get("WANT", "4"))
SEED    = int(os.environ.get("SEED", "7"))
FOUT    = os.environ.get("FOUT")
O.DELAY = float(os.environ.get("DELAY", "0.3"))

CORNERS = {0: (1, 8, 9), 7: (6, 15, 14), 56: (57, 48, 49), 63: (62, 55, 54)}   # corner -> adjacent squares

# (defender's minor, attacker's minor) -- the endings v2 flags drawn where boxed-king mates exist.
ENDINGS = [
    ("KNvKN", chess.KNIGHT, chess.KNIGHT),
    ("KBvKN", chess.KNIGHT, chess.BISHOP),   # defender knight self-blocks, attacker bishop mates
    ("KBvKN", chess.BISHOP, chess.KNIGHT),   # defender bishop self-blocks, attacker knight mates
    ("KBvKB", chess.BISHOP, chess.BISHOP),
]


def candidate(rng):
    name, def_minor, att_minor = rng.choice(ENDINGS)
    def_white = rng.random() < 0.5
    corner = rng.choice(list(CORNERS))
    block_sq = rng.choice(CORNERS[corner])
    ring2 = [s for s in range(64) if O.cheb(s, corner) == 2]
    att_king = rng.choice(ring2)
    free = [s for s in range(64) if s not in (corner, block_sq, att_king)]
    att_minor_sq = rng.choice(free)
    bd = chess.Board.empty()
    dc, ac = (chess.WHITE, chess.BLACK) if def_white else (chess.BLACK, chess.WHITE)
    bd.set_piece_at(corner, chess.Piece(chess.KING, dc))
    bd.set_piece_at(block_sq, chess.Piece(def_minor, dc))
    bd.set_piece_at(att_king, chess.Piece(chess.KING, ac))
    bd.set_piece_at(att_minor_sq, chess.Piece(att_minor, ac))
    bd.turn = ac                              # attacker to move: that is where forced mates are
    if not bd.is_valid() or bd.is_game_over():
        return None, None
    return name, bd


def main():
    # ☠️ The SEARCH demonstration needs positions the SHIPPED C++ `draw_class` actually flags -- a mate in an ending the
    # engine does not flag would never trigger the rule, so "search still finds it" would prove nothing.
    # ⚠️ KEEP THIS SET IN LOCKSTEP WITH eval_v2.cpp draw_class. It was {KvK, KBvK, KNvK, eq_only_minor} for the
    # 2026-09-13 run; KBvKN, KNNvK and the fortress wrong-bishop case were added to the C++ the same day, after
    # that run. ONLY_SHIPPED=0 includes every rule the oracle mirrors, shipped or not.
    only_shipped = os.environ.get("ONLY_SHIPPED", "1") == "1"
    shipped_cases = {"KvK", "KBvK", "KNvK", "eq_only_minor", "KBvKN", "KNNvK", "sf_wrongB"}
    rng = random.Random(SEED)
    found, seen = [], set()
    start_net = O._net[0]
    tries = 0
    while len(found) < WANT and (O._net[0] - start_net) < BUDGET and tries < BUDGET * 200:
        tries += 1
        name, bd = candidate(rng)
        if bd is None:
            continue
        fen = bd.fen()
        if fen in seen:
            continue
        seen.add(fen)
        case = O.v2_case(bd)
        if case is None:                       # must be a position v2's draw rule actually flags
            continue
        if only_shipped and case not in shipped_cases:
            continue
        cat, dtm = O.tb_lookup(fen)
        if cat in ("win", "loss") and dtm is not None and dtm >= MIN_DTM:
            found.append((dtm, name, fen))
            print("  FOUND  %-6s dtm=%-3d %s" % (name, dtm, fen), flush=True)
    O.save_cache()
    print("\n%d found (dtm >= %d); %d new TB queries; %d candidates tried" % (
        len(found), MIN_DTM, O._net[0] - start_net, tries))
    if FOUT and found:
        with open(FOUT, "w") as f:
            for dtm, name, fen in sorted(found):
                f.write(fen + "\n")
        print("wrote %s" % FOUT)
    return 0 if found else 1


if __name__ == "__main__":
    sys.exit(main())
