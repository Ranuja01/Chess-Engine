# -*- coding: utf-8 -*-
"""
Static-eval term attribution — localize WHY our static eval over-reads a position.

Self-play annotation surfaced a systematic over-valuation (the engine reads ~+8-9 pawns hotter than
Stockfish's SEARCH eval in won positions). But our eval is a STATIC (leaf) eval and `sf_cp` is a SEARCH
eval, so comparing them conflates two things: genuine static miscalibration AND the legitimate static-vs-search
gap. The clean, apples-to-apples yardstick is Stockfish's OWN static eval (its NNUE eval, via the UCI `eval`
command). So each position carries THREE numbers, all White-POV:

    our_static   - our handcrafted static eval (ChessAI.ev_breakdown total)
    SF_static    - Stockfish's NNUE static eval (leaf, no search)   <- calibration target
    SF_search    - Stockfish's search eval                          <- the IDEAL value

Decision gate (headline metric = our_static - SF_static):
  - our_static >> SF_static            -> our static eval IS miscalibrated; the per-term breakdown localizes
                                          the offending weight -> taper/calibrate it.
  - our_static ~ SF_static >> SF_search-> our static eval is FINE (+14 is a "correct" static read SF shares);
                                          search pulls it to +6 because the position is only worth +6
                                          (conversion difficulty). The bug is SEARCH/technique, NOT eval weights.
  - our_static ~ SF_static ~ SF_search -> nothing wrong at this FEN.

The per-term breakdown comes from ChessAI.ev_breakdown, which runs the REAL placement_and_piece_eval with a
capture flag (the numbers are exactly what search uses; see EvalBreakdown in cpp_bitboard.h).

Run in WSL, from NN Engine/, at the SAME interpreter that built ChessAI:

    python diagnostics/eval_breakdown.py --fen "8/p4k1p/5Q2/3N4/8/5PK1/Pr2r1PP/8 b - - 0 1" [more FENs...]
    python diagnostics/eval_breakdown.py --tag ship_blitz --top 10
    python diagnostics/eval_breakdown.py --tag ship_standard --top 10 --sf-movetime 0.5

--tag reads the already-annotated self-play games (selfplay/games/<tag>/game_*/game.annotated.jsonl), ranks
plies by engine-vs-SF divergence, and takes the worst --top N. SF_search there is the recorded sf_cp (no
re-search); SF_static is lifted live for just those FENs. --fen computes both SF numbers live (if Stockfish is
found); without Stockfish it still prints our_static + the term breakdown.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # silence TensorFlow startup chatter

import sys
import glob
import json
import argparse

import chess

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.path.dirname(THIS_DIR)            # NN Engine/  (has ChessAI*.so)
SELFPLAY_DIR = os.path.join(ENGINE_DIR, "selfplay")
sys.path.insert(0, ENGINE_DIR)
sys.path.insert(0, SELFPLAY_DIR)

# Engine units are ~ centipawns x 10 (pawn = 1000). Our eval is ABSOLUTE (positive favours Black); the search
# flips it once. Normalize every engine number to White-POV PAWNS: white_pawns = -ev_abs / 1000.
ENGINE_UNITS_PER_PAWN = 1000.0

# The additive terms whose contributions to `total` are captured (material is informational, excluded).
ADDITIVE_TERMS = [
    "pieces", "capture_gains", "passed_pawn_support", "latent_threat", "central",
    "imbalance_white", "imbalance_black", "pair_bonus", "piece_value_boost",
]


def white_pawns(ev_abs):
    """Engine absolute (Black-positive) milli-pawns -> White-POV pawns."""
    return -ev_abs / ENGINE_UNITS_PER_PAWN


def _load_engine():
    # Models unused by the C++ engine; the constructor signature still requires them.
    from ChessAI import ChessAI
    return ChessAI


def _gather_from_tag(tag, top, min_div):
    """Read annotated self-play games under games/<tag>/, rank plies by engine-vs-SF divergence (in playable
    positions), return up to `top` items: dict(fen, provenance, search_white, sf_search_white, div)."""
    logdir = os.path.join(SELFPLAY_DIR, "games", tag)
    paths = sorted(glob.glob(os.path.join(logdir, "game_*", "game.annotated.jsonl")))
    if not paths:
        print("[eval_breakdown] no annotated games under %s "
              "(run selfplay/annotate.py --tag %s first)" % (logdir, tag))
        return []
    items = []
    for p in paths:
        game = os.path.basename(os.path.dirname(p))
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("type") != "move" or r.get("opening"):
                    continue
                ev, cp, fen = r.get("eval_white_pov"), r.get("sf_cp"), r.get("fen")
                if not (isinstance(ev, int) and isinstance(cp, int) and fen):
                    continue
                search_white = ev / 1000.0      # engine search eval, White-POV pawns
                sf_search_white = cp / 100.0    # SF search eval, White-POV pawns
                div = abs(search_white - sf_search_white)
                if div < min_div:
                    continue
                items.append({
                    "fen": fen,
                    "provenance": "%s/%s ply%s" % (tag, game, r.get("ply")),
                    "search_white": search_white,
                    "sf_search_white": sf_search_white,
                    "div": div,
                })
    items.sort(key=lambda d: d["div"], reverse=True)
    return items[:top]


def _gather_from_fens(fens):
    return [{"fen": fen, "provenance": "cli", "search_white": None,
             "sf_search_white": None, "div": None} for fen in fens]


def _reconstruction(bd):
    """Return (ok, delta). additive terms must sum to total, except when advanced_endgame_eval fired (it
    REPLACES total, so total = advanced_endgame_total + pair_bonus + piece_value_boost)."""
    if bd.get("advanced_endgame_fired"):
        expected = bd["advanced_endgame_total"] + bd["pair_bonus"] + bd["piece_value_boost"]
    else:
        expected = sum(bd[k] for k in ADDITIVE_TERMS)
    return expected == bd["total"], bd["total"] - expected


def _fmt(p):
    return "  n/a " if p is None else "%+7.2f" % p


def main():
    ap = argparse.ArgumentParser(description="Static-eval term attribution (our_static vs SF_static vs SF_search).")
    ap.add_argument("--fen", nargs="+", help="explicit FEN(s) to attribute")
    ap.add_argument("--tag", help="rank plies of games/<tag>/ by engine-vs-SF divergence and attribute the worst")
    ap.add_argument("--top", type=int, default=10, help="with --tag: how many top-divergence plies (default 10)")
    ap.add_argument("--min-div", type=float, default=0.0, help="with --tag: minimum divergence in pawns to include")
    ap.add_argument("--sf-path", default=None, help="Stockfish binary (else auto-detected)")
    ap.add_argument("--sf-movetime", type=float, default=0.5, help="SF search movetime for --fen SF_search (s)")
    ap.add_argument("--sf-depth", type=int, default=None, help="SF search fixed depth for --fen SF_search")
    args = ap.parse_args()

    if not args.fen and not args.tag:
        ap.error("need --fen or --tag")

    items = _gather_from_tag(args.tag, args.top, args.min_div) if args.tag else _gather_from_fens(args.fen)
    if not items:
        return

    # Stockfish: needed for SF_static always, and for SF_search in --fen mode. Optional — degrade gracefully.
    arbiter = None
    try:
        from arbiter import Arbiter, find_stockfish
        sf = args.sf_path or find_stockfish()
        if sf:
            arbiter = Arbiter(sf, movetime=args.sf_movetime, depth=args.sf_depth)
        else:
            print("[eval_breakdown] no Stockfish found (set STOCKFISH_PATH or --sf-path); "
                  "SF_static/SF_search will be n/a\n")
    except Exception as e:
        print("[eval_breakdown] Stockfish unavailable (%s); SF columns n/a\n" % e)

    ChessAI = _load_engine()
    seed = chess.Board()
    ai = ChessAI(None, None, seed, seed.turn)  # one warm instance; ev_breakdown reads only the board arg

    try:
        for it in items:
            board = chess.Board(it["fen"])
            bd = ai.ev_breakdown(board)

            sf_static_white = None
            sf_search_white = it["sf_search_white"]
            if arbiter is not None:
                sf_static_white = (lambda c: None if c is None else c / 100.0)(arbiter.evaluate_static(board))
                if sf_search_white is None:  # --fen mode: compute SF search live
                    cp, _, _ = arbiter.evaluate(board)
                    sf_search_white = None if cp is None else cp / 100.0

            if bd.get("checkmate"):
                print("=" * 100)
                print("%s\n  %s\n  CHECKMATE (static eval not defined)" % (it["provenance"], it["fen"]))
                continue

            our_static_white = white_pawns(bd["total"])
            calib_gap = None if sf_static_white is None else our_static_white - sf_static_white
            svs_gap = None if (sf_static_white is None or sf_search_white is None) else sf_static_white - sf_search_white
            ok, delta = _reconstruction(bd)

            print("=" * 100)
            print("%s   %s%s" % (it["provenance"], it["fen"],
                                 ("   [div %.2f]" % it["div"]) if it["div"] is not None else ""))
            print("  our_static %s | SF_static %s | SF_search %s   (White-POV pawns)"
                  % (_fmt(our_static_white), _fmt(sf_static_white), _fmt(sf_search_white)))
            if it["search_white"] is not None:
                print("  (engine SEARCH eval at this ply: %s)" % _fmt(it["search_white"]))
            print("  >> calibration gap our_static - SF_static = %s   |   SF_static - SF_search = %s"
                  % (_fmt(calib_gap), _fmt(svs_gap)))
            phase = "endgame" if bd["is_endgame"] else "midgame"
            print("  phase_score %d (%s)%s" % (bd["phase_score"], phase,
                  "  ADVANCED-ENDGAME REPLACE fired (additive terms are pre-replace)" if bd["advanced_endgame_fired"] else ""))

            # Per-term breakdown, White-POV pawns, sorted by magnitude. material shown separately (informational).
            terms = sorted(((k, white_pawns(bd[k])) for k in ADDITIVE_TERMS), key=lambda kv: abs(kv[1]), reverse=True)
            print("  term contributions (White-POV pawns, + favours White):")
            for k, v in terms:
                print("      %-22s %+8.2f" % (k, v))
            print("      %-22s %+8.2f   [informational, not in sum]" % ("material", white_pawns(bd["material"])))
            if bd["advanced_endgame_fired"]:
                print("      %-22s %+8.2f   [post-replace total]" % ("advanced_endgame_total", white_pawns(bd["advanced_endgame_total"])))
            print("  reconstruction: %s (delta=%d)" % ("OK" if ok else "MISMATCH", delta))
    finally:
        if arbiter is not None:
            arbiter.close()


if __name__ == "__main__":
    main()
