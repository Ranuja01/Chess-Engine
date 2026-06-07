# -*- coding: utf-8 -*-
"""
Eval symmetry probe — two controlled tests to localize the depth-flip color instability.

The 2026-06-05 tournament re-analysis (see memory tournament-color-bias-diagnostics) found that
identical-engine self-play produces a large color win-skew that FLIPS with search depth, from
SF-equal openings, while search EFFORT is color-symmetric. That points the finger at the EVAL, with
two competing hypotheses:

  (a) a side-to-move / TEMPO asymmetry (the eval over-rewards whoever is on move), which an even/odd
      search-depth parity would amplify into a color result skew; or
  (b) a latent COLOR asymmetry baked into the eval (it does not treat White and Black as mirror
      images), which is a direct bug regardless of depth.

This probe discriminates them on CONTROLLED positions (same FEN, not the confounded different
position-sets the game data gave us):

  COLOR-MIRROR test (hypothesis b): the eval is ABSOLUTE / Black-positive, so for any board B a
  perfectly color-symmetric eval must satisfy total(B.mirror()) == -total(B) term-by-term
  (mirror swaps colors + flips ranks + flips side-to-move). The residual total(B)+total(mirror)
  should be ~0; any nonzero per-term residual IS a color asymmetry in that term. (imbalance_white
  and imbalance_black swap roles under mirror, so they are paired accordingly.)

  TEMPO test (hypothesis a): evaluate the SAME position with White to move and with Black to move;
  the White-POV swing measures how much the eval favours the side on move. A symmetric, tempo-free
  static eval gives ~0; the game data suggested ~+0.5 pawn but was confounded.

Run in WSL, from NN Engine/, at the SAME interpreter that built ChessAI (no rebuild needed):

    python diagnostics/eval_symmetry.py --tag nmp_standard --sample 400
    python diagnostics/eval_symmetry.py --tag ship_lightning ship_blitz
    python diagnostics/eval_symmetry.py --fen "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 4"

--tag pulls FENs from the recorded games (annotated or raw JSONL) of games/<tag>/; --fens / --fen take
explicit FENs; with neither, a small built-in corpus (incl. the start position, which MUST read ~0) runs.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # silence TensorFlow startup chatter

import sys
import glob
import json
import argparse
import statistics as st

import chess

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.path.dirname(THIS_DIR)             # NN Engine/  (has ChessAI*.so)
SELFPLAY_DIR = os.path.join(ENGINE_DIR, "selfplay")
sys.path.insert(0, ENGINE_DIR)
sys.path.insert(0, SELFPLAY_DIR)

# eval_breakdown totals/terms are ABSOLUTE (Black-positive) milli-pawns (pawn = 1000).
# White-POV pawns = -ev_abs / 1000.
ENGINE_UNITS_PER_PAWN = 1000.0

ADDITIVE_TERMS = [
    "pieces", "capture_gains", "passed_pawn_support", "latent_threat", "central",
    "imbalance_white", "imbalance_black", "pair_bonus", "piece_value_boost",
]

# Under a color mirror, each term maps to its same-named counterpart EXCEPT the imbalance pair, which
# swaps roles (White's imbalance contribution becomes Black's and vice versa).
MIRROR_PARTNER = {t: t for t in ADDITIVE_TERMS}
MIRROR_PARTNER["imbalance_white"] = "imbalance_black"
MIRROR_PARTNER["imbalance_black"] = "imbalance_white"

# Per-piece-type split of the `pieces` term (midgame path only) — localizes which piece's eval branch
# is color-asymmetric. Under a color mirror a type maps to the same type, so residual = contrib + mirror.
PT_TERMS = ["pt_pawns", "pt_knights", "pt_bishops", "pt_rooks", "pt_queens", "pt_kings"]

BUILTIN_FENS = [
    chess.STARTING_FEN,  # symmetric -> all residuals MUST be ~0 (the probe's own sanity check)
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 4",
    "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 6 5",
    "rnbqk2r/ppp1bppp/4pn2/3p4/2PP4/2N1PN2/PP3PPP/R1BQKB1R w KQkq - 0 6",
    "r2q1rk1/pp2bppp/2n1pn2/2pp4/3P4/2P1PN2/PPB2PPP/RNBQ1RK1 w - - 0 9",
    "8/5pk1/6p1/7p/7P/6P1/5PK1/8 w - - 0 1",
    "8/2k5/3p4/p2P1p2/P2P1P2/8/8/4K3 w - - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
]


def white_pawns(ev_abs):
    return -ev_abs / ENGINE_UNITS_PER_PAWN


def _load_engine():
    from ChessAI import ChessAI
    return ChessAI


def _gather_from_tag(tag, sample):
    """FENs from games/<tag>/ (annotated preferred, raw fallback), skipping opening moves."""
    fens, seen = [], set()
    for sub in ("game.annotated.jsonl", "game.jsonl"):
        for p in sorted(glob.glob(os.path.join(SELFPLAY_DIR, "games", tag, "game_*", sub))):
            gdir = os.path.dirname(p)
            if gdir in seen:           # already took the annotated variant for this game
                continue
            seen.add(gdir)
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    if r.get("type") != "move" or r.get("opening"):
                        continue
                    fen = r.get("fen")
                    if fen:
                        fens.append(fen)
    if sample and len(fens) > sample:                # even stride down to ~sample positions
        step = len(fens) / float(sample)
        fens = [fens[int(i * step)] for i in range(sample)]
    return fens


def _stats(xs):
    if not xs:
        return "n=0"
    return "n=%d mean %+.3f median %+.3f max|%.3f|" % (
        len(xs), st.mean(xs), st.median(xs), max(abs(min(xs)), abs(max(xs))))


def main():
    ap = argparse.ArgumentParser(description="Eval color-mirror + tempo symmetry probe.")
    ap.add_argument("--tag", nargs="+", help="pull FENs from games/<tag>/ recordings")
    ap.add_argument("--fen", nargs="+", help="explicit FEN(s)")
    ap.add_argument("--fens", help="path to a file of FENs (one per line)")
    ap.add_argument("--sample", type=int, default=400, help="with --tag: cap positions per tag (even stride)")
    ap.add_argument("--worst", type=int, default=8, help="how many worst-offender FENs to list")
    args = ap.parse_args()

    fens = []
    if args.tag:
        for t in args.tag:
            fens += _gather_from_tag(t, args.sample)
    if args.fen:
        fens += args.fen
    if args.fens:
        with open(args.fens) as f:
            fens += [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    if not fens:
        fens = list(BUILTIN_FENS)
        print("[eval_symmetry] no --tag/--fen given; running the built-in corpus "
              "(start position residuals MUST be ~0).\n")

    ChessAI = _load_engine()
    seed = chess.Board()
    ai = ChessAI(None, None, seed, seed.turn)

    mirror_total = []                         # total(B)+total(mirror) in White-POV pawns (~0 if symmetric)
    mirror_terms = {t: [] for t in ADDITIVE_TERMS}
    mirror_pt = {t: [] for t in PT_TERMS}     # per-piece-type residual (midgame path)
    tempo = []                                # White-POV swing favouring the side to move (pawns)
    worst_mirror, worst_tempo = [], []

    for fen in fens:
        try:
            board = chess.Board(fen)
        except ValueError:
            continue

        bd = ai.ev_breakdown(board)
        if bd.get("checkmate"):
            continue

        # --- COLOR-MIRROR ---
        mb = ai.ev_breakdown(board.mirror())
        if not mb.get("checkmate"):
            res_total_pawns = white_pawns(bd["total"]) + white_pawns(mb["total"])  # both White-POV; sum~0 if symmetric
            mirror_total.append(res_total_pawns)
            worst_mirror.append((abs(res_total_pawns), res_total_pawns, fen))
            for t in ADDITIVE_TERMS:
                # term(B) + partner(mirror), in absolute milli-pawns; ~0 if that term is color-symmetric
                if t in bd and MIRROR_PARTNER[t] in mb:
                    mirror_terms[t].append((bd[t] + mb[MIRROR_PARTNER[t]]) / ENGINE_UNITS_PER_PAWN)
            for t in PT_TERMS:
                if t in bd and t in mb:
                    mirror_pt[t].append((bd[t] + mb[t]) / ENGINE_UNITS_PER_PAWN)

        # --- TEMPO (same position, flip side to move) ---
        flipped = board.copy(stack=False)
        flipped.turn = not flipped.turn
        fb = ai.ev_breakdown(flipped)
        if not fb.get("checkmate"):
            if board.turn == chess.WHITE:
                e_white_stm, e_black_stm = white_pawns(bd["total"]), white_pawns(fb["total"])
            else:
                e_white_stm, e_black_stm = white_pawns(fb["total"]), white_pawns(bd["total"])
            swing = e_white_stm - e_black_stm     # >0 => eval favours the side to move
            tempo.append(swing)
            worst_tempo.append((abs(swing), swing, fen))

    print("=" * 90)
    print("EVAL SYMMETRY PROBE   (%d positions evaluated)" % len(mirror_total))
    print("=" * 90)
    print("\nCOLOR-MIRROR residual  total(B)+total(mirror), White-POV pawns  (0 = color-symmetric):")
    print("  TOTAL: %s" % _stats(mirror_total))
    print("  per-term residual (White-POV pawns; nonzero => that term is color-asymmetric):")
    for t in sorted(ADDITIVE_TERMS, key=lambda k: (st.mean([abs(x) for x in mirror_terms[k]]) if mirror_terms[k] else 0), reverse=True):
        print("      %-22s %s" % (t, _stats(mirror_terms[t])))

    if any(mirror_pt[t] for t in PT_TERMS):
        print("  per-piece-type residual within `pieces` (midgame path; nonzero => that piece's eval branch is asymmetric):")
        for t in sorted(PT_TERMS, key=lambda k: (st.mean([abs(x) for x in mirror_pt[k]]) if mirror_pt[k] else 0), reverse=True):
            print("      %-22s %s" % (t, _stats(mirror_pt[t])))

    print("\nTEMPO swing  White-POV pawns favouring the side to move  (0 = tempo-free):")
    print("  %s" % _stats(tempo))

    if args.worst:
        print("\nworst COLOR-MIRROR offenders (|residual| pawns):")
        for _, r, fen in sorted(worst_mirror, reverse=True)[:args.worst]:
            print("    %+7.2f  %s" % (r, fen))
        print("worst TEMPO offenders (|swing| pawns):")
        for _, s, fen in sorted(worst_tempo, reverse=True)[:args.worst]:
            print("    %+7.2f  %s" % (s, fen))

    print("\nverdict: large per-term MIRROR residual => color-asymmetry bug (hypothesis b); "
          "small mirror residual but large TEMPO swing => side-to-move/parity (hypothesis a).")


if __name__ == "__main__":
    main()
