# -*- coding: utf-8 -*-
"""Eval invariance tests — ZERO Stockfish, seconds to run.

Two transformations leave the TRUE value of a position unchanged (up to a sign), so our eval must respect
them. Violations are outright bugs, not tuning problems:

  COLOUR SWAP  `board.mirror()` — flip ranks, swap piece colours, swap side to move and castling rights.
               Our eval is absolute Black-positive, so eval(mirror(b)) must equal  -eval(b).
               ⚠️ Colour asymmetry is a LIVE bug class here (see `eval-color-symmetry-fix`), which is why
               this is worth running before any fit: a descent will happily tune around a broken symmetry
               and bake the breakage into the constants.

  FILE MIRROR  a<->h. The rules are mirror-symmetric once castling rights are mirrored too, so we only
               test positions with NO castling rights and no en-passant file, where the transform is
               unambiguous. Our eval MAY legitimately differ here: `CHAIN_F_A..H` / `WALL_F_A..H` are
               per-file tables that are not constrained to be symmetric. So a failure is a QUESTION
               ("is that asymmetry deliberate?"), not automatically a bug — reported separately.

Also used as the augmentation answer: these transforms add no INFORMATION to a fit (they are exact
symmetries of the target, so they only impose constraints), and val/train is 0.988 so regularisation is
not our bottleneck. Their value is as a correctness check, which is what this is.

  pyrun diagnostics/_eval_symmetry.py [IN=ks_sets/diverse_corpus_wide.csv] [N=800] [TOL=0]
"""
import os, sys, csv

for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
import chess
from ChessAI import ChessAI

N = int(os.environ.get("N", "800"))
TOL = int(os.environ.get("TOL", "0"))          # millipawns; 0 = demand exactness
IN = os.environ.get("IN", "ks_sets/diverse_corpus_wide.csv")
if not os.path.isabs(IN):
    IN = os.path.join(THIS, IN)


def main():
    ai = ChessAI(None, None, chess.Board(), True)
    ev = lambda b: ai.ev_breakdown(b).get("total", 0)

    rows = list(csv.DictReader(open(IN, newline="")))[:N]
    col_bad, col_n, col_worst = [], 0, 0
    fil_bad, fil_n, fil_worst = [], 0, 0

    for r in rows:
        try:
            b = chess.Board(r["fen"])
        except Exception:
            continue
        if b.is_game_over(claim_draw=False):
            continue

        # --- colour swap: must negate ---
        try:
            m = b.mirror()
            a, c = ev(b), ev(m)
            col_n += 1
            d = abs(a + c)                      # a and c should sum to zero
            if d > TOL:
                col_worst = max(col_worst, d)
                col_bad.append((d, a, c, r["fen"]))
        except Exception:
            pass

        # --- file mirror: must preserve. Only where castling/ep cannot confuse the transform ---
        if not b.castling_rights and b.ep_square is None:
            try:
                f = b.transform(chess.flip_horizontal)
                a, c = ev(b), ev(f)
                fil_n += 1
                d = abs(a - c)
                if d > TOL:
                    fil_worst = max(fil_worst, d)
                    fil_bad.append((d, a, c, r["fen"]))
            except Exception:
                pass

    print("EVAL INVARIANCE  (tolerance %d millipawns, %d positions)\n" % (TOL, len(rows)))
    def report(title, note, n, bad, worst, lbl):
        print(title)
        if note:
            print(note)
        pctile = sorted(d for d, _, _, _ in bad)
        med = pctile[len(pctile) // 2] if pctile else 0
        print("   tested %d   violations %d (%.1f%%)   median %d mp   worst %d mp"
              % (n, len(bad), 100.0 * len(bad) / max(1, n), med, worst))
        for d, a, c, fen in sorted(bad, reverse=True)[:6]:      # WORST, not first-encountered
            print("     off by %6d mp   ours %+8d   %s %+8d   %s" % (d, a, lbl, c, fen))
        if not bad:
            print("     ✅ clean")

    report("1) COLOUR SWAP — eval(mirror(b)) must equal -eval(b).  A FAILURE HERE IS A BUG.",
           None, col_n, col_bad, col_worst, "mirrored")
    report("\n2) FILE MIRROR — eval(flip_h(b)) must equal eval(b), no-castling/no-ep positions only.",
           "   ⚠️ Our per-file pawn tables (CHAIN_F_*/WALL_F_*) are not constrained symmetric, so a\n"
           "   violation may be deliberate. Treat as a question, not automatically a defect.",
           fil_n, fil_bad, fil_worst, "flipped ")


if __name__ == "__main__":
    main()
