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

# 🚨 THREE different invariants live in this breakdown, and mixing them up manufactures phantom bugs.
#   signed contributions      -> must NEGATE            (the default path below)
#   side-labelled MAGNITUDES  -> plain SWAP: b.w == m.b  (piece-value sums, attack counts, mobility)
#   side-labelled SIGNED      -> NEGATE AND SWAP: b.w == -m.b
# imbalance_white/black are the third kind: they carry the Black-positive sign already. Testing them
# as plain-swap flagged all 87 occurrences at a 1722mp mean and made them look like the single largest
# remaining defect. They are correct -- kaufman_imbalance, the term that actually reaches `total`, is
# perfectly antisymmetric on every one of those positions.
SWAP_PAIRS = [("det_w_pieceval", "det_b_pieceval"), ("det_w_defense", "det_b_defense"),
              ("det_w_offense", "det_b_offense"), ("det_w_mobility", "det_b_mobility"),
              ("det_ks_units_w", "det_ks_units_b")]
SIGNED_SWAP_PAIRS = [("imbalance_white", "imbalance_black")]
SWAP_MEMBERS = {k for pair in SWAP_PAIRS + SIGNED_SWAP_PAIRS for k in pair}
# Not signed quantities at all -- antisymmetry does not apply.
SKIP_TERMS = {"phase_score", "is_endgame", "advanced_endgame_fired", "det_pawn_count", "det_central"}

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
    # TERMS=1 ranks WHICH terms break antisymmetry, across the whole sample. Bisecting one position at a
    # time finds one source; the asymmetry has at least three, and they do not all show on the same
    # position. `total` is listed too as the reference row -- a term whose share approaches total's is
    # the dominant payer. Sub-view fields (pt_*, det_*, material) are reported but are NOT additive with
    # `pieces`, so read them as localisation hints rather than a partition.
    TERMS = os.environ.get("TERMS", "0") == "1"
    term_abs, term_hits = {}, {}

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
            if TERMS:
                db, dm = ai.ev_breakdown(b), ai.ev_breakdown(m)
                # 🚨 Two different invariants. Signed eval contributions must NEGATE under mirror.
                # Side-LABELLED diagnostics (det_w_* / det_b_*, imbalance_white/black) must SWAP --
                # testing those for negation flags every position and made them look like the biggest
                # culprits when they were correct. The tell was det_w_defense and det_b_defense having
                # byte-identical violation sums over identical counts.
                for wk, bk in SWAP_PAIRS:          # magnitudes: b.w == m.b
                    d = abs(db.get(wk, 0) - dm.get(bk, 0)) + abs(db.get(bk, 0) - dm.get(wk, 0))
                    if d:
                        term_abs[wk + "<->" + bk] = term_abs.get(wk + "<->" + bk, 0) + d
                        term_hits[wk + "<->" + bk] = term_hits.get(wk + "<->" + bk, 0) + 1
                for wk, bk in SIGNED_SWAP_PAIRS:   # signed: b.w == -m.b
                    d = abs(db.get(wk, 0) + dm.get(bk, 0)) + abs(db.get(bk, 0) + dm.get(wk, 0))
                    if d:
                        term_abs[wk + "~" + bk] = term_abs.get(wk + "~" + bk, 0) + d
                        term_hits[wk + "~" + bk] = term_hits.get(wk + "~" + bk, 0) + 1
                for k in set(db) | set(dm):
                    x, y = db.get(k, 0), dm.get(k, 0)
                    if not isinstance(x, int) or not isinstance(y, int):
                        continue
                    if k in SKIP_TERMS or k in SWAP_MEMBERS:
                        continue
                    d = abs(x + y)
                    if d:
                        term_abs[k] = term_abs.get(k, 0) + d
                        term_hits[k] = term_hits.get(k, 0) + 1
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

    if TERMS:
        print("\n1b) WHICH TERMS BREAK IT — total |term(b) + term(mirror)| summed over the sample.")
        print("    `total` is the reference row. Sub-views (pt_*, det_*, material) are localisation")
        print("    hints, NOT a partition — they are not additive with `pieces`.")
        print("    %-26s %14s %10s %12s" % ("term", "sum |asym|", "positions", "mean/pos"))
        for k, v in sorted(term_abs.items(), key=lambda kv: -kv[1])[:16]:
            print("    %-26s %14d %10d %12.1f" % (k, v, term_hits[k], v / term_hits[k]))
    report("\n2) FILE MIRROR — eval(flip_h(b)) must equal eval(b), no-castling/no-ep positions only.",
           "   ⚠️ Our per-file pawn tables (CHAIN_F_*/WALL_F_*) are not constrained symmetric, so a\n"
           "   violation may be deliberate. Treat as a question, not automatically a defect.",
           fil_n, fil_bad, fil_worst, "flipped ")


if __name__ == "__main__":
    main()
