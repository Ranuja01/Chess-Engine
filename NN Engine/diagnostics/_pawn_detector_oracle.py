# -*- coding: utf-8 -*-
"""RUNG 2a DETECTOR ORACLE -- compare eval v2's C++ pawn masks against an independent Python reference.

WHY THIS EXISTS (2026-09-12). A detector bug and a scoring bug are indistinguishable from outside: both
read as "the eval moved". Every other instrument we own measures the SCORE, so none of them can tell us
whether `backward` is actually flagging backward pawns.

_pawn_term_overlap.py computes the same nine predicates in Python from the two pawn bitboards, and is
validated 8/8 on hand-checked positions and colour-symmetric 3/3. It was written BEFORE the C++ and from
the reference sources, not from the C++ -- so it is a genuine oracle, not a restatement.

★ This is the only correctness check available to us that does not depend on any constant being right.
⚠️ It exists because of a near-miss: a transcription defect (Ethereal's file-asymmetric isolated table)
reached a built binary and was caught by a symmetry gate rather than by reading the code.

  pyrun diagnostics/_pawn_detector_oracle.py [N=3000] [CORPUS=ks_sets/game_regret_set.csv]

Exit code 0 = every mask matched on every position.
"""
import os, sys, csv

for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v

THIS = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(THIS)
sys.path.insert(0, THIS)
sys.path.insert(0, ENGINE)

N = int(os.environ.get("N", "3000"))
CORPUS = os.environ.get("CORPUS", "ks_sets/game_regret_set.csv")

PREDS = ["isolated", "doubled", "backward", "phalanx", "supported",
         "opposed", "lever", "blocked", "stop_held", "passed", "candidate"]


def main():
    try:
        import chess
    except ImportError:
        print("python-chess required"); return 2
    try:
        import ChessAI
    except ImportError as e:
        print("ChessAI import failed (build first): %s" % e); return 2
    if not hasattr(ChessAI, "pawn_masks"):
        print("☠️ ChessAI.pawn_masks missing -- the build predates the probe. Rebuild."); return 2

    from _pawn_term_overlap import terms as py_terms

    path = os.path.join(ENGINE, CORPUS)
    if not os.path.exists(path):
        path = os.path.join(THIS, CORPUS)
    if not os.path.exists(path):
        print("corpus not found: %s" % CORPUS); return 2

    bad = {p: 0 for p in PREDS}
    first = {}
    rows = 0
    checked = 0

    with open(path, newline='', encoding='utf-8', errors='replace') as fh:
        for rec in csv.DictReader(fh):
            if rows >= N:
                break
            fen = (rec.get("fen") or "").strip()
            if not fen:
                continue
            try:
                bd = chess.Board(fen)
            except Exception:
                continue
            rows += 1

            wp = int(bd.pieces(chess.PAWN, chess.WHITE))
            bp = int(bd.pieces(chess.PAWN, chess.BLACK))
            occ_w = int(bd.occupied_co[chess.WHITE])
            occ_b = int(bd.occupied_co[chess.BLACK])

            cpp = ChessAI.pawn_masks(wp | bp, occ_w, occ_b)
            pw = py_terms(wp, bp, True)
            pb = py_terms(bp, wp, False)

            for p in PREDS:
                checked += 2
                cw, cb = cpp[p]
                if cw != pw[p] or cb != pb[p]:
                    bad[p] += 1
                    if p not in first:
                        first[p] = (fen, cw, pw[p], cb, pb[p])

    total_bad = sum(bad.values())
    print("RUNG 2a DETECTOR ORACLE -- %d positions, %d mask comparisons" % (rows, checked))
    print("C++ eval_v2 build_pawn_entry  vs  Python _pawn_term_overlap.terms\n")
    print("  %-12s %10s" % ("predicate", "mismatches"))
    for p in PREDS:
        mark = "  ✅" if bad[p] == 0 else "  ☠️"
        print("  %-12s %10d%s" % (p, bad[p], mark))

    if total_bad == 0:
        print("\n✅ ALL MASKS IDENTICAL on every position. The detector is what the reference says it is.")
        return 0

    print("\n☠️ %d predicate-positions disagree. First instance of each:" % total_bad)
    for p, (fen, cw, ow, cb, ob) in first.items():
        print("\n  %s" % p)
        print("    fen   %s" % fen)
        print("    white  cpp=0x%016x  py=0x%016x  xor=0x%016x" % (cw, ow, cw ^ ow))
        print("    black  cpp=0x%016x  py=0x%016x  xor=0x%016x" % (cb, ob, cb ^ ob))
    return 1


if __name__ == "__main__":
    sys.exit(main())
