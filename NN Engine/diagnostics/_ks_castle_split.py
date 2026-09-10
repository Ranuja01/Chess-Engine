# -*- coding: utf-8 -*-
"""Does an arm's effect concentrate where the kings are CASTLED? Zero engine calls, zero Stockfish.

WHY THIS EXISTS (2026-09-09). The KS ablation (`KING_SAFETY_MAG=0`) reads +2.2 / +2.9 on the two standard
corpora in the `.opening` stratum (>=26 pieces) and +0.6 on the whacky/960-no-castle variant set. That was
run as a "does it transfer off standard structures" test -- but disabling castling changes TWO things at
once: the opening STRUCTURE (the intended variable) and the KING PLACEMENT (central/exposed instead of
castled behind a shelter). Both predict the same drop, so the variant test cannot separate:

  (a) "KS over-fires on CASTLED kings with intact shelter"  -> LEGITIMATE. Real games have castled kings,
      so a piece-count gate would still be correct where it matters.
  (b) "KS over-fires on patterns memorised from OUR opening book" -> OVERFIT. Gating on it would encode
      the engine's own habits as a phase rule, which is the expensive failure mode because it looks
      principled.

This splits the ALREADY-MEASURED changed sets by king placement and reads (a) directly, on the standard
corpora, where the effect actually lives. It refutes the MECHANISM instead of a proxy.

★ Reads the arm dump AND the matching NULL dump and reports arm - null PER BUCKET. A bucket's null is not
50 and is not the corpus aggregate: nulls move by >3pp across corpora on the same stratum label, so an
unpaired bucket reading is uninterpretable.

  pyrun diagnostics/_ks_castle_split.py ARM=diagnostics/_fpdump_ks_off.csv NULL=diagnostics/_fpdump_null_asp300.csv [MINPC=26]

Dump schema (from _ks_footprint_regret.py): fen, base_move, cand_move, reg_base, reg_cand, delta
where delta = reg_cand - reg_base, so delta < 0 means the CANDIDATE's move is better.
"""
import os, sys, csv

for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chess

ARM = os.environ.get("ARM", "diagnostics/_fpdump_ks_off.csv")
NULL = os.environ.get("NULL", "diagnostics/_fpdump_null_asp300.csv")
MINPC = int(os.environ.get("MINPC", "26"))   # matches phase_of(): "opening" is len(piece_map()) >= 26


def king_state(board, colour):
    """CASTLED / CENTRAL / OTHER for one side, from the position alone.

    A mid-game FEN cannot say whether castling HAPPENED, so classify by where the king actually sits --
    which is what KS reads anyway. Castled-like = the king reached a flank on its own back rank AND has
    no castling rights left on that side. Central = still on the d/e files.
    """
    ksq = board.king(colour)
    if ksq is None:
        return "OTHER"
    f, r = chess.square_file(ksq), chess.square_rank(ksq)
    home = 0 if colour == chess.WHITE else 7
    has_rights = board.has_kingside_castling_rights(colour) or board.has_queenside_castling_rights(colour)
    if r == home and (f >= 5 or f <= 2) and not has_rights:
        return "CASTLED"
    if f in (3, 4):
        return "CENTRAL"
    return "OTHER"


def load(path):
    """-> {fen: better}  where better = True when the candidate's move scored better (delta < 0)."""
    out = {}
    try:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    d = float(row["delta"])
                except (KeyError, ValueError):
                    continue
                if d == 0.0:
                    continue          # ties carry no direction; the tool's bett/wors excludes them too
                out[row["fen"]] = (d < 0.0)
    except FileNotFoundError:
        print("MISSING: %s" % path)
        sys.exit(2)
    return out


def bucket(rows):
    """-> {label: [n_better, n_worse]} keyed by how many kings are castled."""
    agg = {}
    for fen, better in rows.items():
        try:
            b = chess.Board(fen)
        except ValueError:
            continue
        if len(b.piece_map()) < MINPC:
            continue
        w, k = king_state(b, chess.WHITE), king_state(b, chess.BLACK)
        n_cast = (w == "CASTLED") + (k == "CASTLED")
        n_cent = (w == "CENTRAL") + (k == "CENTRAL")
        lab = "2_both_castled" if n_cast == 2 else \
              "1_one_castled" if n_cast == 1 else \
              "0_none_castled" + (" (both central)" if n_cent == 2 else "")
        e = agg.setdefault(lab, [0, 0])
        e[0 if better else 1] += 1
    return agg


arm, null = load(ARM), load(NULL)
ba, bn = bucket(arm), bucket(null)

print("ARM  = %s   (%d changed rows)" % (ARM, len(arm)))
print("NULL = %s   (%d changed rows)" % (NULL, len(null)))
print("filter: >= %d pieces on the board (matches phase_of()'s `opening`)\n" % MINPC)
print("  %-30s %8s %8s   %8s %8s   %9s" % ("king placement", "n_arm", "arm%", "n_null", "null%", "arm-null"))

for lab in sorted(set(ba) | set(bn)):
    a = ba.get(lab, [0, 0])
    n = bn.get(lab, [0, 0])
    ta, tn = a[0] + a[1], n[0] + n[1]
    if ta == 0 or tn == 0:
        print("  %-30s %8d %8s   %8d %8s   %9s" % (lab, ta, "-", tn, "-", "unreadable"))
        continue
    pa, pn = 100.0 * a[0] / ta, 100.0 * n[0] / tn
    # SE of the difference of two independent proportions, at p~0.5 (conservative).
    se = (0.25 / ta + 0.25 / tn) ** 0.5 * 100.0
    sig = (pa - pn) / se if se > 0 else 0.0
    print("  %-30s %8d %7.1f%%   %8d %7.1f%%   %+8.1f  (SE %.1f, %.1f sigma)"
          % (lab, ta, pa, tn, pn, pa - pn, se, sig))

print("\n  READ: arm% is the share of CHANGED positions where the ARM's move scored better per SF18.")
print("  A bucket where arm-null is large and positive is where the ablated term was HURTING us.")
print("  ⚠️ Judge on the SIGMA and the n, never on the delta alone -- these buckets are small by")
print("     construction and the record's sparse-cell failures (n_crit=27, ps3 at n=59) all looked large.")
