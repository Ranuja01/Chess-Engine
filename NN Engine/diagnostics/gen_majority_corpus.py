# -*- coding: utf-8 -*-
"""Build a pawn-MAJORITY test corpus (SF-labeled) for the PAWN_MAJORITY_* PACE loop.

The engine has no pawn-majority / candidate-passer term -- it values a pawn only once it is ACTUALLY
passed, so a wing pawn majority (e.g. a 3v2 kingside majority that will force a passer) reads as pure
material and won pawn-up positions under-convert. To TUNE the new PAWN_MAJORITY_* bundle we need a
corpus that actually EXERCISES it: positions with a wing pawn majority but NO actual passed pawn for
the majority side, spanning phases (midgame structural .. late-endgame conversion) and types
(queenside / kingside / outside).

Mines such positions from recorded self-play games, balances across (phase x type) buckets, adds a few
canonical cases, and labels each with Stockfish (search best-move + cp, and NNUE static). Output:
diagnostics/suites/majorities.csv with columns
    fen_start, sf_best, sf_cp, sf_static, wing, color, surplus, phase, outside, stm
(fen_start so selfplay/fen_vs_sf.py --csv and diagnostics/_majority_match.py consume it directly.)

Run (WSL, from NN Engine/):
    STOCKFISH_PATH=<sf.exe> /home/ranuja/anaconda3/bin/python diagnostics/gen_majority_corpus.py \
        --tags correctness_vs_base improv_r2m150_b placement_bundle --target 400 --sf-movetime 0.3
"""
import os, sys, json, csv, argparse, random
import chess

THIS = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(THIS)
sys.path.insert(0, os.path.join(ENGINE, "selfplay"))
from arbiter import Arbiter, find_stockfish

QS = chess.BB_FILE_A | chess.BB_FILE_B | chess.BB_FILE_C | chess.BB_FILE_D
KS = chess.BB_FILE_E | chess.BB_FILE_F | chess.BB_FILE_G | chess.BB_FILE_H
MAX_PHASE = 24


def phase_score(board):
    """Replicate the engine's material phase: 0 (full material) .. 128 (bare kings)."""
    q = chess.popcount(board.queens)
    r = chess.popcount(board.rooks)
    bn = chess.popcount(board.bishops | board.knights)
    phase = 4 * q + 2 * r + bn
    ps = 128 * (MAX_PHASE - phase) // MAX_PHASE
    return max(0, min(128, ps))


def phase_bucket(ps):
    # Mirror placement_and_piece_eval's split: <=64 midgame, <=96 early endgame, else late endgame.
    return "mid" if ps <= 64 else ("eeg" if ps <= 96 else "leg")


def has_passer(board, color):
    """True if `color` has at least one ACTUAL passed pawn (no enemy pawn in its 3-file forward span).
    Matches the engine's getPPIncrement passed test -- we EXCLUDE these so the corpus isolates the
    not-yet-passed majority (the gap), not actual passers (already valued + the regression guard)."""
    enemy = not color
    for sq in board.pieces(chess.PAWN, color):
        f, r = chess.square_file(sq), chess.square_rank(sq)
        blocked = False
        for ef in (f - 1, f, f + 1):
            if ef < 0 or ef > 7:
                continue
            for er in range(8):
                ahead = (er > r) if color == chess.WHITE else (er < r)
                if not ahead:
                    continue
                p = board.piece_at(chess.square(ef, er))
                if p and p.piece_type == chess.PAWN and p.color == enemy:
                    blocked = True
        if not blocked:
            return True
    return False


def majority_info(board):
    """Return (wing, color, surplus, outside) for the largest non-passed wing pawn majority, or None.
    wing 'qs'|'ks'; color = side with the majority; surplus = own-minus-enemy pawns on that wing;
    outside = the majority wing is opposite the enemy king (the king it would drag away)."""
    wp = board.pieces_mask(chess.PAWN, chess.WHITE)
    bp = board.pieces_mask(chess.PAWN, chess.BLACK)
    best = None
    for wing, mask in (("qs", QS), ("ks", KS)):
        w = chess.popcount(wp & mask)
        b = chess.popcount(bp & mask)
        if w > b:
            color, surplus = chess.WHITE, w - b
        elif b > w:
            color, surplus = chess.BLACK, b - w
        else:
            continue
        if has_passer(board, color):   # isolate the candidate gap, not actual passers
            continue
        if best is None or surplus > best[2]:
            ek = board.king(not color)
            ek_qs = (chess.square_file(ek) <= 3) if ek is not None else False
            outside = (wing == "qs" and not ek_qs) or (wing == "ks" and ek_qs)
            best = (wing, color, surplus, outside)
    return best


CANON = [
    # clean constructed majorities (no actual passer), spanning phases.
    "4k3/pp4pp/8/8/8/8/PP3PPP/4K3 w - - 0 1",                       # 3v2 KS majority, pure pawn endgame
    "4k3/pp4pp/8/8/8/8/PP3PPP/4KB1N w - - 0 1",                     # same majority with minors (early-eg)
    "r3k2r/pp4pp/2n2n2/8/8/2N2N2/PP3PPP/R3K2R w KQkq - 0 1",        # KS majority, full-piece midgame
    "4k3/1p4pp/8/8/8/8/P4PPP/4K3 w - - 0 1",                        # QS majority (a vs b) + KS 3v2
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+",
                    default=["correctness_vs_base", "improv_r2m150_b", "placement_bundle"])
    ap.add_argument("--target", type=int, default=400)
    ap.add_argument("--per-cat", type=int, default=60)   # cap per (phase,wing,outside) bucket for balance
    ap.add_argument("--min-surplus", type=int, default=1)
    ap.add_argument("--sf-movetime", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", default=os.path.join(THIS, "suites", "majorities.csv"))
    args = ap.parse_args()
    random.seed(args.seed)

    def iter_game_fens(tag):
        base = os.path.join(ENGINE, "selfplay", "games", tag)
        for gd in sorted(os.listdir(base)) if os.path.isdir(base) else []:
            jp = os.path.join(base, gd, "game.jsonl")
            if not os.path.isfile(jp):
                continue
            try:
                for line in open(jp):
                    o = json.loads(line)
                    if o.get("type") == "move" and o.get("fen") and not o.get("opening"):
                        yield o["fen"]
            except Exception:
                continue

    seen = set()
    buckets = {}   # (phase,wing,outside) -> list of row dicts
    cand = []
    for tag in args.tags:
        for fen in iter_game_fens(tag):
            key = " ".join(fen.split()[:4])
            if key in seen:
                continue
            seen.add(key)
            cand.append(fen)
    random.shuffle(cand)

    rows = []
    for fen in cand:
        try:
            b = chess.Board(fen)
        except Exception:
            continue
        info = majority_info(b)
        if info is None:
            continue
        wing, color, surplus, outside = info
        if surplus < args.min_surplus:
            continue
        ps = phase_score(b)
        pb = phase_bucket(ps)
        bk = (pb, wing, bool(outside))
        lst = buckets.setdefault(bk, [])
        if len(lst) >= args.per_cat:
            continue
        row = {"fen_start": fen, "wing": wing,
               "color": "w" if color == chess.WHITE else "b",
               "surplus": surplus, "phase": pb, "outside": int(outside),
               "stm": "w" if b.turn == chess.WHITE else "b"}
        lst.append(row); rows.append(row)
        if len(rows) >= args.target:
            break

    for fen in CANON:
        b = chess.Board(fen)
        info = majority_info(b)
        wing, color, surplus, outside = info if info else ("ks", chess.WHITE, 1, False)
        rows.append({"fen_start": fen, "wing": wing,
                     "color": "w" if color == chess.WHITE else "b",
                     "surplus": surplus, "phase": phase_bucket(phase_score(b)),
                     "outside": int(outside), "stm": "w" if b.turn == chess.WHITE else "b"})

    print("mined %d majority positions; bucket spread:" % len(rows))
    for bk in sorted(buckets):
        print("  %-18s %d" % (str(bk), len(buckets[bk])))

    print("SF-labeling (movetime=%.2fs)..." % args.sf_movetime)
    arb = Arbiter(find_stockfish(), movetime=args.sf_movetime)
    out_rows = []
    for i, r in enumerate(rows):
        b = chess.Board(r["fen_start"])
        try:
            cp, best, _d = arb.evaluate(b)
            st = arb.evaluate_static(b)
        except Exception as e:
            print("  skip %s: %s" % (r["fen_start"], e)); continue
        r["sf_best"] = best or ""
        r["sf_cp"] = cp if cp is not None else ""
        r["sf_static"] = st if st is not None else ""
        out_rows.append(r)
        if (i + 1) % 50 == 0:
            print("  %d/%d" % (i + 1, len(rows)))
    try:
        arb.close()
    except Exception:
        pass

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["fen_start", "sf_best", "sf_cp", "sf_static",
                                          "wing", "color", "surplus", "phase", "outside", "stm"])
        w.writeheader()
        w.writerows(out_rows)
    print("wrote %d rows -> %s" % (len(out_rows), args.out))


if __name__ == "__main__":
    main()
