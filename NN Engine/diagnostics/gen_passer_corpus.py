# -*- coding: utf-8 -*-
"""Build a passed-pawn test corpus (SF-labeled) for the Gap-P conditional-passer-scorer PACE loop.

Extracts positions WITH AN ADVANCED PASSED PAWN from recorded self-play games, balances across
categories (our vs enemy passer; blockaded / rook-contested / clear path; midgame vs endgame), adds the
hand-curated gap FENs + a few canonical cases, and labels each with Stockfish (search best-move + cp, and
NNUE static). Output: diagnostics/suites/passers.csv with columns
    fen_start, sf_best, sf_cp, sf_static, cat, dist, stm
(fen_start so selfplay/fen_vs_sf.py --csv can consume it directly.)

Run (WSL, from NN Engine/):
    STOCKFISH_PATH=<sf.exe> /home/ranuja/anaconda3/bin/python diagnostics/gen_passer_corpus.py \
        --tags correctness_vs_base improv_r2m150_b current_vs_old --target 400 --sf-movetime 0.3
"""
import os, sys, json, csv, argparse, random
import chess

THIS = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(THIS)
sys.path.insert(0, os.path.join(ENGINE, "selfplay"))
from arbiter import Arbiter, find_stockfish


def passer_info(board):
    """Return the most-advanced passed pawn's (dist_to_promote, owner_color, blockade_kind) or None.
    blockade_kind: 'clear' (stop sq empty) | 'minor' (enemy N/B on stop sq) | 'major' (enemy R/Q ahead on file) | 'other'."""
    best = None  # (dist, color, kind)
    for color in (chess.WHITE, chess.BLACK):
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
            if blocked:
                continue
            dist = (7 - r) if color == chess.WHITE else r            # squares to promotion
            stop = chess.square(f, r + 1) if color == chess.WHITE else chess.square(f, r - 1)
            sp = board.piece_at(stop)
            if sp is None:
                kind = "clear"
            elif sp.color == enemy and sp.piece_type in (chess.KNIGHT, chess.BISHOP):
                kind = "minor"
            else:
                # enemy major ahead anywhere on the file = rook-contested (the French case)
                kind = "other"
                for er in range(8):
                    ahead = (er > r) if color == chess.WHITE else (er < r)
                    if not ahead:
                        continue
                    q = board.piece_at(chess.square(f, er))
                    if q and q.color == enemy and q.piece_type in (chess.ROOK, chess.QUEEN):
                        kind = "major"
                        break
            if best is None or dist < best[0]:
                best = (dist, color, kind)
    return best


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


CANON = [
    # gap FENs + canonical spectrum cases
    "1r1q4/p2bn1pk/4p1pp/3pPr2/p1pP1NQP/2P2P2/R1PB2P1/3R1K2 w - - 8 33",   # French F33 (a4, rook-contested)
    "8/p2bn1pk/4p1pp/q2pPr2/2pP1NQP/p1P2P2/1rPB2P1/R1R1K3 w - - 0 36",     # French F36 (a3, rook-contested)
    "8/5pk1/8/3P1PP1/4R2P/1p6/2r5/7K w - - 1 47",                          # benoni-47 (b3 racing, unblocked)
    "8/8/8/8/8/1k6/p7/1K6 w - - 0 1",                                       # canonical unstoppable racer
    "8/8/8/3k4/8/1N6/p7/1K6 w - - 0 1",                                     # minor blockade-ish
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["correctness_vs_base", "improv_r2m150_b", "current_vs_old"])
    ap.add_argument("--target", type=int, default=400)
    ap.add_argument("--per-cat", type=int, default=90)   # cap per (kind,color-relative) bucket for balance
    ap.add_argument("--max-dist", type=int, default=4)   # only advanced passers (within N of promotion)
    ap.add_argument("--sf-movetime", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", default=os.path.join(THIS, "suites", "passers.csv"))
    args = ap.parse_args()
    random.seed(args.seed)

    seen = set()
    buckets = {}  # kind -> list of (fen, dist, kind, stm)
    cand = []
    for tag in args.tags:
        for fen in iter_game_fens(tag):
            key = " ".join(fen.split()[:4])
            if key in seen:
                continue
            seen.add(key)
            cand.append(fen)
    random.shuffle(cand)
    for fen in cand:
        try:
            b = chess.Board(fen)
        except Exception:
            continue
        info = passer_info(b)
        if info is None or info[0] > args.max_dist:
            continue
        dist, color, kind = info
        bk = buckets.setdefault(kind, [])
        if len(bk) >= args.per_cat:
            continue
        stm = "w" if b.turn == chess.WHITE else "b"
        bk.append((fen, dist, kind, stm))
        if sum(len(v) for v in buckets.values()) >= args.target:
            break

    rows = []
    for kind, lst in buckets.items():
        for fen, dist, k, stm in lst:
            rows.append({"fen_start": fen, "dist": dist, "cat": k, "stm": stm})
    for fen in CANON:
        b = chess.Board(fen)
        info = passer_info(b)
        rows.append({"fen_start": fen, "dist": info[0] if info else -1,
                     "cat": "canon", "stm": "w" if b.turn == chess.WHITE else "b"})

    print("extracted %d passer positions; buckets: %s" % (len(rows), {k: len(v) for k, v in buckets.items()}))
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
        w = csv.DictWriter(f, fieldnames=["fen_start", "sf_best", "sf_cp", "sf_static", "cat", "dist", "stm"])
        w.writeheader()
        w.writerows(out_rows)
    print("wrote %d rows -> %s" % (len(out_rows), args.out))


if __name__ == "__main__":
    main()
