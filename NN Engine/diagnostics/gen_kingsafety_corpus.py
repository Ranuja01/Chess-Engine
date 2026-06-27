# -*- coding: utf-8 -*-
"""Build a king-safety test corpus (SF-labeled) for the attack-unit king_safety_score PACE loop.

Extracts KING-DANGER positions from recorded self-play games — positions where at least one king's
2-ring is attacked by multiple enemy pieces, in the opening/midgame (where king safety matters and the
crude get_latent_threat_score is blind). Balances across attacker-count and phase buckets, adds a few
canonical king-attack FENs, and labels each with Stockfish (search best-move + cp, and NNUE static).
Output: diagnostics/suites/kingsafety.csv with columns
    fen_start, sf_best, sf_cp, sf_static, cat, attackers, phase_bin, stm
(fen_start so selfplay/fen_vs_sf.py --csv and the static tools can consume it directly.)

Run (WSL, from NN Engine/):
    STOCKFISH_PATH=<sf.exe> /home/ranuja/anaconda3/bin/python diagnostics/gen_kingsafety_corpus.py \
        --tags away_standard correctness_vs_base current_vs_old --target 300 --sf-movetime 0.3
"""
import os, sys, json, csv, argparse, random
import chess

THIS = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(THIS)
sys.path.insert(0, os.path.join(ENGINE, "selfplay"))
from arbiter import Arbiter, find_stockfish

MAX_PHASE = 24  # must match cpp_bitboard.h


def material_phase_score(board):
    """Engine phase: 0 (full material/opening) .. 128 (bare kings/endgame). LOW = midgame.
    Mirrors placement_and_piece_eval (phase = 4Q+2R+(B|N); phase_score = 128*(MAX_PHASE-phase)/MAX_PHASE)."""
    q = chess.popcount(board.queens)
    r = chess.popcount(board.rooks)
    bn = chess.popcount(board.bishops | board.knights)
    phase = 4 * q + 2 * r + bn
    ps = 128 * (MAX_PHASE - phase) // MAX_PHASE
    return max(0, min(128, ps))


def king_ring2(sq):
    """King-centered 2-ring (king moves + their neighbours), matching the engine's king_ring2[]."""
    ring = set(chess.SquareSet(chess.BB_KING_ATTACKS[sq]))
    ring.add(sq)
    for s in list(ring):
        ring |= set(chess.SquareSet(chess.BB_KING_ATTACKS[s]))
    return ring


def king_danger(board, king_color):
    """Light king-danger detector for corpus SELECTION (not the engine's full model): returns the number
    of DISTINCT enemy pieces (non-pawn weight emphasis not applied here) attacking the king's 2-ring."""
    enemy = not king_color
    king_sq = board.king(king_color)
    if king_sq is None:
        return 0
    attacker_squares = set()
    for sq in king_ring2(king_sq):
        attacker_squares |= set(board.attackers(enemy, sq))
    # Count only pieces that can actually pressure a king (exclude pawns from the headline count, though
    # a pawn in the ring still matters; keep it simple and weight by piece presence).
    return len(attacker_squares)


def phase_bin(ps):
    if ps <= 24:
        return "opening"
    if ps <= 48:
        return "early_mid"
    if ps <= 72:
        return "late_mid"
    return "endgame"  # king safety tapers out here; kept as a control bucket


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
    # canonical king-attack spectrum (the crude latent_threat under-reads these)
    "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 b kq - 0 1",   # quiet Italian (baseline)
    "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP1QPPP/R1B2RK1 w - - 0 1",   # both castled, balanced
    "r1bqkb1r/pp3ppp/2n1pn2/2pp4/3P1B2/2PBPN2/PP3PPP/RN1QK2R b KQkq - 0 1",    # semi-open
    "2r2rk1/pp1bppbp/3p1np1/q7/3NP3/1BN1BP2/PPPQ2PP/2KR3R w - - 0 1",          # opposite-side castle attack
    "r2q1rk1/1b1nbppp/p2ppn2/1p6/3NPP2/2N1B3/PPPQB1PP/2KR3R w - - 0 1",        # Sicilian sac setup
    "6k1/5ppp/8/8/8/8/5PPP/3QR1K1 w - - 0 1",                                   # back-rank-ish (low material control)
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["away_standard", "correctness_vs_base", "current_vs_old"])
    ap.add_argument("--target", type=int, default=300)
    ap.add_argument("--per-cat", type=int, default=70)   # cap per (attacker-count, phase) bucket for balance
    ap.add_argument("--min-attackers", type=int, default=2)  # at least this many enemy pieces on a king zone
    ap.add_argument("--max-phase", type=int, default=72)     # only opening..late-mid (king safety lives here)
    ap.add_argument("--sf-movetime", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", default=os.path.join(THIS, "suites", "kingsafety.csv"))
    args = ap.parse_args()
    random.seed(args.seed)

    seen = set()
    buckets = {}  # (attackers_capped, phase_bin) -> list of row dicts
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
        if b.is_checkmate() or b.is_stalemate():
            continue
        ps = material_phase_score(b)
        if ps > args.max_phase:
            continue
        dw, db = king_danger(b, chess.WHITE), king_danger(b, chess.BLACK)
        attackers = max(dw, db)
        if attackers < args.min_attackers:
            continue
        # which king is the danger subject (the more-attacked one)
        subject = "white" if dw >= db else "black"
        acap = min(attackers, 5)  # cap the bucket key so 5+ attackers share one bucket
        pb = phase_bin(ps)
        bk = buckets.setdefault((acap, pb), [])
        if len(bk) >= args.per_cat:
            continue
        bk.append({"fen_start": fen, "cat": "play", "attackers": attackers,
                   "phase_bin": pb, "subject": subject,
                   "stm": "w" if b.turn == chess.WHITE else "b"})
        if sum(len(v) for v in buckets.values()) >= args.target:
            break

    rows = []
    for lst in buckets.values():
        rows.extend(lst)
    for fen in CANON:
        b = chess.Board(fen)
        ps = material_phase_score(b)
        dw, db = king_danger(b, chess.WHITE), king_danger(b, chess.BLACK)
        rows.append({"fen_start": fen, "cat": "canon", "attackers": max(dw, db),
                     "phase_bin": phase_bin(ps), "subject": "white" if dw >= db else "black",
                     "stm": "w" if b.turn == chess.WHITE else "b"})

    dist = {}
    for r in rows:
        dist[(r["attackers"] if r["attackers"] <= 5 else 5, r["phase_bin"])] = \
            dist.get((r["attackers"] if r["attackers"] <= 5 else 5, r["phase_bin"]), 0) + 1
    print("extracted %d king-danger positions; (attackers,phase) buckets: %s" % (len(rows), dict(sorted(dist.items()))))
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
                                          "cat", "attackers", "phase_bin", "subject", "stm"])
        w.writeheader()
        w.writerows(out_rows)
    print("wrote %d rows -> %s" % (len(out_rows), args.out))


if __name__ == "__main__":
    main()
