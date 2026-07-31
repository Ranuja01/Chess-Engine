# -*- coding: utf-8 -*-
"""Characterize WHERE and WHY our eval is optimistic, in chess terms. For a corpus of positions, measure
optimism = our_eval - SF_eval (our POV, pawns; +ve = we think we're better than we are), then explain it two
ways that together name the bias:
  (1) BY TERM  — which eval components are large in high-optimism positions (the over-firing sub-parts).
  (2) BY CHESS FEATURE — named python-chess detectors (advanced pawns, enemy counterplay on our king,
      overextension, material-sac state, passers, ...). For each: mean optimism when the feature is ON vs OFF,
      and the gap. A large positive ON-vs-OFF gap = "we are systematically optimistic in THIS kind of position."

The output is a ranked, named report: "when <feature>, optimism = +X (n=Y), driven by term <T>." That is the
raw material for a CONDITIONAL de-optimism (discount only when the detector fires) — validated later against a
control set of calm/accurate positions so we don't dampen the majority we already get right.

Run: pyrun diagnostics/optimism_diag.py --tags sfelo2400_base200,mediocre_mine [--depth 16] [--source collapses|results]
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import sys
import csv
import glob
import json
import argparse
from collections import defaultdict

import numpy as np
import chess
import chess.engine

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, ENGINE_DIR)

TERMS = ["material", "pieces", "capture_gains", "passed_pawn_support", "latent_threat", "threats", "king_safety",
         "central", "imbalance_white", "imbalance_black", "pair_bonus", "piece_value_boost", "pawn_majority",
         "pawn_struct", "outpost", "mobility", "pt_pawns", "pt_knights", "pt_bishops", "pt_rooks", "pt_queens",
         "pt_kings"]


def load_engine():
    from ChessAI import ChessAI
    seed = chess.Board()
    return ChessAI(None, None, seed, seed.turn)


def king_ring(sq):
    f, r = chess.square_file(sq), chess.square_rank(sq)
    out = []
    for df in (-1, 0, 1):
        for dr in (-1, 0, 1):
            nf, nr = f + df, r + dr
            if 0 <= nf < 8 and 0 <= nr < 8:
                out.append(chess.square(nf, nr))
    return out


def detectors(board, us):
    """Named chess features from OUR (us) point of view. Booleans/ints translated to chess concepts."""
    them = not us
    d = {}
    # advanced own pawns (rank >=5 from our side) — 'are we pushing pawns'
    adv = 0
    for sq in board.pieces(chess.PAWN, us):
        rr = chess.square_rank(sq) if us == chess.WHITE else 7 - chess.square_rank(sq)
        if rr >= 4:  # 5th rank+ (0-indexed 4)
            adv += 1
    d["adv_own_pawns>=2"] = adv >= 2
    d["adv_own_pawns>=3"] = adv >= 3
    # enemy counterplay on OUR king — attackers into our king ring
    ok = board.king(us)
    enemy_ring_attackers = 0
    if ok is not None:
        for sq in king_ring(ok):
            enemy_ring_attackers += len(board.attackers(them, sq))
    d["enemy_pressure_on_our_king>=3"] = enemy_ring_attackers >= 3
    d["enemy_pressure_on_our_king>=5"] = enemy_ring_attackers >= 5
    # our pieces in enemy half without defense (overextension)
    over = 0
    for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        for sq in board.pieces(pt, us):
            rr = chess.square_rank(sq) if us == chess.WHITE else 7 - chess.square_rank(sq)
            if rr >= 4 and not board.attackers(us, sq):   # advanced + undefended by us
                over += 1
    d["overextended_pieces>=1"] = over >= 1
    d["overextended_pieces>=2"] = over >= 2
    # material state (pawns=1..queen=9), our - their
    vals = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    mat = sum(vals[pt] * (len(board.pieces(pt, us)) - len(board.pieces(pt, them)))
              for pt in vals)
    d["we_are_up_material"] = mat >= 2
    d["we_sacked_material"] = mat <= -2
    # passers ours vs theirs (simple: pawn with no enemy pawn ahead on same/adjacent file)
    def passers(side):
        cnt = 0
        for sq in board.pieces(chess.PAWN, side):
            f = chess.square_file(sq)
            blocked = False
            for ef in (f - 1, f, f + 1):
                if not 0 <= ef < 8:
                    continue
                for esq in board.pieces(chess.PAWN, not side):
                    if chess.square_file(esq) != ef:
                        continue
                    er = chess.square_rank(esq)
                    sr = chess.square_rank(sq)
                    ahead = er > sr if side == chess.WHITE else er < sr
                    if ahead:
                        blocked = True
            if not blocked:
                cnt += 1
        return cnt
    d["we_have_passer"] = passers(us) >= 1
    d["enemy_has_passer"] = passers(them) >= 1
    # enemy queen present + our king exposed (open lines) — attack potential
    d["enemy_queen_on"] = len(board.pieces(chess.QUEEN, them)) >= 1
    return d


def collect_positions(tags, source):
    pos = []   # (fen, tag/game)
    for tag in tags:
        gd = os.path.join(THIS_DIR, "..", "selfplay", "games", tag)
        if source in ("collapses", "decision"):
            cp = os.path.join(gd, "collapses.csv")
            fks = ("decision_fen",) if source == "decision" else ("decision_fen", "drop_fen")
            if os.path.exists(cp):
                for r in csv.DictReader(open(cp)):
                    for fk in fks:
                        if r.get(fk):
                            pos.append((r[fk], tag))
        else:
            for jf in sorted(glob.glob(os.path.join(gd, "game_*.jsonl")))[:60]:
                recs = [json.loads(l) for l in open(jf) if l.strip()]
                fens = [rc["fen"] for rc in recs if rc.get("fen") and not rc.get("opening")]
                pos.extend((f, tag) for f in fens[10:-6:5])
    return pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True)
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--source", default="collapses", choices=["collapses", "decision", "results"])
    ap.add_argument("--clip", type=float, default=15.0, help="clip SF/our eval to +/- this (pawns) so mate-scores don't dominate means")
    ap.add_argument("--min-optimism", type=float, default=1.0, help="pawns; a position 'over-reads' if optimism>=this")
    args = ap.parse_args()
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    ai = load_engine()
    sf = chess.engine.SimpleEngine.popen_uci(os.environ["STOCKFISH_PATH"])

    rows = []
    try:
        seen = set()
        for fen, tag in collect_positions(tags, args.source):
            if fen in seen:
                continue
            seen.add(fen)
            try:
                b = chess.Board(fen)
            except ValueError:
                continue
            if b.is_game_over() or b.is_check():
                continue
            us = b.turn
            bd = ai.ev_breakdown(b)
            if bd.get("checkmate"):
                continue
            our_pov = (bd["total"] if us == chess.BLACK else -bd["total"]) / 1000.0
            info = sf.analyse(b, chess.engine.Limit(depth=args.depth))
            sfp = info["score"].pov(us).score(mate_score=100000) / 100.0
            cl = args.clip
            optim = max(-cl, min(cl, our_pov)) - max(-cl, min(cl, sfp))
            sign = 1.0 if us == chess.BLACK else -1.0     # term -> our POV
            terms = {t: sign * float(bd.get(t, 0.0)) / 1000.0 for t in TERMS}
            det = detectors(b, us)
            rows.append({"optim": optim, "terms": terms, "det": det})
    finally:
        sf.quit()

    n = len(rows)
    if not n:
        print("no positions"); return
    opt = np.array([r["optim"] for r in rows])
    print("positions=%d  mean_optimism=%.2f  median=%.2f  frac_over(>= %.1f)=%.0f%%"
          % (n, opt.mean(), float(np.median(opt)), args.min_optimism, 100.0 * np.mean(opt >= args.min_optimism)))

    print("\n=== BY CHESS FEATURE (mean optimism ON vs OFF; gap = how much MORE optimistic when present) ===")
    feats = list(rows[0]["det"].keys())
    res = []
    for f in feats:
        on = np.array([r["optim"] for r in rows if r["det"][f]])
        off = np.array([r["optim"] for r in rows if not r["det"][f]])
        if len(on) < 3 or len(off) < 3:
            continue
        res.append((f, on.mean(), off.mean(), on.mean() - off.mean(), len(on)))
    res.sort(key=lambda x: -x[3])
    print("%-32s %8s %8s %8s %6s" % ("feature", "ON", "OFF", "gap", "n_on"))
    for f, o, of, g, no in res:
        print("%-32s %+8.2f %+8.2f %+8.2f %6d" % (f, o, of, g, no))

    print("\n=== BY TERM (mean term value, our POV, in the OVER-READ subset vs the rest) ===")
    over = [r for r in rows if r["optim"] >= args.min_optimism]
    rest = [r for r in rows if r["optim"] < args.min_optimism]
    if over and rest:
        print("%-22s %10s %10s %10s" % ("term", "over_mean", "rest_mean", "delta"))
        deltas = []
        for t in TERMS:
            om = np.mean([r["terms"][t] for r in over])
            rm = np.mean([r["terms"][t] for r in rest])
            deltas.append((t, om, rm, om - rm))
        deltas.sort(key=lambda x: -abs(x[3]))
        for t, om, rm, dl in deltas:
            print("%-22s %+10.2f %+10.2f %+10.2f" % (t, om, rm, dl))
        print("\ndelta>0 = this term is LARGER (more positive, our POV) in over-optimistic positions = a culprit.")


if __name__ == "__main__":
    main()
