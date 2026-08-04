# -*- coding: utf-8 -*-
"""Apples-to-apples HCE static gap on OUR failure positions: compare our static ev_breakdown to classical
Stockfish-11's static `eval` term table, aggregated over the collapse DECISION fens. SF11 is pre-NNUE (pure
handcraft), so any concept it statically credits that we don't is a BRIDGEABLE handmade gap — not a neural-net
tactic we can never encode. Reuses SF11Eval from eval_vs_sf11.

Per SF11 term and per our term we report the MEAN value in OUR point of view (sign-flipped when we are Black),
so a negative SF11 mean = "SF11 statically thinks this concept is AGAINST us in the positions where we collapse."
Where SF11 assigns big |value| and our mapped term is ~0 = the missing static signal.

Run: pyrun diagnostics/sf11_collapse_gap.py --tags sfelo2400_base200,mediocre_mine [--source decision]
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import sys
import csv
import glob
import json
import random
import argparse
from collections import defaultdict

import numpy as np
import chess

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, ENGINE_DIR)
sys.path.insert(0, THIS_DIR)
from eval_vs_sf11 import SF11Eval, SF11    # reuse the classical-eval term-table reader

OUR_TERMS = ["material", "pieces", "capture_gains", "passed_pawn_support", "latent_threat", "threats",
             "king_safety", "central", "imbalance_white", "imbalance_black", "pair_bonus", "piece_value_boost",
             "pawn_majority", "pawn_struct", "outpost", "mobility",
             "pt_pawns", "pt_knights", "pt_bishops", "pt_rooks", "pt_queens", "pt_kings"]


def load_fens(tags, source):
    fks = ("decision_fen",) if source == "decision" else ("decision_fen", "drop_fen")
    out = []
    for tag in tags:
        cp = os.path.join(THIS_DIR, "..", "selfplay", "games", tag, "collapses.csv")
        if os.path.exists(cp):
            for r in csv.DictReader(open(cp)):
                for fk in fks:
                    if r.get(fk):
                        out.append(r[fk])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="")
    ap.add_argument("--fens-file", default="", help="explicit FEN list ('label<TAB>fen' or 'fen'); overrides --tags")
    ap.add_argument("--source", default="decision", choices=["decision", "collapses"])
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--engine-env", default="", help="'K=V K=V' set into env BEFORE engine init (e.g. KS bundle)")
    args = ap.parse_args()
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    for kv in args.engine_env.split():
        if "=" in kv:
            k, v = kv.split("=", 1); os.environ[k] = v

    from ChessAI import ChessAI
    seed = chess.Board()
    ai = ChessAI(None, None, seed, seed.turn)
    sf = SF11Eval(SF11)

    our_acc = defaultdict(list)
    sf_acc = defaultdict(list)
    gaps = []
    rows = []
    seen = set()
    vals = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    if args.fens_file:
        fen_iter = [ln.rstrip("\n").split("\t", 1)[-1].strip() for ln in open(args.fens_file) if ln.strip()]
    else:
        fen_iter = load_fens(tags, args.source)
    try:
        for fen in fen_iter:
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
            them = not us
            povsign = 1.0 if us == chess.WHITE else -1.0     # White-POV -> our-POV
            bd = ai.ev_breakdown(b)
            if bd.get("checkmate"):
                continue
            sf_total, sf_terms = sf.eval(fen)                # White-POV pawns
            if sf_total is None:
                continue
            our_white = -bd["total"] / 1000.0                # our total in White-POV pawns
            gap = (our_white - sf_total) * povsign
            gaps.append(gap)                                 # + = we read hotter for US than SF11
            our_pov = {t: (-bd.get(t, 0.0) / 1000.0) * povsign for t in OUR_TERMS}
            sf_pov = {lbl: v * povsign for lbl, v in sf_terms.items() if lbl != "Total"}
            for t in OUR_TERMS:
                our_acc[t].append(our_pov[t])
            for lbl, v in sf_pov.items():
                sf_acc[lbl].append(v)
            # light chess context (our POV)
            mat = sum(vals[pt] * (len(b.pieces(pt, us)) - len(b.pieces(pt, them))) for pt in vals)
            ok = b.king(us)
            katt = 0
            if ok is not None:
                f, r = chess.square_file(ok), chess.square_rank(ok)
                for df in (-1, 0, 1):
                    for dr in (-1, 0, 1):
                        nf, nr = f + df, r + dr
                        if 0 <= nf < 8 and 0 <= nr < 8:
                            katt += len(b.attackers(them, chess.square(nf, nr)))
            ksq = b.king(us)
            kfile = chess.square_file(ksq) if ksq is not None else -1
            krank = chess.square_rank(ksq) if ksq is not None else -1
            home_rank = 0 if us == chess.WHITE else 7
            # castled proxy: king on g/c file & home rank; central: king on d/e or off the home rank
            castled = (kfile in (2, 6)) and (krank == home_rank)
            central = (kfile in (3, 4)) or (abs(krank - home_rank) >= 1 and 2 <= kfile <= 5)
            rows.append({"gap": gap, "fen": fen, "us": "W" if us == chess.WHITE else "B",
                         "our_tot": our_white * povsign, "sf_tot": sf_total * povsign,
                         "mat": mat, "katt": katt, "our": our_pov, "sf": sf_pov,
                         "ks_sf": sf_pov.get("King safety", 0.0), "our_ks": our_pov.get("king_safety", 0.0),
                         "ksq": chess.square_name(ksq) if ksq is not None else "?",
                         "castled": castled, "central": central})
    finally:
        sf.close()

    n = len(gaps)
    if not n:
        print("no positions"); return
    print("positions=%d  mean total gap (ours-SF11, OUR POV)=%+.2f  (>0 = we over-read for us)\n"
          % (n, float(np.mean(gaps))))

    print("=== SF11 static terms, mean in OUR POV (negative = SF11 sees this AGAINST us where we collapse) ===")
    print("%-16s %8s %6s" % ("SF11 term", "mean", "n"))
    for lbl in sorted(sf_acc, key=lambda k: -abs(np.mean(sf_acc[k]))):
        v = sf_acc[lbl]
        print("%-16s %+8.2f %6d" % (lbl, float(np.mean(v)), len(v)))

    print("\n=== OUR static terms, mean in OUR POV (compare to SF11 above; ~0 on a concept SF11 credits = a GAP) ===")
    print("%-20s %8s" % ("our term", "mean"))
    for t in sorted(OUR_TERMS, key=lambda k: -abs(np.mean(our_acc[k])) if our_acc[k] else 0):
        if our_acc[t]:
            print("%-20s %+8.2f" % (t, float(np.mean(our_acc[t]))))

    # ---- classification over ALL positions: how many follow the king-safety pattern? ----
    over = [r for r in rows if r["gap"] >= 1.5]
    ksdanger = [r for r in over if r["ks_sf"] <= -1.5]          # SF11 sees our king in real danger
    ks_we_miss = [r for r in ksdanger if abs(r["our_ks"]) < 0.5]  # ...and our KS term ~0
    placement = [r for r in over if r["ks_sf"] > -1.5 and r["mat"] <= 1]  # king ~safe, not a big mat edge
    print("\n=== CLASSIFICATION of all %d positions ===" % len(rows))
    print("over-read (gap>=1.5)                : %d/%d" % (len(over), len(rows)))
    print("  king-danger (SF11 KS<=-1.5)       : %d  (%.0f%% of over-read)"
          % (len(ksdanger), 100.0 * len(ksdanger) / max(1, len(over))))
    print("    ...and OUR king_safety ~0        : %d  (we are BLIND to it)" % len(ks_we_miss))
    print("  king-safe, no big mat edge        : %d  (placement/other over-read)" % len(placement))
    if ksdanger:
        cen = sum(1 for r in ksdanger if r["central"])
        cas = sum(1 for r in ksdanger if r["castled"])
        upm = sum(1 for r in ksdanger if r["mat"] >= 2)
        eqm = sum(1 for r in ksdanger if abs(r["mat"]) <= 1)
        print("  king-danger subset FEN traits: central/exposed king %d/%d, castled %d/%d, up-material %d, ~equal-material %d"
              % (cen, len(ksdanger), cas, len(ksdanger), upm, eqm))
        print("  king-danger squares: " + ", ".join(sorted(r["ksq"] for r in ksdanger)))
        print("  mean over-read in king-danger=%.2f  vs placement=%.2f"
              % (np.mean([r["gap"] for r in ksdanger]),
                 np.mean([r["gap"] for r in placement]) if placement else 0.0))

    # Rank by WIN% error, not centipawn gap. A 2-pawn error at +8 barely changes the expected result while a
    # 2-pawn error at 0.0 flips the game, so raw cp over-weights blowouts and hides the errors that actually
    # cost points. Same Lichess logistic (k=0.00368208) the fit scripts use, so ranking here matches tuning.
    K_LICHESS = 0.00368208

    def winpct(pawns):
        return 100.0 / (1.0 + np.exp(-K_LICHESS * pawns * 100.0))

    for r in rows:
        r["wp_ours"] = winpct(r["our_tot"])
        r["wp_sf"] = winpct(r["sf_tot"])
        r["wp_gap"] = r["wp_ours"] - r["wp_sf"]

    topk = args.top
    rows.sort(key=lambda r: -r["wp_gap"])
    print("\n=== WORST %d OVER-READ collapse positions BY WIN%% ERROR (our POV) — to READ & REASON ===" % topk)
    print("    (win%% via Lichess k=0.00368208, as in the fit scripts; cp gap shown for reference)")
    for r in rows[:topk]:
        print("-" * 100)
        print("win%% %+5.1f (ours %.1f%% vs SF11 %.1f%%)  cp_gap %+5.2f  ours %+5.2f  SF11 %+5.2f"
              " | %s to move, material %+d, enemy king-ring attackers %d"
              % (r["wp_gap"], r["wp_ours"], r["wp_sf"], r["gap"], r["our_tot"], r["sf_tot"],
                 r["us"], r["mat"], r["katt"]))
        print("  %s" % r["fen"])
        ot = sorted(r["our"].items(), key=lambda kv: -abs(kv[1]))
        print("  OURS: " + "  ".join("%s=%+.2f" % (t, v) for t, v in ot if abs(v) > 0.05)[:180])
        st = sorted(r["sf"].items(), key=lambda kv: -abs(kv[1]))
        print("  SF11: " + "  ".join("%s=%+.2f" % (t, v) for t, v in st if abs(v) > 0.05))


if __name__ == "__main__":
    main()
