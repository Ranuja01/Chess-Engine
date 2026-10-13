# -*- coding: utf-8 -*-
"""ENDGAME SIGNATURES (2026-10-13, POT / endgame-knowledge groundwork): how often each of Stockfish's endgame.cpp material
signatures occurs in OUR games, and how our static eval does there vs SF11 / SF15.1 / SF18 static (all against SF18 d14).

  MODE=build [IN=/mnt/e/chess_data/retune/corpus_v2.csv] [PER=400] [OUT=/mnt/e/chess_data/bench1013/eg_signatures.csv]
      held-out positions (split val / stress_hold / excluded-from-train never), up to PER per signature, mates dropped;
      columns fen, target_total (SF18 d14, White-POV pawns), split=val, signature — the _reference_ceiling.py CORPUS format.
  then: _reference_ceiling.py CORPUS=<OUT> N=99999 DUMP=<dump> REFS=1 DUMPLABEL="v2 gen3" V2_PRESET=shipped
  MODE=read DUMP=<dump> SIG=<OUT> [EVALS=…]   per signature: n, win% MSE and side-ahead bias per evaluator.

Signatures (material only, colour-blind, kings implicit; "P" = 1+ pawns): SF11 value fns KXK (lone king vs mating material),
KBNK, KRKP, KRKB, KRKN, KQKP, KQKR, KNNK; scaling fns KBPsK, KQKRPs, KRPKR, KRPKB, KRPPKRP, KPsK, KBPKB (same / opposite
bishops), KBPKN, KNPK, KNPKB, KPKP; plus the families they sit in (rook ending, minor ending, pawn ending, OCB with pawns).
"""
import os, sys, csv, math, random
from collections import defaultdict, Counter
import chess

KV = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
wp = lambda p: 100.0 / (1.0 + math.exp(-0.00368208 * max(-1500.0, min(1500.0, p * 100.0))))


def side(b, c):
    return {t: len(b.pieces(t, c)) for t in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)}


def signature(fen):
    b = chess.Board(fen)
    W, B = side(b, True), side(b, False)
    for strong, weak in ((W, B), (B, W)):
        s_pc = (strong[chess.KNIGHT], strong[chess.BISHOP], strong[chess.ROOK], strong[chess.QUEEN])
        w_pc = (weak[chess.KNIGHT], weak[chess.BISHOP], weak[chess.ROOK], weak[chess.QUEEN])
        sp, wpn = strong[chess.PAWN], weak[chess.PAWN]
        if sum(w_pc) == 0 and wpn == 0 and (sum(s_pc) > 0 and (s_pc[2] or s_pc[3] or sum(s_pc) >= 2)) and s_pc != (1, 1, 0, 0):
            return "KXK (lone king vs mating material)"
        if s_pc == (1, 1, 0, 0) and sp == 0 and sum(w_pc) == 0 and wpn == 0:
            return "KBNK"
        if s_pc == (0, 0, 1, 0) and sp == 0 and sum(w_pc) == 0 and wpn >= 1:
            return "KRKP (rook vs pawns)"
        if s_pc == (0, 0, 1, 0) and sp == 0 and w_pc == (0, 1, 0, 0) and wpn == 0:
            return "KRKB"
        if s_pc == (0, 0, 1, 0) and sp == 0 and w_pc == (1, 0, 0, 0) and wpn == 0:
            return "KRKN"
        if s_pc == (0, 0, 0, 1) and sp == 0 and sum(w_pc) == 0 and wpn >= 1:
            return "KQKP (queen vs pawns)"
        if s_pc == (0, 0, 0, 1) and w_pc == (0, 0, 1, 0):
            return "KQKR(Ps) (queen vs rook)"
        if s_pc == (2, 0, 0, 0) and sp == 0 and sum(w_pc) == 0:
            return "KNNK(P)"
        if s_pc == (0, 1, 0, 0) and sp >= 1 and sum(w_pc) == 0 and wpn == 0:
            return "KBPsK (bishop + pawns vs king)"
        if s_pc == (1, 0, 0, 0) and sp == 1 and sum(w_pc) == 0 and wpn == 0:
            return "KNPK"
        if s_pc == (0, 0, 1, 0) and w_pc == (0, 0, 1, 0) and sp == 1 and wpn == 0:
            return "KRPKR"
        if s_pc == (0, 0, 1, 0) and w_pc == (0, 0, 1, 0) and sp == 2 and wpn == 1:
            return "KRPPKRP"
        if s_pc == (0, 0, 1, 0) and w_pc == (0, 1, 0, 0) and sp >= 1:
            return "KRPKB (rook+pawns vs bishop)"
        if s_pc == (0, 1, 0, 0) and w_pc == (1, 0, 0, 0) and sp >= 1:
            return "KBPKN (bishop+pawns vs knight)"
        if s_pc == (1, 0, 0, 0) and w_pc == (0, 1, 0, 0) and sp >= 1:
            return "KNPKB (knight+pawns vs bishop)"
    W_pc = (W[chess.KNIGHT], W[chess.BISHOP], W[chess.ROOK], W[chess.QUEEN]); B_pc = (B[chess.KNIGHT], B[chess.BISHOP], B[chess.ROOK], B[chess.QUEEN])
    if sum(W_pc) == 0 and sum(B_pc) == 0:
        if (W[chess.PAWN] == 0) != (B[chess.PAWN] == 0):
            return "KPsK (pawns vs bare king)"
        if W[chess.PAWN] == 1 and B[chess.PAWN] == 1:
            return "KPKP"
        return "pawn ending (general)"
    if W_pc == B_pc == (0, 1, 0, 0):
        wb = [chess.square_file(s) + chess.square_rank(s) for s in b.pieces(chess.BISHOP, True)][0] % 2
        bb = [chess.square_file(s) + chess.square_rank(s) for s in b.pieces(chess.BISHOP, False)][0] % 2
        return "KBPKB opposite bishops" if wb != bb else "KBPKB same-colour bishops"
    if W_pc == B_pc == (0, 0, 1, 0):
        return "rook ending (general)"
    if W_pc == B_pc and W_pc[2] == 0 and W_pc[3] == 0:
        return "minor ending (general)"
    return None


def build():
    src = KV.get("IN", "/mnt/e/chess_data/retune/corpus_v2.csv")
    per = int(KV.get("PER", 400))
    out = KV.get("OUT", "/mnt/e/chess_data/bench1013/eg_signatures.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pool = defaultdict(list); cnt_all = Counter(); n_all = 0
    for r in csv.DictReader(open(src, newline="")):
        if r["slice"] not in ("A", "A2", "V1", "GAUNT") or r["mate"] == "1":
            continue
        if r["stage"] not in ("egl", "eg"):
            continue
        s = signature(r["fen"])
        n_all += 1
        if s is None:
            continue
        cnt_all[s] += 1
        if r["split"] == "val":
            pool[s].append(r)
    random.seed(13)
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["fen", "target_total", "split", "signature"])
        print("  %-38s %9s %8s %6s" % ("signature", "in games", "% of EG", "kept"))
        for s, n in cnt_all.most_common():
            rows = pool[s]; random.shuffle(rows); rows = rows[:per]
            for r in rows:
                w.writerow([r["fen"], "%.3f" % (float(r["sf_cp"]) / 100.0), "val", s])
            print("  %-38s %9d %7.2f%% %6d" % (s, n, 100.0 * n / max(1, n_all), len(rows)))
    print("  own-play endgame-leaning + endgame positions scanned: %d → %s" % (n_all, out))


def read():
    sig = {r["fen"]: r["signature"] for r in csv.DictReader(open(KV["SIG"], newline=""))}
    preds = defaultdict(dict); tgt = {}
    for r in csv.DictReader(open(KV["DUMP"], newline="")):
        preds[r["fen"]][r["evaluator"]] = float(r["pred"]); tgt[r["fen"]] = float(r["target"])
    evals = KV.get("EVALS", "v2 gen3,SF11 classical,SF15.1 classical,SF18 static").split(",")
    groups = defaultdict(list)
    for f, s in sig.items():
        if f in preds and all(e in preds[f] for e in evals):
            groups[s].append(f)
    print("  %-38s %5s  " % ("signature", "n") + "  ".join("%-20s" % e for e in evals))
    for s in sorted(groups, key=lambda k: -len(groups[k])):
        fs = groups[s]
        cells = []
        for e in evals:
            err = [wp(preds[f][e]) - wp(tgt[f]) for f in fs]
            mse = sum(x * x for x in err) / len(err)
            big = [(x, tgt[f]) for x, f in zip(err, fs) if abs(tgt[f]) > 0.25]
            bias = sum(x * (1 if t > 0 else -1) for x, t in big) / len(big) if big else float("nan")
            cells.append("MSE %6.1f bias %+5.1f" % (mse, bias))
        print("  %-38s %5d  " % (s, len(fs)) + "  ".join(cells))


if __name__ == "__main__":
    {"build": build, "read": read}[KV.get("MODE", "build")]()
