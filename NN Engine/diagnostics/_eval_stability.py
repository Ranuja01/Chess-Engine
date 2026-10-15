# -*- coding: utf-8 -*-
"""EVAL STABILITY along best play (2026-10-15, POT brief §5.3) — the gate for the POT tempo rule.

  V2_PRESET=shipped pyrun diagnostics/_eval_stability.py [N=20000] [IN=/mnt/e/chess_data/retune/EGX_sf18.csv] [KEY=VAL ...]

Principle: a perfect evaluation does not change along best play (the value is a martingale of the best line). For each
position, play SF18's stored best move (`best_uci`) and read our static eval before and after (White cp):
    swing = static(after) − static(before)
  · mean |swing|              — instability (lower is better)
  · mean swing by side to move — the ODD/EVEN tempo bias: a stable eval has ≈ 0 for both movers; an eval that over-rewards
                                 having the move reads negative after White moves and positive after Black moves.
Stratified by the endgame trajectory markers (opposition · race · breakthrough · tempi_unequal · none). ★ A correct tempo
rule shrinks |swing| ONLY in the strata where it fires and leaves the others EXACTLY unchanged (it contributes 0 there);
run the same N on both arms (one process per setting — knobs latch at init) and compare per stratum.
Mate-scored / terminal / illegal-move rows are skipped; every other row is colour-MIRRORED (EGX is ~96% White to move).
Static only — no search, cheap.
"""
import os, sys, csv, random
for _a in sys.argv[1:]:
    if "=" in _a:
        _k, _v = _a.split("=", 1)
        os.environ[_k] = _v
THIS = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(THIS)
sys.path.insert(0, ENGINE); sys.path.insert(0, THIS); sys.path.insert(0, os.path.join(ENGINE, "selfplay"))
os.environ.setdefault("PRESET", "LONG_FORMAT"); os.environ.setdefault("USE_OPENING_BOOK", "0")
import chess
import ChessAI
from _build_trajectories import unresolved_eg, EG_MARKERS

N = int(os.environ.get("N", "20000"))
IN = os.environ.get("IN", "/mnt/e/chess_data/retune/EGX_sf18.csv")
SEED = int(os.environ.get("SEED", "1015"))


def main():
    rows = [r for r in csv.DictReader(open(IN, newline="")) if r.get("best_uci") and abs(int(float(r["best_cp"] or 0))) < 20000]
    random.seed(SEED)
    random.shuffle(rows)
    ai = ChessAI.ChessAI(None, None, chess.Board(), True)

    def ours(b):
        return -int(ai.ev_breakdown(b)["total"]) / 10.0           # Black-positive mp → White cp

    strata = EG_MARKERS + ["none", "all"]
    acc = {s: {"n": 0, "abs": 0.0, "w_n": 0, "w_sum": 0.0, "b_n": 0, "b_sum": 0.0} for s in strata}
    used = 0
    for r in rows:
        if used >= N:
            break
        b = chess.Board(r["fen"])
        try:
            mv = chess.Move.from_uci(r["best_uci"])
        except ValueError:
            continue
        if b.is_game_over() or mv not in b.legal_moves:
            continue
        if used % 2 == 1:   # EGX is ~96% White to move ⇒ exact colour mirror of every other row balances the movers
            b = b.mirror()
            mv = chess.Move(chess.square_mirror(mv.from_square), chess.square_mirror(mv.to_square), mv.promotion)
        mk = unresolved_eg(b)
        mover_white = b.turn
        s0 = ours(b)
        b.push(mv)
        if b.is_game_over():
            continue
        sw = ours(b) - s0
        used += 1
        tags = [m for m, v in zip(EG_MARKERS, mk) if v] or ["none"]
        for t in tags + ["all"]:
            a = acc[t]
            a["n"] += 1
            a["abs"] += abs(sw)
            if mover_white:
                a["w_n"] += 1; a["w_sum"] += sw
            else:
                a["b_n"] += 1; a["b_sum"] += sw
    print("EVAL STABILITY along SF18 best moves — %d positions from %s" % (used, os.path.basename(IN)))
    print("%-14s %7s %10s %16s %16s" % ("stratum", "n", "mean|swing|", "swing W moved", "swing B moved"))
    for s in strata:
        a = acc[s]
        if not a["n"]:
            continue
        w = a["w_sum"] / a["w_n"] if a["w_n"] else float("nan")
        bb = a["b_sum"] / a["b_n"] if a["b_n"] else float("nan")
        print("%-14s %7d %10.1f cp %10.1f (n=%d) %10.1f (n=%d)" % (s, a["n"], a["abs"] / a["n"], w, a["w_n"], bb, a["b_n"]))


if __name__ == "__main__":
    main()
