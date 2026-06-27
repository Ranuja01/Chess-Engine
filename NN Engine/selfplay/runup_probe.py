# -*- coding: utf-8 -*-
"""Fast run-up collapse probe (the PACE inner loop for the floor).

For each seed FEN (a collapse blindness point), play OUR engine (the side to move = the side that
collapsed) vs Stockfish, ply by ply, and CUT OFF early the moment the position collapses (SF eval for our
side drops >= DROP cp) or survives the window. Reports the collapse-rate over the set. ~3 min per knob
setting (a few plies/seed) vs ~90 min for a full-game playout -> enables SPSA-style knob perturbation.

Knobs are read once at engine init, so set them in the PROCESS env (the dispatcher `runup_probe` sub
forwards trailing KEY=VAL). Compare base vs candidate by running twice. jsonl-free; uses run_one + Arbiter.
Run: overnight_runner.sh runup_probe <seed_csv> <n> <plies> [KEY=VAL ...]
"""
import os, sys, csv, chess

THIS = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(THIS)
sys.path.insert(0, THIS)
sys.path.insert(0, os.path.join(ENGINE, 'diagnostics'))
from arbiter import Arbiter, find_stockfish
from tactical_test import run_one


def main():
    csv_path = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    plies = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    DROP = int(os.environ.get('RUNUP_DROP', '200'))          # cp drop (our POV) = a collapse
    mt = float(os.environ.get('SF_MOVETIME', '0.15'))
    rows = list(csv.DictReader(open(csv_path)))
    if len(rows) > n:
        step = len(rows) / n
        rows = [rows[int(k * step)] for k in range(n)]
    arb = Arbiter(find_stockfish(), movetime=mt)
    collapsed = tested = held = 0
    sum_drift = 0.0
    for r in rows:
        fen = r.get('fen_start') or r.get('fen')
        if not fen:
            continue
        try:
            b = chess.Board(fen)
            cp0, _, _ = arb.evaluate(b)
        except Exception:
            continue
        our = b.turn
        base = cp0 if our == chess.WHITE else -cp0
        tested += 1
        coll = False
        worst = base
        for _ in range(plies):
            if b.is_game_over():
                break
            try:
                if b.turn == our:
                    mv = run_one(b.fen(), set())['uci']
                else:
                    _, mv, _ = arb.evaluate(b)
                b.push_uci(mv)
                cp, _, _ = arb.evaluate(b)
            except Exception:
                break
            ev = cp if our == chess.WHITE else -cp
            worst = min(worst, ev)
            if ev <= base - DROP:
                coll = True
                break
        collapsed += int(coll)
        held += int(not coll)
        sum_drift += (base - worst)                          # avg worst-case drift over the window
    if hasattr(arb, 'close'):
        arb.close()
    if tested:
        print(f"runup: seeds={tested} plies<={plies} drop>={DROP}cp  "
              f"COLLAPSED {collapsed}/{tested} ({100*collapsed/tested:.0f}%)  "
              f"avg_drift={sum_drift/tested:.0f}cp  PRESET={os.environ.get('PRESET','?')} "
              f"MAX_DEPTH={os.environ.get('MAX_DEPTH','?')}")
    else:
        print("runup: no seeds tested")


if __name__ == "__main__":
    main()
