# -*- coding: utf-8 -*-
"""Minimal per-term eval profiler over the cploss_corpus (robust FEN source; avoids the game.jsonl format
dependency of eval_profile.py). Requires a PROFILE_EVAL=1 build (`overnight_runner.sh build_profile`).
Drives placement_and_piece_eval `reps` times per FEN via the ChessAI profile bridge and ranks the
exclusive top-level terms by %share of eval cycles => the next NPS hotspot to optimise.

  overnight_runner.sh build_profile
  overnight_runner.sh pyrun diagnostics/eval_profile_corpus.py [--reps 800] [--n 300]
  overnight_runner.sh build            # restore production (byte-id 247)
"""
import os, sys, csv
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
THIS = os.path.dirname(os.path.abspath(__file__)); ENGINE = os.path.dirname(THIS)
sys.path.insert(0, THIS); sys.path.insert(0, ENGINE)
argv = sys.argv[1:]
REPS = 800; N = 300
if "--reps" in argv: i = argv.index("--reps"); REPS = int(argv[i+1]); del argv[i:i+2]
if "--n" in argv: i = argv.index("--n"); N = int(argv[i+1]); del argv[i:i+2]
corpus = next((a for a in argv if a.endswith(".csv")), os.path.join(ENGINE, "selfplay", "tune_data", "cploss_corpus.csv"))

import chess
from ChessAI import ChessAI


def main():
    seed = chess.Board(); ai = ChessAI(None, None, seed, seed.turn)
    if ai.get_profile() == []:
        print("NOT a profiler build. Run `overnight_runner.sh build_profile` first."); return
    rows = list(csv.DictReader(open(corpus)))
    # even spread across strata/phases
    fens = [r["fen"] for r in rows][::max(1, len(rows) // N)][:N]
    boards = []
    for f in fens:
        try:
            b = chess.Board(f)
            if not (b.is_checkmate() or b.is_stalemate()): boards.append(b)
        except Exception:
            continue
    ai.reset_profile()
    for b in boards:
        ai.profile_eval(b, REPS)
    prof = ai.get_profile()
    excl = [p for p in prof if p.get("exclusive")]
    base = sum(p["cycles"] for p in excl) or 1
    print(f"[eval_profile_corpus] n={len(boards)} FENs x {REPS} reps  ({len(boards)*REPS:,} eval calls)")
    print(f"  {'term':>18} {'%share':>8} {'cyc/call':>10}   (exclusive top-level terms; %share = of eval cycles)")
    for p in sorted(excl, key=lambda x: -x["cycles"]):
        share = 100.0 * p["cycles"] / base
        cpc = p["cycles"] / p["calls"] if p["calls"] else 0
        if share >= 0.5:
            print(f"  {p['term']:>18} {share:7.1f}% {cpc:10.0f}")
    print("\n  nested/excluded terms (not in the %base; 0 unless their line-scopes are active):")
    for p in sorted((p for p in prof if not p.get("exclusive")), key=lambda x: -x["cycles"]):
        if p["cycles"] > 0:
            print(f"    {p['term']:>18} cyc/call {p['cycles']/max(1,p['calls']):.0f}")
    print("\n  READ: the top %share term = the biggest eval cost = the next NPS target (make it cheaper/lazy,")
    print("        byte-identically). Bishop colour-complex was already cheapened (~33% -> less).")


if __name__ == "__main__":
    main()
