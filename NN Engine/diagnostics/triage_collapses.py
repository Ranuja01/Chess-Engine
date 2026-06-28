# -*- coding: utf-8 -*-
"""Triage vs-SF collapse points as DEPTH-based (horizon) vs EVAL-based.

The vs_sf harness flags games where our eval peaked winning then we didn't win, and records the DECISION
fen (the position we chose the over-rated move from). At LIGHTNING those collapses are mostly horizon
blunders (a deeper search would avoid them) — not the self-play-invisible EVAL holes we want. This re-
examines each decision position at DEEP depth + Stockfish, so ~20 candidates are classified in ~10 min
without replaying whole STANDARD-time games:

  HORIZON : a deeper search (same config) picks a DIFFERENT move than the game blunder -> depth fixes it.
  PRUNING : deep search still errs, but re-searching with pruning RELAXED (same depth) finds the better
            move -> the refutation was reachable but pruned (LMP/null-move/LMR). A search-lane fix.
  EVAL    : even deep AND low-prune still plays it / our eval stays far above SF -> a genuine eval hole.

Run (WSL, from NN Engine/, interop up):
  MAX_DEPTH=18 PRESET=LONG_FORMAT python diagnostics/triage_collapses.py selfplay/games/vssf_2400 [sf_time]
"""
import os
import sys
import csv
import subprocess

# Forward-pruning relaxed: LMP off + null-move-extra off + history/extra LMR off + lazy re-sort off.
# Re-searching the decision position under this at the SAME deep depth isolates pruning from eval — if the
# move now corrects, the refutation was being pruned, not mis-evaluated.
LOWPRUNE = {"ENABLE_LMP": "0", "ENABLE_LAZY_RESORT": "0", "NULLMOVE_EXTRA": "0",
            "HISTORY_LMR_SCALE": "0", "LMR_EXTRA": "0"}

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, ENGINE_DIR)

import chess
import chess.engine
from tactical_test import run_one


def find_sf():
    for var in ("STOCKFISH_PATH", "STOCKFISH"):
        p = os.environ.get(var)
        if p and os.path.exists(p):
            return p
    c = "/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/stockfish/stockfish-windows-x86-64-avx2.exe"
    return c if os.path.exists(c) else None


def lowprune_move(fen):
    """Our engine's move on `fen` at the same deep depth but with pruning RELAXED (LOWPRUNE), in a fresh
    process (knobs are read once at engine init). Returns the UCI move, or None on failure."""
    code = ("import sys; sys.path.insert(0,'diagnostics'); from tactical_test import run_one;"
            "r=run_one(sys.argv[1], set()); print(r['uci'])")
    env = dict(os.environ); env.update(LOWPRUNE)
    try:
        out = subprocess.run([sys.executable, "-c", code, fen], env=env, cwd=ENGINE_DIR,
                             capture_output=True, text=True, timeout=240)
        line = [l for l in out.stdout.splitlines() if l.strip()]
        return line[-1].strip() if line else None
    except Exception:
        return None


def main():
    games_dir = sys.argv[1] if len(sys.argv) > 1 else "selfplay/games/vssf_2400"
    sf_time = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
    coll = os.path.join(games_dir, "collapses.csv")
    if not os.path.exists(coll):
        print(f"[triage] no {coll}"); return
    rows = list(csv.DictReader(open(coll)))
    if not rows:
        print("[triage] no collapses to triage"); return

    sf = chess.engine.SimpleEngine.popen_uci(find_sf())
    sf.configure({"Threads": 1})

    n = {"EVAL": 0, "HORIZON": 0, "PRUNING": 0}
    print(f"[triage] {len(rows)} collapse points  (deep re-search + SF @ {sf_time}s; low-prune probe on survivors)\n")
    print(f"{'g':>3} {'col':>5} {'peak':>6} {'gameMv':>7} {'deepMv':>7} {'lpMv':>7} {'ourDeep':>8} {'SF':>7} {'gap':>6}  class")
    results = []
    for r in rows:
        fen = r["decision_fen"]
        board = chess.Board(fen)
        # our engine, deep (depth/preset come from env: MAX_DEPTH / PRESET).
        ro = run_one(fen, set())
        our_uci = ro["uci"]
        our_eval = ro["eval"]                    # mover-POV (= our POV) milli-pawns
        # SF eval of the SAME decision position, our-POV centipawns.
        info = sf.analyse(board, chess.engine.Limit(time=sf_time))
        sf_cp = info["score"].pov(board.turn).score(mate_score=100000)
        sf_best = info["pv"][0].uci() if info.get("pv") else None
        our_cp = (our_eval / 10.0) if isinstance(our_eval, int) else None
        gap = (our_cp - sf_cp) if (our_cp is not None and sf_cp is not None) else None
        game_mv = r["peak_move"]
        lp_uci = ""
        if our_uci == sf_best:
            cls = "EVAL"                          # we play SF's own best move -> not a move error, pure over-read
        elif our_uci != game_mv:
            cls = "HORIZON"                       # deeper search (same config) already picks a different move
        else:
            # we persist in a move SF dislikes even at depth; relax pruning to split pruning vs eval
            lp_uci = lowprune_move(fen) or "?"
            cls = "PRUNING" if (lp_uci not in ("?", "") and lp_uci != game_mv) else "EVAL"
        n[cls] += 1
        print(f"{r['game']:>3} {r['our_color'][:5]:>5} {float(r['peak_eval'])/1000:>+6.1f} "
              f"{game_mv:>7} {our_uci:>7} {lp_uci:>7} {('' if our_cp is None else f'{our_cp/100:+.2f}'):>8} "
              f"{('' if sf_cp is None else f'{sf_cp/100:+.2f}'):>7} {('' if gap is None else f'{gap/100:+.2f}'):>6}  {cls}")
        results.append({**r, "deep_move": our_uci, "lowprune_move": lp_uci, "our_cp": our_cp,
                        "sf_cp": sf_cp, "sf_best": sf_best, "gap_cp": gap, "class": cls})
    sf.quit()

    out = os.path.join(games_dir, "triage.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader(); w.writerows(results)
    print(f"\n[triage] EVAL (eval holes, the targets): {n['EVAL']}   "
          f"PRUNING (search-lane): {n['PRUNING']}   HORIZON (depth-fixable): {n['HORIZON']}")
    print(f"[triage] -> {out}  (sort by gap_cp for the worst eval over-reads)")


if __name__ == "__main__":
    main()
