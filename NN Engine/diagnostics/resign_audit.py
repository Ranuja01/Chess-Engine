# -*- coding: utf-8 -*-
"""Are our resignations JUSTIFIED per SF? For every game that ended 'ours resigned', take the position we
resigned in (the last logged FEN = our turn to move, where our engine returned RESIGN) and get SF's eval from
OUR point of view. Distribution answers: are these truly lost (resign fair) or only mildly bad (resign leaks
losses)? Independent of our own eval's calibration — SF is the yardstick.

Run: pyrun diagnostics/resign_audit.py <games_dir> [sf_depth=18]
  e.g. pyrun diagnostics/resign_audit.py selfplay/games/sfelo2400_base200 18
"""
import os
import sys
import csv
import json
import glob

import chess
import chess.engine


def main():
    gd = sys.argv[1]
    depth = int(sys.argv[2]) if len(sys.argv) > 2 else 18
    sf_path = os.environ["STOCKFISH_PATH"]

    # games that ended in our resignation
    resign_games = {}
    rp = os.path.join(gd, "results.csv")
    for r in csv.DictReader(open(rp)):
        if "resigned" in (r.get("reason") or ""):
            resign_games[str(r["game"])] = r.get("our_color")
    print("resigned games: %d" % len(resign_games))

    eng = chess.engine.SimpleEngine.popen_uci(sf_path)
    try:
        buckets = {"lost(<=-600)": 0, "bad(-600..-300)": 0, "holdable(-300..-150)": 0, "unclear(>-150)": 0}
        cps = []
        for jf in sorted(glob.glob(os.path.join(gd, "game_*.jsonl"))):
            gid = os.path.basename(jf).replace("game_", "").replace(".jsonl", "").lstrip("0") or "0"
            if gid not in resign_games:
                continue
            recs = [json.loads(l) for l in open(jf) if l.strip()]
            fen = next((rc["fen"] for rc in reversed(recs) if rc.get("fen")), None)
            if not fen:
                continue
            b = chess.Board(fen)
            if b.is_game_over():
                continue
            info = eng.analyse(b, chess.engine.Limit(depth=depth))
            sc = info["score"].pov(b.turn)          # b.turn == us (we resigned on our move)
            cp = sc.score(mate_score=100000)
            cps.append(cp)
            if cp <= -600:
                buckets["lost(<=-600)"] += 1
            elif cp <= -300:
                buckets["bad(-600..-300)"] += 1
            elif cp <= -150:
                buckets["holdable(-300..-150)"] += 1
            else:
                buckets["unclear(>-150)"] += 1
    finally:
        eng.quit()

    n = len(cps)
    if not n:
        print("no resign positions evaluated"); return
    cps.sort()
    print("\nSF eval (our POV, cp) of %d resign positions @ depth %d:" % (n, depth))
    for k in ["lost(<=-600)", "bad(-600..-300)", "holdable(-300..-150)", "unclear(>-150)"]:
        print("  %-18s %4d  (%4.1f%%)" % (k, buckets[k], 100.0 * buckets[k] / n))
    med = cps[n // 2]
    print("median=%d  min=%d  max=%d  worse-than--600=%.1f%%"
          % (med, cps[0], cps[-1], 100.0 * sum(1 for c in cps if c <= -600) / n))
    print("\n>= -300cp = arguably NOT resign-worthy vs a 2400 (holdable/unclear) = leaked losses.")


if __name__ == "__main__":
    main()
