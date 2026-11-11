# -*- coding: utf-8 -*-
"""Search v2 watch (plan §7e): what the fixed-depth bench cannot see. Each arm (SEARCH_ARM 0 / 1, the same UCI binary)
plays its own self-play game from the same start positions, one engine process per game, no ucinewgame between moves, so
the history / killer / counter / TT / move-cache state carries from move to move as in a real game. Two limits: a fixed
depth, and a node limit that stops searches part-way through an iteration (the abort paths). Per ply the move, the final
score and the node count are recorded; the arms must agree at every ply.

  pyrun diagnostics/_sv2_game_identity.py [BIN=build_uci/chess_engine_uci_sv2] [STARTS=6] [PLIES=60] [DEPTH=8] [NODES=30000]

Start positions: game_regret_set rows spread over the file (all phases). The opening book is switched off.
REGISTERED (11-11, before the first run): SAME on every game at both limits. Any divergence = a port difference the bench
missed (the first diverging ply is printed with both arms' records).
RESULT (11-11, S1, chess_engine_uci_sv2): 12 of 12 games SAME (depth 8 and nodes 30000; 6 starts; 7-60 plies; games that
end early ended in mate or a draw). Not vacuous: the arm is live (the [toggles] line reads SEARCH_ARM=1, and the arms run
different code: queue #204's instruction counts differ by 1.87% on identical trees).
"""
import os, sys, csv
import chess, chess.engine
KV = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
THIS = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(THIS)
BIN = os.path.join(ENGINE, KV.get("BIN", "build_uci/chess_engine_uci_sv2"))
STARTS, PLIES = int(KV.get("STARTS", 6)), int(KV.get("PLIES", 60))
LIMITS = [("depth %s" % KV.get("DEPTH", 8), chess.engine.Limit(depth=int(KV.get("DEPTH", 8)))),
          ("nodes %s" % KV.get("NODES", 30000), chess.engine.Limit(nodes=int(KV.get("NODES", 30000))))]

rows = list(csv.DictReader(open(os.path.join(THIS, "ks_sets", "game_regret_set.csv"), newline="")))
starts = [rows[k * len(rows) // STARTS]["fen"] for k in range(STARTS)]


def play(fen, limit, arm):
    env = dict(os.environ, SEARCH_ARM=str(arm))
    eng = chess.engine.SimpleEngine.popen_uci(BIN, cwd=ENGINE, env=env)
    try:
        if "OwnBook" in eng.options:
            eng.configure({"OwnBook": False})
        b = chess.Board(fen)
        rec = []
        for _ in range(PLIES):
            if b.is_game_over(claim_draw=True):
                break
            r = eng.play(b, limit, info=chess.engine.INFO_SCORE | chess.engine.INFO_BASIC)
            rec.append((r.move.uci() if r.move else None, str(r.info.get("score")), r.info.get("nodes"), r.info.get("depth")))
            if r.move is None:
                break
            b.push(r.move)
        return rec
    finally:
        eng.quit()


diffs = 0
for name, limit in LIMITS:
    for k, fen in enumerate(starts):
        a0, a1 = play(fen, limit, 0), play(fen, limit, 1)
        if a0 == a1:
            print("%-12s start %d: SAME, %d plies (nodes in the last record %s)" % (name, k, len(a0), a0[-1][2] if a0 else "-"), flush=True)
        else:
            diffs += 1
            j = next((i for i, (x, y) in enumerate(zip(a0, a1)) if x != y), min(len(a0), len(a1)))
            print("%-12s start %d: DIFFERENT at ply %d of %d / %d\n    arm 0 %s\n    arm 1 %s"
                  % (name, k, j, len(a0), len(a1), a0[j] if j < len(a0) else "-", a1[j] if j < len(a1) else "-"), flush=True)
print("game identity: %d of %d games differ" % (diffs, len(LIMITS) * len(starts)))
