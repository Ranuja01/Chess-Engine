# -*- coding: utf-8 -*-
"""EVAL vs DEPTH vs PRUNING triage of the owner's game positions (2026-10-18; owner: "our eval may see it, but our search may be
discarding important branches — the eval's strength has gone past the search's").

  MODE=sf     pyrun diagnostics/_case_triage.py              → SF18 d18 best move + PV (8 plies), SF eval at the PV end,
                                                               OUR static at the PV end (candidate base) → /mnt/e/chess_data/triage_sf.json
  MODE=search pyrun diagnostics/_case_triage.py TAG=normal   → our search move at fixed depth (MAX_DEPTH env) → triage_<TAG>.json
              (run once normally and once with ENABLE_LMR=0 ENABLE_FUTILITY=0 ENABLE_RAZORING=0 ENABLE_LMP=0 ENABLE_RFP=0
               ENABLE_ROOT_RAZOR=0 SEE_PRUNE_CAPTURES=0, TAG=noprune — knobs latch at init, one process per setting)
  MODE=report pyrun diagnostics/_case_triage.py              → per position: verdict
     EVAL     — our static at the end of SF's line disagrees with SF by ≥ 10 win% pts (we would misjudge it even if found)
     PRUNING  — static agrees, the normal search misses SF's move, the no-pruning search FINDS it
     DEPTH    — static agrees, both searches miss it (needs more depth / extensions, not less pruning)
     ok       — the normal search already plays SF's move at this depth
Positions: diagnostics/ks_sets/pot_owner_cases.txt (non-comment lines, tag = first token of the comment).
"""
import os, sys, json, math
KV = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
THIS = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(THIS)
sys.path.insert(0, ENGINE); sys.path.insert(0, THIS)
os.environ.setdefault("PRESET", "LONG_FORMAT"); os.environ.setdefault("USE_OPENING_BOOK", "0")
import chess, chess.engine
MODE = KV.get("MODE", "report")
D = "/mnt/e/chess_data"
K = 0.00368208
wp = lambda cp: 100 / (1 + math.exp(-K * max(-1500, min(1500, cp))))


def cases():
    out = []
    for line in open(os.path.join(THIS, "ks_sets/pot_owner_cases.txt")):
        if line.startswith("#") or "/" not in line:
            continue
        fen, _, note = line.partition("#")
        out.append((fen.strip(), note.split()[0] if note.split() else "?"))
    return out


def mode_sf():
    import ChessAI
    ai = ChessAI.ChessAI(None, None, chess.Board(), True)
    sf = chess.engine.SimpleEngine.popen_uci(os.environ["STOCKFISH_PATH"])
    sf.configure({"Threads": 1})
    res = {}
    for fen, tag in cases():
        b = chess.Board(fen)
        info = sf.analyse(b, chess.engine.Limit(depth=18))
        pv = info.get("pv", [])[:8]
        best = pv[0].uci() if pv else None
        e = b.copy()
        for m in pv:
            e.push(m)
        if e.is_game_over():
            sf_end, ours_end = None, None
        else:
            sf_end = sf.analyse(e, chess.engine.Limit(depth=14))["score"].white().score(mate_score=10000)
            ours_end = -int(ai.ev_breakdown(e)["total"]) / 10.0
        res[fen] = {"tag": tag, "best": best, "pv_len": len(pv), "sf_root": info["score"].white().score(mate_score=10000),
                    "sf_end": sf_end, "ours_end": ours_end}
        print(tag, res[fen], flush=True)
    sf.quit()
    json.dump(res, open(os.path.join(D, "triage_sf.json"), "w"), indent=1)


def mode_search():
    from tactical_test import run_one
    res = {}
    for fen, tag in cases():
        o = run_one(fen, [])
        res[fen] = {"tag": tag, "move": o.get("uci"), "eval": o.get("eval"), "depth": o.get("depth")}
        print(tag, res[fen], flush=True)
    json.dump(res, open(os.path.join(D, "triage_%s.json" % KV.get("TAG", "normal")), "w"), indent=1)


def mode_report():
    sf = json.load(open(os.path.join(D, "triage_sf.json")))
    nm = json.load(open(os.path.join(D, "triage_normal.json")))
    npn = json.load(open(os.path.join(D, "triage_noprune.json")))
    from collections import Counter
    c = Counter()
    print("%-6s %-7s %-7s %-7s %9s %9s  %s" % ("case", "SFbest", "normal", "noprune", "SF@end", "ours@end", "verdict"))
    for fen, s in sf.items():
        n, p = nm.get(fen, {}), npn.get(fen, {})
        if s["sf_end"] is None:
            gap = 0.0
        else:
            gap = abs(wp(s["ours_end"] * 1.0) - wp(s["sf_end"]))
        if n.get("move") == s["best"]:
            v = "ok"
        elif gap >= 10:
            v = "EVAL"
        elif p.get("move") == s["best"]:
            v = "PRUNING"
        else:
            v = "DEPTH"
        c[v] += 1
        print("%-6s %-7s %-7s %-7s %9s %9s  %s (static gap %.0f win%% pts)" % (s["tag"], s["best"], n.get("move"), p.get("move"),
              s["sf_end"], s["ours_end"], v, gap))
    print("VERDICTS:", dict(c))


{"sf": mode_sf, "search": mode_search, "report": mode_report}[MODE]()
