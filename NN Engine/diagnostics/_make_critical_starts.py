# -*- coding: utf-8 -*-
"""CRITICAL-POINT start positions for narrow-item game venues (2026-10-18; owner's refinement of the narrow-item rule).

  pyrun diagnostics/_make_critical_starts.py [GAP=10] [KS=4,8] [CP=200] [OUT=selfplay/eg_critical_fens.txt]

CRITICAL positions = SF18 d14 multi-PV (pe/light_mpv_s*.csv, K=8) where the best move beats the second by ≥ GAP win% points for
the mover (one move keeps the result). Each is traced to its SOURCE GAME (EGX_fens.csv game id → selfplay/games/<id>) and the
position KS plies EARLIER is taken — "a few moves out from a critical point", where an item's knowledge should steer the game.
Starts are SF18-labelled (d14) and kept only if still CLOSE (|cp| ≤ CP) and not over, de-duplicated. tournament.py FEN lines.
⚠️ Run game venues from these starts with USE_OPENING_BOOK=0 (the book holds endgame positions — #75 lesson).
"""
import os, sys, csv, glob, json, math
KV = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
THIS = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(THIS)
import chess, chess.engine
GAP, CP = float(KV.get("GAP", "10")), float(KV.get("CP", "200"))
KS = [int(x) for x in KV.get("KS", "4,8").split(",")]
OUT = KV.get("OUT", os.path.join(ENGINE, "selfplay", "eg_critical_fens.txt"))
G = os.path.join(ENGINE, "selfplay", "games")
K = 0.00368208
wp = lambda cp: 100 / (1 + math.exp(-K * max(-1500, min(1500, cp))))

games = {r["fen"]: r["game"] for r in csv.DictReader(open("/mnt/e/chess_data/retune/EGX_fens.csv", newline=""))}
crit = []
for p in sorted(glob.glob("/mnt/e/chess_data/pe/light_mpv_s*.csv")):
    for r in csv.DictReader(open(p, newline="")):
        mv = [float(x.split(":")[1]) for x in r["moves"].split(";") if ":" in x]
        if len(mv) < 2 or r["fen"] not in games:
            continue
        white = r["fen"].split()[1] == "w"
        w = sorted((wp(c) if white else 100 - wp(c) for c in mv), reverse=True)
        if w[0] - w[1] >= GAP:
            crit.append((r["fen"], games[r["fen"]], w[0] - w[1]))
cand, seen = [], set()
for fen, gid, gap in crit:
    path = os.path.join(G, gid.split("#")[0])
    if not os.path.exists(path):
        continue
    fens = [json.loads(l).get("fen") for l in open(path) if '"type": "move"' in l]
    if fen not in fens:
        continue
    i = fens.index(fen)
    for k in KS:
        if i - k >= 0 and fens[i - k] not in seen:
            seen.add(fens[i - k])
            cand.append((fens[i - k], k, gap))
sf = chess.engine.SimpleEngine.popen_uci(os.environ["STOCKFISH_PATH"])
kept = []
for f, k, gap in cand:
    b = chess.Board(f)
    if b.is_game_over():
        continue
    cp = sf.analyse(b, chess.engine.Limit(depth=14))["score"].white().score(mate_score=100000)
    if abs(cp) <= CP:
        kept.append((f, k, gap, cp))
sf.quit()
with open(OUT, "w") as fh:
    fh.write("# CRITICAL-POINT starts: %d critical positions (gap >= %.0f win%% pts, SF18 multi-PV) → %s plies earlier in the source "
             "game, kept if |SF18 d14| <= %.0f cp. Run with USE_OPENING_BOOK=0.\n" % (len(crit), GAP, "/".join(map(str, KS)), CP))
    for f, k, gap, cp in kept:
        fh.write("%s ; crit k%d gap%.0f sf%+.0f\n" % (f, k, gap, cp))
print("critical %d · candidate starts %d · kept (close) %d → %s" % (len(crit), len(cand), len(kept), OUT))
