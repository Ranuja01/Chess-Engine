# -*- coding: utf-8 -*-
"""
Causal check for the capgain A/B: is the small negative lean circumstantial or causal?

capgain (ENABLE_CAPGAIN_PAWN_FIX) only changes the eval when `approximate_capture_gains` sees a
non-pawn capturing a rank-bonus pawn on the black branch. Measure how OFTEN that actually shifts the
eval on the tournament's own positions, and by how much — that sets the causal ceiling. If it fires in
a few % with small Delta, a -9.3 Elo lean cannot be mostly capgain -> circumstantial.

ev_breakdown reads the COMPILE-TIME default, so off/on is two builds:
    python diagnostics/capgain_causal.py extract           # sample FENs from capgain_vs_base games
    python diagnostics/capgain_causal.py eval off           # current build = capgain OFF
    # flip ENABLE_CAPGAIN_PAWN_FIX default true, rebuild, then:
    python diagnostics/capgain_causal.py eval on
    # restore default false, rebuild, then:
    python diagnostics/capgain_causal.py compare
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys, glob, json
import chess

THIS = os.path.dirname(os.path.abspath(__file__))
ENG = os.path.dirname(THIS)
sys.path.insert(0, ENG); sys.path.insert(0, os.path.join(ENG, "selfplay"))
RES = os.path.join(THIS, "results")
CORPUS = os.path.join(RES, "capgain_corpus.fen")


def extract(every=4, cap=1500):
    fens, seen = [], set()
    for p in sorted(glob.glob(os.path.join(ENG, "selfplay", "games", "capgain_vs_base", "game_*", "game.jsonl"))):
        i = 0
        for line in open(p):
            try: r = json.loads(line)
            except: continue
            if r.get("type") != "move" or r.get("opening"): continue
            fen = r.get("fen")
            if not fen: continue
            i += 1
            if i % every: continue
            key = " ".join(fen.split()[:2])
            if key in seen: continue
            seen.add(key); fens.append(fen)
            if len(fens) >= cap: break
        if len(fens) >= cap: break
    os.makedirs(RES, exist_ok=True)
    open(CORPUS, "w").write("\n".join(fens) + "\n")
    print("[extract] %d FENs -> %s" % (len(fens), CORPUS))


def do_eval(tag):
    from ChessAI import ChessAI
    ai = ChessAI(None, None, chess.Board(), True)
    out = os.path.join(RES, "capgain_causal_%s.jsonl" % tag)
    n = 0
    with open(out, "w") as o:
        for fen in [l.strip() for l in open(CORPUS) if l.strip()]:
            bd = ai.ev_breakdown(chess.Board(fen))
            if bd.get("checkmate"): continue
            o.write(json.dumps({"fen": fen, "cg": bd["capture_gains"], "total": bd["total"]}) + "\n")
            n += 1
    print("[eval %s] %d -> %s" % (tag, n, out))


def compare():
    off = {json.loads(l)["fen"]: json.loads(l) for l in open(os.path.join(RES, "capgain_causal_off.jsonl"))}
    on = {json.loads(l)["fen"]: json.loads(l) for l in open(os.path.join(RES, "capgain_causal_on.jsonl"))}
    n = fired = 0
    dtot = []
    for fen, o in off.items():
        s = on.get(fen)
        if not s: continue
        n += 1
        d = abs(s["total"] - o["total"]) / 1000.0   # eval shift in pawns
        if abs(s["cg"] - o["cg"]) > 0:
            fired += 1
            dtot.append(d)
    import statistics as st
    print("capgain CAUSAL CEILING on %d tournament positions:" % n)
    print("  fired (eval changed): %d / %d = %.1f%%" % (fired, n, 100.0 * fired / n))
    if dtot:
        print("  among fired: mean |eval shift| = %.3f p,  max = %.3f p,  median = %.3f p"
              % (st.mean(dtot), max(dtot), st.median(dtot)))
        big = sum(1 for d in dtot if d >= 0.5)
        print("  fired with |shift| >= 0.5p: %d (%.2f%% of all positions)" % (big, 100.0 * big / n))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "extract": extract()
    elif cmd == "eval": do_eval(sys.argv[2])
    elif cmd == "compare": compare()
