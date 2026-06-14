# -*- coding: utf-8 -*-
"""
Daytime proxy for the symmetrize-up A/B: does raising the white knight-mobility / rook-behind terms
to match black (ENABLE_KNIGHT_MOB_SYM_UP + ENABLE_ROOK_DBLCOUNT_SYM_UP) make our STATIC eval more
accurate vs Stockfish on endgame positions where those terms fire?

This is a tuning question, so SF-divergence is the right signal (the overnight self-play stays the gate).
ev_breakdown reads the COMPILE-TIME default (not env), so the off/on comparison is two builds:

    # corpus once (pure FEN parsing, build-agnostic):
    python diagnostics/symup_accuracy.py extract --tag correctness_vs_base --every 7 --cap 300

    # on the OFF (baseline) build, with SF reference:
    python diagnostics/symup_accuracy.py eval --tag base --sf

    # flip both SYM_UP defaults true, rebuild, then:
    python diagnostics/symup_accuracy.py eval --tag symup

    # restore defaults false, rebuild, then compare (SF taken from the base file):
    python diagnostics/symup_accuracy.py compare --ref sf_search
    python diagnostics/symup_accuracy.py compare --ref sf_static

Everything is White-POV pawns. "FIRED" = positions where symup changed our eval (the terms fired);
that subset is the verdict. correction-helps = how often the white-side raise moves us TOWARD SF.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import glob
import json
import argparse
import statistics as st

import chess

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.path.dirname(THIS_DIR)
SELFPLAY_DIR = os.path.join(ENGINE_DIR, "selfplay")
RESULTS_DIR = os.path.join(THIS_DIR, "results")
sys.path.insert(0, ENGINE_DIR)
sys.path.insert(0, SELFPLAY_DIR)

CORPUS = os.path.join(RESULTS_DIR, "symup_corpus.fen")


def _outpath(tag):
    return os.path.join(RESULTS_DIR, "symup_acc_%s.jsonl" % tag)


def extract(tag, every, cap, max_nonkp):
    """Sample endgame-ish FENs (light material, containing a rook or knight) from games/<tag>/."""
    fens, seen = [], set()
    paths = sorted(glob.glob(os.path.join(SELFPLAY_DIR, "games", tag, "game_*", "game.jsonl")))
    for p in paths:
        i = 0
        with open(p) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("type") != "move" or r.get("opening"):
                    continue
                fen = r.get("fen")
                if not fen:
                    continue
                i += 1
                if i % every:
                    continue
                board = fen.split()[0]
                nonkp = [c for c in board if c.isalpha() and c.lower() not in ("k", "p")]
                if len(nonkp) > max_nonkp:
                    continue
                if not any(c in board for c in "RrNn"):
                    continue
                key = " ".join(fen.split()[:2])
                if key in seen:
                    continue
                seen.add(key)
                fens.append(fen)
                if len(fens) >= cap:
                    break
        if len(fens) >= cap:
            break
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(CORPUS, "w") as o:
        o.write("\n".join(fens) + "\n")
    print("[extract] %d FENs -> %s" % (len(fens), CORPUS))


def do_eval(tag, with_sf):
    from ChessAI import ChessAI
    fens = [l.strip() for l in open(CORPUS) if l.strip()]
    arbiter = None
    if with_sf:
        from arbiter import Arbiter, find_stockfish
        sf = find_stockfish()
        arbiter = Arbiter(sf, movetime=0.3) if sf else None
        if arbiter is None:
            print("[eval] WARNING no Stockfish found; SF columns will be missing")
    seed = chess.Board()
    ai = ChessAI(None, None, seed, seed.turn)
    out = _outpath(tag)
    n = 0
    with open(out, "w") as o:
        for fen in fens:
            b = chess.Board(fen)
            bd = ai.ev_breakdown(b)
            if bd.get("checkmate"):
                continue
            rec = {"fen": fen, "our": -bd["total"] / 1000.0, "pieces": -bd["pieces"] / 1000.0,
                   "cg": -bd["capture_gains"] / 1000.0, "endgame": bd["is_endgame"], "phase": bd["phase_score"]}
            if arbiter is not None:
                cs = arbiter.evaluate_static(b)
                rec["sf_static"] = None if cs is None else cs / 100.0
                cp, _, _ = arbiter.evaluate(b)
                rec["sf_search"] = None if cp is None else cp / 100.0
            o.write(json.dumps(rec) + "\n")
            n += 1
    if arbiter is not None:
        arbiter.close()
    print("[eval] %d positions -> %s" % (n, out))


def compare(ref):
    base = {}
    for l in open(_outpath("base")):
        r = json.loads(l)
        base[r["fen"]] = r
    sym = {}
    for l in open(_outpath("symup")):
        r = json.loads(l)
        sym[r["fen"]] = r

    allp, fired = [], []
    for fen, b in base.items():
        s = sym.get(fen)
        if not s:
            continue
        rv = b.get(ref)
        if rv is None:
            continue
        d = s["our"] - b["our"]          # correction applied by symup
        eb = rv - b["our"]               # error to fix (ref - base)
        es = rv - s["our"]               # residual error after symup
        row = {"fen": fen, "base": b["our"], "symup": s["our"], "ref": rv,
               "corr": d, "err_base": eb, "err_symup": es, "phase": b["phase"]}
        allp.append(row)
        if abs(d) > 1e-9:
            fired.append(row)

    def stats(rows, label):
        if not rows:
            print("  %s n=0" % label)
            return
        mab = st.mean(abs(x["err_base"]) for x in rows)
        mas = st.mean(abs(x["err_symup"]) for x in rows)
        helps = sum(1 for x in rows if x["corr"] * x["err_base"] > 0)
        meancorr = st.mean(x["corr"] for x in rows)
        print("  %s n=%-4d  mean|err| base=%.3f symup=%.3f  delta=%+.3f (neg=better)  "
              "correction-toward-SF=%d/%d  mean_corr=%+.3f"
              % (label, len(rows), mab, mas, mas - mab, helps, len(rows), meancorr))

    print("REF = %s   (White-POV pawns; lower |err| = closer to Stockfish)" % ref)
    stats(allp, "ALL  ")
    stats(fired, "FIRED")
    # worst fired offenders by base error, to eyeball
    fired.sort(key=lambda x: abs(x["err_base"]), reverse=True)
    print("  top fired by |err_base|:")
    for x in fired[:6]:
        print("    %-50s base=%+.2f symup=%+.2f ref=%+.2f corr=%+.2f" %
              (x["fen"][:50], x["base"], x["symup"], x["ref"], x["corr"]))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--tag", required=True)
    e.add_argument("--every", type=int, default=7)
    e.add_argument("--cap", type=int, default=300)
    e.add_argument("--max-nonkp", type=int, default=6)
    v = sub.add_parser("eval")
    v.add_argument("--tag", required=True, help="base | symup")
    v.add_argument("--sf", action="store_true")
    c = sub.add_parser("compare")
    c.add_argument("--ref", default="sf_search", choices=["sf_search", "sf_static"])
    a = ap.parse_args()
    if a.cmd == "extract":
        extract(a.tag, a.every, a.cap, a.max_nonkp)
    elif a.cmd == "eval":
        do_eval(a.tag, a.sf)
    elif a.cmd == "compare":
        compare(a.ref)


if __name__ == "__main__":
    main()
