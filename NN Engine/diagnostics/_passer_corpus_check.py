# -*- coding: utf-8 -*-
"""Have we plugged v1's PASSER weaknesses? (owner, 10-04) — the two v1-era passer corpora, scored against their SF labels.

Corpora (both White-POV): `ks_sets/passer_corpus.csv` (288; `sf18` in PAWNS; tiers blowup_guard / control / under_fire —
`build_passer_corpus.py`) and `suites/passers.csv` (405; `sf_cp` SF search cp + `sf_static`; categories by blockade kind —
`gen_passer_corpus.py`). Metric: mean |win%(evaluator) − win%(SF label)|, per tier / category.
EVALUATOR = the CURRENT env config (knobs latch at init ⇒ one process per config):
  MODE=static  → our static eval (ev_breakdown total)        MODE=search → our search at MAX_DEPTH (run_one)
  MODE=sf11    → SF11's classical static eval (no engine)
Writes one line per (corpus, group) so several runs can be read side by side.

  pyrun diagnostics/_passer_corpus_check.py MODE=static LABEL=v2ship V2_PRESET=shipped
"""
import os, sys, csv, math
import numpy as np

for _a in sys.argv[1:]:
    if "=" in _a:
        _k, _v = _a.split("=", 1)
        os.environ[_k] = _v
THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS); sys.path.insert(0, os.path.dirname(THIS))
os.environ.setdefault("PRESET", "LONG_FORMAT"); os.environ.setdefault("USE_OPENING_BOOK", "0")
import chess

MODE, LABEL = os.environ.get("MODE", "static"), os.environ.get("LABEL", "run")
K = 0.00368208
wp = lambda cp: 100.0 / (1.0 + np.exp(-K * np.clip(cp, -1500, 1500)))


def corpora():
    for r in csv.DictReader(open(os.path.join(THIS, "ks_sets/passer_corpus.csv"), newline="")):
        if r.get("sf18") not in ("", None):
            yield "passer_corpus", r["tier"], r["fen"], 100.0 * float(r["sf18"])
    for r in csv.DictReader(open(os.path.join(THIS, "suites/passers.csv"), newline="")):
        if r.get("sf_cp") not in ("", None):
            yield "passers_suite", r["cat"], r["fen_start"], float(r["sf_cp"])


def main():
    if MODE == "sf11":
        import subprocess, _triangulate_sf11 as T
        p = subprocess.Popen([T.SF11], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                             text=True, bufsize=1)
        p.stdin.write("uci\n"); p.stdin.flush()
        while p.stdout.readline().strip() != "uciok":
            pass
        ev = lambda fen: (lambda t: None if t is None else 100.0 * t)(T.sf11_eval(p, fen)[0])
    elif MODE == "search":
        from tactical_test import run_one
        def ev(fen):
            o = run_one(fen, [])
            if o["eval"] is None:
                return None
            cp = o["eval"] / 10.0
            cp = max(-5000.0, min(5000.0, cp)) if abs(o["eval"]) < 9_000_000 else (5000.0 if o["eval"] > 0 else -5000.0)
            return cp if chess.Board(fen).turn == chess.WHITE else -cp
    else:
        import ChessAI
        ai = ChessAI.ChessAI(None, None, chess.Board(), True)
        ev = lambda fen: -float(ai.ev_breakdown(chess.Board(fen))["total"]) / 10.0
    groups = {}
    for corpus, grp, fen, sf in corpora():
        v = ev(fen)
        if v is None:
            continue
        g = abs(wp(v) - wp(sf))
        for key in ((corpus, "ALL"), (corpus, grp)):
            groups.setdefault(key, []).append(g)
    for (corpus, grp), gs in sorted(groups.items()):
        print("PCC %-10s %-14s %-16s n %4d  mean|gap| %6.2f pp" % (LABEL, corpus, grp, len(gs), float(np.mean(gs))))


if __name__ == "__main__":
    main()
