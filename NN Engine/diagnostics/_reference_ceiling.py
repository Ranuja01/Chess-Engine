# -*- coding: utf-8 -*-
"""What win%-error does a STATIC eval inherently carry against SF18 SEARCH? — the achievable floor.

Our fit objective drives `(winpct(ours) - winpct(sf18_search))^2` toward zero, but zero is not reachable:
no static evaluator can match a deep search, because search sees tactics that no leaf score encodes. So a
`val` of 211.9 is meaningless in isolation -- the question is how it compares to what a good hand-written
eval achieves on the SAME positions against the SAME target.

This runs every reference evaluator we have as a STATIC eval over the fit corpus and scores it with the
IDENTICAL loss `_ks_fit_eval.py` uses, so the numbers sit on one scale beside ours:

    SF11            pure classical, the hand-reachable reference
    SF15.1 classical  the LAST classical-king-safety Stockfish
    SF15.1 NNUE       same binary, NNUE on -- isolates un-encodable-by-hand from simply-missing
    SF18 static       modern NNUE leaf score
    (SF1.1 optional; it is a Windows .exe here so it is skipped unless it runs)

⚠️ This is a REFERENCE, not a hard floor. SF15.1-NNUE will beat any classical eval and we cannot reach it
by hand; SF11/SF15.1-classical are the honest targets. And a static-vs-search gap is partly irreducible
by construction -- the triage measured 29-57% of large disagreements as search properties.

  pyrun diagnostics/_reference_ceiling.py [N=4000] [CORPUS=diagnostics/ks_sets/diverse_corpus_wide.csv]
"""
import os, sys, csv, math, subprocess, re

for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ.setdefault(_k, _v)

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
THIS = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(THIS)
sys.path.insert(0, ENGINE); sys.path.insert(0, THIS)

N = int(os.environ.get("N", "4000"))
CORPUS = os.environ.get("CORPUS", os.path.join(THIS, "ks_sets", "diverse_corpus_wide.csv"))
if not os.path.isabs(CORPUS):
    CORPUS = os.path.join(ENGINE, CORPUS)

# EXACT loss from _ks_fit_eval.py -- must match or the comparison is meaningless.
def winpct(cp):
    return 50.0 + 50.0 * (2.0 / (1.0 + math.exp(-0.00368208 * cp)) - 1.0)


class RawStatic:
    """UCI `eval` -> Final evaluation, White-POV pawns."""
    def __init__(self, path, nnue=None):
        self.p = subprocess.Popen([path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self._send("uci"); self._drain("uciok")
        if nnue is not None:
            self._send("setoption name Use NNUE value %s" % ("true" if nnue else "false"))
            self._send("isready"); self._drain("readyok")

    def _send(self, s):
        self.p.stdin.write(s + "\n"); self.p.stdin.flush()

    def _drain(self, tok):
        while True:
            ln = self.p.stdout.readline()
            if not ln or ln.strip().startswith(tok):
                break

    def ev(self, fen):
        self._send("position fen %s" % fen); self._send("eval"); self._send("isready")
        val = None
        while True:
            ln = self.p.stdout.readline()
            if not ln or ln.strip() == "readyok":
                break
            m = re.search(r"Final evaluation\s*:?\s*([-+]?\d+\.\d+)", ln)
            if m:
                val = float(m.group(1))
        return val

    def close(self):
        try:
            self._send("quit"); self.p.wait(timeout=3)
        except Exception:
            try: self.p.kill()
            except Exception: pass


def main():
    rows = [r for r in csv.DictReader(open(CORPUS, newline=""))][:N]
    SF18 = os.environ["STOCKFISH_PATH"]
    SF15 = os.environ.get("SF15_BIN", "/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/"
                                      "stockfish_15_linux/stockfish_15.1_linux_x64/stockfish-ubuntu-20.04-x86-64")
    from eval_vs_sf11 import SF11Eval, SF11

    engines = []
    try:
        engines.append(("SF11 classical", SF11Eval(SF11).eval, "tuple"))
    except Exception as e:
        print("[skip] SF11: %s" % e)
    for label, path, nnue in [("SF15.1 classical", SF15, False), ("SF15.1 NNUE", SF15, True),
                              ("SF18 static", SF18, None)]:
        try:
            engines.append((label, RawStatic(path, nnue).ev, "scalar"))
        except Exception as e:
            print("[skip] %s: %s" % (label, e))

    # Ours, from the live engine.
    import chess
    from ChessAI import ChessAI
    ai = ChessAI(None, None, chess.Board(), True)
    engines.append(("OURS (current build)", lambda f: -ai.ev(chess.Board(f)) / 1000.0, "scalar"))

    print("REFERENCE CEILING — static eval vs SF18-SEARCH target, %d corpus rows" % len(rows))
    print("Same loss as _ks_fit_eval (win%% squared error, Lichess k=0.00368208). Lower is better.\n")
    print("  %-22s %10s %10s %8s" % ("evaluator", "train", "val", "n"))

    for label, fn, kind in engines:
        sse = {"train": 0.0, "val": 0.0}
        cnt = {"train": 0, "val": 0}
        for r in rows:
            try:
                tgt = float(r["target_total"])
                v = fn(r["fen"])
                if kind == "tuple":
                    v = v[0]
                if v is None:
                    continue
                sp = r.get("split", "train")
                if sp not in sse:
                    continue
                sse[sp] += (winpct(v * 100.0) - winpct(tgt * 100.0)) ** 2
                cnt[sp] += 1
            except Exception:
                continue
        tr = sse["train"] / cnt["train"] if cnt["train"] else float("nan")
        va = sse["val"] / cnt["val"] if cnt["val"] else float("nan")
        print("  %-22s %10.2f %10.2f %8d" % (label, tr, va, cnt["train"] + cnt["val"]))

    print("\nREADING IT")
    print("  SF11 / SF15.1-classical are the HAND-REACHABLE references -- a classical eval genuinely")
    print("  achieves that error against a deep search, so the gap between us and them is the part we")
    print("  can realistically close. SF15.1-NNUE and SF18-static are below what hand-tuning reaches.")
    print("  ⚠️ Not an absolute floor: a static-vs-search gap is partly irreducible by construction.")


if __name__ == "__main__":
    main()
