# -*- coding: utf-8 -*-
"""Localize the SEARCH over-optimism: on positions where our depth-8 search returned a phantom advantage (static
eval + SF agree ~0), toggle each pruning site OFF and see which one collapses our search eval back toward the truth.
The site whose removal drops the phantom = the unsound pruner burying the opponent's refutation.

Self-contained: the parent spawns one worker subprocess per ablation config (env latched per process), each running
our engine search on the fixed peak FENs. Run:  python diagnostics/search_localize.py
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys, subprocess, json

THIS = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(THIS)
sys.path.insert(0, THIS); sys.path.insert(0, os.path.join(ENGINE, "selfplay"))

# Peak positions from bias_profile --peak (our search read +2..+5; SF static & search ~0/negative).
FENS = [
    "6k1/3q1pbp/2b1p3/2P1N3/1P3Pr1/4Q1P1/4R3/4N1K1 b - - 2 35",
    "1rbqk2r/p1pnbppp/2Q5/4N3/4P3/2N5/PPP2PPP/R1B1K2R b KQk - 0 10",
    "r4rk1/1p1b1p2/1qn1p2Q/p2p3n/3P3P/2P2NP1/P3PPB1/2R2K1R b - - 0 16",
    "r1b2b1r/1pNk1ppp/2n1pn2/p2p2N1/5B2/1PP3P1/4PP1P/R3KB1R b KQ - 1 12",
    "3q2k1/p4ppp/4rn2/1p2p3/N1Pp4/bP1P2P1/P3PPK1/R2Q1R2 w - - 0 18",
    "8/1p2bkpp/p1r5/3pPp2/3P1R2/1PNR3P/1P2r3/4B1K1 b - - 0 32",
    "4k1nr/1p3ppp/pq2p3/2rpP3/8/1P1BBb2/P2Q1PPP/4R1K1 w k - 0 19",
    "r1b1k2r/pp2pp2/n1p2b2/N2p3q/3PnB2/P1PQ1NP1/1P2PPB1/R2R2K1 b kq - 2 16",
    "r3k2r/pp2p1bp/2n3p1/2pQ4/q4P2/PP1PP1P1/3B3P/1R1K1B1R b kq - 0 19",
    "3r1k1r/ppp1q2p/5nR1/2P1n2B/3Np2N/4P2b/PP6/R2QK3 w - - 1 22",
]

CONFIGS = [
    ("baseline",   {}),
    ("noLMR",      {"ENABLE_LMR": "0"}),
    ("noNULL",     {"ENABLE_NULLMOVE": "0"}),
    ("noFUTIL",    {"ENABLE_FUTILITY": "0"}),
    ("noRAZOR",    {"ENABLE_RAZORING": "0"}),
    ("verifyFull", {"VERIFY_MARGIN": "30000", "VERIFY_RESEARCH_REDUCTION": "0"}),  # re-search every near-miss at full depth
]

BASE_ENV = {"PRESET": "LONG_FORMAT", "MAX_DEPTH": "8", "USE_OPENING_BOOK": "0",
            "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}


def worker():
    import chess
    from tactical_test import run_one
    for fen in FENS:
        try:
            ev = run_one(fen, set()).get("eval")
        except Exception:
            ev = None
        # our eval is absolute (Black-positive); White-POV pawns = -ev/1000
        wp = None if ev is None else -ev / 1000.0
        # normalize to side-to-move POV (positive = good for the side that just searched / to move)
        stm_white = fen.split()[1] == "w"
        pov = None if wp is None else (wp if stm_white else -wp)
        print(json.dumps({"fen": fen, "white_pawns": wp, "stm_pov": pov}))


def main():
    if "--worker" in sys.argv:
        worker(); return
    print("search-soundness localization: our depth-8 search eval (stm-POV pawns) per ablation.")
    print("truth at these peaks ~ 0 (static + SF agree). The config that DROPS the phantom names the unsound site.\n")
    results = {}
    for name, ov in CONFIGS:
        env = dict(os.environ); env.update(BASE_ENV); env.update(ov)
        out = subprocess.run([sys.executable, os.path.join("diagnostics", "search_localize.py"), "--worker"],
                             cwd=ENGINE, env=env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout.decode("utf-8", "ignore")
        vals = []
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    r = json.loads(line);
                    if r.get("stm_pov") is not None:
                        vals.append(r["stm_pov"])
                except Exception:
                    pass
        results[name] = vals
        mean = sum(vals) / len(vals) if vals else float("nan")
        print("  %-10s mean stm-POV eval = %+6.2f   (n=%d)   %s"
              % (name, mean, len(vals), " ".join("%+.1f" % v for v in vals)))
    base = results.get("baseline", [])
    if base:
        bm = sum(base) / len(base)
        print("\n  drop vs baseline (%+.2f) — biggest negative drop = the unsound pruner:" % bm)
        for name, _ in CONFIGS[1:]:
            v = results.get(name, [])
            if v:
                print("     %-10s %+6.2f  (Δ %+.2f)" % (name, sum(v)/len(v), sum(v)/len(v) - bm))


if __name__ == "__main__":
    main()
