# -*- coding: utf-8 -*-
"""
Mirror self-test for Optimism-Triggered Verification (OTV).

OTV re-searches a child whose backed-up score overshoots the node's OWN static
eval by a margin (the phantom signature). The trigger inequality has an OPPOSITE
sign in the maximizing node (score - static) vs the minimizing node (static -
score). If those signs are inconsistent, OTV fires on one color but not its
mirror image, breaking the exact color-symmetry the rest of the engine has.

This test searches each position AND its chess.Board(fen).mirror() (the exact
color-flip) at a fixed depth with ENABLE_OTV=1 and asserts the node counts are
IDENTICAL. Any asymmetry is an OTV sign bug -- the definitive arbiter that the
min/max overshoot signs are internally consistent.

IMPORTANT (per the design): the mirror test passes for BOTH the correct sign and
a fully-backwards sign, because each is internally symmetric under color-flip.
The mirror test only catches ASYMMETRY between the two functions. The DIRECTION
(does OTV deflate the phantom toward the truth, or inflate it further?) is checked
separately by --direction below on the known phantom FENs: OTV should move the
search eval TOWARD zero, not further from it.

Env knobs are read by the C++ engine via getenv at first search, so this script
sets them in os.environ BEFORE importing ChessAI. Toggles load once per process,
so OTV-on and OTV-off are separate process invocations (choose via MIRROR_OTV).

Run (WSL, from NN Engine/):
    MIRROR_OTV=1 python diagnostics/search_mirror_test.py             # mirror symmetry (OTV on)
    MIRROR_OTV=1 python diagnostics/search_mirror_test.py --direction # phantom eval with OTV on
    MIRROR_OTV=0 python diagnostics/search_mirror_test.py --direction # phantom eval with OTV off
via the dispatcher:
    wsl.exe -e bash -lc "MIRROR_OTV=1 bash '<runner>' pyrun diagnostics/search_mirror_test.py"
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # silence TensorFlow startup chatter

# Fixed-depth control + enable OTV. Set before ChessAI is imported so the engine's
# lazy getenv toggle-load (first search) sees them.
os.environ.setdefault('PRESET', 'LONG_FORMAT')
os.environ.setdefault('MAX_DEPTH', '8')            # searches to fixed depth 8 (exclusive cap is +... literal)
os.environ['ENABLE_OTV'] = os.environ.get('MIRROR_OTV', '1')

import sys
import subprocess
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, ENGINE_DIR)

import chess


def _run_one_in_process(fen):
    """Run a single FEN in THIS process (used by the --fen worker). Imports ChessAI
    lazily so the driver process never loads the engine, keeping each worker's global
    search tables (history/killer/counter/TT) pristine -- the only way the node count
    is a clean function of the position alone and the mirror comparison is meaningful."""
    from tactical_test import run_one
    return run_one(fen, set())


# Two hand-picked phantom FENs (buried-refutation over-optimism signatures) plus a
# spread of WAC tactics so the symmetry check covers ordinary midgame trees too.
PHANTOM_FENS = [
    "6k1/3q1pbp/2b1p3/2P1N3/1P3Pr1/4Q1P1/4R3/4N1K1 b - - 2 35",
    "r3k2r/pp2p1bp/2n3p1/2pQ4/q4P2/PP1PP1P1/3B3P/1R1K1B1R b kq - 0 19",
]

WAC_FENS = [
    "2rr3k/pp3pp1/1nnqbN1p/3pN3/2pP4/2P3Q1/PPB4P/R4RK1 w - - 0 1",
    "8/7p/5k2/5p2/p1p2P2/Pr1pPK2/1P1R3P/8 b - - 0 1",
    "5rk1/1ppb3p/p1pb4/6q1/3P1p1r/2P1R2P/PP1BQ1P1/5RKN w - - 0 1",
    "r1bq2rk/pp3pbp/2p1p1pQ/7P/3P4/2PB1N2/PP3PPR/2KR4 w - - 0 1",
    "5k2/6pp/p1qN4/1p1p4/3P4/2PKP2Q/PP3r2/3R4 b - - 0 1",
    "7k/p7/1R5K/6r1/6p1/6P1/8/8 w - - 0 1",
    "r4q1k/p2bR1rp/2p2Q1N/5p2/5p2/2P5/PP3PPP/R5K1 w - - 0 1",
    "1k5r/pp3p2/2p2q2/3p4/3P3p/2PB1Q1P/PP4P1/5R1K w - - 0 1",
    "r3r1k1/pp1q1ppp/2p5/2P5/3P4/1Q2P3/PP3PPP/R3R1K1 w - - 0 1",
    "r1b1kb1r/1p1n1ppp/p3pn2/8/3NP3/2N5/PPP2PPP/R1BQK2R w KQkq - 0 1",
    "3r2k1/p1q2pp1/1n2rn1p/8/2p1P3/2P2QP1/P1BN1P1P/1R2R1K1 b - - 0 1",
    "2b1r1k1/r4ppp/1qp5/p2p4/P2P4/1P1Q1N2/2P2PPP/R3R1K1 w - - 0 1",
]


def mirror_fen(fen):
    return chess.Board(fen).mirror().fen()


def _worker(fen):
    """Search one FEN in a FRESH subprocess (clean global tables) and return
    (nodes, eval). Env (incl. ENABLE_OTV / PRESET / MAX_DEPTH) is inherited."""
    out = subprocess.run([sys.executable, os.path.abspath(__file__), "--fen", fen],
                         capture_output=True, text=True, env=os.environ).stdout
    nodes = ev = None
    for line in out.splitlines():
        if line.startswith("WORKER "):
            for tok in line.split()[1:]:
                k, _, v = tok.partition("=")
                if k == "nodes":
                    nodes = None if v == "None" else int(v)
                elif k == "eval":
                    ev = None if v == "None" else int(v)
    return nodes, ev


def run_mirror():
    fens = [("phantom", f) for f in PHANTOM_FENS] + [("wac", f) for f in WAC_FENS]
    print(f"OTV mirror self-test  (ENABLE_OTV={os.environ['ENABLE_OTV']}, "
          f"PRESET={os.environ['PRESET']}, MAX_DEPTH={os.environ['MAX_DEPTH']}) "
          f"-- each search in a fresh subprocess\n")
    print(f"{'#':>3}  {'kind':<8} {'nodes':>12} {'mir_nodes':>12}  {'res':<5}  fen")
    npass = nfail = nskip = 0
    for idx, (kind, fen) in enumerate(fens):
        n, _ = _worker(fen)
        nm, _ = _worker(mirror_fen(fen))
        if n is None or nm is None:
            # a book move (no search) makes the node count meaningless -> not a symmetry signal
            res = "SKIP"
            nskip += 1
        elif n == nm:
            res = "PASS"
            npass += 1
        else:
            res = "FAIL"
            nfail += 1
        print(f"{idx:>3}  {kind:<8} {str(n):>12} {str(nm):>12}  {res:<5}  {fen}")
    print()
    print(f"MIRROR SUMMARY: {npass} PASS, {nfail} FAIL"
          + (f", {nskip} SKIP (book)" if nskip else ""))
    return nfail == 0


def run_direction():
    """Report the phantom FENs' search eval so the OTV DIRECTION can be judged
    across two process runs (MIRROR_OTV=0 vs =1). With OTV on, the over-optimistic
    phantom eval should move TOWARD zero (the calibrated truth), not further out."""
    print(f"OTV direction probe  (ENABLE_OTV={os.environ['ENABLE_OTV']}, "
          f"PRESET={os.environ['PRESET']}, MAX_DEPTH={os.environ['MAX_DEPTH']})\n")
    print(f"{'#':>3}  {'eval':>9} {'nodes':>12}  fen")
    for idx, fen in enumerate(PHANTOM_FENS):
        n, ev = _worker(fen)
        ev_s = "" if ev is None else str(ev)
        print(f"{idx:>3}  {ev_s:>9} {str(n):>12}  {fen}")


if __name__ == "__main__":
    if "--fen" in sys.argv:
        # Worker: search a single FEN in this fresh process, emit a parseable line.
        fen = sys.argv[sys.argv.index("--fen") + 1]
        r = _run_one_in_process(fen)
        print(f"WORKER nodes={r['nodes']} eval={r['eval']} uci={r['uci']}")
    elif "--direction" in sys.argv:
        run_direction()
    else:
        ok = run_mirror()
        sys.exit(0 if ok else 1)
