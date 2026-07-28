"""Time-to-depth decomposition: OUR engine vs Stockfish on the SAME FENs at the SAME nominal depths.

We reach a given depth far slower than Stockfish, but "slower" has two independent causes and NPS alone
cannot separate them:

    time_to_depth(d)  =  nodes_to_depth(d)  x  time_per_node

If the time ratio at each depth tracks the NPS ratio, the gap is raw per-node cost (for us, eval is
~65-84% of it). If Stockfish also reaches depth d in far FEWER nodes, its pruning schedule is doing work
ours is not, and tree shape is a second, separate gap.

⚠️ Our engine latches Config knobs ONCE at extension init, so MAX_DEPTH cannot be varied inside a single
process -- an earlier version of this script did exactly that and silently measured the same depth ten
times (identical node counts at every depth were the tell). Each depth therefore runs in its OWN
subprocess, the same discipline diagnostics/bias_sweep uses.

⚠️ Nominal depth is not a common unit across engines: a Stockfish ply is shaped by a different reduction
schedule. This measures how each engine scales in ITS OWN depth units, which is what "they reach depth N
before we do" actually refers to. Stockfish is also NNUE here, which costs it NPS relative to an HCE --
so any per-node speed gap we measure is, if anything, understated.

Usage:
  pyrun diagnostics/depth_race_vs_sf.py [--n 10] [--max-depth 10] [--sf <path>] [--corpus <csv>]
  (internal: --depth D  runs one depth in this process and prints a parseable line)

@author: Ranuja Pinnaduwage
"""

import csv
import os
import statistics
import subprocess
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
THIS = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(THIS)
sys.path.insert(0, THIS)
sys.path.insert(0, ENGINE)


def parse_args(argv):
    o = {"n": 10, "max_depth": 10, "depth": None,
         "sf": os.environ.get("STOCKFISH_PATH", ""),
         "corpus": os.path.join(ENGINE, "selfplay", "tune_data", "cploss_corpus.csv")}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--n":           o["n"] = int(argv[i + 1]); i += 2
        elif a == "--max-depth": o["max_depth"] = int(argv[i + 1]); i += 2
        elif a == "--depth":     o["depth"] = int(argv[i + 1]); i += 2
        elif a == "--sf":        o["sf"] = argv[i + 1]; i += 2
        elif a == "--corpus":    o["corpus"] = argv[i + 1]; i += 2
        else: i += 1
    return o


def load_fens(corpus, n):
    with open(corpus) as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("stratum") in ("game", "neutral", "collapse")]
    return [r["fen"] for r in rows][:n]


def run_our_depth(fens):
    """Child mode: MAX_DEPTH is already latched from the environment for this process."""
    from tactical_test import run_one
    nodes, times = [], []
    for fen in fens:
        try:
            r = run_one(fen, set())
        except Exception:
            continue
        if r.get("booked"):
            continue
        if r.get("nodes") and r.get("time") and r["time"] > 0:
            nodes.append(r["nodes"]); times.append(r["time"])
    if nodes:
        print(f"RESULT nodes={statistics.median(nodes):.0f} time={statistics.median(times):.6f} n={len(nodes)}")
    return 0


def main() -> int:
    o = parse_args(sys.argv[1:])
    fens = load_fens(o["corpus"], o["n"])
    if not fens:
        print("no FENs selected", file=sys.stderr); return 1

    if o["depth"] is not None:
        return run_our_depth(fens)

    if not o["sf"] or not os.path.exists(o["sf"]):
        print(f"stockfish not found at {o['sf']!r}", file=sys.stderr); return 1

    print(f"[depth_race] n={len(fens)} max_depth={o['max_depth']} sf={os.path.basename(o['sf'])}")

    ours = {}
    for d in range(1, o["max_depth"] + 1):
        env = dict(os.environ)
        env.update({"MAX_DEPTH": str(d), "PRESET": "LONG_FORMAT", "USE_OPENING_BOOK": "0"})
        cmd = [sys.executable, os.path.abspath(__file__), "--depth", str(d),
               "--n", str(o["n"]), "--corpus", o["corpus"]]
        try:
            out = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900).stdout
        except subprocess.TimeoutExpired:
            continue
        line = next((l for l in out.splitlines() if l.startswith("RESULT")), None)
        if not line:
            continue
        kv = dict(p.split("=", 1) for p in line.split()[1:])
        ours[d] = (float(kv["nodes"]), float(kv["time"]))
        print(f"  ours d={d}: nodes={ours[d][0]:,.0f} time={1000*ours[d][1]:.1f}ms", flush=True)

    import chess
    import chess.engine
    theirs = {}
    with chess.engine.SimpleEngine.popen_uci(o["sf"]) as sf:
        try:
            sf.configure({"Threads": 1, "Hash": 256})
        except Exception:
            pass
        for d in range(1, o["max_depth"] + 1):
            nodes, times = [], []
            for fen in fens:
                try:
                    info = sf.analyse(chess.Board(fen), chess.engine.Limit(depth=d))
                except Exception:
                    continue
                if info.get("nodes") and info.get("time") and info["time"] > 0:
                    nodes.append(info["nodes"]); times.append(info["time"])
            if nodes:
                theirs[d] = (statistics.median(nodes), statistics.median(times))

    print(f"\n{'d':>3} {'our nodes':>12} {'sf nodes':>11} {'node x':>8} "
          f"{'our ms':>9} {'sf ms':>8} {'time x':>8} {'our nps':>10} {'sf nps':>10} {'nps x':>7}")
    for d in range(1, o["max_depth"] + 1):
        if d not in ours or d not in theirs:
            continue
        on, ot = ours[d]; sn, st = theirs[d]
        onps, snps = (on / ot if ot else 0), (sn / st if st else 0)
        print(f"{d:>3} {on:>12,.0f} {sn:>11,.0f} {on/sn if sn else 0:>8.1f} "
              f"{1000*ot:>9.1f} {1000*st:>8.1f} {ot/st if st else 0:>8.1f} "
              f"{onps:>10,.0f} {snps:>10,.0f} {snps/onps if onps else 0:>7.1f}")

    print("\nRead: 'time x' ~= 'nps x' with 'node x' ~= 1 => the gap is per-node cost alone.")
    print("      'node x' >> 1 => their pruning reaches the same nominal depth in far fewer nodes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
