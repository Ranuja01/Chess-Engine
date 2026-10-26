"""Colour-mirror test of the UCI binary: each position and its colour-flipped twin, fresh process each, `go depth D` with the
given knobs; compares the final score (side to move's view) and node count. Run once with the feature off to see the
search's own symmetry, then on: a sign error in one search half shows as pairs that were symmetric becoming asymmetric.
Windows python (python-chess); engine via wsl.exe.
Usage: python _mirror_test.py KNOBS [DEPTH] [N]   Env: BIN, PGN (positions source)"""
import os, sys, re, math, statistics, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("SEED", "1071")
knobs = sys.argv[1]
depth = int(sys.argv[2]) if len(sys.argv) > 2 else 10
n = int(sys.argv[3]) if len(sys.argv) > 3 else 20
os.environ["N"] = str(n)   # read by _partial_iter_judge at import
import chess
import _partial_iter_judge as J

BIN = os.environ.get("BIN", J.ND + "/build_uci/chess_engine_uci_tm4b")

def run(fen):
    p = subprocess.Popen(["wsl.exe", "-e", BIN], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    cmds = ["setoption name OwnBook value false"] + (["setoption name Knobs value " + knobs] if knobs != "none" else []) + \
           ["isready", "position fen " + fen, f"go depth {depth}"]
    p.stdin.write("\n".join(cmds) + "\n"); p.stdin.flush()
    score, nodes = None, None
    for l in p.stdout:
        m = re.search(r"info depth (\d+) score (cp|mate) (-?\d+) nodes (\d+)", l)
        if m: score, nodes = (m.group(2), int(m.group(3))), int(m.group(4))
        if l.startswith("bestmove"): break
    p.stdin.write("quit\n"); p.stdin.flush(); p.wait()
    return score, nodes

fens = J.sample_positions([os.environ.get("PGN", r"E:\chess_data\fastchess\tm1_60p6.pgn")])
same_score = same_nodes = 0
diffs = []   # cp score (side to move) minus its mirror's, non-mate pairs
for fen in fens:
    a = run(fen); b = run(chess.Board(fen).mirror().fen())
    same_score += a[0] == b[0]; same_nodes += a[1] == b[1]
    if a[0] and b[0] and a[0][0] == "cp" and b[0][0] == "cp":
        diffs.append(a[0][1] - b[0][1])
rms = math.sqrt(sum(d * d for d in diffs) / max(1, len(diffs)))
print(f"knobs {knobs}  depth {depth}: {len(fens)} pairs, same score {same_score}, same nodes {same_nodes}; "
      f"cp diff mean {statistics.mean(diffs):+.1f} (se {statistics.stdev(diffs) / math.sqrt(len(diffs)):.1f}) "
      f"rms {rms:.1f} median |d| {statistics.median([abs(d) for d in diffs]):.0f}  max |d| {max(abs(d) for d in diffs)}")
