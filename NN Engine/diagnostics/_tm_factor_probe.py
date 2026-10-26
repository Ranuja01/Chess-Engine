"""Distributions of the time-manager scaling inputs (TM_SCALE factors) on real middlegame positions.

Runs `go movetime MT` on positions sampled from fastchess PGNs (as _partial_iter_judge.py) with FIX_COUNT_LOG=1 + Debug, reads
the per-iteration TMITER lines (depth >= 5): best-move node share (root main search only, and over all nodes incl. the
pre-search), stability, 3-iteration score drop, and the three factor values. Reports them over the iterations that end inside
the decision window [TLO, THI] seconds, where a 60+0.6 clock's soft stop is decided, and the median factor products that set
TM_NORM for an arm with the same average spend.
Windows python (python-chess); engines through wsl.exe.
Usage: python _tm_factor_probe.py PGN [PGN ...]   Env: BIN, N (600), MT (2500), WORKERS (3), TLO (0.5), THI (2.5), SEED (1041)"""
import os, sys, re, threading, subprocess, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("N", "600"); os.environ.setdefault("SEED", "1041")
import _partial_iter_judge as J

BIN = os.environ.get("BIN", J.ND + "/build_uci/chess_engine_uci_tm4")
MT, WORKERS = int(os.environ.get("MT", 2500)), int(os.environ.get("WORKERS", 3))
TLO, THI = float(os.environ.get("TLO", 0.5)), float(os.environ.get("THI", 2.5))
RX = re.compile(r"TMITER d=(\d+) t=([\d.e+-]+) share=([\d.e+-]+) share_all=([\d.e+-]+) stab=(\d+) drop3=(-?\d+) "
                r"fn=([\d.e+-]+) fs=([\d.e+-]+) fd=([\d.e+-]+)")

def worker(fens, rows, lock):
    p = subprocess.Popen(["wsl.exe", "-e", "env", "FIX_COUNT_LOG=1", BIN], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    def send(s): p.stdin.write(s + "\n"); p.stdin.flush()
    send("setoption name OwnBook value false"); send("setoption name Debug value true"); send("isready")
    for l in p.stdout:
        if "readyok" in l: break
    for fen in fens:
        send("ucinewgame"); send("position fen " + fen); send(f"go movetime {MT}")
        got = []
        for l in p.stdout:
            m = RX.search(l)   # stdout and stderr share the pipe: search anywhere in the line
            if m: got.append(tuple(float(x) for x in m.groups()))
            if re.search(r"bestmove \S+", l): break
        with lock: rows.extend(got)
    send("quit"); p.wait()

def q(v, p): return statistics.quantiles(v, n=100)[p - 1] if len(v) >= 2 else float("nan")

def main():
    fens = J.sample_positions(sys.argv[1:])
    rows, lock = [], threading.Lock()
    ts = [threading.Thread(target=worker, args=(fens[i::WORKERS], rows, lock)) for i in range(WORKERS)]
    for t in ts: t.start()
    for t in ts: t.join()
    win = [r for r in rows if TLO <= r[1] <= THI]
    print(f"positions {len(fens)}  iteration lines {len(rows)}  in window [{TLO}, {THI}] s: {len(win)}")
    names = ["depth", "t", "share", "share_all", "stab", "drop3", "f_nodes", "f_stab", "f_drop"]
    for i, n in enumerate(names):
        v = [r[i] for r in win]
        print(f"  {n:9s} p10 {q(v, 10):8.3f}  p50 {statistics.median(v):8.3f}  p90 {q(v, 90):8.3f}  mean {statistics.mean(v):8.3f}")
    fn = [r[6] for r in win]; sd = [r[7] * r[8] for r in win]; all3 = [r[6] * r[7] * r[8] for r in win]
    for label, v in (("arm A nodes", fn), ("arm B stab*drop", sd), ("all three", all3)):
        med = statistics.median(v)
        print(f"  {label:16s} median product {med:.3f} -> TM_NORM {round(100 / med)}  (p10 {q(v, 10):.3f} p90 {q(v, 90):.3f})")

if __name__ == "__main__":
    main()
