"""Judge TM_PARTIAL (time manager step 3) where it fires.

1. Sample positions from fastchess PGNs (move 10 .. 60, |PGN score| <= 3 pawns).
2. Run each at `go movetime MT` on the UCI binary with Knobs TM_PARTIAL=1 (Debug on, FIX_COUNT_LOG=1), WORKERS engines in
   parallel; collect every PARTIAL_DIFF line (the cut iteration's completed best move replaced the completed iteration's).
3. SF18 scores both moves (root_moves=[m], fixed nodes); report the win% difference adopted - completed.
Windows python (python-chess); engines run through wsl.exe.

Usage: python _partial_iter_judge.py OUT_PREFIX PGN [PGN ...]
Env: BIN (WSL path of the UCI binary), N (positions, 3000), MT (ms, 1000), WORKERS (3), SF_NODES (2000000), SEED (1031)."""
import os, sys, re, math, random, threading, subprocess, statistics
import chess, chess.pgn, chess.engine

ND = "/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
BIN = os.environ.get("BIN", ND + "/build_uci/chess_engine_uci_g61")
SF = "/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/stockfish_18_linux/stockfish-ubuntu-x86-64-avx2"
N, MT, WORKERS = int(os.environ.get("N", 3000)), int(os.environ.get("MT", 1000)), int(os.environ.get("WORKERS", 3))
SF_NODES, SEED = int(os.environ.get("SF_NODES", 2000000)), int(os.environ.get("SEED", 1031))
K = 0.00368208   # the toolkit's win% slope per centipawn; scores clamped to +-1500 cp

def win_pct(cp):
    cp = max(-1500, min(1500, cp))
    return 50 + 50 * (2 / (1 + math.exp(-K * cp)) - 1)

def sample_positions(pgns):
    fens = []
    for path in pgns:
        with open(path) as f:
            while (g := chess.pgn.read_game(f)) is not None:
                board = g.board()
                for node in g.mainline():
                    m = re.match(r"([+-]?\d+\.\d+)/", node.comment or "")
                    if 10 <= board.fullmove_number <= 60 and m and abs(float(m.group(1))) <= 3.0 and not board.is_check():
                        fens.append(board.fen())
                    board.push(node.move)
    random.Random(SEED).shuffle(fens)
    return fens[:N]

def sq_name(s):
    return "abcdefgh"[s % 8] + str(s // 8 + 1)

def to_move(board, frm, to):
    cands = [m for m in board.legal_moves if m.from_square == frm and m.to_square == to]
    if not cands:
        return None
    return next((m for m in cands if m.promotion == chess.QUEEN), cands[0])

totals = [0, 0]   # summed over workers: the engine's TMCNT partial_used, partial_differs

def probe_worker(fens, out, lock):
    tm = [0, 0]
    p = subprocess.Popen(["wsl.exe", "-e", "env", "FIX_COUNT_LOG=1", BIN], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    def send(s): p.stdin.write(s + "\n"); p.stdin.flush()
    send("setoption name OwnBook value false"); send("setoption name Debug value true")
    send("setoption name Knobs value TM_PARTIAL=" + os.environ.get("MODE", "1")); send("isready")
    for l in p.stdout:
        if "readyok" in l: break
    for fen in fens:
        send("ucinewgame"); send("position fen " + fen); send(f"go movetime {MT}")
        # stdout (UCI) and stderr (Debug prints) share one pipe and can interleave mid-line, so markers are searched for
        # anywhere in a line, never only at its start.
        diffs, best = [], None
        for l in p.stdout:
            d = re.search(r"PARTIAL_DIFF depth=\d+ completed=\d+-\d+-\d+ adopted=\d+-\d+-\d+", l)
            if d: diffs.append(d.group(0))
            t = re.search(r"partial_used=(\d+) partial_differs=(\d+)", l)
            if t: tm[0], tm[1] = int(t.group(1)), int(t.group(2))
            b = re.search(r"bestmove (\S+)", l)
            if b: best = b.group(1); break
        with lock:
            out.append((fen, diffs, best))
    send("quit"); p.wait()
    with lock:   # the engine's own cumulative counters: a cross-check that no PARTIAL_DIFF line was lost
        totals[0] += tm[0]; totals[1] += tm[1]

def main():
    prefix, pgns = sys.argv[1], sys.argv[2:]
    fens = sample_positions(pgns)
    print(f"positions {len(fens)}  movetime {MT} ms  workers {WORKERS}  bin {BIN}", flush=True)
    out, lock = [], threading.Lock()
    ts = [threading.Thread(target=probe_worker, args=(fens[i::WORKERS], out, lock)) for i in range(WORKERS)]
    for t in ts: t.start()
    for t in ts: t.join()
    cases = []
    for fen, diffs, best in out:
        if not diffs: continue
        m = re.match(r"PARTIAL_DIFF depth=(\d+) completed=(\d+)-(\d+)-\d+ adopted=(\d+)-(\d+)-\d+", diffs[-1])
        b = chess.Board(fen)
        c, a = to_move(b, int(m.group(2)), int(m.group(3))), to_move(b, int(m.group(4)), int(m.group(5)))
        if c and a and best == a.uci():
            cases.append((fen, int(m.group(1)), c, a))
    print(f"searched {len(out)}  changed moves {len(cases)} ({100 * len(cases) / max(1, len(out)):.1f}%)  "
          f"logged lines {sum(len(d) for _, d, _ in out)}  engine counters: partial_used {totals[0]} partial_differs {totals[1]}",
          flush=True)
    sf = chess.engine.SimpleEngine.popen_uci(["wsl.exe", "-e", SF])
    sf.configure({"Threads": 1, "Hash": 64})
    rows = []
    with open(prefix + "_cases.tsv", "w") as f:
        f.write("fen\tdepth\tcompleted\tadopted\twin_completed\twin_adopted\n")
        for fen, d, c, a in cases:
            b = chess.Board(fen)
            w = []
            for mv in (c, a):
                sf.configure({"Clear Hash": None})
                info = sf.analyse(b, chess.engine.Limit(nodes=SF_NODES), root_moves=[mv])
                w.append(win_pct(info["score"].pov(b.turn).score(mate_score=1500)))
            rows.append(w[1] - w[0])
            f.write(f"{fen}\t{d}\t{c.uci()}\t{a.uci()}\t{w[0]:.2f}\t{w[1]:.2f}\n")
    sf.quit()
    if not rows:
        print("no cases"); return
    better = sum(r > 0.5 for r in rows); worse = sum(r < -0.5 for r in rows)
    se = statistics.stdev(rows) / math.sqrt(len(rows)) if len(rows) > 1 else float("nan")
    print(f"cases {len(rows)}  mean win% adopted - completed {statistics.mean(rows):+.2f} pp (se {se:.2f})  "
          f"median {statistics.median(rows):+.2f}  adopted better {better}  worse {worse}  within 0.5 pp {len(rows) - better - worse}")
    if better + worse:
        print(f"adopted better in {100 * better / (better + worse):.1f}% of non-tied cases")

if __name__ == "__main__":
    main()
