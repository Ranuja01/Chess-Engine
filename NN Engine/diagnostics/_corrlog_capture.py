"""Drive the engine over a FEN corpus with the correction-history logger on, so corrhist_signal.py has a
record stream to analyse.

The engine emits [CORRLOG] records on STDERR (search_engine.cpp corrhist_log, gated by ENABLE_CORRHIST_LOG),
so the caller redirects stderr to the capture file and discards stdout. Positions are walked in FILE ORDER:
a game-ordered corpus then yields records whose pawn keys genuinely recur, which is what the offline replay
needs in order to simulate table accumulation honestly.

☠️ run_one() clears the search tables per position (the fix for the diagnostic-harness history
contamination). That is deliberate and stays: we are collecting (staticEval, bestScore) residual PAIRS here,
not measuring search quality, and corrhist_signal.py simulates table accumulation itself. Letting the engine
accumulate instead would re-contaminate move ordering across unrelated FENs for no gain.

Usage:  python _corrlog_capture.py <corpus.csv|suite.epd> <n>
"""
import sys, os, csv

sys.path.insert(0, "diagnostics")
from tactical_test import run_one


def load_fens(path, n):
    fens = []
    if path.lower().endswith(".csv"):
        with open(path, newline="", errors="ignore") as f:
            r = csv.DictReader(f)
            if not r.fieldnames or "fen" not in r.fieldnames:
                sys.exit(f"[corrlog_capture] {path} has no 'fen' column (got {r.fieldnames})")
            for row in r:
                v = (row.get("fen") or "").strip()
                if v:
                    fens.append(v)
                if len(fens) >= n:
                    break
    else:
        with open(path, errors="ignore") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                # EPD: the FEN is the leading 4 fields; the rest are opcodes.
                parts = ln.split()
                if len(parts) >= 4:
                    fens.append(" ".join(parts[:4]))
                if len(fens) >= n:
                    break
    return fens


def main():
    if len(sys.argv) < 3:
        sys.exit("usage: _corrlog_capture.py <corpus.csv|suite.epd> <n>")
    path, n = sys.argv[1], int(sys.argv[2])
    if not os.path.exists(path):
        sys.exit(f"[corrlog_capture] no such corpus: {path}")
    fens = load_fens(path, n)
    print(f"[corrlog_capture] {len(fens)} positions from {path}")
    done = 0
    for fen in fens:
        try:
            run_one(fen, set())
            done += 1
        except Exception as e:  # a malformed FEN must not abandon the capture
            print(f"[corrlog_capture] skipped: {e}")
        if done % 25 == 0:
            print(f"[corrlog_capture] {done}/{len(fens)}", flush=True)
    print(f"[corrlog_capture] done {done}/{len(fens)}")


if __name__ == "__main__":
    main()
