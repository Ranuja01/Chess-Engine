# -*- coding: utf-8 -*-
"""Item 0: build the over-push refutation map for the pipeline conviction tests.
For each over-push FEN (the position we played the losing push from), play our over-push move to the
opponent-to-move node, then record SF18's refuting reply + the PV a few plies forward (the line THROUGH the
refutation). Feeds:
  - Item 1 (leaf visibility): walk the PV 2-4 plies forward, compare our-static vs SF11-static at the leaf.
  - Item 3 (inject hook): force the refutation first at the opponent node (ordering ceiling).

  overnight_runner.sh pyrun diagnostics/make_refutation_map.py diagnostics/collapse_classified.csv [--depth 16] [--pv 4]
Writes diagnostics/overpush_refutations.csv: overpush_fen, our_move, opp_fen, refutation, pv (space-sep uci).
"""
import os, sys, csv
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
THIS = os.path.dirname(os.path.abspath(__file__)); ENGINE = os.path.dirname(THIS)
sys.path.insert(0, THIS); sys.path.insert(0, os.path.join(ENGINE, "selfplay"))
import chess, chess.engine
from arbiter import find_stockfish

DEPTH = 16; PV = 4
args = sys.argv[1:]
if "--depth" in args: i = args.index("--depth"); DEPTH = int(args[i+1]); del args[i:i+2]
if "--pv" in args: i = args.index("--pv"); PV = int(args[i+1]); del args[i:i+2]
src = next((a for a in args if a.endswith(".csv")), os.path.join(THIS, "collapse_classified.csv"))
out = os.path.join(THIS, "overpush_refutations.csv")


def main():
    rows = [r for r in csv.DictReader(open(src)) if r.get("category") == "overpush"]
    sf = chess.engine.SimpleEngine.popen_uci(find_stockfish()); sf.configure({"Threads": 1})
    n = 0
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["overpush_fen", "our_move", "opp_fen", "refutation", "pv"])
        for r in rows:
            fen = r["fen"]; mv_uci = r["our_move"]
            try:
                b = chess.Board(fen); mv = chess.Move.from_uci(mv_uci)
                if mv not in b.legal_moves: continue
                b.push(mv)                                   # opponent to move now
                opp_fen = b.fen()
                info = sf.analyse(b, chess.engine.Limit(depth=DEPTH))
                pv = info.get("pv") or []
                if not pv: continue
                pv_uci = [m.uci() for m in pv[:PV]]
                w.writerow([fen, mv_uci, opp_fen, pv_uci[0], " ".join(pv_uci)])
                n += 1
            except Exception:
                continue
    sf.quit()
    print(f"[refmap] wrote {n} rows -> {out}")


if __name__ == "__main__":
    main()
