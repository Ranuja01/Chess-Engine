# -*- coding: utf-8 -*-
"""Marginal pawn value on REAL positions — the validity check on the manufactured-backdrop result.

`pawn_truth_ours.py` measured, on manufactured backdrops, that our eval adds ~122-151 cp of POSITIONAL value
per pawn (total 221-251 against SF18's 55-96, of which 100 is material). Taken at face value that says we
price every pawn at ~2.3 pawns, which would distort every capture and sacrifice decision in the engine.

Before acting on a number that large, it has to be shown NOT to be an artifact of the generator. The
backdrops are deliberately unrealistic — random pieces on random squares — and our placement and attacking
layers are tuned on real structures, so they may behave very differently there. This runs the identical
measurement on REAL positions drawn from the fit corpora: remove one pawn, score both engines, compare.

Same conventions as `pawn_truth_ours.py`: SF18 search vs our static eval, White-POV cp, our eval negated
(absolute Black-positive) and scaled by 10.

  pyrun diagnostics/pawn_marginal_real.py [IN=ks_sets/diverse_corpus_wide.csv] [DEPTH=14] [MAX_POS=150]
                                          [PER_POS=2] [SEED=0]
"""
import os, sys, csv, random
from collections import defaultdict

for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
import chess
import chess.engine
from ChessAI import ChessAI

SF18 = os.environ["STOCKFISH_PATH"]
DEPTH = int(os.environ.get("DEPTH", "14"))
MAX_POS = int(os.environ.get("MAX_POS", "150"))
PER_POS = int(os.environ.get("PER_POS", "2"))     # pawns sampled per position
SEED = int(os.environ.get("SEED", "0"))
# Which piece type to price. Default P reproduces the original pawn measurement exactly. N/B/R/Q run the
# IDENTICAL removal experiment on a piece, which is what makes the two comparable: the implied exchange
# rate (piece value / pawn value) is the quantity behind "we think three pawns outweigh a bishop".
PIECE = os.environ.get("PIECE", "P").upper()
_PT = {"P": chess.PAWN, "N": chess.KNIGHT, "B": chess.BISHOP, "R": chess.ROOK, "Q": chess.QUEEN}
# PIECE=ALL prices every type on the SAME positions in ONE process, and reports the global eval-scale
# factor from that same sample. Splicing marginals from separately-sampled runs is what produced three
# retracted exchange-rate numbers: the ratio of two medians drawn from different position sets is not an
# exchange rate. One sample, one scale, one table.
ALL = PIECE == "ALL"
TYPES = [("P", chess.PAWN), ("N", chess.KNIGHT), ("B", chess.BISHOP), ("R", chess.ROOK)] if ALL \
        else [(PIECE, _PT[PIECE])]
PIECE_TYPE = _PT.get(PIECE, chess.PAWN)
IN = os.environ.get("IN", os.path.join(THIS, "ks_sets", "diverse_corpus_wide.csv"))
if not os.path.isabs(IN):
    IN = os.path.join(THIS, IN) if not IN.startswith("ks_sets") else os.path.join(THIS, IN)
MATE_CP = 2000


def median(v):
    if not v:
        return float("nan")
    s = sorted(v); n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def main():
    rows = list(csv.DictReader(open(IN, newline="")))
    rng = random.Random(SEED)
    rng.shuffle(rows)
    ai = ChessAI(None, None, chess.Board(), True)
    engine = chess.engine.SimpleEngine.popen_uci(SF18)
    engine.configure({"Threads": 1, "Hash": 128})
    # The PIECE=ALL branch below `return`s before the tail-end engine.quit(), so the process printed its
    # whole result and then hung on python-chess's non-daemon transport thread -- 82 minutes of held CPU
    # that also starved the following run. atexit fires on early returns and exceptions alike.
    import atexit
    atexit.register(lambda: engine.quit())
    limit = chess.engine.Limit(depth=DEPTH)

    def sf_cp(b):
        return engine.analyse(b, limit)["score"].white().score(mate_score=MATE_CP)

    def our_cp(b):
        return -ai.ev(b) / 10.0

    by_rank = defaultdict(lambda: {"sf": [], "ours": []})
    by_piece = defaultdict(lambda: {"sf": [], "ours": [], "sf11": [], "sf15c": []})
    scale_ours, scale_sf = [], []
    # 🚨 DUMP THE RAW ROWS. A median cannot distinguish "every position is 7% off" (a value defect) from
    # "most are fine and a subset is badly off" (an overfiring predicate) -- and those need opposite fixes.
    # Writing one row per removal, with the position features to slice on, makes every follow-up question
    # free instead of costing another Stockfish run.
    OUT = os.environ.get("OUT", "")
    dump = []
    # Per-term attribution of OUR marginal value, on the same real removals. Answers which terms build the
    # number, so a level correction can be targeted rather than a uniform shrink (uniform shrink flattens
    # the eval -- measured, see `corpus-fit-flattens-eval`).
    TERMS = os.environ.get("TERMS", "0") == "1"
    by_term = defaultdict(list)
    # 🔺 TRIANGULATION. SF18-search prices a piece using tactics no static eval can encode, so a gap against
    # it does NOT by itself mean a hand-fixable defect. SF11 (pure classical) and SF15.1 with NNUE OFF are
    # the ceiling a handcrafted eval can reach. If THEY also disagree with us, the defect is static and
    # fixable; if they sit where we sit, SF18's number is search knowledge and tuning toward it is a trap.
    LADDER = os.environ.get("LADDER", "0") == "1"
    lad = {}
    if LADDER:
        from eval_vs_sf11 import SF11Eval, SF11
        sys.path.insert(0, THIS)
        from probe_fens import NNUEStatic, SF15
        sf11 = SF11Eval(SF11)
        sf15c = NNUEStatic(SF15, nnue=False)
        lad = {"sf11": [], "sf15c": []}
        # These subprocesses were never torn down: the script printed its full result and then HUNG,
        # holding CPU for 82 minutes and starving the next run. Same failure class as the python-chess
        # engine thread. Any long-lived child must be closed on every exit path, not at the happy end.
        import atexit
        atexit.register(lambda: [getattr(sf11, "close", lambda: None)(),
                                 getattr(sf15c, "close", lambda: None)()])
    used = 0
    for r in rows:
        if used >= MAX_POS:
            break
        try:
            board = chess.Board(r["fen"])
        except Exception:
            continue
        if board.is_check() or board.is_game_over(claim_draw=False):
            continue
        # Only WHITE pawns, mirroring the manufactured measurement's White-POV convention. Never a pawn whose
        # removal would leave an illegal position.
        # Global eval scale on this same neutral sample. Without it the marginal columns cannot be
        # compared across engines at all.
        try:
            scale_ours.append(abs(our_cp(board))); scale_sf.append(abs(sf_cp(board)))
        except Exception:
            pass

        squares = []
        for lbl, pt in TYPES:
            got = [s for s in board.pieces(pt, chess.WHITE)]
            rng.shuffle(got)
            squares += [(lbl, s) for s in got[:PER_POS]]
        if not squares:
            continue
        pawns = squares
        rng.shuffle(pawns)
        did = 0
        for lbl, s in pawns:
            if did >= PER_POS * len(TYPES):
                break
            without = board.copy()
            without.remove_piece_at(s)
            if without.status() != chess.STATUS_VALID or without.is_game_over(claim_draw=False):
                continue
            a, c = sf_cp(without), sf_cp(board)
            if a is None or c is None:
                continue
            rank = chess.square_rank(s) + 1
            by_rank[rank]["sf"].append(c - a)
            by_rank[rank]["ours"].append(our_cp(board) - our_cp(without))
            by_piece[lbl]["sf"].append(c - a)
            by_piece[lbl]["ours"].append(our_cp(board) - our_cp(without))
            if OUT:
                bdp = ai.ev_breakdown(board)
                pc = lambda t, col: len(board.pieces(t, col))
                dump.append({
                    "fen": board.fen(), "piece": lbl, "sq": s,
                    "rank": chess.square_rank(s) + 1, "file": chess.square_file(s) + 1,
                    "sf18": round(c - a, 1), "ours": round(our_cp(board) - our_cp(without), 1),
                    "sf18_pos": round(c, 1), "ours_pos": round(our_cp(board), 1),
                    "phase": bdp.get("phase_score", 0), "is_eg": int(bdp.get("is_endgame", 0)),
                    "wp": pc(chess.PAWN, True), "bp": pc(chess.PAWN, False),
                    "wn": pc(chess.KNIGHT, True), "bn": pc(chess.KNIGHT, False),
                    "wb": pc(chess.BISHOP, True), "bb": pc(chess.BISHOP, False),
                    "wr": pc(chess.ROOK, True), "br": pc(chess.ROOK, False),
                    "wq": pc(chess.QUEEN, True), "bq": pc(chess.QUEEN, False),
                })
            if LADDER:
                # Both static evaluators report White-POV pawns; x100 to match the cp convention here.
                try:
                    e0, e1 = sf11.eval(board.fen())[0], sf11.eval(without.fen())[0]
                    if e0 is not None and e1 is not None:
                        lad["sf11"].append((e0 - e1) * 100.0)
                        by_piece[lbl]["sf11"].append((e0 - e1) * 100.0)
                    a15, c15 = sf15c.final_eval(without.fen()), sf15c.final_eval(board.fen())
                    if a15 is not None and c15 is not None:
                        lad["sf15c"].append((c15 - a15) * 100.0)
                        by_piece[lbl]["sf15c"].append((c15 - a15) * 100.0)
                except Exception:
                    pass
            if TERMS:
                bw, bo = ai.ev_breakdown(board), ai.ev_breakdown(without)
                for k, v in bw.items():
                    if isinstance(v, int) and k not in ("phase_score", "is_endgame"):
                        # Negated and /10 to match our_cp: absolute Black-positive millipawns -> White cp.
                        by_term[k].append(-(v - bo.get(k, 0)) / 10.0)
            did += 1
        if did:
            used += 1

    if OUT and dump:
        import csv as _csv
        path = OUT if os.path.isabs(OUT) else os.path.join(THIS, "ks_sets", OUT)
        with open(path, "w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=list(dump[0].keys()))
            w.writeheader(); w.writerows(dump)
        print("[dump] %d rows -> %s\n" % (len(dump), path))

    def pct(v, q):
        if not v:
            return float("nan")
        s = sorted(v)
        return s[max(0, min(len(s) - 1, int(q * (len(s) - 1))))]

    if ALL:
        k = (median(scale_sf) / (median(scale_ours) or 1.0)) if scale_ours else 1.0
        print("\n  ERROR DISTRIBUTION, ours-minus-SF18 per removal (cp). A tight band = a VALUE defect;")
        print("  a fat tail = the term OVERFIRES on a subset and a uniform damp would break the rest.")
        print("  %-6s %8s %8s %8s %8s %8s %8s" % ("piece", "p10", "p25", "p50", "p75", "p90", "n"))
        for lbl, _ in TYPES:
            d = by_piece[lbl]
            if len(d["sf"]) < 5:
                continue
            e = [d["ours"][i] - d["sf"][i] for i in range(len(d["sf"]))]
            print("  %-6s %8.0f %8.0f %8.0f %8.0f %8.0f %8d"
                  % (lbl, pct(e, .10), pct(e, .25), pct(e, .50), pct(e, .75), pct(e, .90), len(e)))
        print()
        print("ONE-SAMPLE PIECE PRICING (%s), %d positions, SF18 d%d\n" % (os.path.basename(IN), used, DEPTH))
        print("  GLOBAL EVAL SCALE on these same positions: ours x%.2f -> SF18"
              "   (median |eval| ours %.0f vs SF18 %.0f cp)"
              % (k, median(scale_ours), median(scale_sf)))
        print("  A factor near 1.00 means the marginal columns below ARE directly comparable.\n")
        hdr = "  %-6s %8s %8s %8s %8s %8s %7s" % ("piece", "SF18", "ours", "ours*k", "SF11", "SF15.1c", "n")
        print(hdr)
        base = {}
        for lbl, _ in TYPES:
            d = by_piece[lbl]
            if len(d["sf"]) < 5:
                continue
            base[lbl] = (median(d["sf"]), median(d["ours"]) * k,
                         median(d["sf11"]) if d["sf11"] else float("nan"),
                         median(d["sf15c"]) if d["sf15c"] else float("nan"))
            print("  %-6s %8.0f %8.0f %8.0f %8.0f %8.0f %7d"
                  % (lbl, median(d["sf"]), median(d["ours"]), median(d["ours"]) * k,
                     base[lbl][2], base[lbl][3], len(d["sf"])))
        if "P" in base:
            print("\n  EXCHANGE RATE (piece / pawn, same sample -- the scale-invariant quantity)")
            print("  %-6s %8s %8s %8s %8s" % ("piece", "SF18", "ours", "SF11", "SF15.1c"))
            p = base["P"]
            for lbl in ("N", "B", "R"):
                if lbl in base:
                    q = base[lbl]
                    f = lambda a, b: (a / b) if b else float("nan")
                    print("  %-6s %8.2f %8.2f %8.2f %8.2f"
                          % (lbl, f(q[0], p[0]), f(q[1], p[1]), f(q[2], p[2]), f(q[3], p[3])))
        return
    print("MARGINAL %s VALUE ON REAL POSITIONS (%s)" % (PIECE, os.path.basename(IN)))
    print("%d positions, SF18 d%d, medians, White-POV cp. Both columns include the piece's material.\n"
          % (used, DEPTH))
    print("  %-6s %10s %10s %10s %8s" % ("rank", "SF18", "ours", "gap", "n"))
    allsf, allours = [], []
    for rank in sorted(by_rank):
        sf, ours = by_rank[rank]["sf"], by_rank[rank]["ours"]
        allsf += sf; allours += ours
        if len(sf) < 3:
            continue
        print("  %-6d %10.0f %10.0f %10.0f %8d" % (rank, median(sf), median(ours),
                                                   median(sf) - median(ours), len(sf)))
    if allsf:
        print("  %-6s %10.0f %10.0f %10.0f %8d" % ("ALL", median(allsf), median(allours),
                                                   median(allsf) - median(allours), len(allsf)))
        NOMINAL = {"P": 100, "N": 305, "B": 333, "R": 563, "Q": 950}[PIECE]
        print("\n  ours minus nominal material (%d cp) = %+.0f cp of POSITIONAL value per %s"
              % (NOMINAL, median(allours) - NOMINAL, PIECE))
        print("  SF18 minus nominal material         = %+.0f cp" % (median(allsf) - NOMINAL))
        if LADDER:
            print("\n  🔺 CLASSICAL LADDER — marginal %s value by evaluator (median cp)" % PIECE)
            print("    %-22s %+8.0f   (n=%d)" % ("SF11 static (classical)", median(lad["sf11"]),
                                                 len(lad["sf11"])))
            print("    %-22s %+8.0f   (n=%d)" % ("SF15.1 static, NNUE off", median(lad["sf15c"]),
                                                 len(lad["sf15c"])))
            print("    %-22s %+8.0f" % ("ours (static)", median(allours)))
            print("    %-22s %+8.0f" % ("SF18 search", median(allsf)))
            print("    ⇒ If the two CLASSICAL rows sit near SF18, the gap is static and hand-fixable.")
            print("      If they sit near OURS, SF18's number is search knowledge and tuning to it is a trap.")
        if TERMS:
            print("\n  OUR MARGINAL VALUE BY TERM (median cp per %s removed, |median| >= 1)" % PIECE)
            rows = [(k, median(v)) for k, v in by_term.items()]
            for k, m in sorted(rows, key=lambda r: -abs(r[1])):
                if abs(m) >= 1:
                    print("    %-24s %+8.1f" % (k, m))
        print("\n  ⇒ Divide this run's ALL median by the P run's to get the implied EXCHANGE RATE")
        print("    (how many pawns we think the piece is worth). Ours vs SF18's is the number behind")
        print("    'we think three pawns outweigh a bishop'.")
    engine.quit()


if __name__ == "__main__":
    main()
