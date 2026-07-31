# -*- coding: utf-8 -*-
"""Diagnose the LOSING collapse decisions: at the peak decision FEN, did we play SF's move (so the loss is
downstream, not this decision), or a worse one? And if worse, is it an OVER-PUSH into counterplay (we play
an aggressive capture/push where SF defends) = the counterplay-blindness signature, vs another error type?

Answers the fork: is the collapse an eval-magnitude problem, a counterplay-blindness (missing-danger) problem,
or a tactic-miscalc (search) problem -- from the actual losing moves, not from static residuals.

  overnight_runner.sh pyrun diagnostics/diagnose_collapse_moves.py <collapses.csv> [collapses2.csv ...] [--depth 14] [--limit N]
"""
import os, sys, csv
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
THIS = os.path.dirname(os.path.abspath(__file__)); ENGINE = os.path.dirname(THIS)
sys.path.insert(0, THIS); sys.path.insert(0, os.path.join(ENGINE, "selfplay"))
import chess
from arbiter import Arbiter, find_stockfish

DEPTH = 14; LIMIT = 0; DUMP = None
args = [a for a in sys.argv[1:]]
if "--depth" in args: i = args.index("--depth"); DEPTH = int(args[i+1]); del args[i:i+2]
if "--limit" in args: i = args.index("--limit"); LIMIT = int(args[i+1]); del args[i:i+2]
if "--dump" in args: i = args.index("--dump"); DUMP = args[i+1]; del args[i:i+2]
paths = [a for a in args if a.endswith(".csv")]


def we_lost(result, color):
    if result == "1-0": return color == "black"
    if result == "0-1": return color == "white"
    return False


def aggressive(board, mv):
    if board.is_capture(mv): return True
    p = board.piece_at(mv.from_square)
    if p and p.piece_type == chess.PAWN: return True          # any pawn move = push/advance
    if board.gives_check(mv): return True
    return False


def main():
    seen = set(); rows = []
    for p in paths:
        for r in csv.DictReader(open(p)):
            key = r.get("decision_fen", "")
            if key and key not in seen and we_lost(r.get("result",""), r.get("our_color","")):
                seen.add(key); rows.append(r)
    if LIMIT: rows = rows[:LIMIT]
    arb = Arbiter(find_stockfish(), depth=DEPTH)
    n = 0; cat = {"downstream": 0, "overpush": 0, "other_error": 0, "minor": 0, "bad_fen": 0}
    refutation_hits = 0; details = []; dump_rows = []
    for r in rows:
        fen = r.get("decision_fen", ""); mv_uci = r.get("peak_move", "")
        try:
            b = chess.Board(fen); mv = chess.Move.from_uci(mv_uci)
            if mv not in b.legal_moves: cat["bad_fen"] += 1; continue
        except Exception:
            cat["bad_fen"] += 1; continue
        us_white = (b.turn == chess.WHITE)
        cp_before, sf_best, _ = arb.evaluate(b)
        if cp_before is None or sf_best is None: cat["bad_fen"] += 1; continue
        our_pov_before = cp_before if us_white else -cp_before
        agg = aggressive(b, mv)
        b.push(mv)
        cp_after, sf_reply, _ = arb.evaluate(b)          # opponent to move now
        our_pov_after = (cp_after if us_white else -cp_after) if cp_after is not None else our_pov_before
        cploss = our_pov_before - our_pov_after           # >0 = our move gave ground
        # counterplay confirmation: SF's refutation of our move is a capture/check on us
        refut = False
        if sf_reply:
            try:
                rm = chess.Move.from_uci(sf_reply); refut = b.is_capture(rm) or b.gives_check(rm)
            except Exception: pass
        n += 1
        if mv_uci == sf_best:
            category = "downstream"                        # we played SF's move -> loss is not this decision
        elif cploss < 30:
            category = "minor"                             # our move only slightly worse -> not the cause
        elif agg:
            category = "overpush"                          # aggressive move, big loss = over-push into counterplay
        else:
            category = "other_error"
        cat[category] += 1
        if cploss >= 30 and refut: refutation_hits += 1
        dump_rows.append({"fen": fen, "category": category, "our_move": mv_uci, "sf_best": sf_best,
                          "cploss": f"{cploss}", "refut": "1" if refut else "0", "aggressive": "1" if agg else "0",
                          "drop_fen": r.get("drop_fen", ""), "game": r.get("game", ""), "our_color": r.get("our_color", "")})
        if len(details) < 25 and mv_uci != sf_best and cploss >= 30:
            details.append(f"  {('AGG' if agg else 'qui')} cploss={cploss:>4} our={mv_uci} sf={sf_best} refut={'Y' if refut else 'n'}  {fen}")
    print(f"[diagnose] LOSING collapse decisions analysed: {n}  (SF depth {DEPTH})")
    for k in ("downstream", "minor", "overpush", "other_error", "bad_fen"):
        pct = 100.0*cat[k]/max(1,n)
        print(f"    {k:>12}: {cat[k]:>4}  ({pct:4.0f}%)")
    print(f"    of the real errors (cploss>=30, move!=SF), {refutation_hits} walked into a capture/check refutation (counterplay).")
    print("\n  READ: downstream high => the PEAK decision is fine, loss accumulates later (not a decision-point fix).")
    print("        overpush high + refutation high => counterplay-blindness (we push, SF defends, we get hit) => LANE 2 danger term.")
    print("        other_error high => positional drift or tactic-miscalc (search).")
    print("\n  sample real errors:")
    for d in details: print(d)
    if DUMP and dump_rows:
        import csv as _csv
        with open(DUMP, "w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=list(dump_rows[0].keys())); w.writeheader(); w.writerows(dump_rows)
        print(f"\n  dumped {len(dump_rows)} classified rows -> {DUMP}")


if __name__ == "__main__":
    main()
