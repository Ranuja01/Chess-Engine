# -*- coding: utf-8 -*-
"""Eval divergence hunt vs classical Stockfish 11 — find WHERE our HCE judges positions worst, term-by-term.

The us-vs-SF11 equal-depth result (~590 Elo gap at depth 12) localized the dominant weakness to PER-NODE
JUDGMENT — our eval, not search speed. SF11 is a great pure-HCE, apples-to-apples with ours, and its `eval`
command prints a LABELED term table (Material/Pawns/Mobility/King safety/Threats/Passed/Space/...) + a final
White-POV total. So this ranks positions by |our_total − SF11_total| (where our eval is most wrong vs a great
HCE) and dumps the worst with BOTH breakdowns side-by-side — our terms (ChessAI.ev_breakdown) and SF11's
labeled terms — so the agent can read which term a strong HCE values differently and PACE-tune it.

Run in WSL from NN Engine/ (needs SF11 interop; STOCKFISH_PATH/SF18 unused here):
    pyrun diagnostics/eval_vs_sf11.py [N=400]    # N STS positions (diverse, positional)
Outputs: aggregate signed bias (do we systematically over/under-read vs SF11?) + the top-divergence positions.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
# Optional: evaluate OUR side with the m4000ctl king-safety anchor ON, to see if our KS term (when active)
# closes the king-safety divergence vs SF11. Must be set before ChessAI init reads the env. `... 400 anchor`.
if "anchor" in sys.argv:
    # Defaults only — an outer env (KING_SAFETY_MAG=.. KS_ZONE2=1 KS_WEAK=.. etc.) overrides, so a
    # candidate KS bundle can be probed without the script clobbering its knobs.
    os.environ.setdefault("ENABLE_KS_REPLACE_LT", "1")
    os.environ.setdefault("KING_SAFETY_MAG", "4000")
    os.environ.setdefault("MOD_KS_CONTROL", "256")
import re
import random
import subprocess

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, ENGINE_DIR)
sys.path.insert(0, THIS_DIR)

import chess  # noqa: E402
from sts_test import load_sts_epd  # noqa: E402

SF11 = "/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/stockfish_11/stockfish-11-win/Windows/stockfish_20011801_x64_bmi2.exe"
STS = os.path.join(THIS_DIR, "suites", "STS1-STS15_LAN_v3.epd")

# Our additive terms (ChessAI.ev_breakdown), in Black-positive milli-pawns -> White-POV pawns = -v/1000.
OUR_TERMS = ["material", "pieces", "capture_gains", "passed_pawn_support", "latent_threat",
             "king_safety", "central", "imbalance_white", "imbalance_black", "pair_bonus", "piece_value_boost"]
# Rough map to SF11's labels (for the agent's reading; not a strict 1:1):
#   pieces~Knights/Bishops/Rooks/Queens(placement), king_safety~King safety, latent_threat~Threats,
#   passed_pawn_support~Passed, central~Space, imbalance~Imbalance, (mobility is folded into ours).


class SF11Eval:
    def __init__(self, path):
        self.p = subprocess.Popen([path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.p.stdin.write("uci\n"); self.p.stdin.flush()
        while True:
            ln = self.p.stdout.readline()
            if not ln or ln.strip() == "uciok":
                break

    def eval(self, fen):
        """Return (total_white_pov_pawns, {term: total_mg}) from SF11's classical `eval`."""
        self.p.stdin.write("position fen %s\neval\nisready\n" % fen); self.p.stdin.flush()
        total, terms, lines = None, {}, []
        while True:
            ln = self.p.stdout.readline()
            if not ln or ln.strip() == "readyok":
                break
            lines.append(ln.rstrip())
        for s in lines:
            m = re.search(r"Total evaluation:\s*([-+]?\d+\.\d+)", s)
            if m:
                total = float(m.group(1))
            # term rows: " Term | wMG wEG | bMG bEG | tMG tEG " ; grab label + the final two (Total MG/EG)
            mm = re.match(r"\s*([A-Za-z ]+?)\s*\|.*\|.*\|\s*([-+]?\d+\.\d+|----)\s+([-+]?\d+\.\d+|----)\s*$", s)
            if mm:
                lbl = mm.group(1).strip()
                try:
                    terms[lbl] = float(mm.group(2))   # Total-MG column
                except ValueError:
                    pass
        return total, terms

    def close(self):
        try:
            self.p.stdin.write("quit\n"); self.p.stdin.flush(); self.p.wait(timeout=2)
        except Exception:
            try: self.p.kill()
            except Exception: pass


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    # Seeded DISJOINT train/holdout split (fixed seed => train and holdout are always the same disjoint
    # halves): tune the KS bundle on `train`, confirm the KS-fit gain GENERALIZES on `holdout`.
    split = "train" if "train" in sys.argv else ("holdout" if "holdout" in sys.argv else "all")
    allpos = load_sts_epd(STS)
    if split != "all":
        random.Random(1234).shuffle(allpos)
        half = len(allpos) // 2
        allpos = allpos[:half] if split == "train" else allpos[half:]
    pos = allpos[:n]
    print("[split=%s  n=%d]" % (split, len(pos)))
    from ChessAI import ChessAI
    seed = chess.Board()
    ai = ChessAI(None, None, seed, seed.turn)
    sf = SF11Eval(SF11)
    rows = []
    bias = 0.0
    try:
        for fen, _, _, _, _ in pos:
            board = chess.Board(fen)
            bd = ai.ev_breakdown(board)
            if bd.get("checkmate"):
                continue
            our = -bd["total"] / 1000.0            # White-POV pawns
            sf_total, sf_terms = sf.eval(fen)
            if sf_total is None:
                continue
            gap = our - sf_total                   # + => we read hotter for White than SF11
            bias += gap
            rows.append((gap, fen, our, sf_total, bd, sf_terms))
    finally:
        sf.close()
    if not rows:
        print("no positions evaluated"); return 1
    # King-safety term fit: our king_safety (White-POV pawns = -v/1000) vs SF11's "King safety" term.
    ks = []
    for gap, fen, our, sft, bd, sft_terms in rows:
        ours_ks = -bd.get("king_safety", 0) / 1000.0
        sf_ks = sft_terms.get("King safety", 0.0)
        ks.append((ours_ks - sf_ks, fen, ours_ks, sf_ks))
    ks.sort(key=lambda r: abs(r[0]), reverse=True)
    n_under = sum(1 for d, _, o, s in ks if abs(s) > 1.0 and abs(o) < abs(s) / 2)
    print("KS-FIT vs SF11: mean|our_KS - SF11_KS|=%.3f  (positions where SF11 sees >1p KS but we read <half: %d/%d)"
          % (sum(abs(d) for d, _, _, _ in ks) / len(ks), n_under, len(ks)))
    print("  worst KS-under-fire (SF11 sees the attack, we don't) — the structural-detection targets:")
    for d, fen, o, s in [r for r in ks if abs(r[3]) > abs(r[2])][:8]:
        print("    SF11_KS=%+5.2f  ours_KS=%+5.2f  %s" % (s, o, fen))
    print()
    rows.sort(key=lambda r: abs(r[0]), reverse=True)
    print("eval-vs-SF11 divergence  n=%d  mean signed gap (ours-SF11, White-POV pawns)=%+.3f  mean |gap|=%.3f"
          % (len(rows), bias / len(rows), sum(abs(r[0]) for r in rows) / len(rows)))
    print("(+ = we read the position HOTTER for White than a great HCE does)\n")
    for gap, fen, our, sft, bd, sft_terms in rows[:18]:
        print("=" * 96)
        print("gap %+6.2f   ours %+6.2f  SF11 %+6.2f   %s" % (gap, our, sft, fen))
        ot = sorted(((t, -bd[t] / 1000.0) for t in OUR_TERMS), key=lambda kv: abs(kv[1]), reverse=True)
        print("  OURS : " + "  ".join("%s=%+.2f" % (t, v) for t, v in ot[:6] if abs(v) > 0.04))
        st = sorted(sft_terms.items(), key=lambda kv: abs(kv[1]), reverse=True)
        print("  SF11 : " + "  ".join("%s=%+.2f" % (t, v) for t, v in st if abs(v) > 0.04 and t != "Total"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
