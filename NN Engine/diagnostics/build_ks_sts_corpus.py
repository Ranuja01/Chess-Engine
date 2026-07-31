# -*- coding: utf-8 -*-
"""Build an SF11-static-anchored KS fit corpus from the STS300 positions. For each position:
target_ks = SF11-static 'King safety' term; target_total = our OFF-config total with KS swapped to the
SF11 target (= our_off_total - our_off_ks + sf11_ks), so the win%-fit drives ONLY the KS term toward
SF11's apply/don't-apply pattern while leaving the rest of our eval fixed.

Tiers encode the DISCRIMINATION question:
  quiet_neg : |sf11_ks| < 0.3  (SF says do NOT apply KS)   -> GUARD: our KS must stay ~0 here
  attack    : |sf11_ks| >= 0.75 (SF says real danger)      -> TARGET: our KS must fire here
  mid       : in between                                    -> context

Writes ks_sets/ks_sts_corpus.csv. Uses our OFF (default) config for the base eval.
  pyrun diagnostics/build_ks_sts_corpus.py
"""
import os, sys, csv, re, subprocess
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # OFF/default config -> base eval anchor

THIS = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(THIS, "results")
OUT = os.path.join(THIS, "ks_sets", "ks_sts_corpus.csv")
sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
SF11_PATH = "/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/stockfish_11/stockfish-11-win/Windows/stockfish_20011801_x64_bmi2.exe"
import chess
from ChessAI import ChessAI
ai = ChessAI(None, None, chess.Board(), True)


class SF11:
    def __init__(self, path):
        self.p = subprocess.Popen([path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.p.stdin.write("uci\n"); self.p.stdin.flush()
        for _ in range(200):
            if (self.p.stdout.readline() or "").strip() == "uciok":
                break

    def ks(self, fen):
        self.p.stdin.write("position fen %s\neval\nisready\n" % fen); self.p.stdin.flush()
        val = None
        while True:
            ln = self.p.stdout.readline()
            if not ln or ln.strip() == "readyok":
                break
            mm = re.match(r"\s*([A-Za-z ]+?)\s*\|.*\|.*\|\s*([-+]?\d+\.\d+|----)\s+([-+]?\d+\.\d+|----)\s*$", ln)
            if mm and mm.group(1).strip() == "King safety":
                try:
                    val = float(mm.group(2))
                except ValueError:
                    pass
        return val


def phase_bucket(ps):
    ps = float(ps)
    return "opening" if ps < 24 else "midgame" if ps < 64 else "endgame" if ps < 104 else "adveg"


src = os.path.join(RES, "sts_results_ksoff.csv")
sf = SF11(SF11_PATH)
rows = []
with open(src) as f:
    for r in csv.DictReader(f):
        fen = r["fen"]
        bd = ai.ev_breakdown(chess.Board(fen))
        if bd.get("checkmate"):
            continue
        our_total = -bd.get("total", 0.0) / 1000.0
        our_ks = -bd.get("king_safety", 0.0) / 1000.0
        sf_ks = sf.ks(fen)
        if sf_ks is None:
            continue
        ps = bd.get("phase_score", 64)
        target_total = our_total - our_ks + sf_ks     # swap ONLY the KS term to SF11's
        a = abs(sf_ks)
        tier = "quiet_neg" if a < 0.3 else ("attack" if a >= 0.75 else "mid")
        split = "val" if (hash(fen) % 5 == 0) else "train"
        rows.append({"fen": fen, "target_ks": round(sf_ks, 3), "target_total": round(target_total, 3),
                     "our_total_base": round(our_total, 3), "our_ks_base": round(our_ks, 3),
                     "tier": tier, "phase_bucket": phase_bucket(ps), "split": split})
sf.close() if hasattr(sf, "close") else None

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["fen", "target_ks", "target_total", "our_total_base",
                                      "our_ks_base", "tier", "phase_bucket", "split"])
    w.writeheader(); w.writerows(rows)
from collections import Counter
print("ks_sts corpus: %d rows -> %s" % (len(rows), OUT))
print("tiers:", dict(Counter(c["tier"] for c in rows)))
print("phases:", dict(Counter(c["phase_bucket"] for c in rows)))
