# -*- coding: utf-8 -*-
"""Does the per-pawn bonus CLAMP bind? The precondition for the pawn-representation redesign.

Both pawn evaluators cap the per-pawn bonus before applying it:
    midgame  total -/+= min(225, structural + positional)
    endgame  total -/+= min(175, structural)
A richer file x rank scheme adds terms that compete for exactly that headroom. Where the clamp already
saturates, an added bonus contributes NOTHING -- and a term worth 0 because it was truncated is
indistinguishable, from a corpus fit or a game result, from a term that is worth 0 because it is wrong.
So measure headroom BEFORE designing the table (owner's constraint in the redesign brief).

Note the two paths clamp different quantities: the midgame bundles the placement and attacking layers
(`positional`) in with `structural`, while the endgame applies those layers straight to `total` and clamps
`structural` alone. Endgame records therefore carry positional=0 by construction. The midgame's positional
component is NOT a pawn-structure signal, so if it alone exhausts the cap, the structural terms are being
crowded out by placement -- a different problem from "the cap is too low".

Reports, per position set and per phase path:
  BIND%       share of pawns whose raw offer already meets or exceeds the cap
  SURVIVAL    of a hypothetical +X mp added to a pawn, how much actually reaches `total`
The survival curve is the decision number: if +50 mp survives at 30%, granularity is mostly wasted and the
cap is the lever, not the table.

Run across MANY sets -- a clamp rate is a property of the corpus, not of the engine
([[corpus-composition-decides-the-optimum]]).

  pyrun diagnostics/_pawn_clamp_headroom.py [MAX_POS=1500] [SETS=a,b]
"""
import os, sys, csv

for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
import chess
from ChessAI import ChessAI

MAX_POS = int(os.environ.get("MAX_POS", "1500"))
ADDS = [int(v) for v in os.environ.get("ADDS", "25,50,100").split(",")]
KS = os.path.join(THIS, "ks_sets")
ENG = os.path.dirname(THIS)

# (label, path, kind). Deliberately spans general play, tactical, positional and pawn-only material, so a
# clamp rate that holds everywhere can be separated from one that is an artifact of one corpus.
SETS = [
    ("diverse_wide",  os.path.join(KS, "diverse_corpus_wide.csv"), "csv"),
    ("position_bank", os.path.join(KS, "position_bank.csv"),       "csv"),
    ("collapse",      os.path.join(KS, "collapse_dataset.csv"),    "csv_collapse"),
    ("sts300",        os.path.join(THIS, "suites", "sts300.epd"),  "epd"),
    ("wac",           os.path.join(THIS, "suites", "wac.epd"),     "epd"),
    ("kp_mixed",      os.path.join(ENG, "selfplay", "kp_fens.txt"),       "fens"),
    ("kp_dense",      os.path.join(ENG, "selfplay", "kp_dense_fens.txt"), "fens"),
]


def load(path, kind, limit):
    """Return up to `limit` FEN strings, or [] if the set is unavailable."""
    out = []
    if not os.path.exists(path):
        return out
    if kind.startswith("csv"):
        col = "decision_fen" if kind == "csv_collapse" else "fen"
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                fen = (row.get(col) or "").strip()
                if fen:
                    out.append(fen)
                if len(out) >= limit:
                    break
        return out
    for ln in open(path):
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        toks = ln.split()
        start = next((i for i, t in enumerate(toks) if "/" in t), 0)
        if kind == "epd":
            # EPD carries operations after the four board fields, not a full FEN; supply move counters.
            out.append(" ".join(toks[start:start + 4]) + " 0 1")
        else:
            out.append(" ".join(toks[start:start + 6]))
        if len(out) >= limit:
            break
    return out


class Bucket(object):
    def __init__(self):
        self.n = 0
        self.bind = 0
        self.raw_sum = 0
        self.struct_sum = 0
        self.pos_sum = 0
        self.head_sum = 0          # cap - raw, floored at 0
        self.survived = {a: 0 for a in ADDS}
        self.pos_alone_binds = 0   # midgame: placement/attacking layers exhaust the cap on their own

    def add(self, rec):
        cap, raw = rec["cap"], rec["raw"]
        self.n += 1
        self.raw_sum += raw
        self.struct_sum += rec["structural"]
        self.pos_sum += rec["positional"]
        if raw >= cap:
            self.bind += 1
        self.head_sum += max(0, cap - raw)
        if rec["positional"] >= cap:
            self.pos_alone_binds += 1
        base = min(cap, raw)
        for a in ADDS:
            self.survived[a] += min(cap, raw + a) - base

    def report(self, label):
        if not self.n:
            return "  %-10s (no pawns)" % label
        parts = ["  %-10s n=%-6d bind=%5.1f%%  mean_raw=%6.1f (struct %5.1f + pos %5.1f)  mean_headroom=%5.1f"
                 % (label, self.n, 100.0 * self.bind / self.n, float(self.raw_sum) / self.n,
                    float(self.struct_sum) / self.n, float(self.pos_sum) / self.n,
                    float(self.head_sum) / self.n)]
        surv = "  ".join("+%d->%4.0f%%" % (a, 100.0 * self.survived[a] / (a * self.n)) for a in ADDS)
        parts.append("             survival of an added bonus:  %s" % surv)
        if self.pos_sum:
            parts.append("             positional alone already caps: %.1f%% of pawns"
                         % (100.0 * self.pos_alone_binds / self.n))
        return "\n".join(parts)


def main():
    want = os.environ.get("SETS")
    sets = [s for s in SETS if not want or s[0] in want.split(",")]
    ai = ChessAI(None, None, chess.Board(), True)

    print("per-pawn clamp headroom  (midgame cap 225 on structural+positional, endgame cap 175 on structural)")
    print("survival = fraction of an added bonus that still reaches `total` after clamping\n")

    overall = {"mid": Bucket(), "end": Bucket()}
    for label, path, kind in sets:
        fens = load(path, kind, MAX_POS)
        if not fens:
            print("%s: UNAVAILABLE (%s)\n" % (label, path))
            continue
        buckets = {"mid": Bucket(), "end": Bucket()}
        bad = 0
        for fen in fens:
            try:
                b = chess.Board(fen)
            except Exception:
                bad += 1
                continue
            if b.is_checkmate():
                continue
            for rec in ai.pawn_clamp_records(b):
                key = "end" if rec["endgame"] else "mid"
                buckets[key].add(rec)
                overall[key].add(rec)
        print("%s  (%d positions%s)" % (label, len(fens), ", %d unparseable" % bad if bad else ""))
        print(buckets["mid"].report("midgame"))
        print(buckets["end"].report("endgame"))
        print("")

    print("ALL SETS COMBINED")
    print(overall["mid"].report("midgame"))
    print(overall["end"].report("endgame"))


if __name__ == "__main__":
    main()
