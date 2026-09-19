# -*- coding: utf-8 -*-
"""Measure an eval arm's POSITIONAL SPREAD -- the scale any new positional term must be sized against.

WHY: reference positional constants converted "by the pawn" landed 8x too large on 2026-09-13 (tempo),
because v2's whole midgame positional signal was only 5-35 mp on a hand probe. Mobility will be the largest
positional term v2 owns, so its magnitude is chosen against THIS number, never against a pawn ratio.
Memory: convert-reference-constants-by-positional-scale-not-by-the-pawn.

WHAT IT MEASURES (static eval only, no search):
  1. Hand rows -- the 09-13 probe positions, for continuity, plus a sign check (White up a rook must be < 0,
     the eval being Black-positive).
  2. SIBLING SPREAD over quiet moves: for each root position, evaluate every legal quiet child (no capture,
     no promotion, no check, no castling) from the ROOT MOVER's point of view and report the range and std
     across siblings. ★ This is the quantity that ORDERS moves, so it is what a mobility term competes with.
     Siblings share material, so the spread is purely positional.
  3. The same spread for SF18's multi-PV scores on the quiet moves the corpus lists (cp x10 -> nominal mp),
     restricted to the same moves, so "how compressed is our positional signal vs SF's" is one ratio.
  4. Spread by MOVED PIECE TYPE -- where a per-piece term will land.

USAGE (knobs latch at engine init => ONE PROCESS PER ARM; KEY=VAL args are exported before the engine loads):
  pyrun diagnostics/_v2_positional_spread.py EVAL_ARM=1 <shipped v2 knobs> [N=400]
  pyrun diagnostics/_v2_positional_spread.py EVAL_ARM=0 [N=400]
The echoed [knobs] header is the only place a malformed value is visible -- read it before the numbers.
"""
import os, sys, csv, statistics

# KEY=VAL args -> environment, BEFORE ChessAI is imported or constructed.
N = 400
for a in sys.argv[1:]:
    if "=" not in a:
        continue
    k, v = a.split("=", 1)
    if k == "N":
        N = int(v)
    else:
        os.environ[k] = v

ENGINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # diagnostics/ -> NN Engine/
sys.path.insert(0, ENGINE)
os.chdir(ENGINE)
import chess, ChessAI

knobs = [a for a in sys.argv[1:] if "=" in a]
print("[knobs] " + (" ".join(knobs) if knobs else "(none)"))

seed = chess.Board()
ai = ChessAI.ChessAI(None, None, seed, seed.turn)

HAND = [
    ("start position            ", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("W knight rim a3           ", "rnbqkbnr/pppppppp/8/8/8/N7/PPPPPPPP/R1BQKBNR b KQkq - 1 1"),
    ("W knight central e5       ", "rnbqkbnr/pppppppp/8/4N3/8/8/PPPPPPPP/R1BQKBNR b KQkq - 1 1"),
    ("Italian, W bishop c4      ", "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"),
    ("Italian, W bishop f1      ", "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 3 3"),
    ("W rook open file d1 vs a1 ", "r3k3/ppp2ppp/8/8/8/8/PPP2PPP/3RK3 b - - 0 1"),
    ("W rook closed a1          ", "r3k3/ppp2ppp/8/8/8/8/PPP2PPP/R3K3 b - - 0 1"),
    ("W up a clean rook (sign)  ", "4k3/pppppppp/8/8/8/8/PPPPPPPP/R3K3 w - - 0 1"),
]

print("\n== hand rows (Black-positive mp) ==")
hv = {}
for name, fen in HAND:
    v = ai.ev(chess.Board(fen))
    hv[name] = v
    print("  %s %8d" % (name, v))
sign_ok = hv["W up a clean rook (sign)  "] < 0
print("  sign check (White up a rook < 0): %s" % ("OK" if sign_ok else "<<< FAILED -- convention differs, numbers below are mis-signed"))


def quiet(b, m):
    return not (b.is_capture(m) or m.promotion or b.is_castling(m) or b.gives_check(m))


rows = []
with open(os.path.join(ENGINE, "diagnostics", "ks_sets", "game_regret_set.csv"), newline="") as f:
    rows = list(csv.DictReader(f))
stride = max(1, len(rows) // N)
sample = rows[::stride][:N]

PT = {chess.PAWN: "P", chess.KNIGHT: "N", chess.BISHOP: "B", chess.ROOK: "R", chess.QUEEN: "Q", chess.KING: "K"}
ours_range, ours_std, sf_range, ours_range_same, ratio = [], [], [], [], []
by_pt = {k: [] for k in PT.values()}
by_phase = {}

for r in sample:
    b = chess.Board(r["fen"])
    if b.is_check():
        continue
    mover_white = b.turn
    child = {}
    for m in b.legal_moves:
        if not quiet(b, m):
            continue
        b.push(m)
        v = ai.ev(b)
        b.pop()
        child[m.uci()] = (-v if mover_white else v)            # root mover's POV, mp
    if len(child) < 3:
        continue
    vals = list(child.values())
    mean = statistics.fmean(vals)
    ours_range.append(max(vals) - min(vals))
    ours_std.append(statistics.pstdev(vals))
    by_phase.setdefault(r["phase_bucket"], []).append(statistics.pstdev(vals))
    for u, v in child.items():
        p = b.piece_at(chess.parse_square(u[:2]))
        by_pt[PT[p.piece_type]].append(v - mean)

    sf = {}
    for tok in (r.get("moves") or "").split(";"):
        if ":" in tok:
            u, cp = tok.split(":")
            if u in child:
                sf[u] = int(cp) * 10
    if len(sf) >= 3:
        srange = max(sf.values()) - min(sf.values())
        orange = max(child[u] for u in sf) - min(child[u] for u in sf)
        sf_range.append(srange)
        ours_range_same.append(orange)
        if srange > 0:
            ratio.append(orange / srange)


def pct(xs, p):
    if not xs:
        return float("nan")
    s = sorted(xs)
    return s[min(len(s) - 1, int(p * len(s)))]


def line(name, xs):
    print("  %-34s n=%5d  p10 %7.1f  p50 %7.1f  p90 %7.1f  mean %7.1f" %
          (name, len(xs), pct(xs, .1), pct(xs, .5), pct(xs, .9), statistics.fmean(xs) if xs else float("nan")))


print("\n== sibling spread over QUIET moves, root mover POV (mp) ==  positions=%d" % len(ours_range))
line("ours  range (all quiet siblings)", ours_range)
line("ours  std   (all quiet siblings)", ours_std)
line("ours  range (SF-listed quiet only)", ours_range_same)
line("SF18  range (same moves, cp x10)", sf_range)
line("ours/SF range ratio (per position)", ratio)

print("\n== std across siblings by phase bucket (mp) ==")
for k in sorted(by_phase):
    line(k, by_phase[k])

print("\n== |child - sibling mean| by MOVED piece (mp) ==")
for k in "PNBRQK":
    line(k, [abs(x) for x in by_pt[k]])

sys.exit(0 if sign_ok else 1)
