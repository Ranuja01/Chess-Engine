# -*- coding: utf-8 -*-
"""Build a STRUCTURE-INDEPENDENT regret set: whacky legal-chess positions from piece-replacement and 960-style
starting arrays (castling disabled, so no UCI_Chess960 support needed), reached by a short random walk, then
labelled with SF18 multi-PV. Same schema as game_regret_set.csv, so the regret/footprint tools consume it as a
THIRD cross-validation set that tests whether a detector is genuine chess knowledge vs standard-structure overfit.

Why: the 15k + v2 sets are both STANDARD selfplay ⇒ shared opening structures a term can memorize. Variant
positions have no such scaffolding but still require correct evaluation, and SF18 (computes, not books) is a
valid arbiter. Use as a GENERALIZATION check (does a detector transfer?), NOT a primary tuning target — our
PSTs/king-zone tables are standard-tuned, so variants also expose those (a confound to respect).

  pyrun diagnostics/_build_variant_regret_set.py [N=6000] [K=8] [SF_DEPTH=14] [OUT=ks_sets/variant_regret_set.csv]
        [SHARD=0/1] [WALK_MIN=6] [WALK_MAX=20] [MAT_TOL=500] [SEED=12345]   (needs STOCKFISH_PATH -> Linux SF18)

Resumable: re-running skips FENs already in OUT. Shardable (SHARD=i/n) for parallel labeling on spare cores.
"""
import os, sys, csv, random, atexit
from collections import defaultdict

for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
THIS = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(THIS)
sys.path.insert(0, ENGINE); sys.path.insert(0, THIS)
sys.path.insert(0, os.path.join(ENGINE, "selfplay"))

import chess
import chess.engine

N = int(os.environ.get("N", "6000"))
K = int(os.environ.get("K", "8"))
SF_DEPTH = int(os.environ.get("SF_DEPTH", "14"))
OUT = os.environ.get("OUT", "ks_sets/variant_regret_set.csv")
if not os.path.isabs(OUT):
    OUT = os.path.join(THIS, OUT)
WALK_MIN = int(os.environ.get("WALK_MIN", "6"))
WALK_MAX = int(os.environ.get("WALK_MAX", "20"))
MAT_TOL = int(os.environ.get("MAT_TOL", "500"))     # reject a walk that ended >MAT_TOL cp lopsided (keep non-trivial)
_si, _sn = (int(x) for x in os.environ.get("SHARD", "0/1").split("/"))
rng = random.Random(int(os.environ.get("SEED", "12345")) + _si)

# ---- Starting-array zoo: symmetric back ranks (black = white lowercased), castling disabled ----
# Standard multiset is 2R 2N 2B 1Q 1K. Variants replace/permute; both sides get the SAME rank => material balanced.
PIECE_VAL = {'p': 100, 'n': 320, 'b': 330, 'r': 500, 'q': 900, 'k': 0}


def named_arrays():
    a = []
    a.append(("std",         "RNBQKBNR"))
    a.append(("all_NtoB",    "RBBQKBBR"))   # every knight -> bishop (4 bishops)
    a.append(("all_BtoN",    "RNNQKNNR"))   # every bishop -> knight (4 knights)
    a.append(("one_NtoB",    "RBBQKBNR"))   # one knight -> bishop
    a.append(("one_BtoN",    "RNNQKBNR"))   # one bishop -> knight
    a.append(("swap_NB",     "RBNQKNBR"))   # knights<->bishops swapped in place
    a.append(("outer_swap",  "RNBQKBNR"[::-1]))  # mirror the standard rank
    return a


def random_arrays(n):
    """960-style: random permutations of the standard multiset AND of the all-swapped multisets. Exactly one K."""
    out = []
    pools = [list("RNBQKBNR"), list("RBBQKBBR"), list("RNNQKNNR"), list("RRNNBBQK")]
    for i in range(n):
        pool = list(rng.choice(pools))
        rng.shuffle(pool)
        out.append(("rand%d" % i, "".join(pool)))
    return out


def start_fen(rank):
    black = rank.lower()
    white = rank.upper()
    return "%s/pppppppp/8/8/8/8/PPPPPPPP/%s w - - 0 1" % (black, white)


def material_cp(board):
    w = b = 0
    for _, pc in board.piece_map().items():
        v = PIECE_VAL[pc.symbol().lower()]
        if pc.color == chess.WHITE:
            w += v
        else:
            b += v
    return w - b


def phase_of(board):
    pc = len(board.piece_map())
    return "opening" if pc >= 26 else ("midgame" if pc >= 14 else "endgame")


def walk_to_position(rank):
    """Random legal walk from a variant start; return a non-terminal, not-too-lopsided FEN or None."""
    try:
        board = chess.Board(start_fen(rank))
    except Exception:
        return None
    plies = rng.randint(WALK_MIN, WALK_MAX)
    for _ in range(plies):
        if board.is_game_over(claim_draw=False):
            return None
        moves = list(board.legal_moves)
        if not moves:
            return None
        board.push(rng.choice(moves))
    if board.is_game_over(claim_draw=False):
        return None
    if abs(material_cp(board)) > MAT_TOL:
        return None
    return board.fen()


# Build the candidate FEN list (deduped), sharded.
ARRAYS = named_arrays() + random_arrays(max(1, N // 40))
picked, seen = [], set()
attempts = 0
while len(picked) < N and attempts < N * 40:
    attempts += 1
    _, rank = rng.choice(ARRAYS)
    fen = walk_to_position(rank)
    if not fen or fen in seen:
        continue
    seen.add(fen)
    picked.append(fen)
picked = picked[_si::_sn]

# Resume.
done = {}
if os.path.exists(OUT):
    try:
        for r in csv.DictReader(open(OUT, newline="")):
            if r.get("fen"):
                done[r["fen"]] = r
    except Exception:
        pass

FIELDS = ["fen", "phase_bucket", "best_uci", "best_cp", "moves"]
out_rows = list(done.values())


def flush():
    tmp = OUT + ".tmp"
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in out_rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    os.replace(tmp, OUT)


atexit.register(flush)

sfpath = os.environ.get("STOCKFISH_PATH")
if not sfpath:
    sys.exit("STOCKFISH_PATH required (native-ELF Linux SF18)")
eng = chess.engine.SimpleEngine.popen_uci(sfpath)
try:
    eng.configure({"Threads": 1})
except Exception:
    pass

print("  %d variant positions (%d already done)  SF d%d multipv %d -> %s"
      % (len(picked), len(done), SF_DEPTH, K, os.path.basename(OUT)), flush=True)

n_new = 0
for i, fen in enumerate(picked, 1):
    if fen in done:
        continue
    board = chess.Board(fen)
    try:
        infos = eng.analyse(board, chess.engine.Limit(depth=SF_DEPTH), multipv=K)
    except Exception as e:
        print("  [%5d/%5d] SKIP %s" % (i, len(picked), type(e).__name__), flush=True)
        continue
    pairs = []
    for info in infos:
        pv = info.get("pv"); sc = info.get("score")
        if not pv or sc is None:
            continue
        pairs.append((pv[0].uci(), sc.pov(chess.WHITE).score(mate_score=100000)))
    if not pairs:
        continue
    out_rows.append({"fen": fen, "phase_bucket": phase_of(board), "best_uci": pairs[0][0],
                     "best_cp": pairs[0][1], "moves": ";".join("%s:%d" % (u, c) for u, c in pairs)})
    n_new += 1
    if n_new % 100 == 0:
        flush()
        print("  [%5d/%5d] labelled %d new" % (i, len(picked), n_new), flush=True)

flush()
try:
    eng.quit()
except Exception:
    pass
ph = defaultdict(int)
for r in out_rows:
    ph[r.get("phase_bucket", "?")] += 1
print("\n  DONE: %d total -> %s" % (len(out_rows), OUT))
print("  by phase: %s" % "  ".join("%s=%d" % (k, ph[k]) for k in sorted(ph)), flush=True)
