# -*- coding: utf-8 -*-
"""Build/extend the DURABLE labeled position bank from the saved-game archive (selfplay/games/*/game_*.jsonl).
A reusable asset: sample positions stratified by phase, label each with CHEAP signals at scale (phase_score,
our eval + king_safety term, SF11-static total + KS term, geometry class, king-zone piece density). SF18-search
truth is added to a curated subset by a separate pass (add_sf18_labels.py). Append-only + deterministic (seeded
stride) so it accumulates across sessions and feeds every future eval-fit (KS now, pawn lever next).

Writes ks_sets/position_bank.csv (schema in dev_notes/position-bank-schema-2026-07-21.md).
Run: pyrun diagnostics/build_position_bank.py [N=3000] [SEED=7] [PER_GAME=3]
"""
import os, sys, json, glob, csv
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ.setdefault('KS_SAFE_CHECK_DEF', '5')
for _a in sys.argv[1:]:
    if '=' in _a: _k, _v = _a.split('=', 1); os.environ.setdefault(_k, _v)
N        = int(os.environ.get('N', '3000'))
SEED     = int(os.environ.get('SEED', '7'))
PER_GAME = int(os.environ.get('PER_GAME', '3'))
THIS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
import random
import chess
from ChessAI import ChessAI
from eval_vs_sf11 import SF11Eval, SF11
from classify_collapses import king_ring_pressure, king_ring, PIECE_VAL

GAMES = os.path.join(THIS, "..", "selfplay", "games")
OUT   = os.path.join(THIS, "ks_sets", "position_bank.csv")
COLS  = ["fen", "src", "phase_score", "our_total", "our_ks", "sf11_total", "sf11_ks",
         "geo_class", "kzone_w", "kzone_b", "in_check_w", "in_check_b", "sf18"]

def phase_bucket(board):
    # cheap pre-label proxy for stratified sampling: non-pawn material count (0..14ish)
    npm = sum(1 for _, p in board.piece_map().items() if p.piece_type not in (chess.PAWN, chess.KING))
    return npm

# ---- collect candidate fens from the archive, stratified by the cheap phase proxy ----
rng = random.Random(SEED)
jsonls = sorted(glob.glob(os.path.join(GAMES, "*", "game_*.jsonl")))
buckets = {}            # npm -> list of (fen, src)
seen = set()
for jf in jsonls:
    src = os.path.basename(os.path.dirname(jf))
    try:
        lines = open(jf).read().splitlines()
    except Exception:
        continue
    picks = rng.sample(lines, min(PER_GAME, len(lines))) if lines else []
    for ln in picks:
        try:
            fen = json.loads(ln).get("fen", "").strip()
            if not fen or fen in seen: continue
            b = chess.Board(fen)
        except Exception:
            continue
        seen.add(fen)
        buckets.setdefault(phase_bucket(b), []).append((fen, src))

# even sample across phase buckets up to N
order = sorted(buckets)
sample = []
i = 0
while len(sample) < N and any(buckets[k] for k in order):
    k = order[i % len(order)]; i += 1
    if buckets[k]: sample.append(buckets[k].pop())
print("archive: %d games, %d unique fens; sampling %d across %d phase-buckets" %
      (len(jsonls), len(seen), len(sample), len(order)))

# ---- label ----
ai = ChessAI(None, None, chess.Board(), True)
sf11 = SF11Eval(SF11)

def geo_class(board):
    # lightweight: is either king under heavy ring pressure / check
    wc = 1 if board.attackers(chess.BLACK, board.king(chess.WHITE)) else 0
    bc = 1 if board.attackers(chess.WHITE, board.king(chess.BLACK)) else 0
    kw = king_ring_pressure(board, chess.WHITE)
    kb = king_ring_pressure(board, chess.BLACK)
    cls = "ks" if (wc or bc or kw >= 3 or kb >= 3) else "quiet"
    return cls, kw, kb, wc, bc

rows = []
for fen, src in sample:
    try:
        b = chess.Board(fen)
        if b.is_game_over():                       # terminal: no static eval
            continue
        bd = ai.ev_breakdown(b)
        our_total = -bd.get("total", 0.0) / 1000.0            # white-POV pawns
        our_ks = -bd.get("king_safety", 0.0) / 1000.0
        ph = round(bd.get("phase_score", 0), 3)   # raw engine phase_score (0 opening .. 128 endgame); inspect distn
        sf11_total, terms = sf11.eval(fen)
        if sf11_total is None:                     # SF11 'eval' unparsable for this position
            continue
        sf11_ks = terms.get("King safety", 0.0)
        cls, kw, kb, wc, bc = geo_class(b)
    except Exception:
        continue
    rows.append({"fen": fen, "src": src, "phase_score": ph, "our_total": round(our_total, 3),
                 "our_ks": round(our_ks, 3), "sf11_total": round(sf11_total, 3), "sf11_ks": round(sf11_ks, 3),
                 "geo_class": cls, "kzone_w": kw, "kzone_b": kb, "in_check_w": wc, "in_check_b": bc, "sf18": ""})
sf11.close()

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(rows)

from collections import Counter
print("labeled %d positions -> %s" % (len(rows), OUT))
print("geo_class:", dict(Counter(r["geo_class"] for r in rows)))
# where SF11 sees KS and we read ~0 (candidate KS targets) vs both agree (working)
big_gap = sum(1 for r in rows if abs(r["sf11_ks"]) >= 1.0 and abs(r["our_ks"]) < 0.3)
agree   = sum(1 for r in rows if abs(r["sf11_ks"]) >= 0.5 and abs(r["our_ks"] - r["sf11_ks"]) < 0.5)
print("KS gap (SF11>=1.0, ours<0.3): %d   |   KS agree (both fire, close): %d" % (big_gap, agree))
