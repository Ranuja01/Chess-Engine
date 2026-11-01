# -*- coding: utf-8 -*-
"""TRANSITION starts: late-middlegame / early-endgame positions with a modest edge, for judging drawishness items (endgame
caps, the damp threshold, scaling) where they decide games (2026-11-01, owner).

A drawishness term pays mostly BEFORE the endgame, in the trade-down decision: the side ahead should avoid trading into an
ending the eval knows is drawn, the side behind should steer into one. UHO starts reach those choices rarely and eg_critical
starts after them, so neither venue tests it.

Source: fastchess PGNs of gen 6.4-era self-play (UHO, 10+0.1). Per game, one position chosen at random (seeded) among plies
past MIN_PLY whose engine phase256 lies in PH (non-pawn material of both sides, N 3250 B 3450 R 5000 Q 10000, between
EVAL_V2_EG_LIMIT 15,800 and EVAL_V2_MG_LIMIT 61,700 — phase 128 is the mgl / egl boundary), that are quiet (side to move not
in check; neither the move that reached it nor the move played from it a capture or promotion) and whose side to move's own
search score (the PGN comment of the move it played) has an edge |cp| within CP. At most one start per source opening, then N
sampled. Output: fastchess EPD (4 fields + id), with the phase / edge / class mix printed.

  python diagnostics/_make_transition_starts.py PGNS=E:/chess_data/fastchess/rfp450.pgn,... [OUT=E:/chess_data/fastchess/transition.epd]
         [N=500] [PH=64,160] [CP=40,200] [MIN_PLY=20] [SEED=1101]
"""
import os, sys, re, random, collections
import chess, chess.pgn

KV = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
PGNS = KV["PGNS"].split(",")
OUT = KV.get("OUT", "E:/chess_data/fastchess/transition.epd")
N = int(KV.get("N", "500"))
PH_LO, PH_HI = (int(x) for x in KV.get("PH", "64,160").split(","))
CP_LO, CP_HI = (int(x) for x in KV.get("CP", "40,200").split(","))
MIN_PLY = int(KV.get("MIN_PLY", "20"))
rng = random.Random(int(KV.get("SEED", "1101")))

NPM = {chess.KNIGHT: 3250, chess.BISHOP: 3450, chess.ROOK: 5000, chess.QUEEN: 10000}
LO, HI = 15800, 61700
SCORE = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)/\d+")


def phase256(b):
    npm = sum(len(b.pieces(pt, c)) * v for pt, v in NPM.items() for c in (chess.WHITE, chess.BLACK))
    return 0 if npm <= LO else 256 if npm >= HI else 256 * (npm - LO) // (HI - LO)


def quiet_move(b, m):
    return not b.is_capture(m) and m.promotion is None


by_opening = collections.defaultdict(list)
games = 0
for path in PGNS:
    with open(path, encoding="utf-8", errors="ignore") as f:
        while True:
            g = chess.pgn.read_game(f)
            if g is None:
                break
            games += 1
            opening = g.headers.get("FEN", "startpos")
            b = g.board()
            node, prev, ply, cands = g, None, 0, []
            while node.variations:
                nxt = node.variation(0)
                m = nxt.move
                if ply >= MIN_PLY and PH_LO <= phase256(b) <= PH_HI and not b.is_check() \
                        and (prev is None or prev) and quiet_move(b, m):
                    s = SCORE.match(nxt.comment or "")
                    if s:
                        cp = abs(float(s.group(1))) * 100
                        if CP_LO <= cp <= CP_HI:
                            cands.append((b.epd(), phase256(b), cp))
                prev = quiet_move(b, m)
                b.push(m)
                node, ply = nxt, ply + 1
            if cands:
                by_opening[opening].append(rng.choice(cands))

pool, seen = [], set()
for opening in sorted(by_opening):
    c = rng.choice(by_opening[opening])
    if c[0] not in seen:
        seen.add(c[0]); pool.append(c)
rng.shuffle(pool)
pick = pool[:N]
with open(OUT, "w", newline="\n") as f:
    for i, (epd, _, _) in enumerate(pick):
        f.write('%s id "tr%04d";\n' % (epd, i))

print("%d games read, %d openings with a candidate, %d starts written to %s" % (games, len(by_opening), len(pick), OUT))
ph = [p for _, p, _ in pick]; cp = [c for _, _, c in pick]
print("phase256: %d below 128 (early endgame) / %d at 128+ (late middlegame); quartiles %s" % (
    sum(p < 128 for p in ph), sum(p >= 128 for p in ph), sorted(ph)[len(ph) // 4::len(ph) // 4][:3]))
print("edge |cp|: quartiles %s" % [round(x) for x in sorted(cp)[len(cp) // 4::len(cp) // 4][:3]])
cls = collections.Counter()
for epd, _, _ in pick:
    b = chess.Board(epd + " 0 1")
    q =bool(b.pieces(chess.QUEEN, chess.WHITE) or b.pieces(chess.QUEEN, chess.BLACK))
    wb, bb = b.pieces(chess.BISHOP, chess.WHITE), b.pieces(chess.BISHOP, chess.BLACK)
    ocb = len(wb) == 1 and len(bb) == 1 and (chess.square_file(list(wb)[0]) + chess.square_rank(list(wb)[0])) % 2 \
        != (chess.square_file(list(bb)[0]) + chess.square_rank(list(bb)[0])) % 2
    cls["queens on" if q else "queens off"] += 1
    if ocb: cls["single opposite-coloured bishops"] += 1
    if b.pieces(chess.ROOK, chess.WHITE) and b.pieces(chess.ROOK, chess.BLACK): cls["rooks both sides"] += 1
print("classes: " + " · ".join("%s %d" % kv for kv in cls.most_common()))
