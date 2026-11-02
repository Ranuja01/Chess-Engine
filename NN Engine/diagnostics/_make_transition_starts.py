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

NEAR=caps (2026-11-02): starts around the onset of an endgame-cap class instead, for judging the caps where they fire. A game's
onset is the first ply from which a cap class holds for PERSIST plies in a row (so a half-finished exchange does not count);
the classes mirror eval_v2.cpp win_scale_adjust: opposite-coloured bishops with other pieces on, queen vs no queen, the
one-flank rook ending (leader at most one pawn up, defending king touching one of its pawns) and a pawnless leader with at most
a lone minor; the leader is the side the mover's score favours. Candidates are the quiet plies within WIN (plies relative to the
onset, e.g. -10,6 = five moves before to three moves after) that pass CP; PH then applies to the onset ply, not the start.
EXCLUDE drops starts already in another EPD file (e.g. the transition starts).

  python diagnostics/_make_transition_starts.py PGNS=E:/chess_data/fastchess/rfp450.pgn,... [OUT=E:/chess_data/fastchess/transition.epd]
         [N=500] [PH=64,160] [CP=40,200] [MIN_PLY=20] [SEED=1101] [ID=tr]
         [NEAR=caps WIN=-10,6 PERSIST=6] [EXCLUDE=E:/chess_data/fastchess/transition.epd]
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
ID = KV.get("ID", "tr")
NEAR = KV.get("NEAR", "")
WIN_LO, WIN_HI = (int(x) for x in KV.get("WIN", "-10,6").split(","))
PERSIST = int(KV.get("PERSIST", "6"))
EXCLUDE = set()
if KV.get("EXCLUDE"):
    with open(KV["EXCLUDE"], encoding="utf-8") as f:
        EXCLUDE = {" ".join(line.split()[:4]) for line in f if line.strip()}

NPM = {chess.KNIGHT: 3250, chess.BISHOP: 3450, chess.ROOK: 5000, chess.QUEEN: 10000}
LO, HI = 15800, 61700
SCORE = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)/\d+")
QS, KS = chess.BB_FILE_A | chess.BB_FILE_B | chess.BB_FILE_C | chess.BB_FILE_D, \
    chess.BB_FILE_E | chess.BB_FILE_F | chess.BB_FILE_G | chess.BB_FILE_H


def phase256(b):
    npm = sum(len(b.pieces(pt, c)) * v for pt, v in NPM.items() for c in (chess.WHITE, chess.BLACK))
    return 0 if npm <= LO else 256 if npm >= HI else 256 * (npm - LO) // (HI - LO)


def quiet_move(b, m):
    return not b.is_capture(m) and m.promotion is None


def light(sq):
    return (chess.square_file(sq) + chess.square_rank(sq)) % 2


def cap_classes(b, cp):
    """Cap classes holding on board b; cp = the mover's signed score in cp (None if unknown) picks the leader."""
    W, B = chess.WHITE, chess.BLACK
    out = []
    wb, bb = b.pieces(chess.BISHOP, W), b.pieces(chess.BISHOP, B)
    others = b.knights | b.rooks | b.queens
    if len(wb) == 1 and len(bb) == 1 and others and light(next(iter(wb))) != light(next(iter(bb))):
        out.append("ocbx")
    if chess.popcount(b.queens) == 1:
        out.append("qnoq")
    if cp:
        lead = b.turn if cp > 0 else not b.turn
        own, them = b.occupied_co[lead], b.occupied_co[not lead]
        sp = b.pawns & own
        if not (b.knights | b.bishops | b.queens) and chess.popcount(b.rooks & own) == 1 \
                and chess.popcount(b.rooks & them) == 1:
            tk = b.king(not lead)
            if chess.popcount(sp) - chess.popcount(b.pawns & them) <= 1 and sp and bool(sp & KS) != bool(sp & QS) \
                    and tk is not None and chess.BB_KING_ATTACKS[tk] & b.pawns & them:
                out.append("rooke")
        if not sp and not ((b.rooks | b.queens) & own) and chess.popcount((b.knights | b.bishops) & own) <= 1:
            out.append("loneminor")
    return out


by_opening = collections.defaultdict(list)
games = onsets = 0
onset_cls = collections.Counter()
for path in PGNS:
    with open(path, encoding="utf-8", errors="ignore") as f:
        while True:
            g = chess.pgn.read_game(f)
            if g is None:
                break
            games += 1
            opening = g.headers.get("FEN", "startpos")
            b = g.board()
            node, prev, ply, rows = g, None, 0, []
            while node.variations:
                nxt = node.variation(0)
                m = nxt.move
                ok = not b.is_check() and (prev is None or prev) and quiet_move(b, m)
                s = SCORE.match(nxt.comment or "")
                cp = float(s.group(1)) * 100 if s else None
                rows.append((b.epd(), phase256(b), ok, cp, cap_classes(b, cp) if NEAR == "caps" else []))
                prev = quiet_move(b, m)
                b.push(m)
                node, ply = nxt, ply + 1
            if NEAR == "caps":
                t = next((i for i in range(len(rows) - PERSIST + 1) if all(rows[j][4] for j in range(i, i + PERSIST))), None)
                if t is None or not PH_LO <= rows[t][1] <= PH_HI:
                    continue
                onsets += 1
                onset_cls.update(rows[t][4])
                cands = [(r[0], r[1], abs(r[3]), i - t, rows[t][4][0]) for i, r in enumerate(rows)
                         if t + WIN_LO <= i <= t + WIN_HI and i >= MIN_PLY and r[2] and r[3] is not None
                         and CP_LO <= abs(r[3]) <= CP_HI and r[0] not in EXCLUDE]
            else:
                cands = [(r[0], r[1], abs(r[3])) for i, r in enumerate(rows)
                         if i >= MIN_PLY and PH_LO <= r[1] <= PH_HI and r[2] and r[3] is not None
                         and CP_LO <= abs(r[3]) <= CP_HI and r[0] not in EXCLUDE]
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
    for i, c in enumerate(pick):
        f.write('%s id "%s%04d";\n' % (c[0], ID, i))

print("%d games read, %d openings with a candidate, %d starts written to %s" % (games, len(by_opening), len(pick), OUT))
ph = [c[1] for c in pick]; cp = [c[2] for c in pick]
print("phase256: %d below 128 (early endgame) / %d at 128+ (late middlegame); quartiles %s" % (
    sum(p < 128 for p in ph), sum(p >= 128 for p in ph), sorted(ph)[len(ph) // 4::len(ph) // 4][:3]))
print("edge |cp|: quartiles %s" % [round(x) for x in sorted(cp)[len(cp) // 4::len(cp) // 4][:3]])
if NEAR == "caps":
    print("%d games with a cap-class onset (PH at onset %d-%d); onset classes: %s" % (
        onsets, PH_LO, PH_HI, " · ".join("%s %d" % kv for kv in onset_cls.most_common())))
    print("picked starts by onset class: %s" % " · ".join(
        "%s %d" % kv for kv in collections.Counter(c[4] for c in pick).most_common()))
    off = collections.Counter(c[3] for c in pick)
    print("plies from onset: %d before, %d at or after; by ply %s" % (
        sum(v for k, v in off.items() if k < 0), sum(v for k, v in off.items() if k >= 0), sorted(off.items())))
cls = collections.Counter()
for c in pick:
    b = chess.Board(c[0] + " 0 1")
    q =bool(b.pieces(chess.QUEEN, chess.WHITE) or b.pieces(chess.QUEEN, chess.BLACK))
    wb, bb = b.pieces(chess.BISHOP, chess.WHITE), b.pieces(chess.BISHOP, chess.BLACK)
    ocb = len(wb) == 1 and len(bb) == 1 and (chess.square_file(list(wb)[0]) + chess.square_rank(list(wb)[0])) % 2 \
        != (chess.square_file(list(bb)[0]) + chess.square_rank(list(bb)[0])) % 2
    cls["queens on" if q else "queens off"] += 1
    if ocb: cls["single opposite-coloured bishops"] += 1
    if b.pieces(chess.ROOK, chess.WHITE) and b.pieces(chess.ROOK, chess.BLACK): cls["rooks both sides"] += 1
print("classes: " + " · ".join("%s %d" % kv for kv in cls.most_common()))
