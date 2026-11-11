# -*- coding: utf-8 -*-
"""Bench-format copies ("fen | uci moves") of EPD suites for the search v2 identity check (diagnostics/_sv2_identity.sh):
wac_mirror, sts300, sts300_mirror, perpetual_probe, plus 48 random legal bare-king endings (KBNK / KQK / KRK / KBBK, 12 each,
seed 7).
  pyrun diagnostics/_sv2_sets.py <suites dir> <out dir>      (11-11: /mnt/e/chess_data/sv2_id)
"""
import sys, os, chess
SUITES = sys.argv[1]
OUT = sys.argv[2]
os.makedirs(OUT, exist_ok=True)
for name in ("wac_mirror.epd", "sts300.epd", "sts300_mirror.epd", "perpetual_probe.epd"):
    n = 0
    with open(os.path.join(OUT, name.replace(".epd", ".txt")), "w") as out:
        for line in open(os.path.join(SUITES, name), errors="replace"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                b, ops = chess.Board.from_epd(line)
            except Exception:
                continue
            bm = []
            for m in ops.get("bm", []) or []:
                try:
                    bm.append(m.uci() if isinstance(m, chess.Move) else b.parse_san(str(m)).uci())
                except Exception:
                    pass
            out.write("%s | %s\n" % (b.fen(), " ".join(bm) or "0000"))
            n += 1
    print(name, n)
# Bare-king endings: KBNK / KQK / KRK / KBBK, the side with the pieces to move, legal and not already mate.
import random
rng = random.Random(7)
sets = {"KBNK": "BN", "KQK": "Q", "KRK": "R", "KBBK": "BB"}
with open(os.path.join(OUT, "barekings.txt"), "w") as out:
    for sig, pcs in sets.items():
        k = 0
        while k < 12:
            b = chess.Board(None)
            sq = rng.sample(range(64), 2 + len(pcs))
            b.set_piece_at(sq[0], chess.Piece(chess.KING, chess.WHITE))
            b.set_piece_at(sq[1], chess.Piece(chess.KING, chess.BLACK))
            for s, p in zip(sq[2:], pcs):
                b.set_piece_at(s, chess.Piece(chess.Piece.from_symbol(p).piece_type, chess.WHITE))
            if sig == "KBBK" and (chess.square_file(sq[2]) + chess.square_rank(sq[2])) % 2 == (chess.square_file(sq[3]) + chess.square_rank(sq[3])) % 2:
                continue
            b.turn = rng.random() < 0.5
            if not b.is_valid() or b.is_game_over():
                continue
            out.write("%s | 0000\n" % b.fen())
            k += 1
print("barekings.txt", 48)
