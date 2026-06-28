# -*- coding: utf-8 -*-
"""Replay the chess.com 2200-bot conversion-failure PGN (our engine = White, won a pawn early then
drew the rook/pawn ending) and dump White-to-move FENs + a material tally for the endgame run-up, so
the collapse diagnosis can feed them to ourmove / fen_vs_sf / eval_breakdown.

Run (WSL, from NN Engine/):  /home/ranuja/anaconda3/bin/python diagnostics/_chesscom_gap_fens.py
Writes diagnostics/_chesscom_gap_fens.csv (movno,played_san,played_uci,wmat,bmat,fen) + prints a summary.
"""
import io, csv, os, chess, chess.pgn

PGN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "selfplay", "external",
                        "chesscom_2200_white.pgn")

# White-to-move fullmove ranges to dump: the whole convertible phase from the won pawn (29.Bxa7) into
# the rook ending and the final K+P-vs-K rook-pawn draw.
RANGES = [(29, 69)]

VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def material(board, color):
    return sum(VAL[pt] * len(board.pieces(pt, color)) for pt in VAL)


with open(PGN_PATH) as f:
    game = chess.pgn.read_game(f)

rows = []
board = game.board()
for mv in game.mainline_moves():
    if board.turn == chess.WHITE:
        mn = board.fullmove_number
        if any(lo <= mn <= hi for lo, hi in RANGES):
            rows.append({"movno": mn, "side": "w",
                         "played_san": board.san(mv), "played_uci": mv.uci(),
                         "wmat": material(board, chess.WHITE), "bmat": material(board, chess.BLACK),
                         "fen": board.fen()})
    board.push(mv)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_chesscom_gap_fens.csv")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["movno", "side", "played_san", "played_uci", "wmat", "bmat", "fen"])
    w.writeheader(); w.writerows(rows)

for r in rows:
    print(f"{r['movno']:>3}.  played={r['played_san']:7} ({r['played_uci']})  "
          f"W{r['wmat']}-B{r['bmat']} (+{r['wmat']-r['bmat']})  {r['fen']}")
print(f"\n{len(rows)} White-to-move FENs -> {out}")
