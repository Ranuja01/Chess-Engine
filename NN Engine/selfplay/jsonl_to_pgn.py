# -*- coding: utf-8 -*-
"""Convert a recorded game.jsonl (or game.annotated.jsonl) into a verified python-chess PGN.

Replays the UCI moves through python-chess so the exported SAN is guaranteed legal (avoids any
hand-transcription / engine-writer errors). When the annotated JSONL is present, per-move comments
carry our White-POV eval and Stockfish's eval/best-move.

Run from NN Engine/selfplay/:
    python jsonl_to_pgn.py --tag jun6_standard --game 1
    python jsonl_to_pgn.py --path games/away_lightning/game_013/game.annotated.jsonl
"""
import os, json, argparse
import chess, chess.pgn

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def to_pgn(jsonl_path):
    recs = [json.loads(l) for l in open(jsonl_path) if l.strip()]
    meta = recs[0] if recs and recs[0].get("type") == "meta" else {}
    start = meta.get("start_fen") or chess.STARTING_FEN
    board = chess.Board(start)
    game = chess.pgn.Game()
    game.headers["Event"] = os.path.basename(os.path.dirname(jsonl_path))
    game.headers["White"] = meta.get("white", "?")
    game.headers["Black"] = meta.get("black", "?")
    if start != chess.STARTING_FEN:
        game.headers["FEN"] = start
        game.headers["SetUp"] = "1"
    node = game
    result = "*"
    for r in recs:
        if r.get("type") == "result":
            result = r.get("result", "*")
            continue
        if r.get("type") != "move" or not r.get("uci"):
            continue
        mv = chess.Move.from_uci(r["uci"])
        if mv not in board.legal_moves:
            raise ValueError(f"illegal {r['uci']} at ply {r.get('ply')} fen={board.fen()}")
        node = node.add_variation(mv)
        board.push(mv)
        # comment: our eval (pawns, White-POV) + SF if annotated
        parts = []
        ewp = r.get("eval_white_pov")
        if isinstance(ewp, int):
            parts.append(f"ours {ewp/1000:+.2f}/d{r.get('depth')}")
        if isinstance(r.get("sf_cp"), int):
            parts.append(f"SF {r['sf_cp']/100:+.2f}/d{r.get('sf_depth')} bm={r.get('sf_best')}")
        if parts:
            node.comment = "  ".join(parts)
    game.headers["Result"] = result
    return game


def main():
    ap = argparse.ArgumentParser(description="Recorded game.jsonl -> verified python-chess PGN.")
    ap.add_argument("--path", help="direct path to a game(.annotated).jsonl")
    ap.add_argument("--tag", help="tournament tag under games/")
    ap.add_argument("--game", type=int, help="game index (with --tag)")
    args = ap.parse_args()
    if args.path:
        path = args.path
    elif args.tag is not None and args.game is not None:
        gd = os.path.join(THIS_DIR, "games", args.tag, f"game_{args.game:03d}")
        path = os.path.join(gd, "game.annotated.jsonl")
        if not os.path.exists(path):
            path = os.path.join(gd, "game.jsonl")
    else:
        ap.error("need --path or (--tag and --game)")
    print(to_pgn(path), end="\n\n")


if __name__ == "__main__":
    main()
