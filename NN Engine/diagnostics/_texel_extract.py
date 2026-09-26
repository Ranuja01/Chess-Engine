# -*- coding: utf-8 -*-
"""TEXEL DATASET, STAGE 1: positions + game results from our own stored v2 self-play games.

WHY (2026-09-25). A textbook Texel fit -- logistic of the static eval against GAME RESULTS, on QUIET positions,
over the whole vector including tapered PSTs -- has never been run on v1 or v2 (EVAL-V2-INVENTORY-2026-09-25.md
§3). The "corpus fit is anti-correlated with Elo" record was SF-label distillation over v1 knob subsets. This
builds the dataset for the recipe that was never tried, from games already on disk.

SOURCE. selfplay/games/<tag>_kNN/game_NNN/game.jsonl, written by tournament.py: a `meta` record (both configs),
one `move` record per ply (fen AFTER the move, `opening` flag, `eval_white_pov` of the d6 search), and a
`result` record. Only games whose BOTH configs are v2 (`V2_PRESET=shipped` or `EVAL_ARM=1`) are kept.

QUIET PROXY (stage 1, no engine needed). A position is kept if:
  - it is past the book (`opening` is false for the move that reached it) and past ply MIN_PLY;
  - the side to move is not in check;
  - the move the engine then PLAYED from it (a d6 search choice) is not a capture or promotion -- the standard
    "best move is quiet" filter: if a recapture or winning capture were pending, the search would have taken it.
Stage 2 (after the SPRTs free the machine) confirms quietness with the engine: static eval == qsearch value.

OUTPUT (gzip CSV, deliberately OUTSIDE OneDrive -- tens of MB of generated data must not sync):
  game_id, split, ply, fen, result_white (1 / 0.5 / 0), search_white_mp (d6 score of the move played from here,
  White POV, empty if none), pieces (non-king piece count), phase_hint (non-pawn material count).
`split` is decided per GAME (hash of game_id): a by-game holdout, so train and validation never share a game.

  python diagnostics/_texel_extract.py [TAGS=spsaeval3,spsarun2,spsaks1,spsaks2] [OUT=E:/chess_data/texel/v2_stage1.csv.gz]
                                       [MIN_PLY=16] [HOLDOUT_PCT=10]
"""
import os, sys, json, gzip, csv, hashlib, glob, time
import chess

ENGINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAMES_DIR = os.path.join(ENGINE_DIR, "selfplay", "games")
KV = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
TAGS = KV.get("TAGS", "spsaeval3,spsarun2,spsaks1,spsaks2").split(",")
OUT = KV.get("OUT", "E:/chess_data/texel/v2_stage1.csv.gz")
MIN_PLY = int(KV.get("MIN_PLY", 16))
HOLDOUT_PCT = int(KV.get("HOLDOUT_PCT", 10))
RESULT_WHITE = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}


def is_v2(cfg):
    return "V2_PRESET=shipped" in cfg or "EVAL_ARM=1" in cfg


def split_of(game_id):
    h = int(hashlib.md5(game_id.encode()).hexdigest()[:8], 16)
    return "val" if (h % 100) < HOLDOUT_PCT else "train"


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    stats = {"games": 0, "skipped_not_v2": 0, "skipped_no_result": 0, "skipped_bad": 0,
             "positions_seen": 0, "kept": 0, "drop_book": 0, "drop_check": 0, "drop_noisy": 0, "drop_last": 0}
    results = {1.0: 0, 0.5: 0, 0.0: 0}
    t0 = time.time()
    with gzip.open(OUT, "wt", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["game_id", "split", "ply", "fen", "result_white", "search_white_mp", "pieces", "phase_hint"])
        for tag in TAGS:
            for gdir in sorted(glob.glob(os.path.join(GAMES_DIR, tag + "_k*", "game_*"))):
                jf = os.path.join(gdir, "game.jsonl")
                if not os.path.exists(jf):
                    continue
                try:
                    recs = [json.loads(l) for l in open(jf) if l.strip()]
                except Exception:
                    stats["skipped_bad"] += 1
                    continue
                meta = next((r for r in recs if r.get("type") == "meta"), None)
                res = next((r for r in recs if r.get("type") == "result"), None)
                if not meta or not is_v2(meta.get("config_white", "")) or not is_v2(meta.get("config_black", "")):
                    stats["skipped_not_v2"] += 1
                    continue
                if not res or res.get("result") not in RESULT_WHITE:
                    stats["skipped_no_result"] += 1
                    continue
                rw = RESULT_WHITE[res["result"]]
                game_id = os.path.relpath(gdir, GAMES_DIR).replace("\\", "/")
                sp = split_of(game_id)
                stats["games"] += 1
                results[rw] += 1
                moves = [r for r in recs if r.get("type") == "move"]
                # moves[i]["fen"] is the position AFTER move i; the move played FROM it is moves[i+1].
                for i in range(len(moves) - 1):
                    cur, nxt = moves[i], moves[i + 1]
                    stats["positions_seen"] += 1
                    if cur.get("opening") or nxt.get("opening") or cur.get("ply", 0) < MIN_PLY:
                        stats["drop_book"] += 1
                        continue
                    uci = nxt.get("uci")
                    if not uci or not cur.get("fen"):
                        stats["drop_last"] += 1
                        continue
                    try:
                        b = chess.Board(cur["fen"])
                        mv = chess.Move.from_uci(uci)
                    except Exception:
                        stats["drop_last"] += 1
                        continue
                    if b.is_check():
                        stats["drop_check"] += 1
                        continue
                    if b.is_capture(mv) or mv.promotion:
                        stats["drop_noisy"] += 1
                        continue
                    sw = nxt.get("eval_white_pov")
                    pieces = len(b.piece_map()) - 2
                    npm = sum(len(b.pieces(pt, c)) * v for pt, v in ((chess.KNIGHT, 1), (chess.BISHOP, 1),
                              (chess.ROOK, 2), (chess.QUEEN, 4)) for c in (chess.WHITE, chess.BLACK))
                    w.writerow([game_id, sp, cur.get("ply"), cur["fen"], rw,
                                "" if sw is None else int(sw), pieces, npm])
                    stats["kept"] += 1
            print("[texel] %s done: %d games, %d kept, %.0fs" % (tag, stats["games"], stats["kept"], time.time() - t0),
                  flush=True)
    print("[texel] DONE ->", OUT)
    print("[texel]", json.dumps(stats))
    n = max(1, stats["games"])
    print("[texel] results  W %.1f%%  D %.1f%%  L %.1f%%" % (100 * results[1.0] / n, 100 * results[0.5] / n,
                                                          100 * results[0.0] / n))
    return 0


if __name__ == "__main__":
    sys.exit(main())
