# -*- coding: utf-8 -*-
"""Sample positions from recorded self-play games and label each with our per-term static eval AND classical
Stockfish 11's per-term `eval` breakdown. Output feeds the conditioning fitter (tune_cond.py) and the scale
fitter (tune_fit.py).

Why SF11 (not SF18): SF11 is the last classical HCE — apples-to-apples with ours and, uniquely, its `eval`
command prints a LABELED per-term table (Material/Imbalance/Mobility/King safety/Threats/Passed/Space/...).
That per-term decomposition is the supervision signal: it tells each of OUR terms what a strong HCE reads in
the same position. We use SF11's total for the win/loss strata too, so only ONE engine is launched (half the
labelling cost, less WSL-interop surface). NOT NNUE distillation — classical HCE, in the original spirit.

Labels are STATIC (our `ev_breakdown` wraps the real eval; SF11's `eval` is its static leaf) so they are
depth-independent — old recorded games are a valid position source. Per-position columns also carry the cheap
DETECTOR inputs (det_*) and each tunable term's RAW value, so the conditioning fit can replay mod_gain offline
without re-running the engine.

Run in WSL from NN Engine/ (needs SF11 interop — keep it warm, see wsl-sf-interop-binfmt):
    python selfplay/tune_corpus.py --tag newstack_uho,placement_bundle --n 40000 \
        --out selfplay/tune_data/cond_corpus.csv --resume

--resume appends and skips FENs already in --out, so the run is killable/resumable (hand cores back anytime).
"""

import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import sys
import csv
import glob
import json
import random
import argparse

import chess

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, ENGINE_DIR)
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, os.path.join(ENGINE_DIR, "diagnostics"))

# Reuse the proven SF11 classical-eval term-table parser (diagnostics/eval_vs_sf11.py).
from eval_vs_sf11 import SF11Eval, SF11

# Our additive terms recorded per position (the conditioning cluster + the rest, to reconstruct the total).
# `material`/`pieces` carry the material/placement bulk; the tunable subset is chosen in the fitter.
TERMS = [
    "material", "pieces", "capture_gains", "passed_pawn_support", "latent_threat", "threats", "king_safety",
    "central", "imbalance_white", "imbalance_black", "pair_bonus", "piece_value_boost", "pawn_majority",
    "pawn_struct", "outpost", "mobility",
    # Per-piece placement (PST) contributions -> finer fit parameters than the aggregate "pieces" scale.
    "pt_pawns", "pt_knights", "pt_bishops", "pt_rooks", "pt_queens", "pt_kings",
]

# Cheap detector inputs the mod_gain conditioning hooks consume (already computed in ev_breakdown).
DETS = [
    "det_w_offense", "det_b_offense", "det_w_defense", "det_b_defense",
    "det_w_pieceval", "det_b_pieceval", "det_central", "det_pawn_count",
    "det_ks_units_w", "det_ks_units_b", "det_w_mobility", "det_b_mobility",
]

# SF11 labeled term -> our column name (White-POV pawns, Total-MG column). Missing terms -> blank.
SF11_TERMS = {
    "Material": "sf11_material", "Imbalance": "sf11_imbalance", "Mobility": "sf11_mobility",
    "King safety": "sf11_kingsafety", "Threats": "sf11_threats", "Passed": "sf11_passed",
    "Space": "sf11_space", "Pawns": "sf11_pawns", "Knights": "sf11_knights",
    "Bishops": "sf11_bishops", "Rooks": "sf11_rooks", "Queens": "sf11_queens",
}
SF11_COLS = ["sf11_total"] + list(SF11_TERMS.values())

RESULT_WHITE = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}
NEAR_EQUAL_CP = 150


def load_engine():
    from ChessAI import ChessAI
    seed = chess.Board()
    return ChessAI(None, None, seed, seed.turn)


def opposite_bishop_signal(board):
    """Classic opposite-coloured-bishops detector: 1 when each side has exactly one bishop and they sit on
    opposite square colours (the case where a material/placement edge is hardest to convert), else 0."""
    wb = board.bishops & board.occupied_co[chess.WHITE]
    bb = board.bishops & board.occupied_co[chess.BLACK]
    if chess.popcount(wb) != 1 or chess.popcount(bb) != 1:
        return 0
    w_sq = chess.lsb(wb)
    b_sq = chess.lsb(bb)
    w_light = (chess.BB_LIGHT_SQUARES >> w_sq) & 1
    b_light = (chess.BB_LIGHT_SQUARES >> b_sq) & 1
    return 1 if w_light != b_light else 0


def mirror_fen(fen):
    """Left-right (file a<->h) mirror — a true eval symmetry (kingside<->queenside). Castling rights become
    invalid after the flip, so clear them; the caller RE-LABELS the result rather than copying labels."""
    b = chess.Board(fen)
    m = b.transform(chess.flip_horizontal)
    m.castling_rights = 0
    return m.fen()


def sample_positions(tags, n, per_game, rng):
    """Collect up to `n` (fen, result_white, game_id) triples across the given tags, capped at `per_game` per
    game. game_id = "<tag>/game_xxx" so the fitter can hold out whole GAMES (positions from one game share the
    outcome label and are near-duplicates -> a by-position split leaks; by-game does not)."""
    jsonls = []
    for tag in tags:
        logdir = os.path.join(THIS_DIR, "games", tag)
        jsonls += sorted(glob.glob(os.path.join(logdir, "game_*", "game.jsonl")))
    rng.shuffle(jsonls)
    out = []
    for path in jsonls:
        if len(out) >= n:
            break
        try:
            lines = open(path).read().splitlines()
        except OSError:
            continue
        game_id = "/".join(path.replace("\\", "/").split("/games/")[-1].split("/")[:2])
        result_white = None
        fens = []
        for ln in lines:
            try:
                rec = json.loads(ln)
            except ValueError:
                continue
            t = rec.get("type")
            if t == "result":
                result_white = RESULT_WHITE.get(rec.get("result"))
            elif t == "move" and not rec.get("opening") and rec.get("fen"):
                fens.append(rec["fen"])
        if result_white is None or not fens:
            continue
        pick = fens if len(fens) <= per_game else rng.sample(fens, per_game)
        for fen in pick:
            out.append((fen, result_white, game_id))
    rng.shuffle(out)
    return out[:n]


def load_done_fens(path):
    """FENs already labelled in an existing --out file (for --resume)."""
    done = set()
    if not os.path.exists(path):
        return done
    try:
        with open(path, newline="") as fh:
            r = csv.DictReader(fh)
            for row in r:
                if row.get("fen"):
                    done.add(row["fen"])
    except OSError:
        pass
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="overnight_speed", help="comma-separated list of game tags")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--per-game", type=int, default=8)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", default=os.path.join(THIS_DIR, "tune_data", "cond_corpus.csv"))
    ap.add_argument("--resume", action="store_true", help="append and skip FENs already in --out")
    ap.add_argument("--mirror", action="store_true", help="also emit a left-right mirror of each position")
    ap.add_argument("--no-sf11", action="store_true",
                    help="skip SF11 per-term labelling (outcome-Texel fits OUR terms to result_white, not SF11) "
                         "-> ~10x faster/larger corpus. status strata come from our own eval sign; sf columns blank.")
    args = ap.parse_args()

    # Launch SF11 FIRST — the interop exe-launch is the binfmt-staleness failure point, so do it right at
    # WSL boot (freshest interop) before the slow 40k JSONL sampling. Doubles as a fail-fast warm-check.
    # --no-sf11 skips it entirely (outcome-Texel doesn't need SF11) -> no interop surface, ~10x faster.
    sf = None if args.no_sf11 else SF11Eval(SF11)
    ai = load_engine()

    tags = [t.strip() for t in args.tag.split(",") if t.strip()]
    if tags == ["ALL"]:   # every recorded self-play tag (broad outcome diversity; also the NNUE data superset)
        gdir = os.path.join(THIS_DIR, "games")
        tags = sorted(d for d in os.listdir(gdir) if os.path.isdir(os.path.join(gdir, d)))
        print("ALL tags -> %d game directories" % len(tags))
    rng = random.Random(args.seed)
    positions = sample_positions(tags, args.n, args.per_game, rng)
    if args.mirror:   # mirror shares the source game_id (same game, symmetric board) -> stays with it in the split
        positions += [(mirror_fen(f), r, g) for f, r, g in positions]
    print("sampled %d positions from tags=%s%s" % (len(positions), tags, " (+mirror)" if args.mirror else ""))

    done = load_done_fens(args.out) if args.resume else set()
    if done:
        print("resume: %d FENs already labelled in %s — skipping them" % (len(done), args.out))

    cols = (["fen", "game", "phase_score", "is_endgame", "status", "result_white", "our_total", "sf_static_cp"]
            + TERMS + DETS + ["oppb"] + SF11_COLS)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fresh = not (args.resume and os.path.exists(args.out))
    written = skipped = 0
    try:
        with open(args.out, "a" if not fresh else "w", newline="") as fh:
            w = csv.writer(fh)
            if fresh:
                w.writerow(cols)
            seen = set(done)
            for i, (fen, result_white, game_id) in enumerate(positions):
                if fen in seen:
                    skipped += 1
                    continue
                seen.add(fen)
                board = chess.Board(fen)
                if board.is_check():
                    skipped += 1
                    continue
                bd = ai.ev_breakdown(board)
                if bd.get("checkmate"):
                    skipped += 1
                    continue
                if sf is None:
                    # No SF11: status from OUR eval sign (Black-positive milli-pawns -> White-POV pawns = -v/1000).
                    wp = -bd["total"] / 1000.0
                    status = ("white_winning" if wp * 1000.0 >= NEAR_EQUAL_CP
                              else "black_winning" if wp * 1000.0 <= -NEAR_EQUAL_CP else "near_equal")
                    sf_cp = 0
                    sf_cols = [""] * len(SF11_COLS)
                else:
                    sf_total, sf_terms = sf.eval(fen)
                    if sf_total is None:
                        skipped += 1
                        continue
                    sf_cp = int(round(sf_total * 100.0))  # White-POV cp from SF11's total (status strata + tune_fit)
                    if sf_cp >= NEAR_EQUAL_CP:
                        status = "white_winning"
                    elif sf_cp <= -NEAR_EQUAL_CP:
                        status = "black_winning"
                    else:
                        status = "near_equal"
                    sf_cols = [round(sf_total, 3)] + [
                        (round(sf_terms[lbl], 3) if lbl in sf_terms else "") for lbl in SF11_TERMS
                    ]
                row = ([fen, game_id, bd["phase_score"], int(bd["is_endgame"]), status, result_white,
                        bd["total"], sf_cp]
                       + [bd[t] for t in TERMS]
                       + [bd[d] for d in DETS]
                       + [opposite_bishop_signal(board)]
                       + sf_cols)
                w.writerow(row)
                fh.flush()  # per-row flush => killable/resumable with zero loss
                written += 1
                if written % 500 == 0:
                    print("  %d/%d  (written %d, skipped %d)" % (i + 1, len(positions), written, skipped))
    finally:
        if sf is not None:
            sf.close()
    print("wrote %d rows (skipped %d) -> %s" % (written, skipped, args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
