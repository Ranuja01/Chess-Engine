# -*- coding: utf-8 -*-
"""Build a WIDE, tiered cploss corpus from annotated tournament PGNs.

The tournament harness writes one PGN per tag with every move annotated as
`{ <label> <eval>/d<depth> }`, and in a base-vs-candidate tournament BOTH sides are our engine -- so a mistake by
either colour is our mistake. That gives ~45 labelled decision points per game with no SF pass and no replay cost.

Tiers are derived from the GAME RECORD (the engine's own eval trace), deliberately NOT from cploss. If positions
were selected because cploss was high under some config, that config's loss on the tier would be inflated by
selection and every other config would look better through regression to the mean. Classifying from the trace
keeps selection independent of the metric that later scores it.

  general  - uniform sample of decision points            -> "is this config better overall"
  blunder  - the mover's own eval fell by >= BLUNDER_CP   -> "did this config fix our failures"
  stable   - eval steady while clearly winning            -> regression guard: keep what already works

Usage (via the runner's pyrun sub):
  pyrun diagnostics/build_cploss_wide.py [--out CSV] [--per-tier N] [--blunder-cp 100] [--stable-cp 15]
        [--min-adv 50] [--tags tag1,tag2,...]
Writes `fen,stratum` (the schema cploss_frozen.py expects) and prints tier counts.
"""
import argparse, csv, glob, io, os, random, re, sys

THIS = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(THIS)
sys.path.insert(0, ENGINE)
import chess, chess.pgn

# `{ base +0.49/d10 }` / `{ fast -1.09/d14 }` / `{ base +9999.90/d8 }`
COMMENT_RE = re.compile(r"(\S+)\s+([+-]?\d+\.\d+)\s*/\s*d(\d+)")
MATE_ABS = 9000.0          # |eval| at/above this is a mate score: excluded from swing math
MAX_PLY = 400


def parse_games(pgn_path):
    """Yield [(fen_before, mover_label, eval_cp, depth), ...] per game."""
    with open(pgn_path, encoding="utf-8", errors="replace") as fh:
        while True:
            try:
                game = chess.pgn.read_game(fh)
            except Exception:
                break
            if game is None:
                break
            board = game.board()
            rows, ply = [], 0
            for node in game.mainline():
                if ply > MAX_PLY:
                    break
                m = COMMENT_RE.search(node.comment or "")
                fen = board.fen()
                try:
                    board.push(node.move)
                except Exception:
                    break
                ply += 1
                if m:
                    label, ev, dep = m.group(1), float(m.group(2)), int(m.group(3))
                    rows.append((fen, label, ev, dep))
            if rows:
                yield rows


def parse_jsonl_games(game_dir):
    """Yield the same (fen_before, label, eval_pawns, depth) rows from vs_sf/gauntlet per-game jsonl.

    Two format differences from the PGN path: each line stores the FEN *after* its own move (so a move's decision
    point is the PREVIOUS line's FEN), and only OUR moves carry `our_pov_eval` -- opponent lines are marked
    `sf: true` and are skipped for scoring but still advance the FEN cursor. Evals are engine-internal
    millipawns (pawn=1000), converted to pawns so both sources share one threshold scale.
    """
    import json
    for path in sorted(glob.glob(os.path.join(game_dir, "game_*.jsonl"))):
        rows, prev_fen = [], None
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    ev = rec.get("our_pov_eval")
                    if ev is not None and prev_fen:
                        rows.append((prev_fen, "ours", float(ev) / 1000.0, int(rec.get("depth") or 0)))
                    if rec.get("fen"):
                        prev_fen = rec["fen"]
        except Exception:
            continue
        if rows:
            yield rows


TIERS = ("general", "inaccuracy", "mistake", "blunder", "found")


def classify(rows, inacc_cp, mistake_cp, blunder_cp, found_cp):
    """Tag each decision point by comparing a mover's eval to its OWN previous eval.

    Evals are from the mover's own point of view, so a FALL means that side just got worse by its own reckoning --
    it had been over-optimistic about the move it chose. Mate scores are skipped: the jump to +9999 is not an
    error signal and would swamp every threshold.

    Severity is GRADED rather than a single blunder flag. That matters because the signal that tracks Elo lives in
    severity, not in error count: measuring default vs qdelta-OFF, the >5% and >10% loss buckets were identical
    while >20% differed by 60%. A single threshold averages that distinction away.

      inaccuracy  small fall      mistake  medium fall      blunder  large fall
      found       eval ROSE sharply after our move -- a proxy for "the engine found something hard to see".
                  Regression guard with teeth: a pruning change is far likelier to cost us a hard-to-find move
                  than a routine one. (Weaker than the ideal shallow-vs-deep disagreement test, which the stored
                  PGNs cannot support -- they record only the final depth per move, not the PV per iteration.)
    """
    out = []
    prev = {}   # label -> that side's previous eval
    for fen, label, ev, dep in rows:
        tier = "general"
        if abs(ev) < MATE_ABS and label in prev:
            delta = ev - prev[label]          # negative = this side's position deteriorated
            if delta <= -blunder_cp / 100.0:
                tier = "blunder"
            elif delta <= -mistake_cp / 100.0:
                tier = "mistake"
            elif delta <= -inacc_cp / 100.0:
                tier = "inaccuracy"
            elif delta >= found_cp / 100.0:
                tier = "found"
        out.append((fen, tier, label, ev, dep))
        if abs(ev) < MATE_ABS:
            prev[label] = ev
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ENGINE, "selfplay", "tune_data", "cploss_corpus_wide.csv"))
    ap.add_argument("--per-tier", type=int, default=2000)
    ap.add_argument("--inacc-cp", type=int, default=40, help="eval fall marking an inaccuracy (centipawns)")
    ap.add_argument("--mistake-cp", type=int, default=100, help="eval fall marking a mistake")
    ap.add_argument("--blunder-cp", type=int, default=250, help="eval fall marking a blunder")
    ap.add_argument("--found-cp", type=int, default=75, help="eval RISE marking a hard-to-find move")
    ap.add_argument("--tags", default="", help="comma-separated game tags; default = every annotated tournament")
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    games_dir = os.path.join(ENGINE, "selfplay", "games")
    if args.tags:
        pgns = [os.path.join(games_dir, t.strip(), "tournament.pgn") for t in args.tags.split(",") if t.strip()]
        pgns = [p for p in pgns if os.path.exists(p)]
    else:
        pgns = sorted(glob.glob(os.path.join(games_dir, "*", "tournament.pgn")))
    if not pgns:
        print("no annotated tournament.pgn found under selfplay/games/"); return

    # vs_sf / gauntlet runs write per-game jsonl instead of an annotated PGN; that is where the historic
    # archive lives, so scan both shapes.
    jsonl_dirs = []
    if not args.tags:
        jsonl_dirs = sorted({os.path.dirname(p) for p in glob.glob(os.path.join(games_dir, "*", "game_*.jsonl"))})
    else:
        for t in args.tags.split(","):
            d = os.path.join(games_dir, t.strip())
            if glob.glob(os.path.join(d, "game_*.jsonl")):
                jsonl_dirs.append(d)

    buckets, n_games, n_pts = {t: [] for t in TIERS}, 0, 0
    seen = set()

    def absorb(rows, tag):
        nonlocal n_games, n_pts
        n_games += 1
        for fen, tier, label, ev, dep in classify(rows, args.inacc_cp, args.mistake_cp,
                                                  args.blunder_cp, args.found_cp):
            n_pts += 1
            key = " ".join(fen.split()[:4])      # dedupe on board+side+castling+ep
            if key in seen:
                continue
            seen.add(key)
            buckets[tier].append((fen, tag))

    for p in pgns:
        tag = os.path.basename(os.path.dirname(p))
        for rows in parse_games(p):
            absorb(rows, tag)
        print("  pgn   %-24s games=%d points=%d uniq=%d" % (tag, n_games, n_pts, len(seen)))
    for d in jsonl_dirs:
        tag = os.path.basename(d)
        for rows in parse_jsonl_games(d):
            absorb(rows, tag)
        print("  jsonl %-24s games=%d points=%d uniq=%d" % (tag, n_games, n_pts, len(seen)))

    rng = random.Random(args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["fen", "stratum"])
        # Shuffle the tiers together before writing: cploss_frozen's --limit takes a PREFIX, so tier-grouped rows
        # would make any limited run read one stratum only. Interleaved, a prefix stays balanced across tiers.
        chosen = []
        for tier in TIERS:
            pool = buckets[tier]
            rng.shuffle(pool)
            take = pool[: args.per_tier]
            chosen.extend((fen, tier) for fen, _tag in take)
            print("  %-8s pool=%-7d taken=%d" % (tier, len(pool), len(take)))
        rng.shuffle(chosen)
        for fen, tier in chosen:
            w.writerow([fen, tier])
        total = len(chosen)
    print("\nscanned %d games / %d decision points / %d unique positions" % (n_games, n_pts, len(seen)))
    print("wrote %d rows -> %s" % (total, args.out))
    print("next: cploss_frozen 10 12 0 all PRESET=LIGHTNING MAX_DEPTH=64 %s" % args.out)


if __name__ == "__main__":
    main()
