# -*- coding: utf-8 -*-
"""Build a regret set from EXTERNAL PGNs (FICS games), same schema as game_regret_set.csv.

WHY (2026-09-09). Our regret sets are mined from OUR OWN self-play: one engine, one book. That caps two
things at once --

  YIELD:        `_build_game_regret_set.py` is limited to ~2500 self-play game files (hardcoded
                `len(files)//2500`), which yields only ~15k positions however high N goes.
  INDEPENDENCE: every position comes from the same engine's openings, so a term can score by matching our
                own habits. The whacky/960 corpus tests that, but it disables castling, which confounds
                "no book structure" with "no castled king" -- the two hypotheses we most need to separate.

`PGNs/` holds ~34k UNIQUE games across 300-450 distinct opponents. That is real structural variety with
REAL CASTLED KINGS, which is exactly the pair of properties the variant corpus cannot provide.

⚠️ NOT filtered on `WhiteIsComp`/`BlackIsComp`, deliberately. Most of these are engine games (FICS has a
computer section) and that was investigated and found NOT to be a defect:
  - the population only has to differ from OUR self-play, and 300+ distinct opponents does;
  - FICS engines run 2600-2800, ABOVE our ~1900-1950, so the play is better than our corpus's, not worse;
  - "mid-strength engines make mistakes" is a feature -- we want positions where a decision is at stake.
  ⚠️ Footnote, not a filter: `IFDStock` is a Stockfish derivative, so SF18 judging SF-family games is
  mildly on-distribution. Worth remembering if a result looks strange; not worth discarding 10k games over.
★ The real independence risk is CONCENTRATION, not engine-ness: in SuperSet.pgn `ArasanX` holds ~30% of
player slots, so its book recurs. `testgames7.pgn` (450 players, top 5.5%) and `testgames.pgn` (327,
includes real GM accounts) are far better spread. Prefer those; take SuperSet as bulk.

☠️ DEDUPE IS MANDATORY. `LargeSet - Copy.pgn` is byte-identical to `SuperSet.pgn`, and `LargeSet.pgn` to
`ficsgamesdb_search_393412.pgn` (md5-confirmed). Repeated engine openings duplicate positions too. Rows are
deduped BY FEN here and against the existing corpora, so a duplicate cannot inflate n and make the set look
more powerful than it is.

⚠️ KEEP THIS AS A SEPARATE CROSS-SET. Do NOT blend it into game_regret_set -- composition decides the
optimum and the mix ratio would become an invisible knob. It needs its OWN measured null before any arm is
read against it (nulls differ by >3pp between corpora on the same stratum label).

⚠️ SF18 @ d14, K=8 -- identical to the existing corpora, or the rows cannot pool. NOT SF19: the judge is
the target column.

  pyrun diagnostics/_build_pgn_regret_set.py [N=20000] [K=8] [SF_DEPTH=14] [PLY_STRIDE=7]
        [PGN_GLOB=../PGNs/testgames7.pgn] [SHARD=i/n] [OUT=ks_sets/pgn_regret_set.csv] [MIN_ELO=0]

Resumable: re-running skips FENs already in OUT. Flushes every 100 (atomic replace), so a kill loses <=99.
⚠️ With SHARD, give each shard its OWN OUT -- flush() rewrites the whole file from that process's rows, so
shards sharing one OUT would clobber each other. Shards are disjoint by game index; just concat.
"""
import os, sys, csv, glob, atexit
from collections import defaultdict

for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)

import chess
import chess.pgn
import chess.engine
from arbiter import find_stockfish

N = int(os.environ.get("N", "20000"))
K = int(os.environ.get("K", "8"))
SF_DEPTH = int(os.environ.get("SF_DEPTH", "14"))
PLY_STRIDE = int(os.environ.get("PLY_STRIDE", "7"))
MIN_ELO = int(os.environ.get("MIN_ELO", "0"))
OUT = os.environ.get("OUT", "ks_sets/pgn_regret_set.csv")
if not os.path.isabs(OUT):
    OUT = os.path.join(THIS, OUT)
PGN_GLOB = os.environ.get("PGN_GLOB", os.path.join(THIS, "..", "..", "PGNs", "*.pgn"))

# Keep disjoint from every existing corpus, exactly as the self-play builder does.
exclude = set()
for rel in ("ks_sets/game_regret_set.csv", "ks_sets/game_regret_set_v2.csv",
            "ks_sets/variant_regret_set.csv", "ks_sets/regret_set.csv",
            "ks_sets/lowdepth_tuneset.csv", "ks_sets/diverse_corpus_wide.csv"):
    p = os.path.join(THIS, rel)
    if os.path.exists(p):
        try:
            for r in csv.DictReader(open(p, newline="")):
                if r.get("fen"):
                    exclude.add(r["fen"])
        except Exception:
            pass


def phase_of(board):
    """IDENTICAL to _build_game_regret_set.py:56 -- a PAWN-INCLUSIVE total piece count, NOT our
    phase_score (which counts only 4Q+2R+1(B|N) and ignores pawns). The two disagree; keep them keyed the
    same way as the existing corpora or the strata are not comparable."""
    pc = len(board.piece_map())
    return "opening" if pc >= 26 else ("midgame" if pc >= 14 else "endgame")


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

_sh = os.environ.get("SHARD", "0/1")
_si, _sn = (int(x) for x in _sh.split("/"))

files = sorted(glob.glob(PGN_GLOB))
# md5-confirmed duplicate pairs; keep one of each so a position is not sampled (and paid for) twice.
_DUPES = ("LargeSet - Copy.pgn", "LargeSet.pgn")
files = [f for f in files if os.path.basename(f) not in _DUPES]

picked, seen, gi = [], set(), 0
src_counts = defaultdict(int)
for fp in files:
    try:
        fh = open(fp, encoding="utf-8", errors="replace")
    except Exception:
        continue
    while len(picked) < N:
        try:
            game = chess.pgn.read_game(fh)
        except Exception:
            continue
        if game is None:
            break
        gi += 1
        if gi % _sn != _si:            # shard by GAME INDEX -> disjoint by construction
            continue
        if MIN_ELO:
            try:
                if min(int(game.headers.get("WhiteElo", 0) or 0),
                       int(game.headers.get("BlackElo", 0) or 0)) < MIN_ELO:
                    continue
            except ValueError:
                pass
        board = game.board()
        for ply, mv in enumerate(game.mainline_moves()):
            board.push(mv)
            if ply % PLY_STRIDE:
                continue
            if board.is_game_over():
                continue
            fen = board.fen()
            if fen in seen or fen in exclude:
                continue
            seen.add(fen)
            picked.append((fen, phase_of(board)))
            src_counts[os.path.basename(fp)] += 1
            if len(picked) >= N:
                break
    fh.close()
    if len(picked) >= N:
        break

picked = [p for p in picked if p[0] not in done][:N]

sfpath = os.environ.get("STOCKFISH_PATH") or find_stockfish()
print("  %d pgn files, %d games scanned, sampled %d positions (%d already done)  SF d%d multipv %d -> %s"
      % (len(files), gi, len(picked), len(done), SF_DEPTH, K, os.path.basename(OUT)), flush=True)
for k, v in sorted(src_counts.items(), key=lambda kv: -kv[1])[:6]:
    print("      %-44s %6d" % (k, v), flush=True)

eng = chess.engine.SimpleEngine.popen_uci(sfpath)
n_new = 0
for i, (fen, ph) in enumerate(picked):
    board = chess.Board(fen)
    try:
        infos = eng.analyse(board, chess.engine.Limit(depth=SF_DEPTH), multipv=K)
    except Exception as e:
        print("  [%5d/%5d] SKIP %s" % (i, len(picked), type(e).__name__), flush=True)
        continue
    pairs = []
    for info in infos:
        pv = info.get("pv")
        sc = info.get("score")
        if not pv or sc is None:
            continue
        pairs.append((pv[0].uci(), sc.pov(chess.WHITE).score(mate_score=100000)))
    if not pairs:
        continue
    out_rows.append({"fen": fen, "phase_bucket": ph, "best_uci": pairs[0][0], "best_cp": pairs[0][1],
                     "moves": ";".join("%s:%d" % (u, c) for u, c in pairs)})
    n_new += 1
    if n_new % 100 == 0:
        flush()
        print("  [%5d/%5d] labelled %d new" % (i, len(picked), n_new), flush=True)

flush()
try:
    eng.quit()
except Exception:
    pass
ph_counts = defaultdict(int)
for r in out_rows:
    ph_counts[r.get("phase_bucket", "?")] += 1
print("\n  DONE: %d total -> %s" % (len(out_rows), OUT))
print("  phases: %s" % dict(ph_counts))
print("  ⚠️ SEPARATE cross-set. Measure its OWN null before reading any arm against it.")
