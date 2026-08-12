# -*- coding: utf-8 -*-
"""Build a GAME-REPRESENTATIVE multi-PV regret set: sample real positions from the stored self-play game
jsonls (the natural distribution the engine actually faces), then cache SF18 top-K move evals ONCE.

Why games, not the curated corpus: `diverse_corpus_wide` over-represents STS/passer themes and overlaps the
STS validation suite. Real game positions are representative AND naturally disjoint from the benches -- the
clean, leak-free foundation for the broad regret program. Positions are deduped and excluded from the
move-match validation sets and the existing corpus regret/tune sets.

  pyrun diagnostics/_build_game_regret_set.py [N=6000] [K=8] [SF_DEPTH=14] [PLY_STRIDE=7]
        [OUT=ks_sets/game_regret_set.csv]   (needs STOCKFISH_PATH -> native-ELF Linux Stockfish)

Resumable: re-running skips FENs already in OUT.
"""
import os, sys, csv, json, glob, atexit
from collections import defaultdict

for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
THIS = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(THIS)
sys.path.insert(0, ENGINE); sys.path.insert(0, THIS)
sys.path.insert(0, os.path.join(ENGINE, "selfplay"))

import chess
import chess.engine
from arbiter import find_stockfish

N = int(os.environ.get("N", "6000"))
K = int(os.environ.get("K", "8"))
SF_DEPTH = int(os.environ.get("SF_DEPTH", "14"))
PLY_STRIDE = int(os.environ.get("PLY_STRIDE", "7"))
OUT = os.environ.get("OUT", "ks_sets/game_regret_set.csv")
if not os.path.isabs(OUT):
    OUT = os.path.join(THIS, OUT)
GAMES = os.path.join(ENGINE, "selfplay", "games")

# Exclude: move-match validation FENs + existing corpus tune/regret sets (keep everything disjoint).
exclude = set()
for rel, col in (("_mp_target.csv", "fen_start"), ("_mp_holdout.csv", "fen_start"),
                 ("ks_sets/lowdepth_tuneset.csv", "fen"), ("ks_sets/regret_set.csv", "fen"),
                 ("ks_sets/game_regret_set.csv", "fen")):   # keep a v2 set DISJOINT from the standing 15k
    p = os.path.join(THIS, rel)
    if os.path.exists(p):
        try:
            for r in csv.DictReader(open(p, newline="")):
                f = (r.get(col) or "").strip()
                if f:
                    exclude.add(f)
        except Exception:
            pass


def phase_of(board):
    pc = len(board.piece_map())
    return "opening" if pc >= 26 else ("midgame" if pc >= 14 else "endgame")


# Sample positions spread across MANY distinct games (diversity) and across plies within each game.
files = sorted(glob.glob(os.path.join(GAMES, "*", "*.jsonl")))
files = [f for f in files if "_archive" not in f and "_bak" not in f]
stride_f = max(1, len(files) // 2500)          # spread over up to ~2500 distinct games
sel_files = files[::stride_f]
# Optional file-SHARD (SHARD=i/n): split the selected game files across parallel labeler instances so spare
# cores can be used (each SF engine is single-threaded). Shards are disjoint by construction -> just concat.
_sh = os.environ.get("SHARD", "0/1")
_si, _sn = (int(x) for x in _sh.split("/"))
sel_files = sel_files[_si::_sn]

picked = []
seen = set()
for fp in sel_files:
    try:
        lines = open(fp).read().splitlines()
    except Exception:
        continue
    for ln in lines[::PLY_STRIDE]:
        try:
            fen = (json.loads(ln).get("fen") or "").strip()
        except Exception:
            continue
        if not fen or fen in seen or fen in exclude:
            continue
        try:
            board = chess.Board(fen)
        except Exception:
            continue
        if board.is_game_over():
            continue
        seen.add(fen)
        picked.append((fen, phase_of(board)))
    if len(picked) >= N:
        break
picked = picked[:N]

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

sfpath = os.environ.get("STOCKFISH_PATH") or find_stockfish()
eng = chess.engine.SimpleEngine.popen_uci(sfpath)
try:
    eng.configure({"Threads": 1})
except Exception:
    pass

print("  %d files, sampled %d positions (%d already done)  SF d%d multipv %d -> %s"
      % (len(sel_files), len(picked), len(done), SF_DEPTH, K, os.path.basename(OUT)), flush=True)

n_new = 0
for i, (fen, ph) in enumerate(picked, 1):
    if fen in done:
        continue
    board = chess.Board(fen)
    try:
        infos = eng.analyse(board, chess.engine.Limit(depth=SF_DEPTH), multipv=K)
    except Exception as e:
        print("  [%5d/%5d] SKIP %s" % (i, len(picked), type(e).__name__), flush=True)
        continue
    pairs = []
    for info in infos:
        pv = info.get("pv"); sc = info.get("score")
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
print("  by phase: %s" % "  ".join("%s=%d" % (k, ph_counts[k]) for k in sorted(ph_counts)), flush=True)
