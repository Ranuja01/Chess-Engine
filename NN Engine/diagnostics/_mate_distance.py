"""Mate-finding and mate-distance check for TT mate storage (FIX_TT_MATE) and mate-distance pruning (FIX_MDP).

MODE=find (default): positions where Stockfish reports a forced mate for the side to move (WAC by default; labels cached in
LABELS). Each arm searches each position from a fresh game at NODES and reports, per arm:
  found    - the final score is a mate for the side to move
  exact    - the reported distance equals Stockfish's
  shorter  - a distance SHORTER than Stockfish's: impossible when Stockfish's is the shortest, so a wrong stored distance
  longer   - a mate, but a longer one
  nodes to the first mate report (median over the positions every arm found), the "how fast" read.
MODE=play: basic mating endgames (KQK, KRK; random legal positions, the strong side to move, Stockfish distance known).
Each arm plays both sides at NODES per move in one process per game (the TT carries across moves, as in a real game) until
mate, a draw, or 120 plies, and reports mates, draws and the plies used beyond Stockfish's shortest mate.

Run (WSL, through the runner so STOCKFISH_PATH is set):
  bash selfplay/overnight_runner.sh pyrun diagnostics/_mate_distance.py BIN=<uci binary> [MODE=find|play]
       [ARMS='off:;ttm:FIX_TT_MATE=1;mdp:FIX_MDP=1;both:FIX_TT_MATE=1,FIX_MDP=1'] [NODES=1000000] [SF_NODES=5000000]
       [SUITE=diagnostics/suites/wac_uci.txt] [N=60] [SEED=11] [JOBS=2]
Arms are environment settings (comma-separated KEY=VALUE), one engine process per position or game.
"""
import csv, os, random, statistics, sys
from concurrent.futures import ThreadPoolExecutor

import chess, chess.engine

O = {'MODE': 'find', 'ARMS': 'off:;ttm:FIX_TT_MATE=1;mdp:FIX_MDP=1;both:FIX_TT_MATE=1,FIX_MDP=1', 'NODES': '1000000',
     'SF_NODES': '5000000', 'SUITE': 'diagnostics/suites/wac_uci.txt', 'N': '60', 'SEED': '11', 'JOBS': '2',
     'LABELS': '/mnt/e/chess_data/mate_labels_wac.csv'}
for a in sys.argv[1:]:
    if '=' in a:
        k, v = a.split('=', 1)
        O[k] = v
BIN, NODES, SF_NODES, JOBS = O['BIN'], int(O['NODES']), int(O['SF_NODES']), int(O['JOBS'])
SF = os.environ.get('STOCKFISH_PATH') or O.get('SF')
ARMS = []
for part in O['ARMS'].split(';'):
    name, _, kv = part.partition(':')
    ARMS.append((name, dict(x.split('=', 1) for x in kv.split(',') if x)))


def arm_env(knobs):
    env = dict(os.environ)
    env.update(knobs)
    return env


def sf_mate(board):
    """Stockfish's mate distance in moves for the side to move (positive = it mates), or None."""
    with chess.engine.SimpleEngine.popen_uci(SF) as sf:
        sf.configure({'Threads': 1, 'Hash': 64})
        info = sf.analyse(board, chess.engine.Limit(nodes=SF_NODES))
    m = info['score'].relative.mate()
    return m


def run_find():
    fens = []
    for raw in open(O['SUITE']):
        line = raw.split('|')[0].strip()
        if line and not line.startswith('#'):
            fens.append(line)
    labels = {}
    if os.path.exists(O['LABELS']):
        for r in csv.DictReader(open(O['LABELS'])):
            labels[r['fen']] = int(r['mate']) if r['mate'] else None
    todo = [f for f in fens if f not in labels]
    if todo:
        with ThreadPoolExecutor(max_workers=JOBS) as ex:
            for f, m in zip(todo, ex.map(lambda f: sf_mate(chess.Board(f)), todo)):
                labels[f] = m
        with open(O['LABELS'], 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['fen', 'mate'])
            for f in fens:
                w.writerow([f, '' if labels.get(f) is None else labels[f]])
    mates = [(f, labels[f]) for f in fens if labels.get(f) is not None and labels[f] > 0]
    print(f'positions: {len(fens)}; Stockfish mates for the side to move: {len(mates)} '
          f'(distance median {statistics.median([m for _, m in mates]) if mates else 0})', flush=True)

    def one(job):
        (name, knobs), (fen, m_sf) = job
        with chess.engine.SimpleEngine.popen_uci(BIN, env=arm_env(knobs), stderr=open(os.devnull, 'w')) as eng:
            eng.configure({'OwnBook': False})
            first, final = None, None
            with eng.analysis(chess.Board(fen), chess.engine.Limit(nodes=NODES)) as an:
                for info in an:
                    sc = info.get('score')
                    if sc is None:
                        continue
                    m = sc.relative.mate()
                    final = m
                    if m is not None and m > 0 and first is None:
                        first = info.get('nodes')
            return name, fen, m_sf, final, first

    jobs = [(arm, pos) for arm in ARMS for pos in mates]
    res = {}
    with ThreadPoolExecutor(max_workers=JOBS) as ex:
        for name, fen, m_sf, final, first in ex.map(one, jobs):
            res[(name, fen)] = (m_sf, final, first)
    common = [f for f, _ in mates if all(res[(a, f)][1] is not None and res[(a, f)][1] > 0 for a, _ in ARMS)]
    print(f'{"arm":6s} {"found":>6s} {"exact":>6s} {"shorter":>8s} {"longer":>7s}   nodes to first mate (median over the '
          f'{len(common)} found by every arm)')
    for name, _ in ARMS:
        found = exact = shorter = longer = 0
        for f, m_sf in mates:
            _, final, _ = res[(name, f)]
            if final is not None and final > 0:
                found += 1
                exact += final == m_sf
                shorter += final < m_sf
                longer += final > m_sf
        nodes = [res[(name, f)][2] for f in common if res[(name, f)][2] is not None]
        print(f'{name:6s} {found:6d} {exact:6d} {shorter:8d} {longer:7d}   {int(statistics.median(nodes)) if nodes else 0:,}')
    shorter_cases = [(a, f, res[(a, f)]) for a, _ in ARMS for f, _ in mates
                     if res[(a, f)][1] is not None and 0 < res[(a, f)][1] < res[(a, f)][0]]
    for a, f, (m_sf, final, _) in shorter_cases[:10]:
        print(f'  SHORTER than Stockfish: {a} mate {final} vs {m_sf}: {f}')


def random_endgame(rng, piece):
    while True:
        b = chess.Board(None)
        sq = rng.sample(range(64), 3)
        b.set_piece_at(sq[0], chess.Piece(chess.KING, chess.WHITE))
        b.set_piece_at(sq[1], chess.Piece(chess.KING, chess.BLACK))
        b.set_piece_at(sq[2], chess.Piece(piece, chess.WHITE))
        b.turn = chess.WHITE
        if b.is_valid() and not b.is_game_over() and not b.is_check():
            return b


def run_play():
    rng = random.Random(int(O['SEED']))
    n = int(O['N'])
    starts = [random_endgame(rng, chess.QUEEN) for _ in range(n // 2)] + [random_endgame(rng, chess.ROOK) for _ in range(n - n // 2)]
    with ThreadPoolExecutor(max_workers=JOBS) as ex:
        dist = list(ex.map(sf_mate, starts))
    print(f'starts: {n} (KQK {n // 2}, KRK {n - n // 2}); Stockfish distance median {statistics.median([d for d in dist if d])}',
          flush=True)

    def one(job):
        (name, knobs), (b0, m_sf) = job
        b = b0.copy()
        with chess.engine.SimpleEngine.popen_uci(BIN, env=arm_env(knobs), stderr=open(os.devnull, 'w')) as eng:
            eng.configure({'OwnBook': False})
            while not b.is_game_over(claim_draw=True) and b.ply() < 120:
                mv = eng.play(b, chess.engine.Limit(nodes=NODES), game=b0).move
                b.push(mv)
        mated = b.is_checkmate()
        return name, mated, b.ply(), m_sf

    jobs = [(arm, (s, d)) for arm in ARMS for s, d in zip(starts, dist) if d and d > 0]
    out = {name: [] for name, _ in ARMS}
    with ThreadPoolExecutor(max_workers=JOBS) as ex:
        for name, mated, plies, m_sf in ex.map(one, jobs):
            out[name].append((mated, plies, m_sf))
    print(f'{"arm":6s} {"mates":>6s} {"draws":>6s}   plies beyond Stockfish\'s shortest (mean, median, max, over mates)')
    for name, _ in ARMS:
        r = out[name]
        extra = [p - (2 * m - 1) for mated, p, m in r if mated]
        print(f'{name:6s} {sum(1 for x in r if x[0]):6d} {sum(1 for x in r if not x[0]):6d}   '
              f'{statistics.mean(extra) if extra else 0:.2f}, {statistics.median(extra) if extra else 0}, {max(extra) if extra else 0}')


run_find() if O['MODE'] == 'find' else run_play()
