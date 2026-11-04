"""Build a venue of positions where the 50-move rule decides the game.

Source: every game in the given fastchess PGNs that ended "Draw by fifty moves rule". In such a game the last 100 half-moves
have no capture or pawn move, so the position CLOCK half-moves into that final stretch is one where the side to convert had
100 - CLOCK half-moves left to reset the clock and did not. Each such position is labelled by Stockfish at its real clock
(Stockfish knows the 50-move rule, so a high score means the win is still reachable in time). Positions where one side is
ahead by at least WIN_CP are kept, one per game, deduplicated by placement / side / castling / ep.

SOURCE=all widens the source to every game that reaches the clock at all (the first position with halfmove clock CLOCK,
whatever the result): positions with the 50-move clock running where one side is winning, converted or not. Games shorter
than CLOCK half-moves are skipped from their PlyCount tag; MAXPOS caps the positions labelled (a seeded random sample).

Output: OUT.pgn (fastchess openings with the real move number and halfmove clock, format=pgn) and OUT.csv
(index, fen, ahead = w|b, sf_cp from the side ahead, source pgn).

Usage (from NN Engine/, through the runner so STOCKFISH_PATH is set):
  bash selfplay/overnight_runner.sh pyrun diagnostics/_r50_venue.py OUT=/mnt/e/chess_data/fastchess/r50_crit \
       [PGNS='/mnt/e/chess_data/fastchess/*.pgn'] [CLOCK=78] [WIN_CP=200] [NODES=1000000] [JOBS=2] [SOURCE=fifty|all]
       [MAXPOS=0]
"""
import csv, glob, io, os, re, sys
from concurrent.futures import ThreadPoolExecutor

import chess, chess.engine, chess.pgn

args = dict(a.split('=', 1) for a in sys.argv[1:] if '=' in a)
OUT = args['OUT']
PGNS = args.get('PGNS', '/mnt/e/chess_data/fastchess/*.pgn')
CLOCK = int(args.get('CLOCK', 78))
WIN_CP = int(args.get('WIN_CP', 200))
NODES = int(args.get('NODES', 1000000))
JOBS = int(args.get('JOBS', 2))
SOURCE = args.get('SOURCE', 'fifty')
MAXPOS = int(args.get('MAXPOS', 0))
SF = os.environ.get('STOCKFISH_PATH') or args.get('SF')

# 1. candidate positions (text pre-filter, then python-chess on the survivors only)
cands, seen, n_games = [], set(), 0
for path in sorted(glob.glob(PGNS)):
    base = os.path.basename(path)
    if base.startswith(os.path.basename(OUT)) or base.startswith(('r50A_', 'r50B_')):
        continue   # never mine the venue's own games or the games played on it
    txt = open(path, errors='replace').read()
    for chunk in txt.split('[Event ')[1:]:
        fifty = 'Draw by fifty moves rule' in chunk
        if SOURCE == 'fifty' and not fifty:
            continue
        pc = re.search(r'\[PlyCount "(\d+)"\]', chunk)
        if pc and int(pc.group(1)) < CLOCK:
            continue
        n_games += 1
        game = chess.pgn.read_game(io.StringIO('[Event ' + chunk))
        if game is None:
            continue
        board = game.board()
        pos = None
        if SOURCE == 'fifty':
            # the position 100 - CLOCK half-moves before the end: keep a short window of recent positions
            from collections import deque
            recent = deque([board.copy(stack=False)], maxlen=100 - CLOCK + 1)
            for mv in game.mainline_moves():
                board.push(mv)
                recent.append(board.copy(stack=False))
            if board.halfmove_clock >= 100 and len(recent) == recent.maxlen:
                pos = recent[0]
        else:
            if board.halfmove_clock == CLOCK:
                pos = board.copy(stack=False)
            else:
                for mv in game.mainline_moves():
                    board.push(mv)
                    if board.halfmove_clock == CLOCK:
                        pos = board.copy(stack=False)
                        break
        if pos is None or pos.halfmove_clock != CLOCK or pos.is_game_over():
            continue
        key = pos.epd()
        if key in seen:
            continue
        seen.add(key)
        cands.append((pos.fen(), os.path.basename(path)))
print(f'{"fifty-move draws" if SOURCE == "fifty" else "games"} read: {n_games}; distinct positions at clock {CLOCK}: {len(cands)}',
      flush=True)
if MAXPOS and len(cands) > MAXPOS:
    import random
    random.Random(1478).shuffle(cands)
    cands = cands[:MAXPOS]
    print(f'  sampled {MAXPOS} of them', flush=True)

# 2. Stockfish label at the real clock
def label(item):
    fen, src = item
    eng = getattr(label_local, 'eng', None)
    if eng is None:
        eng = chess.engine.SimpleEngine.popen_uci(SF)
        eng.configure({'Threads': 1, 'Hash': 64})
        label_local.eng = eng
        engines.append(eng)
    info = eng.analyse(chess.Board(fen), chess.engine.Limit(nodes=NODES))
    return fen, src, info['score'].white().score(mate_score=100000)

import threading
label_local = threading.local()
engines = []
rows = []
with ThreadPoolExecutor(max_workers=JOBS) as ex:
    for k, (fen, src, cp) in enumerate(ex.map(label, cands)):
        if cp is not None and abs(cp) >= WIN_CP:
            rows.append((fen, 'w' if cp > 0 else 'b', abs(cp), src))
        if (k + 1) % 200 == 0:
            print(f'  labelled {k + 1} / {len(cands)}, kept {len(rows)}', flush=True)

# 3. write the venue
with open(OUT + '.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['index', 'fen', 'ahead', 'sf_cp', 'source'])
    for i, (fen, ahead, cp, src) in enumerate(rows):
        w.writerow([i, fen, ahead, cp, src])
with open(OUT + '.pgn', 'w') as f:
    for i, (fen, ahead, cp, src) in enumerate(rows):
        f.write('[Event "r50_crit %d"]\n[SetUp "1"]\n[FEN "%s"]\n\n*\n\n' % (i, fen))
wa = sum(1 for r in rows if r[1] == 'w')
print(f'venue: {len(rows)} positions (white ahead {wa}, black ahead {len(rows) - wa}); '
      f'SF cp median {sorted(r[2] for r in rows)[len(rows) // 2] if rows else 0} -> {OUT}.pgn / .csv', flush=True)
for eng in engines:
    eng.quit()
