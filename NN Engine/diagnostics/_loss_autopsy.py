# -*- coding: utf-8 -*-
"""Batch autopsy of the games we LOST in a fastchess PGN: where did each game collapse, and why?

For every lost game, SF (default SF18) analyses each position at DEPTH with WDL on, giving our expected score E (W + D/2,
our POV) before and after every one of our moves. Labels are WDL, not cp (memory sf-cp-labels-are-material-scaled-use-wdl).

Pre-registered classification (2026-10-20, before any data was read):
  * lost_from_start  E < START_LOST at our first move (UHO openings are unbalanced on purpose).
  * decisive move    among our moves made while E_before >= ALIVE, the one with the largest drop E_before - E_after;
                     if that drop < BLUNDER it is a `drift` game, and the decisive move is the first one leaving E < ALIVE.
  * phase            of the position before the decisive move: non-pawn material of both sides (N,B 3, R 5, Q 9):
                     > 40 middlegame, 21-40 late_mg, 1-20 endgame, 0 pawn_ending.
  * kind             tactical if SF's PV after our move (<= 8 plies) wins >= 2 pawns of material from us or mates; else positional.
  * cause            our binary re-searches the position before the decisive move at 1 s (replay) and at LONG s:
                     search  - the LONG-s move is SF-acceptable (E_after >= E_before - ACCEPT): more search fixes it;
                     eval    - the LONG-s move is still a bad one: it persists with depth, so the eval misjudges it.
                     replay_differs flags a 1 s re-search that does not reproduce the game move (timing noise).
  * saw_it           our own in-game score (PGN comment) fell >= 1 pawn within our next 3 moves = we noticed only afterwards.

Run (WSL):  bash <runner> pyrun diagnostics/_loss_autopsy.py PGN=<pgn> OURS=<engine name in PGN> BIN=<uci binary>
            [SF=<stockfish path>] [DEPTH=14] [LONG=10] [LIMIT=<n games>] [OUT=<dir>]
Writes <OUT>/autopsy_games.csv (one row per lost game) and prints the summary tables.
"""
import os, sys, re, csv, time, collections
OPTS = {'DEPTH': '14', 'LONG': '10', 'LIMIT': '0', 'OUT': '/mnt/e/chess_data/autopsy',
        'START_LOST': '0.25', 'ALIVE': '0.25', 'BLUNDER': '0.10', 'ACCEPT': '0.05'}
for a in sys.argv[1:]:
    if '=' in a:
        k, v = a.split('=', 1)
        OPTS[k] = v
import chess, chess.pgn, chess.engine

PGN, OURS, BIN = OPTS['PGN'], OPTS['OURS'], OPTS['BIN']
SF = OPTS.get('SF') or os.environ.get('STOCKFISH_PATH')
DEPTH, LONG, LIMIT = int(OPTS['DEPTH']), float(OPTS['LONG']), int(OPTS['LIMIT'])
START_LOST, ALIVE = float(OPTS['START_LOST']), float(OPTS['ALIVE'])
BLUNDER, ACCEPT = float(OPTS['BLUNDER']), float(OPTS['ACCEPT'])
OUT = OPTS['OUT']; os.makedirs(OUT, exist_ok=True)
VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
SCORE_RE = re.compile(r'^\s*([+-]?(?:\d+\.\d+|M\d+|-M\d+))/(\d+)')


def material(board, colour):
    return sum(VAL[p.piece_type] * (1 if p.color == colour else -1) for p in board.piece_map().values())


def npm(board):
    return sum(VAL[p.piece_type] for p in board.piece_map().values() if p.piece_type not in (chess.PAWN, chess.KING))


def phase_of(board):
    n = npm(board)
    return 'middlegame' if n > 40 else 'late_mg' if n > 20 else 'endgame' if n > 0 else 'pawn_ending'


def our_score(comment):
    """Our in-game score in pawns, own POV (fastchess writes the engine's own view); mates clamp to +-20."""
    m = SCORE_RE.match(comment or '')
    if not m:
        return None
    s = m.group(1)
    if 'M' in s:
        return -20.0 if s.startswith('-') else 20.0
    return float(s)


sf = chess.engine.SimpleEngine.popen_uci(SF)
sf.configure({'Threads': 1, 'Hash': 64, 'UCI_ShowWDL': True})
cache = {}


def sf_info(board):
    key = board.fen()
    if key not in cache:
        cache[key] = sf.analyse(board, chess.engine.Limit(depth=DEPTH))
    return cache[key]


def expect(board, us):
    if board.is_checkmate():
        return 0.0 if board.turn == us else 1.0
    if board.is_game_over():
        return 0.5
    info = sf_info(board)
    w = info.get('wdl')
    return w.pov(us).expectation() if w is not None else (1.0 if info['score'].pov(us).score(mate_score=10**6) > 0 else 0.0)


def tactical(board_after, us):
    """SF's PV from the position after our move: does it win >= 2 pawns from us (or mate) within 8 plies?"""
    info = sf_info(board_after)
    if info['score'].pov(us).is_mate():
        return True
    b = board_after.copy(); m0 = material(b, us)
    for mv in info.get('pv', [])[:8]:
        b.push(mv)
    return material(b, us) - m0 <= -2


def ours_move(board, secs):
    eng = chess.engine.SimpleEngine.popen_uci(BIN)       # fresh process: no memory growth, no warm state
    try:
        eng.configure({'OwnBook': False})
        return eng.play(board, chess.engine.Limit(time=secs)).move
    finally:
        eng.quit()


rows, t0 = [], time.time()
with open(PGN) as fh:
    gi = 0
    while True:
        game = chess.pgn.read_game(fh)
        if game is None:
            break
        gi += 1
        h = game.headers
        us = chess.WHITE if h.get('White') == OURS else chess.BLACK if h.get('Black') == OURS else None
        if us is None:
            continue
        lost = (h.get('Result') == '0-1') if us == chess.WHITE else (h.get('Result') == '1-0')
        if not lost:
            continue
        # walk the game: per our move (ply, board_before, move, E_before, E_after, our_score)
        board, ours = game.board(), []
        for node in game.mainline():
            mv = node.move
            if board.turn == us:
                before = board.copy()
                board.push(mv)
                ours.append([len(ours), before, mv, expect(before, us), expect(board, us), our_score(node.comment)])
            else:
                board.push(mv)
        if not ours:
            continue
        r = {'game': gi, 'round': h.get('Round'), 'colour': 'white' if us == chess.WHITE else 'black',
             'termination': h.get('Termination', ''), 'plies': h.get('PlyCount', ''), 'start_E': round(ours[0][3], 3)}
        if ours[0][3] < START_LOST:
            r['class'] = 'lost_from_start'
            rows.append(r); print('[autopsy] game %d: lost_from_start E=%.2f' % (gi, ours[0][3]), flush=True)
            continue
        alive = [o for o in ours if o[3] >= ALIVE]
        dec = max(alive, key=lambda o: o[3] - o[4])
        if dec[3] - dec[4] < BLUNDER:
            r['class'] = 'drift'
            dec = next((o for o in ours if o[4] < ALIVE), dec)
        else:
            r['class'] = 'blunder'
        i, before, mv, eb, ea, sc = dec
        after = before.copy(); after.push(mv)
        best = sf_info(before).get('pv', [None])[0]
        later = [o[5] for o in ours[i + 1:i + 4] if o[5] is not None]
        r.update({'move_no': before.fullmove_number, 'fen': before.fen(), 'move': before.san(mv),
                  'sf_best': before.san(best) if best else '', 'E_before': round(eb, 3), 'E_after': round(ea, 3),
                  'drop': round(eb - ea, 3), 'phase': phase_of(before), 'npm': npm(before),
                  'kind': 'tactical' if tactical(after, us) else 'positional',
                  'our_score': sc, 'our_score_3later': later[-1] if later else '',
                  'saw_it': 'late' if (sc is not None and later and sc - min(later) >= 1.0) else 'no_drop_seen'})
        rep = ours_move(before, 1.0); lng = ours_move(before, LONG)
        b2 = before.copy(); b2.push(lng); e_long = expect(b2, us)
        r.update({'replay_1s': before.san(rep), 'replay_differs': int(rep != mv), 'long_move': before.san(lng),
                  'E_long': round(e_long, 3), 'cause': 'search' if e_long >= eb - ACCEPT else 'eval'})
        rows.append(r)
        print('[autopsy] game %d (%s): %s mv %d %s (SF %s) E %.2f->%.2f %s %s cause=%s  [%.0fs]' % (
            gi, r['colour'], r['class'], r['move_no'], r['move'], r['sf_best'], eb, ea, r['phase'], r['kind'],
            r['cause'], time.time() - t0), flush=True)
        if LIMIT and len(rows) >= LIMIT:
            break
sf.quit()

cols = ['game', 'round', 'colour', 'termination', 'plies', 'start_E', 'class', 'move_no', 'move', 'sf_best', 'E_before',
        'E_after', 'drop', 'phase', 'npm', 'kind', 'cause', 'replay_1s', 'replay_differs', 'long_move', 'E_long',
        'our_score', 'our_score_3later', 'saw_it', 'fen']
path = os.path.join(OUT, 'autopsy_games.csv')
with open(path, 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
    for r in rows:
        w.writerow({c: r.get(c, '') for c in cols})

print('\n=== %d lost games  (SF depth %d, long re-search %.0f s)  -> %s' % (len(rows), DEPTH, LONG, path))
for field in ('class', 'colour'):
    print('%-10s %s' % (field, dict(collections.Counter(r[field] for r in rows))))
judged = [r for r in rows if r['class'] != 'lost_from_start']
for field in ('phase', 'kind', 'cause', 'saw_it', 'replay_differs'):
    print('%-14s %s' % (field, dict(collections.Counter(r[field] for r in judged))))
print('\nphase x kind x cause:')
for k, n in sorted(collections.Counter((r['phase'], r['kind'], r['cause']) for r in judged).items(), key=lambda x: -x[1]):
    print('  %-12s %-10s %-7s %3d' % (k[0], k[1], k[2], n))
print('[autopsy] AUTOPSY DONE %.0fs' % (time.time() - t0))
