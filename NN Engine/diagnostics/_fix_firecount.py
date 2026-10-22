# -*- coding: utf-8 -*-
"""Do the correctness fixes FIRE, and where? Self-play the UCI binary from start sets, fixes OFF vs ON, and count.

A narrow fix can only be judged where it fires (memory narrow-items-are-judged-where-they-fire) and a test of a
mechanism that never fires passes vacuously (a-detector-gate-passes-vacuously-unless-the-term-is-proved-to-fire).
For each start position this plays one self-play game per mode at a fixed node budget (deterministic, fresh process
per game) and reports, per start set:
  * the binary's FIXCNT counters (FIX_COUNT_LOG): nodes created at halfmove clock 100 (fifty), repetition hits (rep),
    iterations meeting the draw-exit condition (drawexit), mates caught at the q-depth cap (qsmate) — OFF vs ON;
  * how many games DIVERGE between the modes, and at which ply — the changed-rate of the fixes in play.

Run (WSL): bash <runner> pyrun diagnostics/_fix_firecount.py BIN=<uci binary> [N=30] [NODES=30000] [PLIES=200]
           [SETS=eg:selfplay/eg_critical_fens.txt,kp:selfplay/kp_dense_fens.txt,uho:/mnt/e/chess_data/fastchess/openings_uho.epd]
           [ON=FIX_HALFMOVE=1;FIX_REP_KEY=1;FIX_DRAW_EXIT=1;FIX_QS_CHECK=1] [SEED=7] [JOBS=2]
"""
import os, sys, random, re, statistics
from concurrent.futures import ThreadPoolExecutor
OPTS = {'N': '30', 'NODES': '30000', 'PLIES': '200', 'SEED': '7', 'JOBS': '2',
        'SETS': 'eg:selfplay/eg_critical_fens.txt,kp:selfplay/kp_dense_fens.txt,uho:/mnt/e/chess_data/fastchess/openings_uho.epd',
        'ON': 'FIX_HALFMOVE=1;FIX_REP_KEY=1;FIX_DRAW_EXIT=1;FIX_QS_CHECK=1'}
for a in sys.argv[1:]:
    if '=' in a:
        k, v = a.split('=', 1)
        OPTS[k] = v
import chess, chess.engine

BIN, N, NODES, PLIES, JOBS = OPTS['BIN'], int(OPTS['N']), int(OPTS['NODES']), int(OPTS['PLIES']), int(OPTS['JOBS'])
ON_ENV = dict(kv.split('=', 1) for kv in OPTS['ON'].split(';') if kv)
CNT_RE = re.compile(r'FIXCNT fifty=(\d+) rep=(\d+) drawexit=(\d+) qsmate=(\d+)')
TM_RE = re.compile(r'TMCNT moves=(\d+) total_s=([\d.e+-]+) wasted_s=([\d.e+-]+) cut=(\d+) cut_differs=(\d+)')
KEYS = ('fifty', 'rep', 'drawexit', 'qsmate')
# MOVETIME=<s> plays at a fixed time per move instead of NODES (needs a quiet box); OFF_ONLY=1 skips the ON pass (accounting only)
MOVETIME = float(OPTS.get('MOVETIME', 0))
OFF_ONLY = OPTS.get('OFF_ONLY', '0') == '1'
tm_totals = [0, 0.0, 0.0, 0, 0]


def load(path):
    boards = []
    for raw in open(path):
        raw = raw.strip()
        line = raw.split(';')[0].strip()
        if not line or line.startswith('#'):
            continue
        try:
            boards.append(chess.Board(line))
        except ValueError:
            try:
                boards.append(chess.Board.from_epd(raw)[0])     # EPD with opcodes (id "...";)
            except ValueError:
                pass
    return boards


def play(board, env_extra, tag):
    """One self-play game; returns (moves, result, counters)."""
    env = dict(os.environ, FIX_COUNT_LOG='1', **env_extra)
    errpath = '/tmp/fixcnt_%d_%s.err' % (os.getpid(), tag)
    b = board.copy()
    moves = []
    with open(errpath, 'w') as err:
        eng = chess.engine.SimpleEngine.popen_uci(BIN, env=env, stderr=err)
        try:
            eng.configure({'OwnBook': False, 'Debug': True})   # Debug=false routes the engine's std::cerr to nowhere
            while not b.is_game_over(claim_draw=True) and len(moves) < PLIES:
                mv = eng.play(b, chess.engine.Limit(time=MOVETIME) if MOVETIME > 0 else chess.engine.Limit(nodes=NODES)).move
                moves.append(mv.uci()); b.push(mv)
            result = b.result(claim_draw=True) if b.is_game_over(claim_draw=True) else '*'
        except chess.engine.EngineTerminatedError as e:
            # a crashed engine is a finding, not a harness failure: record how to reproduce it and keep going
            result = 'CRASH'
            print('  !! CRASH (%s) mode=%s start=%s moves=%s' % (e, tag, board.fen(), ' '.join(moves)), flush=True)
        finally:
            try:
                eng.quit()
            except chess.engine.EngineTerminatedError:
                pass
    if result == 'CRASH':
        tail = [l.rstrip() for l in open(errpath, errors='replace')][-6:]
        print('  !! stderr tail:\n     ' + '\n     '.join(tail), flush=True)
    last, tm = None, None
    for line in open(errpath, errors='replace'):
        m = CNT_RE.search(line)
        if m:
            last = m
        t = TM_RE.search(line)
        if t:
            tm = t
    if tm and tag.startswith('off'):
        g = tm.groups()
        for i, v in enumerate((int(g[0]), float(g[1]), float(g[2]), int(g[3]), int(g[4]))):
            tm_totals[i] += v
    os.remove(errpath)
    return moves, result, dict(zip(KEYS, (int(x) for x in last.groups()))) if last else dict.fromkeys(KEYS, 0)


rng = random.Random(int(OPTS['SEED']))
divergences = []   # (set, ply, fen at the first differing ply, move without fixes, move with fixes)
for spec in OPTS['SETS'].split(','):
    name, path = spec.split(':', 1)
    boards = load(path)
    sample = rng.sample(boards, min(N, len(boards)))
    with ThreadPoolExecutor(JOBS) as ex:
        offs = list(ex.map(lambda ib: play(ib[1], {}, 'off%d' % ib[0]), enumerate(sample)))
        ons = offs if OFF_ONLY else list(ex.map(lambda ib: play(ib[1], ON_ENV, 'on%d' % ib[0]), enumerate(sample)))
    div_plies, fired_games = [], dict.fromkeys(KEYS, 0)
    tot_off, tot_on = dict.fromkeys(KEYS, 0), dict.fromkeys(KEYS, 0)
    res_change = 0
    for start, (mo, ro, co), (mn, rn, cn) in zip(sample, offs, ons):
        k = next((i for i, (a, b) in enumerate(zip(mo, mn)) if a != b), None)
        if k is None and len(mo) != len(mn):
            k = min(len(mo), len(mn))
        if k is not None:
            div_plies.append(k)
            if k < len(mo) and k < len(mn):
                b = start.copy()
                for u in mo[:k]:
                    b.push_uci(u)
                divergences.append((name, k, b.fen(), mo[k], mn[k]))
        res_change += ro != rn
        crashes = sum(r == 'CRASH' for _, r, _ in offs + ons)
        for key in KEYS:
            tot_off[key] += co[key]; tot_on[key] += cn[key]
            fired_games[key] += (co[key] + cn[key]) > 0
    print('=== %s (%s): %d games/mode, %d nodes/move, <= %d plies' % (name, path, len(sample), NODES, PLIES))
    print('  diverged: %d / %d games%s; result changed: %d; engine crashes: %d' % (
        len(div_plies), len(sample),
        ' (divergence ply median %d, min %d)' % (statistics.median(div_plies), min(div_plies)) if div_plies else '', res_change,
        crashes))
    for key in KEYS:
        print('  %-9s OFF %8d   ON %8d   games where it fired (either mode): %d' % (key, tot_off[key], tot_on[key], fired_games[key]))
    sys.stdout.flush()

if tm_totals[0]:
    n, tot, wasted, cut, differs = tm_totals
    print('\n=== TIME ACCOUNTING (OFF pass, %s): %d moves, %.1f s total, %.1f s in cut iterations = %.1f%% of move time; '
          'cut on %d moves (%.0f%%); cut iteration had a DIFFERENT best-so-far move on %d (%.0f%% of cuts)' % (
              'movetime %.2f s' % MOVETIME if MOVETIME else '%d nodes' % NODES, n, tot, wasted, 100.0 * wasted / max(tot, 1e-9),
              cut, 100.0 * cut / n, differs, 100.0 * differs / max(cut, 1)))

# Move test at the divergence points: which move — without or with the fixes — does SF prefer? Expected score (WDL) of each
# move for the side to move, searched as the only root move at JUDGE_DEPTH. Registered 2026-10-21: the fixed move is better
# more often than worse, mean delta > 0.
SF = OPTS.get('SF')
if SF and divergences:
    import csv
    sf = chess.engine.SimpleEngine.popen_uci(SF)
    sf.configure({'Threads': 1, 'Hash': 64, 'UCI_ShowWDL': True})
    depth = int(OPTS.get('JUDGE_DEPTH', 16))

    def exp_of(board, uci):
        info = sf.analyse(board, chess.engine.Limit(depth=depth), root_moves=[chess.Move.from_uci(uci)])
        w = info.get('wdl')
        return w.pov(board.turn).expectation() if w is not None else 0.5

    rows = []
    for name, k, fen, m_off, m_on in divergences:
        b = chess.Board(fen)
        e_off, e_on = exp_of(b, m_off), exp_of(b, m_on)
        rows.append((name, k, fen, m_off, m_on, round(e_off, 3), round(e_on, 3), round(e_on - e_off, 3)))
    sf.quit()
    out = OPTS.get('OUT', '/mnt/e/chess_data/fix_divergences.csv')
    with open(out, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['set', 'ply', 'fen', 'move_off', 'move_on', 'E_off', 'E_on', 'delta'])
        w.writerows(rows)
    print('\n=== MOVE TEST at %d divergence points (SF depth %d, expected score for the mover) -> %s' % (len(rows), depth, out))
    for name in sorted({r[0] for r in rows}) + ['ALL']:
        d = [r[7] for r in rows if name == 'ALL' or r[0] == name]
        better, worse = sum(x > 0.02 for x in d), sum(x < -0.02 for x in d)
        se = statistics.stdev(d) / len(d) ** 0.5 if len(d) > 1 else 0
        print('  %-4s n=%3d  fixed better %3d  worse %3d  ~equal %3d   mean delta %+.4f (se %.4f)' % (
            name, len(d), better, worse, len(d) - better - worse, statistics.mean(d), se))
print('[firecount] DONE')
