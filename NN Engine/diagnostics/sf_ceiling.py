# -*- coding: utf-8 -*-
"""SF fixed-depth CEILING reference on our benches.

Answers "how much does even Stockfish solve at depth N, and why does it miss the rest?"
— the reference ceiling for our own fixed-depth WAC/STS numbers. SF at a given depth
searches a FAR narrower tree than we do (EBF ~2 vs ~4.65), so the same "depth 10" label
means very different things; the node counts below make that concrete.

For WAC it also splits SF's misses into:
  - DEPTH-GATED   : SF plays the bm once re-run at the higher depth (literally needed depth)
  - NOT-DEPTH     : SF still plays a different move even deep — usually it prefers another
                    (often also-winning) move, i.e. disagrees with the puzzle's nominal answer,
                    NOT a search failure.

Run (WSL, from NN Engine/):
    /home/ranuja/anaconda3/bin/python diagnostics/sf_ceiling.py [d_low=10] [d_high=20]

SF is full strength (UCI_LimitStrength off), single-thread, fixed hash, no book/TB — a clean,
reproducible anchor. Pin the SF version (printed below) for cross-time comparisons.
"""

import os
import sys
import chess
import chess.engine

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SUITES = os.path.join(THIS_DIR, 'suites')
SF = "/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/stockfish/stockfish-windows-x86-64-avx2.exe"

D_LOW = int(sys.argv[1]) if len(sys.argv) > 1 else 10
D_HIGH = int(sys.argv[2]) if len(sys.argv) > 2 else 20


def load_bm(path):
    """[(fen, {bm_uci,...}, id)] from a WAC-style EPD (bm operation)."""
    out = []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        b = chess.Board()
        try:
            ops = b.set_epd(line)
        except Exception:
            continue
        bm = ops.get('bm')
        if not bm:
            continue
        out.append((b.fen(), {m.uci() for m in bm}, ops.get('id', '')))
    return out


def load_sts(path):
    """[(fen, {uci:score}, max, id)] from an STS EPD (c8/c9 ops)."""
    import re
    c8re, c9re, idre = re.compile(r'\bc8\s+"([^"]*)"'), re.compile(r'\bc9\s+"([^"]*)"'), re.compile(r'\bid\s+"([^"]*)"')
    out = []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        c8, c9 = c8re.search(line), c9re.search(line)
        if not (c8 and c9):
            continue
        scores, moves = c8.group(1).split(), c9.group(1).split()
        if len(scores) != len(moves):
            continue
        try:
            sm = {mv: int(sc) for mv, sc in zip(moves, scores)}
        except ValueError:
            continue
        b = chess.Board()
        try:
            b.set_epd(line)
        except Exception:
            continue
        idm = idre.search(line)
        out.append((b.fen(), sm, max(sm.values()), idm.group(1) if idm else ''))
    return out


def best_move_and_nodes(eng, board, depth):
    info = eng.analyse(board, chess.engine.Limit(depth=depth))
    pv = info.get('pv') or []
    mv = pv[0].uci() if pv else None
    return mv, int(info.get('nodes', 0) or 0), int(info.get('depth', 0) or 0)


def main():
    eng = chess.engine.SimpleEngine.popen_uci(SF)
    eng.configure({'Threads': 1, 'Hash': 64})
    print(f"SF = {eng.id.get('name')}  |  d_low={D_LOW}  d_high={D_HIGH}  (single-thread, 64MB, no book/TB)\n")

    # ---- WAC ----
    wac = load_bm(os.path.join(SUITES, 'wac.epd'))
    solved = 0
    total_nodes = 0
    misses = []
    for fen, bm, pid in wac:
        mv, nodes, _ = best_move_and_nodes(eng, chess.Board(fen), D_LOW)
        total_nodes += nodes
        if mv in bm:
            solved += 1
        else:
            misses.append((fen, bm, pid, mv))
    n = len(wac)
    print(f"WAC @ d{D_LOW}: solved {solved}/{n}  |  total nodes {total_nodes:,}  mean {total_nodes // max(1,n):,}/pos")

    depth_gated = 0
    not_depth = []
    for fen, bm, pid, lowmv in misses:
        mv, _, _ = best_move_and_nodes(eng, chess.Board(fen), D_HIGH)
        if mv in bm:
            depth_gated += 1
        else:
            not_depth.append((pid, lowmv, mv, sorted(bm)))
    print(f"  of {len(misses)} misses @ d{D_LOW}: {depth_gated} become solved @ d{D_HIGH} (DEPTH-GATED), "
          f"{len(not_depth)} still differ (NOT depth — SF prefers another move)")
    if not_depth:
        print(f"  not-depth sample (id: SF_d{D_LOW} -> SF_d{D_HIGH} vs bm):")
        for pid, lowmv, highmv, bm in not_depth[:20]:
            print(f"    {pid}: {lowmv} -> {highmv}  vs {','.join(bm)}")

    # ---- STS ----
    sts_path = os.path.join(SUITES, 'sts300.epd')
    if os.path.exists(sts_path):
        sts = load_sts(sts_path)
        pts = mx = sts_nodes = 0
        for fen, sm, m, pid in sts:
            mv, nodes, _ = best_move_and_nodes(eng, chess.Board(fen), D_LOW)
            sts_nodes += nodes
            pts += sm.get(mv, 0)
            mx += m
        pct = 100.0 * pts / mx if mx else 0.0
        print(f"\nSTS @ d{D_LOW}: {pts}/{mx} ({pct:.1f}%)  |  total nodes {sts_nodes:,}  mean {sts_nodes // max(1,len(sts)):,}/pos")

    eng.quit()


if __name__ == "__main__":
    main()
