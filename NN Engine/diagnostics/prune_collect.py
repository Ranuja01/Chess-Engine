# -*- coding: utf-8 -*-
"""
Prune-fire collector (Step 2, arbitrary corpus) — search each FEN in a CSV/EPD with ENABLE_PRUNE_LOG on so
the engine emits [PRUNEFIRE] stderr records. Point stderr at a file:
    ... pyrun diagnostics/prune_collect.py <corpus> [MAX_DEPTH] [STRIDE]  2> /tmp/prunelog_<tag>.err
then label with prune_verify.py + prune_discriminate.py.

Corpus = a .csv with a `fen` column, or a .epd/.txt with one FEN per line (first 4-6 fields).
This is the messy-corpus counterpart to the `wac` runner sub (which only runs wac.epd).
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['ENABLE_PRUNE_LOG'] = '1'
os.environ['USE_OPENING_BOOK'] = '0'

import sys
import csv

import chess

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, ENGINE_DIR)
sys.path.insert(0, THIS_DIR)

corpus = sys.argv[1]
os.environ['MAX_DEPTH'] = sys.argv[2] if len(sys.argv) > 2 else '10'
os.environ['PRUNE_LOG_STRIDE'] = sys.argv[3] if len(sys.argv) > 3 else '5'


def load_fens(path):
    fens = []
    if path.endswith('.csv'):
        for r in csv.DictReader(open(path)):
            f = r.get('fen') or r.get('FEN')
            if f:
                fens.append(f.strip())
    else:
        for line in open(path):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 4:
                fens.append(' '.join(parts[:6]) if len(parts) >= 6 else ' '.join(parts[:4]))
    return fens


def main():
    fens = load_fens(corpus)
    from tactical_test import run_one
    print("[prune_collect] %s: %d fens, MAX_DEPTH=%s stride=%s" %
          (os.path.basename(corpus), len(fens), os.environ['MAX_DEPTH'], os.environ['PRUNE_LOG_STRIDE']),
          file=sys.stderr)
    done = 0
    for f in fens:
        try:
            chess.Board(f)             # validate
            run_one(f, [])
        except Exception:
            continue
        done += 1
        if done % 200 == 0:
            print("[prune_collect] %d/%d searched" % (done, len(fens)), file=sys.stderr)
    print("[prune_collect] done: %d searched" % done, file=sys.stderr)


if __name__ == '__main__':
    main()
