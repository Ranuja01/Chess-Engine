# -*- coding: utf-8 -*-
"""Full-partition + SF15.1-classical probe for a short FEN list.

probe_fens.py truncates our breakdown to the top-9 terms by magnitude, which hid +8.23 pawns of offsetting
terms on the 2026-07-30 collapse position and made term attribution impossible. This prints EVERY field of
ev_breakdown with a sum-vs-total check, and adds SF15.1's CLASSICAL (non-NNUE) static eval plus its full
labeled term table as the second classical witness alongside SF11.

Run: pyrun diagnostics/_collapse_full_probe.py <label_tab_fen_file>
"""
import os, sys, re, subprocess
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
# KEY=VAL argv -> environ BEFORE the extension import; Config latches at init and pyrun forwards argv, not env.
for _a in sys.argv[1:]:
    if '=' in _a and not _a.endswith('.fens'):
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v
sys.argv = [a for a in sys.argv if '=' not in a or a.endswith('.fens')]

THIS = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.path.dirname(THIS)
sys.path.insert(0, ENGINE_DIR)
os.chdir(ENGINE_DIR)

import chess
from ChessAI import ChessAI

SF15 = os.environ.get("SF15_BIN",
    "/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/stockfish_15_linux/"
    "stockfish_15.1_linux_x64/stockfish-ubuntu-20.04-x86-64")

rows = []
for line in open(sys.argv[1]):
    line = line.rstrip("\n")
    if not line.strip():
        continue
    label, fen = line.split("\t", 1)
    rows.append((label.strip(), fen.strip()))

# --- SF15.1 classical, NNUE off -------------------------------------------------------------------
p = subprocess.Popen([SF15], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=subprocess.STDOUT, text=True, bufsize=1)
def send(s): p.stdin.write(s + "\n"); p.stdin.flush()
send("uci"); send("setoption name Use NNUE value false"); send("isready")
while True:
    ln = p.stdout.readline()
    if not ln or ln.startswith("readyok"):
        break

_total_re = re.compile(r"Classical evaluation\s+([+-]?\d+\.\d+)")
def sf15_eval(fen):
    """Return (total, [raw table lines]) for the classical eval of fen."""
    send("position fen " + fen); send("eval"); send("isready")
    total, lines = None, []
    while True:
        ln = p.stdout.readline()
        if not ln:
            break
        m = _total_re.search(ln)
        if m:
            total = float(m.group(1))
        if ln.startswith("readyok"):
            break
        lines.append(ln.rstrip("\n"))
    return total, lines

seed = chess.Board(rows[0][1])
ai = ChessAI(None, None, seed, seed.turn)  # one warm instance; ev_breakdown reads only the board arg

for label, fen in rows:
    board = chess.Board(fen)
    print("=" * 100)
    print(f"{label}   [{fen}]")

    bd = ai.ev_breakdown(board)
    # Our eval is ABSOLUTE Black-positive; every number below is NEGATED to WHITE-POV so it is
    # directly comparable with SF11/SF15/SF18 output and with the owner's own tooling.
    total = -bd.get("total")
    parts = {k: -v for k, v in bd.items() if k != "total"}
    # `pieces` is the sum of pt_* and already contains `material`, so pt_*/material are SUB-VIEWS,
    # not partition members (see diagnostics/breakdown_partition_check.py). det_*/ae_*/phase_score
    # and the per-side imbalance_* details are diagnostics. Sum only the real members.
    PART = ['pieces', 'capture_gains', 'passed_pawn_support', 'latent_threat', 'threats', 'king_safety',
            'central', 'imbalance_white', 'imbalance_black', 'pair_bonus', 'piece_value_boost',
            'kaufman_imbalance', 'pawn_majority', 'pawn_struct', 'outpost', 'space', 'mobility', 'rook_cond']
    ae_delta = (parts.get('advanced_endgame_total', 0) - parts.get('ae_input', 0)) \
        if bd.get('advanced_endgame_fired') else 0
    psum = sum(parts.get(k, 0) for k in PART) + ae_delta
    print(f"\n  OURS total = {total/1000.0:+.2f}  [WHITE-POV]   "
          f"partition sum = {psum/1000.0:+.2f}   residual = {(total - psum)/1000.0:+.2f}")
    print("  PARTITION MEMBERS (pawns, White-POV):")
    for k in PART:
        v = parts.get(k, 0)
        if v:
            print(f"      {k:<28} {v/1000.0:+8.2f}")
    if ae_delta:
        print(f"      {'advanced_endgame(delta)':<28} {ae_delta/1000.0:+8.2f}")
    print("  SUB-VIEWS of `pieces` (NOT additive on top of it):")
    for k in ['material', 'pt_pawns', 'pt_knights', 'pt_bishops', 'pt_rooks', 'pt_queens', 'pt_kings']:
        v = parts.get(k, 0)
        if v:
            print(f"      {k:<28} {v/1000.0:+8.2f}")
    print(f"      {'det_w_pieceval':<28} {parts.get('det_w_pieceval',0)/1000.0:+8.2f}   "
          f"{'det_b_pieceval':<28} {parts.get('det_b_pieceval',0)/1000.0:+8.2f}")

    t15, tbl = sf15_eval(fen)
    print(f"\n  SF15.1 CLASSICAL total = {t15:+.2f}" if t15 is not None else "\n  SF15.1 classical: n/a")
    print("  SF15.1 term table:")
    for ln in tbl:
        if ln.strip():
            print("      " + ln)

send("quit")
