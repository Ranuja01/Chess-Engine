# -*- coding: utf-8 -*-
"""Dump SF11's RAW `eval` term table (verbatim) for FENs, to verify our SF11Eval parser isn't misattributing
columns (user caught a suspicious 'Passed' sign). No parsing — just the engine's own output.
  pyrun diagnostics/raw_sf11_dump.py
"""
import os, sys, subprocess
THIS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, THIS)
from eval_vs_sf11 import SF11
FENS = {
    "3 central-wall": "r3k3/pp2pp2/2p3p1/1q1p2b1/3P2n1/BP1N4/P1P1Kp2/Q2R1N2 b q - 1 21",
    "5 up4-passers":  "4k3/3qb1p1/2np3P/p3p3/Q3P3/p3B3/5P1K/5B2 b - - 0 32",
    "6 e4-promotes":  "4B3/8/P3k3/2p5/2P1pp1P/1P2P3/3r4/1K6 w - - 0 60",
}
p = subprocess.Popen([SF11], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                     text=True, bufsize=1)
p.stdin.write("uci\n"); p.stdin.flush()
while True:
    if (p.stdout.readline() or "").strip() == "uciok":
        break
for label, fen in FENS.items():
    p.stdin.write("position fen %s\neval\nisready\n" % fen); p.stdin.flush()
    print("\n########## %s\n%s" % (label, fen))
    while True:
        ln = p.stdout.readline()
        if not ln or ln.strip() == "readyok":
            break
        print(ln.rstrip())
p.stdin.write("quit\n"); p.stdin.flush()
