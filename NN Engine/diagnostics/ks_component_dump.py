# -*- coding: utf-8 -*-
"""Scope WHY our king_safety under-fires: dump the per-component king-danger breakdown (attacked-zone squares,
weak squares, safe checks, attacker pieces, defender pieces, open files, raw units, danger) for OUR king on the
danger set, and compare to the opponent king. Needs the engine built with the KS_DEBUG_DUMP diagnostic.

Worker writes 'FEN\\t<fen>' to stderr then evaluates (C++ prints two 'KSD ...' lines to stderr, W then B),
so the parent can pair each position with its two component lines in order.

Run: pyrun diagnostics/ks_component_dump.py
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import sys
import subprocess

THIS = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
KNOBS = dict(KS_DEBUG_DUMP="1", ENABLE_KS_REPLACE_LT="1", KING_SAFETY_MAG="3000", KS_DEFENDER="0",
             ENABLE_KS_SF_WEAK="1", ENABLE_KS_SF_SAFECHECK="1", KS_FLOOR="0", KS_NO_QUEEN="6")  # bundle, FLOOR=0 to see raw units


def load_danger():
    out = []
    for ln in open(os.path.join(THIS, "ks_sets", "danger.txt")):
        ks, fen = ln.rstrip("\n").split("\t", 1)
        out.append((float(ks), fen))
    return out


def worker():
    import chess
    sys.path.insert(0, os.path.dirname(THIS))
    from ChessAI import ChessAI
    ai = ChessAI(None, None, chess.Board(), True)
    for _ks, fen in load_danger():
        sys.stderr.write("FEN\t%s\n" % fen); sys.stderr.flush()
        try:
            ai.ev_breakdown(chess.Board(fen))
        except Exception as e:
            sys.stderr.write("ERR\t%s\n" % e); sys.stderr.flush()


def parse_ksd(line):
    d = {}
    for kv in line.split()[1:]:
        if "=" in kv:
            k, v = kv.split("=")
            d[k] = int(v)
    d["side"] = line.split()[1] if not line.split()[1].startswith(("attsq",)) else None
    return d


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        worker(); return
    env = os.environ.copy(); env.update(KNOBS)
    p = subprocess.run([PY, os.path.abspath(__file__), "worker"], env=env, capture_output=True, text=True)
    lines = p.stderr.splitlines()
    # pair: FEN line, then KSD W, KSD B
    recs = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("FEN\t"):
            fen = lines[i].split("\t", 1)[1]
            ks = {}
            j = i + 1
            while j < len(lines) and lines[j].startswith("KSD "):
                parts = lines[j].split()
                side = parts[1]
                comp = {kv.split("=")[0]: int(kv.split("=")[1]) for kv in parts[2:] if "=" in kv}
                ks[side] = comp
                j += 1
            recs.append((fen, ks))
            i = j
        else:
            i += 1

    danger = load_danger()
    sfks = {fen: k for k, fen in danger}
    import chess
    print("Our king's KS components on the danger set (OUR king = side to move):\n")
    print("%-4s %-8s | attsq weak safe attpc defpc openf | units danger | opp_units" % ("sf", "our_pov"))
    print("-" * 82)
    agg = {"attsq": [], "weak": [], "safe": [], "attpc": [], "defpc": [], "units": [], "danger": []}
    for fen, ks in recs:
        us = "W" if chess.Board(fen).turn == chess.WHITE else "B"
        them = "B" if us == "W" else "W"
        o = ks.get(us, {}); t = ks.get(them, {})
        if not o:
            continue
        for k in agg:
            agg[k].append(o.get(k, 0))
        print("%+4.1f          | %4d %4d %4d %5d %5d %5d | %5d %6d | %6d"
              % (sfks.get(fen, 0), o.get("attsq", 0), o.get("weak", 0), o.get("safe", 0), o.get("attpc", 0),
                 o.get("defpc", 0), o.get("openf", 0), o.get("units", 0), o.get("danger", 0), t.get("units", 0)))
    n = max(1, len(agg["units"]))
    print("\nMEANS over %d danger positions (OUR king):" % n)
    for k in ("attsq", "weak", "safe", "attpc", "defpc", "units", "danger"):
        print("  %-7s %.1f" % (k, sum(agg[k]) / n))
    zero_units = sum(1 for u in agg["units"] if u == 0)
    print("  positions with units==0 (detector totally blind): %d/%d" % (zero_units, n))


if __name__ == "__main__":
    main()
