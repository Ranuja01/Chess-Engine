# -*- coding: utf-8 -*-
"""Quiet-threat regression set (2026-11-17): does the search find, or avoid, the quiet moves that decided real games?
Each position in diagnostics/ks_sets/quiet_threat_positions.txt has a key move set: `find` passes when the engine plays one of
them, `avoid` when it plays none. Run through a UCI binary at fixed depths, a fresh game per search (no carried tables).
  pyrun diagnostics/_quiet_threat_set.py [BIN=<uci binary>] [KNOBS=A=1,B=2] [DEPTHS=12,14,16] [CAP=60] [SET=<file>]
One process per call: knobs latch at init. Prints one row per position and depth, then the pass count per depth.
The set is a pruning check, not an Elo instrument: games decide (memory pruning-quality-is-what-it-leaves-behind).
"""
import os
import sys
import chess
import chess.engine

KV = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
THIS = os.path.dirname(os.path.abspath(__file__))
BIN = KV.get("BIN", os.path.join(os.path.dirname(THIS), "build_uci", "chess_engine_uci"))
SET = KV.get("SET", os.path.join(THIS, "ks_sets", "quiet_threat_positions.txt"))
DEPTHS = [int(d) for d in KV.get("DEPTHS", "12,14,16").split(",")]
CAP = float(KV.get("CAP", "60"))
KNOBS = KV.get("KNOBS", "")

rows = []
for line in open(SET, encoding="utf-8"):
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    label, fen, kind, moves, note = [p.strip() for p in line.split("|", 4)]
    rows.append((label, fen, kind, [m.strip() for m in moves.split(",")], note))

eng = chess.engine.SimpleEngine.popen_uci(BIN, timeout=300)
eng.configure({"OwnBook": False})
if KNOBS:
    eng.configure({"Knobs": KNOBS})
print("quiet-threat set: %d positions · %s · knobs [%s] · depths %s · cap %.0f s" % (len(rows), os.path.basename(BIN), KNOBS, DEPTHS, CAP), flush=True)
passes = {d: 0 for d in DEPTHS}
for label, fen, kind, keys, note in rows:
    b = chess.Board(fen)
    for d in DEPTHS:
        info = eng.analyse(b, chess.engine.Limit(depth=d, time=CAP), game=object())
        mv = b.san(info["pv"][0])
        ok = (mv in keys) if kind == "find" else (mv not in keys)
        passes[d] += ok
        print("  %-11s %-5s %-10s d%-2d (reached %s): %-6s %+6d  %s  nodes %11s" % (
            label, kind, ",".join(keys), d, info.get("depth", "?"), mv, info["score"].white().score(mate_score=10000),
            "PASS" if ok else "fail", "{:,}".format(info.get("nodes", 0))), flush=True)
eng.quit()
print("passes by depth: " + " · ".join("d%d %d/%d" % (d, passes[d], len(rows)) for d in DEPTHS), flush=True)
