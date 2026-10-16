# -*- coding: utf-8 -*-
"""CLOSURE for the C++ POT race (2026-10-16): the engine's published race features (ev_breakdown pot_race_*, POT_V2_RACE=1)
must EQUAL the Python prototype's (diagnostics/_pot_race_proto.py features()) on every position — exact integers.

  pyrun diagnostics/_pot_race_closure.py [N=3000] [IN=/mnt/e/chess_data/retune/EGX_sf18.csv]

Pure pawn endings from EGX, every other row colour-mirrored. Prints the mismatch count per feature and the first mismatches.
Also checks the race term is absent (no pot_race_* keys) on a non-pawn ending, and that POT_V2_RACE=1 with zero weights leaves
the total unchanged versus the published total (pot_race_mp == 0).
"""
import os, sys, csv
os.environ.setdefault("V2_PRESET", "shipped")
os.environ["POT_V2_RACE"] = "1"
for _a in sys.argv[1:]:
    if "=" in _a:
        _k, _v = _a.split("=", 1)
        os.environ[_k] = _v
THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
import chess
import _pot_race_proto as P

N = int(os.environ.get("N", "3000"))
IN = os.environ.get("IN", "/mnt/e/chess_data/retune/EGX_sf18.csv")
KEYS = ["pot_race_best", "pot_race_second", "pot_race_promo", "pot_race_break", "pot_race_decisive", "pot_race_over",
        "pot_race_tempi", "pot_race_stm", "pot_race_contested"]
P.ours(chess.Board())                                    # build the engine instance
ai = P.AI
bad = {k: 0 for k in KEYS}
shown, n, nz_mp, kpk_owned = 0, 0, 0, 0
for r in csv.DictReader(open(IN, newline="")):
    b = chess.Board(r["fen"])
    if b.is_game_over() or not b.pawns or (b.occupied & ~(b.pawns | b.kings)):
        continue
    if n % 2 == 1:
        b = b.mirror()
    n += 1
    eb = ai.ev_breakdown(b)
    py = P.features(b)
    if "pot_race_best" not in eb:
        if chess.popcount(b.pawns) == 1:                 # K+P vs K: the exact KPK bitbase owns it and returns first (correct)
            kpk_owned += 1
            continue
        bad["pot_race_best"] += 1
        if shown < 5:
            print("MISSING race keys:", b.fen()); shown += 1
        continue
    nz_mp += eb["pot_race_mp"] != 0
    diffs = [(k, eb[k], int(v)) for k, v in zip(KEYS, py) if eb[k] != int(v)]
    for k, _, _ in diffs:
        bad[k] += 1
    if diffs and shown < 10:
        print("MISMATCH %s  %s" % (b.fen(), diffs)); shown += 1
    if n >= N:
        break
eb = ai.ev_breakdown(chess.Board("8/5k2/8/3n4/8/2P5/1K6/8 w - - 0 1"))
print("CLOSURE: %d pawn endings (%d K+P vs K owned by the exact KPK bitbase, skipped) · mismatches per feature: %s · nonzero pot_race_mp with zero weights: %d · race keys on a knight ending: %s"
      % (n, kpk_owned, " ".join("%s=%d" % (k.replace("pot_race_", ""), v) for k, v in bad.items()), nz_mp, "pot_race_best" in eb))
print("CLOSURE %s" % ("PASS" if not any(bad.values()) and not nz_mp and "pot_race_best" not in eb else "FAIL"))
