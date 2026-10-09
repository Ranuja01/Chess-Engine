# -*- coding: utf-8 -*-
"""Feature export for the PASSER-SYSTEM fit, restricted to the SF18-LABELLED rows (mg + eg samples), instead of a full
1.84M-row feature pass. Per FEN: the v2_features diff (Black − White, all V2F_PER_SIDE columns incl. the 51 PX cells at
184-234), v2's phase256 and the flags word. Needs a build with the PX block (V2F_PER_SIDE = 235).

  pyrun diagnostics/_px_export.py V2_PRESET=shipped [OUT=/mnt/e/chess_data/texel/px_labelled.npz]
"""
import os, sys, csv
import numpy as np

for _a in sys.argv[1:]:
    if "=" in _a:
        _k, _v = _a.split("=", 1)
        os.environ[_k] = _v
THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS))
os.environ.setdefault("PRESET", "LONG_FORMAT"); os.environ.setdefault("USE_OPENING_BOOK", "0")
import chess
import ChessAI

OUT = os.environ.get("OUT", "/mnt/e/chess_data/texel/px_labelled.npz")
assert ChessAI.V2F_PER_SIDE >= 235, "this build predates the PX block (V2F_PER_SIDE %d)" % ChessAI.V2F_PER_SIDE   # 337 since the CONN cells (2026-10-09); all columns are exported
ai = ChessAI.ChessAI(None, None, chess.Board(), True)
fens, diffs, phases, flags = [], [], [], []
seen = set()
for lab in ("fitC_mg_sf18.csv", "fitC_eg_sf18.csv"):
    for r in csv.DictReader(open(os.path.join(THIS, "ks_sets", lab), newline="")):
        f = r["fen"]
        if f in seen or not r.get("best_cp"):
            continue
        seen.add(f)
        b = chess.Board(f)
        wc, bc, fl = ChessAI.v2_feature_counts(b)
        fens.append(f)
        diffs.append([x - y for x, y in zip(bc, wc)])
        phases.append(int(ai.ev_breakdown(b).get("v2_phase256", -1)))
        flags.append(fl)
tmg, teg = ChessAI.v2_feature_theta()      # the LIVE starting values (shipped constants; PX cells 0 unless loaded)
np.savez_compressed(OUT, fen=np.array(fens), diff=np.array(diffs, dtype=np.int16), phase=np.array(phases, np.int16),
                    flags=np.array(flags, np.int8), theta_mg=np.array(tmg), theta_eg=np.array(teg))
print("PX EXPORT  %d labelled FENs → %s" % (len(fens), OUT))
