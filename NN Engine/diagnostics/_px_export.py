# -*- coding: utf-8 -*-
"""Feature export for the PASSER-SYSTEM fit, restricted to the SF18-LABELLED rows (mg + eg samples), instead of a full
1.84M-row feature pass. Per FEN: the v2_features diff (Black − White, all V2F_PER_SIDE columns incl. the 51 PX cells at
184-234), v2's phase256 and the flags word. Needs a build with the PX block (V2F_PER_SIDE = 235).

  pyrun diagnostics/_px_export.py V2_PRESET=shipped [OUT=/mnt/e/chess_data/texel/px_labelled.npz]

IN=<csv with a fen column> (2026-10-10, retune corpus): export THOSE rows instead of the fitC sets, in resumable chunks
of CHUNK rows (default 100,000) → OUT with _cNN.npz suffixes; a chunk already on disk is skipped. Each chunk also
carries `total` = the engine's static eval (ev_breakdown total, Black-positive mp) — the static dump from the same pass.
  V2_PRESET=shipped pyrun diagnostics/_px_export.py IN=/mnt/e/chess_data/retune/corpus_v1.csv OUT=/mnt/e/chess_data/retune/feat_ship.npz
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
tmg, teg = ChessAI.v2_feature_theta()      # the LIVE starting values (shipped constants; PX cells 0 unless loaded)

if os.environ.get("IN"):
    chunk = int(os.environ.get("CHUNK", "100000"))
    fens = [r["fen"] for r in csv.DictReader(open(os.environ["IN"], newline="")) if r.get("fen")]
    nf = ChessAI.V2F_PER_SIDE
    print("CORPUS EXPORT  %d FENs · %d columns · chunks of %d · V2_PRESET=%s" % (len(fens), nf, chunk,
          os.environ.get("V2_PRESET", "(unset)")), flush=True)
    for c0 in range(0, len(fens), chunk):
        path = OUT.replace(".npz", "_c%02d.npz" % (c0 // chunk))
        if os.path.exists(path):
            continue
        part = fens[c0:c0 + chunk]
        diff = np.zeros((len(part), nf), np.int16)
        phase = np.zeros(len(part), np.int16); flags = np.zeros(len(part), np.int8); total = np.zeros(len(part), np.int32)
        for i, f in enumerate(part):
            b = chess.Board(f)
            wc, bc, fl = ChessAI.v2_feature_counts(b)
            diff[i] = np.subtract(bc, wc)
            bd = ai.ev_breakdown(b)
            phase[i] = int(bd.get("v2_phase256", -1)); total[i] = int(bd["total"]); flags[i] = fl
        tmp = path + ".tmp.npz"
        np.savez_compressed(tmp, fen=np.array(part), diff=diff, phase=phase, flags=flags, total=total,
                            theta_mg=np.array(tmg), theta_eg=np.array(teg))
        os.replace(tmp, path)
        print("  chunk %02d: %d rows → %s" % (c0 // chunk, len(part), path), flush=True)
    print("CORPUS EXPORT DONE", flush=True)
    sys.exit(0)

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
np.savez_compressed(OUT, fen=np.array(fens), diff=np.array(diffs, dtype=np.int16), phase=np.array(phases, np.int16),
                    flags=np.array(flags, np.int8), theta_mg=np.array(tmg), theta_eg=np.array(teg))
print("PX EXPORT  %d labelled FENs → %s" % (len(fens), OUT))
