# -*- coding: utf-8 -*-
"""TEXEL FIT K, engine side: every king-safety CHANNEL per king, plus the engine's own KS score, per stage-1 row.

WHY (2026-09-28). The joint KS fit (dev_notes/TEXEL-C3-DETECTORS-DESIGN-2026-09-27.md §7, §8b) re-prices king safety
TOGETHER with the PSTs and the C3 detectors. KS is not linear -- units -> onset clamp -> saturating curve -- so the
fitter reproduces it in Python from the channel counts instead of treating it as a fixed block. That only works if
the Python model is EXACTLY the engine's, so this pass also runs the gate: rebuild shipped KS from the counts and
compare to ev_breakdown's king_safety on every row. Any residual is a model bug, never noise (the KS arithmetic is
integer and deterministic). Units are checked against the probe's own post-onset `units` too, so a mismatch can be
localised to the units stage or the curve stage.

Channels (per king, index 0 = White's king): n_att w_att weak adj chk_r chk_q chk_b chk_n enemy_queen units, and the
2026-09-27 balance channels n_att_x adj_inst unsafe blockers flank_att flank_def knight_def contest_excess contest_sq
w_att_contest gate. All are raw and unconditional (ks_counts contract); only `units` is post-onset.

  pyrun diagnostics/_texel_ks_pass.py V2_PRESET=shipped [IN=/mnt/e/chess_data/texel/fitC_stage1.csv.gz]
        [OUT=/mnt/e/chess_data/texel/fitC_ks.npz] [LIMIT=0]
Output .npz: row, phase (int16), ks_engine (int32, Black-positive mp), ch (int16, n x 2 x NCH), names.
"""
import os, sys, csv, gzip, time
import numpy as np

for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v
THIS = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(THIS)
sys.path.insert(0, ENGINE)
os.environ.setdefault("PRESET", "LONG_FORMAT")
os.environ.setdefault("USE_OPENING_BOOK", "0")
IN = os.environ.get("IN", "/mnt/e/chess_data/texel/fitC_stage1.csv.gz")
OUT = os.environ.get("OUT", "/mnt/e/chess_data/texel/fitC_ks.npz")
LIMIT = int(os.environ.get("LIMIT", "0"))

import chess
import ChessAI

NAMES = ["n_att", "w_att", "weak", "adj", "chk_r", "chk_q", "chk_b", "chk_n", "enemy_queen", "units",
         "n_att_x", "adj_inst", "unsafe", "blockers", "flank_att", "flank_def", "knight_def", "contest_excess",
         "contest_sq", "w_att_contest", "gate"]
IX = {n: i for i, n in enumerate(NAMES)}

# The SHIPPED KS parameters (search_engine.cpp V2_PRESET=shipped block). The gate below fails loudly if the live
# engine disagrees, so a stale copy here cannot pass silently.
SHIP = dict(COORD=256, WEAK=57, ADJ=61, NO_QUEEN=321, CHK_R=122, CHK_Q=126, CHK_B=80, CHK_N=152, ONSET=450,
            MAX=4000, HALF=600, EG_PCT=100)


def ks_units_py(ch, p):
    """Shipped-form units from channel counts (ch: n x NCH int array for ONE king), numpy int64, post-onset clamp.
    Mirrors eval_v2.cpp ks_units with ATT_XRAY=0, DEFAWARE=0, CHK_COUNT=0, PAWN_ATT=0 and every balance weight 0."""
    ch = ch.astype(np.int64)
    n, w = ch[:, IX["n_att"]], ch[:, IX["w_att"]]
    u = np.where(n > 0, (w * (256 + (n - 1) * p["COORD"])) >> 8, 0)
    u = u + p["WEAK"] * ch[:, IX["weak"]] + p["ADJ"] * ch[:, IX["adj"]]
    for t in ("R", "Q", "B", "N"):
        u = u + p["CHK_" + t] * (ch[:, IX["chk_" + t.lower()]] > 0)
    u = u - p["NO_QUEEN"] * (ch[:, IX["enemy_queen"]] == 0)
    u = u - p["ONSET"]
    return np.maximum(u, 0)


def ks_danger_py(u, p):
    uu = u.astype(np.int64) ** 2
    return np.where(u > 0, (p["MAX"] * uu) // (uu + p["HALF"] * p["HALF"]), 0)


def ks_mp_py(chw, chb, phase, p):
    """Black-positive KS millipawns: a dangerous WHITE king favours Black. Per-side phase blend as the engine does."""
    dw, db = ks_danger_py(ks_units_py(chw, p), p), ks_danger_py(ks_units_py(chb, p), p)
    ph = phase.astype(np.int64)
    ew, eb = dw * p["EG_PCT"] // 100, db * p["EG_PCT"] // 100
    return ((dw * ph + ew * (256 - ph)) >> 8) - ((db * ph + eb * (256 - ph)) >> 8)


if __name__ == "__main__":
    ai = ChessAI.ChessAI(None, None, chess.Board(), True)
    rows, phases, kse, chs = [], [], [], []
    t0 = time.time()
    with gzip.open(IN, "rt") as f:
        for i, r in enumerate(csv.DictReader(f)):
            if LIMIT and i >= LIMIT:
                break
            b = chess.Board(r["fen"])
            bd = ai.ev_breakdown(b)
            k = ChessAI.ks_counts(b.pawns, b.knights, b.bishops, b.rooks, b.queens, b.kings,
                                  b.occupied_co[True], b.occupied_co[False])
            rows.append(i)
            phases.append(int(bd.get("v2_phase256", -1)))
            kse.append(int(bd.get("king_safety", 0)) if "total" in bd and "v2_phase256" in bd else 0)
            chs.append([[k[n][s] for n in NAMES] for s in (0, 1)])
            if (i + 1) % 200000 == 0:
                sys.stderr.write("[ks pass] %d rows, %.0f/s\n" % (i + 1, (i + 1) / (time.time() - t0)))
                sys.stderr.flush()

    CH = np.array(chs, dtype=np.int32)
    ph = np.array(phases, dtype=np.int32)
    KE = np.array(kse, dtype=np.int64)
    assert CH.max() < 32767 and CH.min() > -32768, "a channel overflows int16"
    np.savez_compressed(OUT, row=np.array(rows, dtype=np.int32), phase=ph.astype(np.int16),
                        ks_engine=KE.astype(np.int32), ch=CH.astype(np.int16), names=np.array(NAMES))

    # ---- the gate: does the Python KS reproduce the engine exactly? ------------------------------------------
    ok = ph >= 0                       # draw-classifier / terminal rows publish no phase and no KS
    uw, ub = ks_units_py(CH[:, 0], SHIP), ks_units_py(CH[:, 1], SHIP)
    du = np.concatenate([np.abs(uw - CH[:, 0, IX["units"]])[ok], np.abs(ub - CH[:, 1, IX["units"]])[ok]])
    pred = ks_mp_py(CH[:, 0], CH[:, 1], ph, SHIP)
    dk = np.abs(pred - KE)[ok]
    live = (KE != 0)[ok].sum()
    print("KS PASS  %d rows (%d scored)  %.0fs" % (len(rows), ok.sum(), time.time() - t0))
    print("  units  vs probe:   mismatched kings %d   max |diff| %d" % ((du != 0).sum(), du.max() if du.size else 0))
    print("  KS mp  vs engine:  live rows %d   mismatched rows %d   max |diff| %d mp"
          % (live, (dk != 0).sum(), dk.max() if dk.size else 0))
    print("  VERDICT: %s" % ("EXACT" if (du == 0).all() and (dk == 0).all() else "☠️ MODEL DIVERGES -- do not fit"))
    print("wrote", OUT)
