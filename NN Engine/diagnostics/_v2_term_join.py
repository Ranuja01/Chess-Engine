# -*- coding: utf-8 -*-
"""v2 TERM TRIANGULATION -- join eval v2's per-term breakdown onto the cached SF11 per-term columns.

WHY THIS EXISTS (2026-09-19). `selfplay/tune_data/cond_corpus_v2.csv` already carries 13 `sf11_*` per-term
columns for 37,222 positions, and SF is arm-independent, so that half NEVER needs regenerating. What it
carries on OUR side is v1's partition, from before the rebuild. This recomputes our side under EVAL_ARM=1
and joins, so v2 can finally be compared term-by-term against a strong classical HCE.
★ Needs the 2026-09-19 publication change (17/48 fields): before it, pawn structure, passers, placement and
rook files were folded into `total` and three of SF's rows had no counterpart at all.

☠️☠️ READ THIS BEFORE READING THE OUTPUT -- the obvious reading of it is a CLOSED LANE.
  * The aggregate mean-gap form ("our term reads X, SF reads Y, so scale by Y/X") is a RESOLVED NULL
    (2026-08-19: capture_gains -0.08 in collapses AND -0.08 in the quiet control => "the
    eval-calibration-by-aggregate lane is CLOSED"), and tuning a term toward SF's magnitude is a resolved
    NEGATIVE (KS, 2026-09-11: best fit, +9.70% worst-case error). Only SHAPE and CHANNEL SET transfer.
  * The ONE triangulation result that ever reached Elo (+45, the threats ship) came from a SILENCE:
    `threats` read 0.00 where SF read -1.48/-0.87 on a class where it mattered.
  => This tool therefore reports SILENCE, not mean gap: how often does SF see something substantial where
     we read ~nothing? Means cancel; zeros do not.

Term pairing is dev_notes/our_eval_reference.md's v2 map. ⚠️ `material + pieces` <-> SF `Material` (SF folds
PSQT into Material; v2 splits them, and v1's material-inclusive `pieces` bug does NOT apply to v2).
⚠️ Our placement is ONE aggregate against SF's FOUR piece rows -- compare sum-to-sum only.
⚠️ ABSENT != ZERO: a parked term is missing from ev_breakdown entirely. Those pairs are reported as PARKED
and must not be read as "we score zero here".

  pyrun diagnostics/_v2_term_join.py [IN=selfplay/tune_data/cond_corpus_v2.csv] [OUT=...] [N=0] [SILENT=0.05] [SEES=0.50]
"""
import os, sys, csv

for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
THIS = os.path.dirname(os.path.abspath(__file__))
ENG = os.path.dirname(THIS)
sys.path.insert(0, ENG); sys.path.insert(0, THIS)
import chess
from ChessAI import ChessAI

IN     = os.environ.get("IN", os.path.join(ENG, "selfplay", "tune_data", "cond_corpus_v2.csv"))
OUT    = os.environ.get("OUT", os.path.join(THIS, "ks_sets", "v2_term_join.csv"))
if not os.path.isabs(OUT):
    OUT = os.path.join(THIS, OUT)     # the runner cds to NN Engine/, so resolve against diagnostics/
N      = int(os.environ.get("N", "0"))            # 0 = all
SILENT = float(os.environ.get("SILENT", "0.05"))  # |ours| below this = we say nothing
SEES   = float(os.environ.get("SEES", "0.50"))    # |SF| above this = SF says something substantial

# SF row -> the v2 field(s) that carry the same concept. [] = we deliberately do not carry it.
PAIRS = [
    ("sf11_material",   ["material", "pieces"]),   # SF folds PSQT into Material; v2 splits them
    ("sf11_mobility",   ["mobility"]),
    ("sf11_kingsafety", ["king_safety"]),
    ("sf11_pawns",      ["pawn_struct"]),
    ("sf11_passed",     ["v2_passers"]),
    ("sf11_pieces4",    ["v2_placement", "v2_rookfile"]),   # synthesised: knights+bishops+rooks+queens
    ("sf11_imbalance",  []),                                # PARKED (Kaufman)
    ("sf11_threats",    []),                                # PARKED
    ("sf11_space",      []),                                # PARKED
]
SF4 = ["sf11_knights", "sf11_bishops", "sf11_rooks", "sf11_queens"]


def wp(mp):
    """our Black-positive millipawns -> White-POV pawns, SF's convention."""
    return -mp / 1000.0


def main():
    ai = ChessAI(None, None, chess.Board(), True)
    rows = list(csv.DictReader(open(IN, newline="")))
    if N:
        rows = rows[:N]
    print("v2 TERM TRIANGULATION  --  %d positions from %s\n" % (len(rows), os.path.basename(IN)))

    out_rows, skipped = [], 0
    v2keys = ["total", "material", "pieces", "king_safety", "mobility",
              "pawn_struct", "v2_passers", "v2_placement", "v2_rookfile"]
    present = None
    for i, r in enumerate(rows):
        try:
            b = chess.Board(r["fen"])
        except Exception:
            skipped += 1; continue
        if b.is_game_over(claim_draw=False):
            skipped += 1; continue
        bd = ai.ev_breakdown(b)
        if present is None:
            present = {k for k in v2keys if k in bd}
        o = {"fen": r["fen"], "phase_score": r.get("phase_score", ""),
             "is_endgame": r.get("is_endgame", ""), "result_white": r.get("result_white", "")}
        for k in v2keys:
            o["v2_" + k] = ("%.4f" % wp(bd[k])) if k in bd else ""
        for c in [p[0] for p in PAIRS if p[0] != "sf11_pieces4"] + SF4 + ["sf11_total"]:
            o[c] = r.get(c, "")
        try:
            o["sf11_pieces4"] = "%.4f" % sum(float(r[c]) for c in SF4 if r.get(c) not in (None, ""))
        except Exception:
            o["sf11_pieces4"] = ""
        out_rows.append(o)
        if (i + 1) % 5000 == 0:
            print("  ... %d / %d" % (i + 1, len(rows)))

    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)
    print("\nwrote %s  (%d rows, %d skipped)" % (OUT, len(out_rows), skipped))
    print("v2 published: %s" % ", ".join(sorted(present or [])))

    # ---------------- the SILENCE report ----------------
    print("\n%s" % ("=" * 104))
    print("SILENCE REPORT -- where SF sees >= %.2f pawns, how often do we read < %.2f?" % (SEES, SILENT))
    print("★ This, not the mean gap, is the shape that once produced +45 Elo. Means cancel; zeros do not.")
    print("%s" % ("=" * 104))
    # ☠️ THE ONE-SIDED VIEW MANUFACTURES FINDINGS. "We are silent 78% of the time" means nothing without
    # knowing how often SF is silent, and how often WE fire where SF does not. Report the full 2x2 plus
    # sign agreement on the cells where both speak -- a term can be loud, aligned in magnitude, and still
    # point the wrong way, which no magnitude column can show.
    # ★★ THE WIN% COLUMN sizes each row against the ONE statistic that matters, at each position's ACTUAL
    # operating point. For every position: what would our win% be if THIS ONE TERM matched SF's and nothing
    # else changed? |Δwin%| averaged over the corpus. That handles the nonlinearity a pawn-space mean cannot
    # -- 0.5 pawns near equality moves win% far more than 0.5 pawns at +6 -- which is exactly why a
    # directionally-correct-but-mis-scaled term can still be cheap or expensive depending on WHERE it errs.
    # ☠️ READ IT AS AN UPPER BOUND ON DISAGREEMENT, NOT AS EXPECTED GAIN. It assumes SF's value is the right
    # one, which is precisely the assumption this project has falsified: tuning KS toward SF's magnitude
    # gave +9.70% worst-case error, and SIX concepts sized this way went on to measure MOVE-NULL. Calibrate
    # against the whole-eval prize (~7pp) and the cross-set resolution bar (~2-2.5pp).
    def winpct(cp):
        import math
        cp = max(-1500.0, min(1500.0, cp))
        return 100.0 / (1.0 + math.exp(-0.00368208 * cp))

    print("\n  %-14s %-24s %7s %7s %7s %7s %6s %8s %8s %9s" %
          ("SF row", "v2 counterpart", "both", "SFonly", "usonly", "neither",
           "sign+", "mean|SF|", "mean|us|", "Δwin% pp"))
    for sfcol, ours in PAIRS:
        label = " + ".join(ours) if ours else "(none -- PARKED)"
        c_both = c_sf = c_us = c_no = c_sign = 0
        s_sf = s_us = 0.0
        dw_sum = 0.0
        dw_n = 0
        for o in out_rows:
            try:
                sf = float(o[sfcol])
            except Exception:
                continue
            vals = [o.get("v2_" + k, "") for k in ours] if ours else []
            mine = None if (not ours or all(v == "" for v in vals)) \
                else sum(float(v) for v in vals if v != "")
            # A parked or gated-off term genuinely contributes 0 to `total`, so 0 is its honest comparand
            # for the counterfactual even though it is ABSENT from the breakdown.
            contrib = mine if mine is not None else 0.0
            try:
                tot = float(o["v2_total"])
            except Exception:
                tot = None
            if tot is not None:
                base = winpct(tot * 100.0)
                swapped = winpct((tot - contrib + sf) * 100.0)
                dw_sum += abs(swapped - base); dw_n += 1
            sf_on = abs(sf) >= SEES
            us_on = (mine is not None and abs(mine) >= SEES)
            if sf_on and us_on:
                c_both += 1; s_sf += abs(sf); s_us += abs(mine)
                if (sf > 0) == (mine > 0):
                    c_sign += 1
            elif sf_on:
                c_sf += 1
            elif us_on:
                c_us += 1
            else:
                c_no += 1
        tot = c_both + c_sf + c_us + c_no
        if not tot:
            continue
        print("  %-14s %-24s %7d %7d %7d %7d %5s %8.2f %8.2f %9.2f" %
              (sfcol.replace("sf11_", ""), label, c_both, c_sf, c_us, c_no,
               ("%.0f%%" % (100.0 * c_sign / c_both)) if c_both else "-",
               (s_sf / c_both) if c_both else 0.0, (s_us / c_both) if c_both else 0.0,
               (dw_sum / dw_n) if dw_n else 0.0))

    print("\n  ⚠️ `kingsafety` IS NOT A VALID PAIR: SF's row is shelter + storm + kingDanger")
    print("     (evaluate.cpp:383-384 seeds it with pe->king_safety, the pawn shelter), while v2's KS is")
    print("     attack-units ONLY. SF is non-silent almost everywhere because shelter always evaluates.")
    print("     The apparent gap is mostly DEFINITIONAL. ★ The real observation is that v2 carries no")
    print("     shelter/storm concept at all.\n")

    print("  (legacy one-sided view, kept for continuity -- read the 2x2 above instead)")
    print("  %-16s %-26s %8s %9s %9s %10s %10s" %
          ("SF row", "v2 counterpart", "n |SF|>=", "SILENT", "%", "mean|SF|", "mean|ours|"))
    for sfcol, ours in PAIRS:
        label = " + ".join(ours) if ours else "(none -- PARKED)"
        n_sees = n_sil = 0
        sum_sf = sum_ours = 0.0
        for o in out_rows:
            try:
                sf = float(o[sfcol])
            except Exception:
                continue
            if abs(sf) < SEES:
                continue
            n_sees += 1; sum_sf += abs(sf)
            if not ours:
                n_sil += 1
                continue
            # ☠️ Sum the PRESENT members only. The first version counted the pair silent if ANY member was
            # absent, which made `v2_placement + v2_rookfile` read 100% silent purely because rook files are
            # gated off -- masking what placement actually scores. A gated-off member genuinely contributes
            # 0 to `total`, so summing the present ones IS the honest comparand; only an ENTIRELY absent
            # pair means "we carry nothing here".
            vals = [o.get("v2_" + k, "") for k in ours]
            if all(v == "" for v in vals):
                n_sil += 1
                continue
            mine = sum(float(v) for v in vals if v != "")
            sum_ours += abs(mine)
            if abs(mine) < SILENT:
                n_sil += 1
        if not n_sees:
            continue
        print("  %-16s %-26s %8d %9d %8.1f%% %10.2f %10.2f" %
              (sfcol.replace("sf11_", ""), label, n_sees, n_sil, 100.0 * n_sil / n_sees,
               sum_sf / n_sees, (sum_ours / n_sees) if ours else 0.0))
    print("\n⚠️ PARKED rows read 100%% silent BY CONSTRUCTION -- imbalance/threats/space were each measured")
    print("   and parked this month. They are pre-answered, NOT discoveries.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
