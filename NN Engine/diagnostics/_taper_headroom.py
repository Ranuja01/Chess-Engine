# -*- coding: utf-8 -*-
"""How much could a PHASE-TAPERED piece-value table actually buy us? (headroom screen, no rebuild)

Our values are FLAT (cpp_bitboard.h:141 -> P 1000, N 3250, B 3450, R 5000, Q 10000). SF taperes them
(SF15 types.h:189-193): pawn +65% MG->EG, pieces only +6-11%, so its knight/pawn ratio falls 6.20 -> 4.11.
The 2026-07-30 collapse showed us reading a 3-extra-pawns-vs-a-knight endgame as winning for the pawns
while SF11/SF15/SF18 all read it for the knight.

Rather than guess, this measures the CEILING: for each banked position, add the delta a tapered table WOULD
have produced to our existing total, then score win%-space loss against the SF18 label. alpha sweeps from
0 (our current flat table) to 1 (full SF-shaped ratios). If the loss curve is flat or rises, tapering cannot
help and we stop; if it dips, the dip size is the most it could ever buy STATICALLY.

    delta(alpha) = SUM_pt  (count_w - count_b) * alpha * (V_phase(pt) - V_flat(pt))
    V_phase interpolates our flat value toward the SF-shaped MG/EG pair using the stored phase_score.

⚠️ A static-fit gain is NOT Elo (see corpus-fit-flattens-eval, eval-accuracy-payoff-is-pruning). This
screens OUT a dead idea cheaply; it cannot promote one.

Run: pyrun diagnostics/_taper_headroom.py
"""
import os, sys, csv, math
THIS = os.path.dirname(os.path.abspath(__file__))
BANK = os.path.join(THIS, "ks_sets", "position_bank.csv")
import chess

WIN_K = 0.00368208                       # Lichess cp->win% sigmoid, matches cploss_frozen.py / moves_dump.py
def winpct(cp): return 50.0 + 50.0 * (2.0 / (1.0 + math.exp(-WIN_K * cp)) - 1.0)

# ours, millipawns (flat). PHASE convention: 0 = opening ... 128 = endgame.
FLAT = {chess.PAWN: 1000, chess.KNIGHT: 3250, chess.BISHOP: 3450, chess.ROOK: 5000, chess.QUEEN: 10000}
# SF15 types.h, rescaled so the EG pawn == our 1000 (keeps our units; only the SHAPE is borrowed).
SF_MG = {chess.PAWN: 126, chess.KNIGHT: 781, chess.BISHOP: 825, chess.ROOK: 1276, chess.QUEEN: 2538}
SF_EG = {chess.PAWN: 208, chess.KNIGHT: 854, chess.BISHOP: 915, chess.ROOK: 1380, chess.QUEEN: 2682}
SCALE = 1000.0 / SF_EG[chess.PAWN]
SF_MG = {k: v * SCALE for k, v in SF_MG.items()}
SF_EG = {k: v * SCALE for k, v in SF_EG.items()}

def v_phase(pt, phase, alpha):
    """Blend our flat value toward the SF-shaped tapered value by alpha; phase 0=open..128=eg."""
    t = max(0.0, min(1.0, phase / 128.0))
    sf = SF_MG[pt] * (1.0 - t) + SF_EG[pt] * t
    return FLAT[pt] * (1.0 - alpha) + sf * alpha

rows = []
for r in csv.DictReader(open(BANK)):
    if not r.get("sf18") or not r.get("our_total"):
        continue
    try:
        b = chess.Board(r["fen"])
        rows.append((b, float(r["our_total"]), float(r["sf18"]), float(r.get("phase_score") or 0)))
    except Exception:
        continue

# Bank columns are in PAWNS and already WHITE-POV (our_total agrees in sign with sf11/sf15/sf18).
# sf18 is clamped at +-99 for mates; drop |sf18| > 20 where win% is saturated and the fit is meaningless.
rows = [r for r in rows if abs(r[2]) <= 20.0]
print(f"positions with SF18 labels, |sf18| <= 20: {len(rows)}")
print("units: pawns, WHITE-POV; win% via Lichess k=0.00368208 on centipawns.\n")

def counts(b):
    return {pt: len(b.pieces(pt, chess.WHITE)) - len(b.pieces(pt, chess.BLACK)) for pt in FLAT}

CT = [(b, counts(b), ours, sf18, ph) for b, ours, sf18, ph in rows]

print(f"  {'alpha':>6} {'meanWin%err':>12} {'MSE(win%)':>11} {'median|err|':>12}")
best = None
for i in range(0, 11):
    alpha = i / 10.0
    errs = []
    for b, c, ours, sf18, ph in CT:
        # delta is in millipawns -> pawns; `ours` and `sf18` are already pawns, White-POV.
        delta = sum(c[pt] * (v_phase(pt, ph, alpha) - FLAT[pt]) for pt in FLAT) / 1000.0
        e = winpct((ours + delta) * 100.0) - winpct(sf18 * 100.0)
        errs.append(e)
    mse = sum(e * e for e in errs) / len(errs)
    mae = sum(abs(e) for e in errs) / len(errs)
    med = sorted(abs(e) for e in errs)[len(errs) // 2]
    print(f"  {alpha:>6.1f} {mae:>12.3f} {mse:>11.1f} {med:>12.3f}")
    if best is None or mse < best[1]:
        best = (alpha, mse)
print(f"\n  best alpha = {best[0]:.1f}  (MSE {best[1]:.1f})")
print("  alpha 0.0 = our current flat table; a flat/rising curve means tapering cannot help here.")
