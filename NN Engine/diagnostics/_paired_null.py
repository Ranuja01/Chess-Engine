# -*- coding: utf-8 -*-
"""POPULATION-MATCHED null: read an arm against the null ON THE ARM'S OWN CHANGED POSITIONS. Zero CPU.

WHY THIS EXISTS (2026-09-09). `_ks_footprint_regret` reports win% on the positions where the CANDIDATE
changed our move, and we compare it to a null arm's GLOBAL win%. Those are different populations, and the
difference is systematic, not incidental:

    arm            flip%   reg_base
    noise100       37.9%    3.633     <- neutral arms all sit HERE
    asp300         34.7%    3.652
    noise30        34.8%    3.617
    threats A      26.7%    3.938     <- every candidate sits HERE
    ks_off         25.9%    4.029
    ks_1500        21.6%    4.073
    ks_4500        21.7%    4.073

Lower flip rate => systematically HIGHER base regret: a knob that flips fewer moves flips only where the
base's top two were closest, i.e. where the base is most likely wrong and ANY perturbation regresses toward
better. So every candidate today was read against a null measured on a materially EASIER population.

☠️ Attempts to build a rate-matched neutral arm FAILED: EVAL_NOISE_SIGMA saturates (30 -> 34.8%,
100 -> 37.9%) because a large population of near-tied moves flips under any perturbation. You cannot dial a
neutral arm down to 22%.

▶️ So match the POPULATION instead of the rate. For the FENs where BOTH the arm and the null changed the
move, compute both win%s on that identical set. Same positions, same selection, no rate mismatch.

⚠️ LIMITATION, stated up front: the intersection is biased toward EASY-TO-FLIP positions (both arms found
them marginal). It is not the arm's full changed set. It is much closer to matched than a global null, and
it is the only matched comparison available without a neutral arm we cannot construct.
★ Run SEVERAL nulls -- the neutral arms disagree by ~1pp among themselves (49.0 / 50.0 / 50.3), so one
paired null gives a number and several give the band.

  pyrun diagnostics/_paired_null.py ARM=diagnostics/_fpdump_ks_off.csv \
      NULLS=diagnostics/_fpdump_null_asp300.csv,diagnostics/_fpdump_null_noise30.csv,diagnostics/_fpdump_null_noise100.csv

Dump schema: fen, base_move, cand_move, reg_base, reg_cand, delta   (delta = reg_cand - reg_base)
"""
import os, sys, csv

for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v

ARM = os.environ.get("ARM", "diagnostics/_fpdump_ks_off.csv")
NULLS = [p for p in os.environ.get("NULLS", "").split(",") if p.strip()]
if not NULLS:
    print("need NULLS=a.csv,b.csv")
    sys.exit(2)


def load(path):
    """-> {fen: (better, base_regret)}; better = candidate's move scored better (delta < 0). Ties dropped.

    ⚠️ The dump's header is LEGACY KS-SPECIFIC -- `_ks_footprint_regret` writes
    [fen, ks_on_move, ks_off_move, reg_ks_on, reg_ks_off, delta] for EVERY arm, not just KS ones, so
    `reg_ks_on` is the BASE regret whatever the candidate was. Accept both namings rather than assume:
    reading `reg_base` here silently dropped every row on the first run.
    """
    out = {}
    try:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    d = float(row["delta"])
                except (KeyError, ValueError):
                    continue
                rb = row.get("reg_ks_on", row.get("reg_base", ""))
                try:
                    rb = float(rb)
                except ValueError:
                    rb = float("nan")
                if d == 0.0:
                    continue
                out[row["fen"]] = (d < 0.0, rb)
    except FileNotFoundError:
        print("MISSING: %s" % path)
        sys.exit(2)
    return out


def winpct(rows, keys):
    b = sum(1 for k in keys if rows[k][0])
    n = len(keys)
    return (100.0 * b / n if n else float("nan")), n


MINPC = int(os.environ.get("MINPC", "0"))   # 26 == phase_of()'s `opening` (pawn-INCLUSIVE total pieces)
if MINPC:
    import chess

    def _keep(fen):
        try:
            return len(chess.Board(fen).piece_map()) >= MINPC
        except ValueError:
            return False
else:
    def _keep(fen):
        return True

arm = load(ARM)
if MINPC:
    arm = {f: v for f, v in arm.items() if _keep(f)}
print("ARM: %s  (%d changed%s, mean reg_base %.4f)\n"
      % (os.path.basename(ARM), len(arm), (" at >=%d pieces" % MINPC) if MINPC else "",
         sum(v[1] for v in arm.values()) / max(1, len(arm))))
print("  %-26s %7s   %8s %8s %9s   %8s" % ("paired against", "n_both", "arm%", "null%", "arm-null", "sigma"))

deltas = []
for np_ in NULLS:
    nul = load(np_)
    both = sorted(set(arm) & set(nul))
    if not both:
        print("  %-26s %7d   no overlap" % (os.path.basename(np_), 0))
        continue
    pa, n = winpct(arm, both)
    pn, _ = winpct(nul, both)
    # Paired on the SAME positions, but the two arms' outcomes are distinct measurements; treat as
    # independent proportions on n. Conservative at p~0.5.
    se = (0.5 / n) ** 0.5 * 100.0
    d = pa - pn
    deltas.append(d)
    print("  %-26s %7d   %7.1f%% %7.1f%%  %+8.1f   %6.1f" % (os.path.basename(np_), n, pa, pn, d, d / se))

if len(deltas) > 1:
    m = sum(deltas) / len(deltas)
    spread = max(deltas) - min(deltas)
    print("\n  mean arm-null across %d nulls: %+.1fpp   (spread %.1fpp)" % (len(deltas), m, spread))
    print("  ⚠️ The neutral arms disagree by ~1pp among THEMSELVES. If the spread here is comparable to")
    print("     the mean, the reading is inside the null band and is NOT a result.")
print("\n  ⚠️ Intersection is biased toward easy-to-flip positions. Compare the ARM's paired win% to its")
print("     GLOBAL win% from _ks_footprint_regret: a large gap means the pairing changed the population,")
print("     and the paired number is the more honest one.")
