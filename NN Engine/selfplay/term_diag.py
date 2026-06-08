# -*- coding: utf-8 -*-
"""Eval term-attribution — WHICH of our static-eval terms drive the divergence from NNUE.

deep_diag.py established that our static eval's win%-gap vs SF's NNUE static eval is largest in the
endgame (and is a pure static-vs-static, depth-independent gap). This localizes the culprit using the
per-term + per-piece-type partials already recorded in each annotated position's `eval_breakdown`
(no rebuild, no re-run, no engine — pure offline read).

All term values are stored ABSOLUTE Black-positive milli-pawns and sum to `total`; we convert each to
White-POV cp = -x/10 (so + favours White), matching sf_static_cp (White-POV cp, NNUE static).

  [A] NEAR-EQUAL precision (PRIORITY): positions NNUE calls ~level (|sf_static|<50cp). Our eval should
      read ~0 there; whatever it reads is miscalibration. After the color-symmetry fix the SIGNED mean
      ~0 over a balanced set, so mean|.| / std are the tells -- the term with the biggest spread in
      objectively-equal positions injects the scatter. Ranked for endgame vs midgame.
  [B] WINNING-SIDE bias: in won positions, which term over-rewards the winning side (the +-80..120cp
      winning-side-hot read). A term large-+ when White's winning AND large-- when Black's winning is a
      symmetric driver.
  [C] endgame-eval activation: is advanced_endgame_eval even firing where we diverge?

Correlational (NNUE has no terms): localizes which of OUR terms drives OUR divergence, not a term diff.

Run from NN Engine/:  python selfplay/term_diag.py --tag hlmr_more1_lightning [more tags ...]
"""

import os
import sys
import glob
import argparse
import statistics as st

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
from deep_diag import load_game, winpct, phase_of, piece_count, our_static_cp  # noqa: E402

GAMES_DIR = os.path.join(THIS_DIR, "games")
NEAR_EQ_CP = 50            # |sf_static| below this => NNUE says ~level
WIN_CP = 100               # |sf_static| above this => one side winning (Cut B)

ADDITIVE = ["pieces", "central", "capture_gains", "passed_pawn_support", "latent_threat",
            "imbalance_white", "imbalance_black", "pair_bonus", "piece_value_boost"]
PT = ["pt_pawns", "pt_knights", "pt_bishops", "pt_rooks", "pt_queens", "pt_kings"]
ALLTERMS = ADDITIVE + PT
ENDGAME_PH = {"4.early-end ", "5.late-end  "}
PHASES = ["1.opening   ", "2.early-mid ", "3.late-mid  ", "4.early-end ", "5.late-end  "]


def cp(x):
    return -(x or 0) / 10.0


def _stat_block(label, rows):
    # rows: term -> list of White-POV cp; print ranked by mean|.|
    print(f"  {label}")
    print(f"     {'term':<20} {'mean':>8} {'mean|.|':>8} {'std':>8}")
    ranked = sorted(rows, key=lambda t: (st.mean([abs(v) for v in rows[t]]) if rows[t] else 0), reverse=True)
    for t in ranked:
        xs = rows[t]
        if not xs:
            continue
        print(f"     {t:<20} {st.mean(xs):>8.1f} {st.mean([abs(v) for v in xs]):>8.1f} {st.pstdev(xs):>8.1f}")


def analyze(tag):
    tdir = os.path.join(GAMES_DIR, tag)
    # accumulators
    neq = {ph: {t: [] for t in ALLTERMS} for ph in PHASES}     # near-equal, per phase, per term (cp)
    neq_static = {ph: [] for ph in PHASES}                      # near-equal |our_static| & winpct gap
    neq_scatter = {ph: [] for ph in PHASES}                     # |our_winpct - sf_winpct| in near-eq
    ww = {t: [] for t in ALLTERMS}                              # white-winning per term (cp, signed)
    bw = {t: [] for t in ALLTERMS}                              # black-winning per term
    adv = {ph: [0, 0, []] for ph in PHASES}                     # [fired, total, [adv_total when fired]]
    n = 0

    for gdir in sorted(glob.glob(os.path.join(tdir, "game_*"))):
        moves = load_game(gdir)
        if not moves:
            continue
        for m in moves:
            if m.get("label") in (None, "opening") or m.get("booked"):
                continue
            eb = m.get("eval_breakdown")
            sf = m.get("sf_static_cp")
            if eb is None or sf is None:
                continue
            ph = phase_of(piece_count(m["fen"]))
            adv[ph][1] += 1
            if eb.get("advanced_endgame_fired"):
                adv[ph][0] += 1
                adv[ph][2].append(cp(eb.get("advanced_endgame_total", 0)))
            n += 1
            our = our_static_cp(eb)
            if abs(sf) < NEAR_EQ_CP:
                neq_static[ph].append(our)
                neq_scatter[ph].append(abs(winpct(our) - winpct(sf)))
                for t in ALLTERMS:
                    neq[ph][t].append(cp(eb.get(t, 0)))
            if sf > WIN_CP:
                for t in ALLTERMS:
                    ww[t].append(cp(eb.get(t, 0)))
            elif sf < -WIN_CP:
                for t in ALLTERMS:
                    bw[t].append(cp(eb.get(t, 0)))

    print("=" * 80)
    print(f"{tag} — eval term-attribution  ({n} non-opening positions; terms in White-POV cp)")
    print("=" * 80)

    print("\n[A] NEAR-EQUAL precision  (positions |SF_static| < %dcp; our eval should read ~0)" % NEAR_EQ_CP)
    print("  how hot we read when NNUE says level, by phase:")
    print(f"     {'phase':<14} {'n':>5} {'mean our_cp':>12} {'mean|our_cp|':>13} {'win%-scatter':>13}")
    for ph in PHASES:
        xs = neq_static[ph]; sc = neq_scatter[ph]
        if xs:
            print(f"     {ph:<14} {len(xs):>5} {st.mean(xs):>12.1f} {st.mean([abs(v) for v in xs]):>13.1f} {st.mean(sc):>12.1f}%")

    def merge(phs):
        out = {t: [] for t in ALLTERMS}
        for ph in phs:
            for t in ALLTERMS:
                out[t] += neq[ph][t]
        return out

    print("\n  -- term contributions in near-equal ENDGAME positions (ranked by mean|.|) --")
    _stat_block("ENDGAME (early-end + late-end):", merge(ENDGAME_PH))
    print("\n  -- term contributions in near-equal MIDGAME positions (for contrast) --")
    _stat_block("MIDGAME (opening + early/late-mid):", merge([p for p in PHASES if p not in ENDGAME_PH]))

    print("\n[B] WINNING-SIDE bias  (mean signed White-POV cp per term; |SF_static|>%dcp)" % WIN_CP)
    print(f"     {'term':<20} {'White-winning':>14} {'Black-winning':>14} {'symmetric drive':>16}")
    ranked = sorted(ALLTERMS, key=lambda t: ((st.mean(ww[t]) if ww[t] else 0) - (st.mean(bw[t]) if bw[t] else 0)), reverse=True)
    for t in ranked:
        w = st.mean(ww[t]) if ww[t] else 0.0
        b = st.mean(bw[t]) if bw[t] else 0.0
        print(f"     {t:<20} {w:>14.1f} {b:>14.1f} {w - b:>16.1f}")
    print(f"     (n: White-winning {len(ww['pieces'])}, Black-winning {len(bw['pieces'])})")

    print("\n[C] endgame-eval activation (advanced_endgame_eval)")
    print(f"     {'phase':<14} {'fired/total':>14} {'%':>7} {'mean adv_total(cp) when fired':>30}")
    for ph in PHASES:
        f, tot, vals = adv[ph]
        if tot:
            mt = (st.mean(vals) if vals else 0.0)
            print(f"     {ph:<14} {f:>6}/{tot:<7} {100*f/tot:>6.1f}% {mt:>30.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", nargs="+", required=True)
    args = ap.parse_args()
    for t in args.tag:
        analyze(t)


if __name__ == "__main__":
    main()
