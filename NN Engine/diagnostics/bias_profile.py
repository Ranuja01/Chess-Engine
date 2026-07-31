# -*- coding: utf-8 -*-
"""Aggregate over-optimism profile: our static eval vs SF static eval, bucketed by who is winning, with the
per-term breakdown that localizes the driver. Reproduces the 2026-06-05 winning-side-bias measurement on a fresh
corpus at the current build (Step 0 of the eval-calibration program). SF_static (NNUE, UCI `eval`) is the
apples-to-apples reference for our STATIC eval (both leaf, no search).

    python diagnostics/bias_profile.py --tag vssf1_def_d8 --n 400 [--stride 7]

Samples ply FENs from selfplay/games/<tag>/game_*.jsonl, computes our_static (ChessAI.ev_breakdown) + SF_static
per FEN, buckets by SF_static sign (white/black winning vs balanced), and reports the mean calibration gap
(our_static - SF_static) + mean per-term contribution per bucket. If our eval over-rewards the winning side, the
white-winning bucket gap is strongly +, the black-winning bucket strongly - , and one term (expected: `pieces`)
dominates the split.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys, glob, json, argparse
import chess

THIS = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(THIS)
sys.path.insert(0, ENGINE); sys.path.insert(0, os.path.join(ENGINE, "selfplay"))

UPP = 1000.0  # engine units per pawn (absolute, Black-positive)
def wp(ev):   # engine abs -> White-POV pawns
    return -ev / UPP
TERMS = ["pieces", "capture_gains", "passed_pawn_support", "latent_threat", "king_safety", "central",
         "imbalance_white", "imbalance_black", "pair_bonus", "piece_value_boost"]


def sample_fens(tag, n, stride):
    paths = sorted(glob.glob(os.path.join(ENGINE, "selfplay", "games", tag, "game_*.jsonl")))
    fens = []
    for p in paths:
        try:
            lines = [l for l in open(p) if l.strip()]
        except Exception:
            continue
        for i, l in enumerate(lines):
            if i % stride:
                continue
            try:
                r = json.loads(l)
            except Exception:
                continue
            f = r.get("fen")
            if f:
                fens.append(f)
        if len(fens) >= n * 3:
            break
    # even spread across the collected pool
    if len(fens) > n:
        step = len(fens) / n
        fens = [fens[int(k * step)] for k in range(n)]
    return fens


def peak_fens(tag, n):
    """Per game, the OUR-move record with the highest our_pov_eval (the over-optimistic peak before collapse).
    Returns list of (fen, recorded_search_peak_white_pawns)."""
    paths = sorted(glob.glob(os.path.join(ENGINE, "selfplay", "games", tag, "game_*.jsonl")))
    out = []
    for p in paths:
        best = None
        try:
            for l in open(p):
                if not l.strip():
                    continue
                r = json.loads(l)
                e = r.get("our_pov_eval")
                if isinstance(e, int) and r.get("fen") and abs(e) < 90000:
                    if best is None or e > best[1]:
                        best = (r["fen"], e)
        except Exception:
            continue
        if best and best[1] >= 1500:          # only games where we thought we were clearly winning (+1.5)
            out.append((best[0], best[1] / 1000.0))
        if len(out) >= n:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--stride", type=int, default=7)
    ap.add_argument("--win", type=float, default=1.0, help="|SF_static| pawns to count as winning")
    ap.add_argument("--peak", action="store_true", help="profile per-game over-optimistic PEAK positions (3-way)")
    ap.add_argument("--sf-depth", type=int, default=12, help="SF search depth for SF_search in --peak")
    a = ap.parse_args()

    from arbiter import Arbiter, find_stockfish
    sf = find_stockfish()
    arb = Arbiter(sf, depth=None) if sf else None
    if arb is None:
        print("[bias] no Stockfish (set STOCKFISH_PATH) — cannot compute SF_static"); return

    from ChessAI import ChessAI
    seed = chess.Board(); ai = ChessAI(None, None, seed, seed.turn)

    if a.peak:
        arb_s = Arbiter(sf, depth=a.sf_depth)   # SF search
        rows = peak_fens(a.tag, a.n)
        g_os_ss, g_ss_sr, g_os_sr = [], [], []
        os_list, ss_list, sr_list, peak_list = [], [], [], []
        pieces_gap = []
        print("PEAK profile tag=%s  n=%d  (White-POV pawns; our over-optimistic peak positions)" % (a.tag, len(rows)))
        print("  our_static | SF_static | SF_search | our_SEARCH_peak   (+favours White)")
        shown = 0
        for fen, search_peak in rows:
            try:
                b = chess.Board(fen)
                bd = ai.ev_breakdown(b)
                if bd.get("checkmate"): continue
                sfs = arb.evaluate_static(b); sfr, _, _ = arb_s.evaluate(b)
            except Exception:
                continue
            if sfs is None or sfr is None: continue
            clamp = lambda x: max(-15.0, min(15.0, x))   # kill mate-score outliers
            os_, ss, sr = clamp(wp(bd["total"])), clamp(sfs / 100.0), clamp(sfr / 100.0)
            g_os_ss.append(os_ - ss); g_ss_sr.append(ss - sr); g_os_sr.append(os_ - sr)
            os_list.append(os_); ss_list.append(ss); sr_list.append(sr); peak_list.append(search_peak)
            pieces_gap.append(wp(bd["pieces"]))
            if shown < 12:
                print("  %+8.2f  %+8.2f  %+8.2f   %+8.2f     %s" % (os_, ss, sr, search_peak, fen))
                shown += 1
        arb.close(); arb_s.close()
        def mean(v): return sum(v)/len(v) if v else 0.0
        print("\n  MEANS over %d peaks:  our_static %+.2f | SF_static %+.2f | SF_search %+.2f | our_SEARCH_peak %+.2f"
              % (len(os_list), mean(os_list), mean(ss_list), mean(sr_list), mean(peak_list)))
        print("    our_static - SF_static  = %+.3f   (>0 => our STATIC over-reads = static miscalibration)" % mean(g_os_ss))
        print("    SF_static  - SF_search  = %+.3f   (>0 => even SF's static over-reads; pos worth less = conversion)" % mean(g_ss_sr))
        print("    our_static - SF_search  = %+.3f   (the full static gap to truth)" % mean(g_os_sr))
        print("    our_SEARCH_peak - our_static = %+.3f   (>>0 => our SEARCH inflates beyond the static read)"
              % (mean(peak_list) - mean(os_list)))
        return

    fens = sample_fens(a.tag, a.n, a.stride)
    buckets = {"white_win": [], "black_win": [], "balanced": []}
    term_sums = {b: {t: 0.0 for t in TERMS} for b in buckets}
    n_ok = 0
    for fen in fens:
        try:
            b = chess.Board(fen)
            bd = ai.ev_breakdown(b)
            if bd.get("checkmate"):
                continue
            sfc = arb.evaluate_static(b)
            if sfc is None:
                continue
        except Exception:
            continue
        sf_static = sfc / 100.0                 # White-POV pawns
        our_static = wp(bd["total"])
        gap = our_static - sf_static
        bk = "white_win" if sf_static > a.win else "black_win" if sf_static < -a.win else "balanced"
        buckets[bk].append(gap)
        for t in TERMS:
            term_sums[bk][t] += wp(bd[t])
        n_ok += 1

    if arb: arb.close()

    def mean(v): return sum(v) / len(v) if v else 0.0
    print("bias_profile tag=%s  n=%d  (White-POV pawns; gap = our_static - SF_static)" % (a.tag, n_ok))
    for bk in ("white_win", "balanced", "black_win"):
        v = buckets[bk]
        print("\n=== %s  (n=%d) ===  mean calibration gap = %+.3f" % (bk, len(v), mean(v)))
        if v:
            tm = sorted(((t, term_sums[bk][t] / len(v)) for t in TERMS), key=lambda kv: abs(kv[1]), reverse=True)
            print("  mean per-term contribution (driver = largest split white_win vs black_win):")
            for t, m in tm[:6]:
                print("     %-22s %+7.3f" % (t, m))


if __name__ == "__main__":
    main()
