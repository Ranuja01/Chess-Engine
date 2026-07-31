# -*- coding: utf-8 -*-
"""Item 1 (conviction test A) + Item 2 validation — STAGE 4 (leaf visibility). Walk each over-push line N
plies forward THROUGH the refutation (overpush_refutations.csv PV), then compare OUR static vs SF11 static
at the post-refutation leaf (totals, our-POV cp). If SF11 registers the damage (leaf bad for us) and WE
don't (we read it higher), stage 4 is convicted: our shallow leaf doesn't see the refutation's consequence.

  # baseline (no threats):
  overnight_runner.sh pyrun diagnostics/leaf_visibility_test.py [--ply 4]
  # Item-2 medicine validation (ours-ON): pass the knobs (parsed into env before ChessAI init)
  overnight_runner.sh pyrun diagnostics/leaf_visibility_test.py --ply 4 ENABLE_THREATS=true THREATS_STANDING_ONLY=true

gap = our_static_ourpov - sf11_static_ourpov at the leaf.  gap>0 = WE over-read the leaf (miss the damage).
Medicine works (Item 2) if turning threats on moves our leaf reading DOWN toward SF11 by >= 1/3 of the gap.
"""
import os, sys, csv
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
THIS = os.path.dirname(os.path.abspath(__file__)); ENGINE = os.path.dirname(THIS)
sys.path.insert(0, THIS); sys.path.insert(0, ENGINE)
for _kv in [a for a in sys.argv[1:] if "=" in a and not a.startswith("-") and not a.endswith(".csv")]:
    _k, _v = _kv.split("=", 1); os.environ[_k] = _v
argv = [a for a in sys.argv[1:] if not ("=" in a and not a.startswith("-") and not a.endswith(".csv"))]
PLY = 4
if "--ply" in argv: i = argv.index("--ply"); PLY = int(argv[i+1]); del argv[i:i+2]

import chess
from ChessAI import ChessAI
from eval_vs_sf11 import SF11Eval, SF11

REFMAP = os.path.join(THIS, "overpush_refutations.csv")


def our_ourpov(ai, fen, us_white):
    bd = ai.ev_breakdown(chess.Board(fen))
    if bd.get("checkmate"): return None
    return (-bd["total"] / 1000.0) * 100.0 * (1 if us_white else -1)     # our-POV cp


def sf11_ourpov(sf11, fen, us_white):
    tot, _ = sf11.eval(fen)
    return None if tot is None else (tot if us_white else -tot) * 100.0  # our-POV cp


def main():
    rows = list(csv.DictReader(open(REFMAP)))
    seed = chess.Board(); ai = ChessAI(None, None, seed, seed.turn)
    sf11 = SF11Eval(SF11)
    knobs = {k: os.environ[k] for k in ("ENABLE_THREATS", "THREATS_STANDING_ONLY") if k in os.environ}
    print(f"[leaf-vis] ply={PLY}  n_rows={len(rows)}  knobs={knobs or '(baseline)'}")
    gaps = []; our_l = []; sf_l = []; skipped = 0
    for r in rows:
        us_white = chess.Board(r["overpush_fen"]).turn == chess.WHITE   # mover at the push = us
        pv = r["pv"].split()
        try:
            b = chess.Board(r["opp_fen"])
            for u in pv[:PLY]:
                b.push(chess.Move.from_uci(u))
            leaf = b.fen()
            o = our_ourpov(ai, leaf, us_white); s = sf11_ourpov(sf11, leaf, us_white)
        except Exception:
            skipped += 1; continue
        if o is None or s is None: skipped += 1; continue
        gaps.append(o - s); our_l.append(o); sf_l.append(s)
    sf11.close()
    if not gaps:
        print("  no leaves evaluated"); return
    n = len(gaps)
    over = sum(1 for g in gaps if g >= 50)
    print(f"  leaves n={n} (skipped {skipped})")
    print(f"  mean OUR leaf (our-POV cp) = {sum(our_l)/n:+.0f}   mean SF11 leaf = {sum(sf_l)/n:+.0f}")
    print(f"  mean GAP (our - sf11) = {sum(gaps)/n:+.0f}   % we over-read leaf (gap>=50): {100*over/n:.0f}%")
    print("  READ: SF11 leaf << OUR leaf (big +gap) => WE don't register the refutation's damage => stage 4 convicted.")
    print("        For Item-2: compare this mean-OUR-leaf baseline vs threats-ON; ON must drop >= 1/3 of the gap toward SF11.")


if __name__ == "__main__":
    main()
