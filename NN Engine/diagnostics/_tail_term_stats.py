# -*- coding: utf-8 -*-
"""Per-term stats for two FEN sets, with the two statistics the signed-mean view cannot show.

WHY THIS EXISTS. `sf11_collapse_gap.py` reports the MEAN SIGNED value per term. That is the right statistic
for "does SF11 think this concept is against us", but it hides two things this investigation needs:

  (1) MAGNITUDE. If a component is -6 in half the positions and +6 in the other half, its signed mean is ~0
      and it looks harmless -- yet a term that large is a large error source whenever it is slightly wrong.
      Our per-piece placement components (pt_*) were observed at +/-3..6 pawns against SF11's +/-0.1..0.9 in
      hand-read positions; mean |value| is what tests whether that holds across the set.
      ⚠️ pt_* are a DECOMPOSITION of `pieces` (verified: they sum to it), not additive extra terms.

  (2) SPREAD. A difference in signed means across two 40-position sets means nothing without it. The
      observed SF11 King-safety split (-0.48 tail A vs -0.11 tail B) needs a standard error before it is a
      result -- the record is full of narrativised aggregates that dissolved (per-theme STS deltas, the
      LMR_SHAPE story). This prints mean, SD and SE so the separation can be judged, not eyeballed.

Reuses SF11Eval + ev_breakdown exactly as sf11_collapse_gap does, so numbers are comparable to it.

  pyrun diagnostics/_tail_term_stats.py A=<fensA.txt> B=<fensB.txt> [LABEL_A=..] [LABEL_B=..]
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import sys
import math

OPTS = {}
for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        OPTS[_k] = _v

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, ENGINE_DIR)
sys.path.insert(0, THIS_DIR)

import chess
from eval_vs_sf11 import SF11Eval, SF11


def load(path):
    out = []
    with open(path) as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            if '\t' in ln:
                ln = ln.split('\t')[-1].strip()
            out.append(ln)
    return out


def stats(vals):
    n = len(vals)
    if n == 0:
        return 0.0, 0.0, 0.0
    m = sum(vals) / n
    var = sum((v - m) ** 2 for v in vals) / n if n > 1 else 0.0
    sd = math.sqrt(var)
    return m, sd, sd / math.sqrt(n) if n else 0.0


def collect(fens, ai, sf11):
    """Return (our_terms, sf_terms) as {term: [values in OUR pov]}."""
    ours, sfs = {}, {}
    for fen in fens:
        b = chess.Board(fen)
        pov = 1.0 if b.turn == chess.WHITE else -1.0
        try:
            br = ai.ev_breakdown(b)
        except Exception:
            continue
        for k, v in br.items():
            if k == 'total':
                continue
            ours.setdefault(k, []).append(pov * (v / 1000.0))
        try:
            _tot, terms = sf11.eval(fen)
        except Exception:
            terms = {}
        for k, v in (terms or {}).items():
            if k.lower() == 'total':
                continue
            sfs.setdefault(k, []).append(pov * v)
    return ours, sfs


def main():
    pa, pb = OPTS.get('A'), OPTS.get('B')
    if not pa or not pb:
        sys.exit('need A=<fens> B=<fens>')
    la = OPTS.get('LABEL_A', os.path.basename(pa))
    lb = OPTS.get('LABEL_B', os.path.basename(pb))

    # Construct exactly as sf11_collapse_gap.py:64-66 does — the two keras models are unused by
    # ev_breakdown but the ctor signature requires them, so None/None with a seed board is the house form.
    from ChessAI import ChessAI
    seed = chess.Board()
    ai = ChessAI(None, None, seed, seed.turn)

    sf11 = SF11Eval(SF11)
    try:
        fa, fb = load(pa), load(pb)
        oa, sa = collect(fa, ai, sf11)
        ob, sb = collect(fb, ai, sf11)
    finally:
        sf11.close()

    print("A = %s (n=%d)   B = %s (n=%d)\n" % (la, len(fa), lb, len(fb)))

    print("=== SF11 terms: SIGNED mean +/- SE, both sets. Separation is real only if |dA-dB| > ~2x the SEs ===")
    print("  %-16s %18s %18s %10s" % ("SF11 term", "A mean+/-SE", "B mean+/-SE", "A-B"))
    for k in sorted(set(sa) | set(sb), key=lambda x: -abs(stats(sa.get(x, []))[0] - stats(sb.get(x, []))[0])):
        ma, _, ea = stats(sa.get(k, []))
        mb, _, eb = stats(sb.get(k, []))
        print("  %-16s %+10.3f+/-%-5.3f %+10.3f+/-%-5.3f %+10.3f" % (k, ma, ea, mb, eb, ma - mb))

    print("\n=== MAGNITUDE: mean |value| per term (what the signed view hides) ===")
    print("  %-22s %10s %10s   %-16s %10s %10s" % ("our term", "|A|", "|B|", "SF11 term", "|A|", "|B|"))
    ours_mag = sorted(set(oa) | set(ob), key=lambda x: -stats([abs(v) for v in oa.get(x, [0])])[0])
    sf_mag = sorted(set(sa) | set(sb), key=lambda x: -stats([abs(v) for v in sa.get(x, [0])])[0])
    for i in range(max(len(ours_mag), len(sf_mag))):
        lhs = rhs = ""
        if i < len(ours_mag):
            k = ours_mag[i]
            lhs = "  %-22s %10.3f %10.3f" % (k, stats([abs(v) for v in oa.get(k, [])])[0],
                                             stats([abs(v) for v in ob.get(k, [])])[0])
        else:
            lhs = " " * 45
        if i < len(sf_mag):
            k = sf_mag[i]
            rhs = "   %-16s %10.3f %10.3f" % (k, stats([abs(v) for v in sa.get(k, [])])[0],
                                              stats([abs(v) for v in sb.get(k, [])])[0])
        print(lhs + rhs)


if __name__ == '__main__':
    main()
