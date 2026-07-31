# -*- coding: utf-8 -*-
"""Is the over-push danger STATICALLY-CLASSICALLY expressible? Use SF11 (classical HCE, no NNUE) as the
finest-HCE reference. For each over-push FEN we compare SF11's STATIC eval (depth 0) of the position AFTER
our over-push move vs AFTER SF18's chosen move (from the classified dump):
  static_gap = sf11(after SF's move) - sf11(after our move), our-POV cp.
  static_gap >> 0  => SF11's STATIC classical eval already sees our move as worse => the danger is
                      classical-static-expressible => an HCE term CAN capture it (the eval lane is viable).
  static_gap ~ 0   => only SEARCH distinguishes them => tactical/NNUE => a static HCE term won't fix it.
Also reports SF11's per-term signature (King safety / Threats / Passed) on the pre-move position, to show
WHICH classical term flags the danger = what to build.

  overnight_runner.sh pyrun diagnostics/sf11_overpush_test.py diagnostics/collapse_classified.csv [--cat overpush]
"""
import os, sys, csv
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
THIS = os.path.dirname(os.path.abspath(__file__)); ENGINE = os.path.dirname(THIS)
sys.path.insert(0, THIS); sys.path.insert(0, ENGINE)
import chess
from eval_vs_sf11 import SF11Eval, SF11
from ChessAI import ChessAI      # our engine, for the totals-level (valid) comparison

CAT = "overpush"
args = sys.argv[1:]
if "--cat" in args: i = args.index("--cat"); CAT = args[i+1]; del args[i:i+2]
csv_path = next((a for a in args if a.endswith(".csv")), os.path.join(THIS, "collapse_classified.csv"))


def wp_ourpov(white_pov_pawns, us_white):
    return (white_pov_pawns if us_white else -white_pov_pawns) * 100.0     # -> our-POV cp


def sf11_bestmove(sf11, fen, depth):
    """Raw-UCI best move from the SF11 subprocess at fixed depth (classical search)."""
    p = sf11.p
    p.stdin.write("ucinewgame\nposition fen %s\ngo depth %d\n" % (fen, depth)); p.stdin.flush()
    while True:
        ln = p.stdout.readline()
        if not ln: return None
        if ln.startswith("bestmove"):
            parts = ln.split()
            return parts[1] if len(parts) > 1 else None


def main():
    rows = [r for r in csv.DictReader(open(csv_path)) if r.get("category") == CAT]
    try:
        sf11 = SF11Eval(SF11)
    except Exception as e:
        print(f"[sf11] unavailable ({type(e).__name__}: {e}) — cannot run"); return
    seed = chess.Board(); ai = ChessAI(None, None, seed, seed.turn)
    def our_static_ourpov(fen_str, us_white):
        bd = ai.ev_breakdown(chess.Board(fen_str))
        if bd.get("checkmate"): return None
        return (-bd["total"] / 1000.0) * 100.0 * (1 if us_white else -1)   # our-POV cp
    gaps = []; ks_us = []; thr_us = []; n = 0; sees = 0
    per = []          # (sf11_gap, our_gap, avoid10)
    avoid = {6: 0, 10: 0, 14: 0}; searched = 0
    for r in rows:
        fen = r["fen"]; our = r["our_move"]; sfb = r.get("sf_best", "")
        try:
            b = chess.Board(fen); us_white = (b.turn == chess.WHITE)
            om = chess.Move.from_uci(our); sm = chess.Move.from_uci(sfb) if sfb else None
            if om not in b.legal_moves: continue
            searched += 1
            avoided10 = False
            for d in (6, 10, 14):
                bm = sf11_bestmove(sf11, fen, d)
                if bm and bm != our:
                    avoid[d] += 1     # SF11 at depth d does NOT play our over-push
                    if d == 10: avoided10 = True
            # pre-move per-term (King safety/Threats are White-POV MG; convert to our-POV so - = bad for us)
            tot0, terms0 = sf11.eval(fen)
            b_ours = b.copy(); b_ours.push(om); tot_ours, _ = sf11.eval(b_ours.fen())
            if not (sm and sm in b.legal_moves):
                continue
            b_sf = b.copy(); b_sf.push(sm); tot_sf, _ = sf11.eval(b_sf.fen())
            if tot_ours is None or tot_sf is None or tot0 is None:
                continue                                  # SF11 refuses static eval in check -> skip
        except Exception:
            continue
        gap = wp_ourpov(tot_sf, us_white) - wp_ourpov(tot_ours, us_white)   # >0 = SF11-static: our move worse
        gaps.append(gap); n += 1
        if gap >= 50: sees += 1
        ks = terms0.get("King safety", 0.0) * 100.0; thr = terms0.get("Threats", 0.0) * 100.0
        ks_us.append(ks if us_white else -ks); thr_us.append(thr if us_white else -thr)  # our-POV cp, - = against us
        # our OWN static totals gap (valid total-vs-total comparison; do WE also statically prefer the over-push?)
        our_after_ours = our_static_ourpov(b_ours.fen(), us_white)
        our_after_sf = our_static_ourpov(b_sf.fen(), us_white)
        our_gap = (our_after_sf - our_after_ours) if (our_after_ours is not None and our_after_sf is not None) else None
        per.append((gap, our_gap, avoided10))
    if not n:
        print("[sf11] no rows evaluated"); sf11.close(); return
    m = sum(gaps)/n
    print(f"[sf11-overpush] n={n}  category={CAT}")
    print(f"  mean STATIC gap (sf11_after_SF - sf11_after_ours), our-POV cp = {m:+.0f}")
    print(f"  % where SF11-static sees our move >= 50cp worse: {100*sees/n:.0f}%  ({sees}/{n})")
    print(f"  pre-move SF11 King safety (our-POV, - = our king in danger): mean {sum(ks_us)/n:+.0f}cp")
    print(f"  pre-move SF11 Threats     (our-POV, - = threats against us):  mean {sum(thr_us)/n:+.0f}cp")
    if searched:
        print(f"  SF11 classical SEARCH avoids our over-push: d6 {100*avoid[6]/searched:.0f}%  "
              f"d10 {100*avoid[10]/searched:.0f}%  d14 {100*avoid[14]/searched:.0f}%  (n={searched})")
    # BROAD-vs-SMALL-SET: distribution of the SF11 static gap + cross-tab with search-avoidance.
    b_lo = sum(1 for g,_,_ in per if g < -50); b_mid = sum(1 for g,_,_ in per if -50 <= g < 50); b_hi = sum(1 for g,_,_ in per if g >= 50)
    print(f"\n  SF11 static-gap distribution (n={len(per)}): SF11 prefers OURS(<-50) {b_lo}  neutral(-50..50) {b_mid}  flags ours(>=50) {b_hi}")
    # the key pattern: static does NOT flag (gap<50) BUT search avoids (broad search-correction)?
    corr = sum(1 for g,_,av in per if g < 50 and av)
    print(f"  positions where SF11 static does NOT flag (gap<50) BUT d10 search avoids ours: {corr}/{len(per)} "
          f"({100*corr/max(1,len(per)):.0f}%) = the search-correction pattern (broad if high)")
    our_gaps = [og for _,og,_ in per if og is not None]
    if our_gaps:
        our_pref = sum(1 for og in our_gaps if og < 0)
        print(f"  OUR static totals gap (our-POV cp): mean {sum(our_gaps)/len(our_gaps):+.0f}; "
              f"we too statically PREFER our over-push in {our_pref}/{len(our_gaps)} ({100*our_pref/len(our_gaps):.0f}%)")
    print("\n  READ: if 'SF11 prefers ours'+'neutral' dominate AND search-avoid is high across MANY positions,")
    print("        the search-correction is BROAD (not a few outliers). If OUR gap also mostly negative, we and")
    print("        SF11's STATIC eval agree (both over-value) -> the fix is SEARCH, not a static term.")
    sf11.close()


if __name__ == "__main__":
    main()
