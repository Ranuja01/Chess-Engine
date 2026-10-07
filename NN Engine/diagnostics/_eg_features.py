# -*- coding: utf-8 -*-
"""ENDGAME FEATURE READ (owner, 2026-10-07) — two questions on the DEPTH-target endgame rows (v2 phase256 < 128; SF18 d14
label; our d10 search of the 10-04 ship), before any POT endgame design:

  A  WINNABILITY: how much does the shipped POT eg scale factor (`v2_winnab`) actually do, per endgame type? How often it
     fires, its mean size, and the counterfactual: win% error and side-ahead BIAS with it vs with it removed.
  B  PAWN ENDINGS: hand-checkable structural labels per row, oriented to the LEADER (= the side SF18's search says is
     ahead; |SF18| > 25 cp), and which labels carry our UNDER-rating. UNDER = (win%(SF18) − win%(ours)) · sign(SF18):
     positive = we are too drawish. Labels (python-chess, no engine):
       pawn_diff        leader pawns − other pawns
       passers L/O      passed pawns per side;  protected = a passer defended by an own pawn
       outside          a passer at least 3 files from every enemy pawn (the decoy/outside passer)
       unstoppable L/O  square rule: a passer whose promotion race the enemy king cannot catch (side to move counted;
                        own king not in the path); pawn-only ending so no other defenders
       kdist            king activity: leader king's distance to the NEAREST ENEMY pawn minus the other king's distance
                        to the nearest LEADER pawn (negative = the leader's king is the closer raider)
       kadv             leader king's rank advancement minus the other's (relative ranks, 0..7)
       opposition       kings on one file/rank (or diagonal) with an odd number of squares between them and the side NOT
                        to move holds it: +1 leader holds, −1 other holds, 0 none
       majority         leader has more pawns than the other on a wing half (a-d or e-h) — can create a passer
       mobile_majority  ...and at least one of those pawns is not blocked head-on by an enemy pawn
       blocked_frac     fraction of all pawns blocked head-on (a locked structure)
  pyrun diagnostics/_eg_features.py [QTOL=8] [GAP=10]   (run with V2_PRESET=shipped)
"""
import os, sys
from collections import defaultdict
import numpy as np
import chess
THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
sys.path.insert(0, os.path.dirname(THIS))
os.environ.setdefault("PRESET", "LONG_FORMAT"); os.environ.setdefault("USE_OPENING_BOOK", "0")
import ChessAI
import _revival_screen as RS
from _endgame_types import classify

KV = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
wp = lambda cp: 100.0 / (1.0 + np.exp(-0.00368208 * np.clip(cp, -1500, 1500)))


def passed(b, sq, c):
    f, r = chess.square_file(sq), chess.square_rank(sq)
    for e in b.pieces(chess.PAWN, not c):
        ef, er = chess.square_file(e), chess.square_rank(e)
        if abs(ef - f) <= 1 and (er > r if c == chess.WHITE else er < r):
            return False
    return True


def unstoppable(b, sq, c):
    """Square rule for a pawn-only ending: plies-to-queen vs the enemy king's distance to the queening square."""
    f, r = chess.square_file(sq), chess.square_rank(sq)
    qsq = chess.square(f, 7 if c == chess.WHITE else 0)
    steps = (7 - r) if c == chess.WHITE else r
    if (c == chess.WHITE and r == 1) or (c == chess.BLACK and r == 6):
        steps -= 1                                                     # double step
    path = [chess.square(f, rr) for rr in (range(r + 1, 8) if c == chess.WHITE else range(r - 1, -1, -1))]
    if any(b.piece_at(s) for s in path):
        return False
    ek = b.king(not c)
    dk = chess.square_distance(ek, qsq) - (1 if b.turn != c else 0)    # the defender moves first if it is its turn
    return dk > steps


def labels(b, L):
    O = not L
    pl, po = list(b.pieces(chess.PAWN, L)), list(b.pieces(chess.PAWN, O))
    pasL = [s for s in pl if passed(b, s, L)]
    pasO = [s for s in po if passed(b, s, O)]
    att = lambda s, c: any(chess.square_file(d) != chess.square_file(s) for d in b.attackers(c, s) if b.piece_type_at(d) == chess.PAWN)
    kL, kO = b.king(L), b.king(O)
    near = lambda k, ps: min((chess.square_distance(k, s) for s in ps), default=8)
    rel = lambda k, c: chess.square_rank(k) if c == chess.WHITE else 7 - chess.square_rank(k)
    fd, rd = abs(chess.square_file(kL) - chess.square_file(kO)), abs(chess.square_rank(kL) - chess.square_rank(kO))
    opp = 0
    if (fd == 0 or rd == 0 or fd == rd) and max(fd, rd) % 2 == 0:     # odd number of squares BETWEEN ⇔ even distance
        opp = 1 if b.turn == O else -1
    blocked = lambda s, c: b.piece_at(s + (8 if c == chess.WHITE else -8)) is not None and \
        b.piece_type_at(s + (8 if c == chess.WHITE else -8)) == chess.PAWN
    maj = mob = 0
    for files in (range(0, 4), range(4, 8)):
        a = [s for s in pl if chess.square_file(s) in files]
        o = [s for s in po if chess.square_file(s) in files]
        if len(a) > len(o):
            maj = 1
            if any(not blocked(s, L) for s in a):
                mob = 1
    allp = pl + po
    pot = ChessAI.pawn_masks(int(b.pawns), int(b.occupied_co[chess.WHITE]), int(b.occupied_co[chess.BLACK]))["potential"]
    pot_L, pot_O = (pot[0], pot[1]) if L == chess.WHITE else (pot[1], pot[0])
    return dict(
        pot_L=bin(pot_L).count("1"), pot_O=bin(pot_O).count("1"),
        pawn_diff=len(pl) - len(po), passers_L=len(pasL), passers_O=len(pasO),
        protected_L=sum(att(s, L) for s in pasL), protected_O=sum(att(s, O) for s in pasO),
        outside_L=sum(all(abs(chess.square_file(s) - chess.square_file(e)) >= 3 for e in po) for s in pasL),
        outside_O=sum(all(abs(chess.square_file(s) - chess.square_file(e)) >= 3 for e in pl) for s in pasO),
        unstop_L=sum(unstoppable(b, s, L) for s in pasL), unstop_O=sum(unstoppable(b, s, O) for s in pasO),
        kdist=near(kL, po) - near(kO, pl), kadv=rel(kL, L) - rel(kO, O), opposition=opp,
        majority=maj, mobile_majority=mob,
        blocked_frac=(sum(blocked(s, chess.WHITE if b.color_at(s) else chess.BLACK) for s in allp) / max(len(allp), 1)))


def main():
    QTOL, GAP = float(KV.get("QTOL", 8)), float(KV.get("GAP", 10))
    fens, _, ph, sfl, base, _, _ = RS.load_rows()
    ai = ChessAI.ChessAI(None, None, chess.Board(), True)
    win = defaultdict(list)
    pe, ruled = [], []
    for f, phase, tgt, d10 in zip(fens, ph, sfl, base):
        if phase >= 128:
            continue
        c = classify(f)
        if c is None:
            continue
        b = chess.Board(f)
        bd = ai.ev_breakdown(b)
        stat = -float(bd["total"]) / 10.0
        w = -float(bd.get("v2_winnab") or 0) / 10.0
        win[c[0]].append((tgt, stat, w))
        if c[0] == "pawn ending" and abs(stat) >= 50:
            # the owner's rule, tested directly: OUR leader (sign of our static) with NO passer potential ⇒ drawish?
            Lo = chess.WHITE if stat > 0 else chess.BLACK
            lo = labels(b, Lo)
            ruled.append((lo["pot_L"] > 0, lo["pot_O"] > 0, stat, tgt, d10, f))
        if c[0] == "pawn ending" and abs(tgt) > 25:
            L = chess.WHITE if tgt > 0 else chess.BLACK
            s = 1.0 if tgt > 0 else -1.0
            pe.append(dict(fen=f, under=(wp(tgt) - wp(stat)) * s, under_d10=(wp(tgt) - wp(d10)) * s,
                           quiet=abs(wp(d10) - wp(stat)) <= QTOL, **labels(b, L)))

    print("A ▶ WINNABILITY (`v2_winnab`, the POT eg scale factor) on the endgame depth rows")
    print("  %-26s %6s %8s %10s %13s %13s %15s" % ("type", "rows", "fires", "mean |w| cp", "MSE with", "MSE without",
                                                  "bias with→without"))
    for t, R in sorted(win.items()):
        tg, st, w = (np.array(x) for x in zip(*R))
        a = np.abs(tg) > 25
        bias = lambda p: float(np.mean((wp(p[a]) - wp(tg[a])) * np.sign(tg[a])))
        fires = np.abs(w) >= 1
        print("  %-26s %6d %7.1f%% %10.1f %13.1f %13.1f %+7.2f → %+6.2f" % (
            t, len(R), 100 * fires.mean(), float(np.abs(w[fires]).mean()) if fires.any() else 0.0,
            float(np.mean((wp(st) - wp(tg)) ** 2)), float(np.mean((wp(st - w) - wp(tg)) ** 2)), bias(st), bias(st - w)))

    print("\nB ▶ PAWN ENDINGS — %d rows with a side ahead (|SF18| > 25 cp); UNDER = how much too drawish we are (win%% pts)" % len(pe))
    for scope, R in (("ALL", pe), ("QUIET (d10 agrees with static ≤ %.0f pp)" % QTOL, [r for r in pe if r["quiet"]])):
        if not R:
            continue
        u = np.array([r["under"] for r in R])
        big = u >= GAP
        print("\n  %s: %d rows · mean UNDER static %+.1f, d10 %+.1f · rows UNDER ≥ %.0f: %d" % (
            scope, len(R), u.mean(), np.mean([r["under_d10"] for r in R]), GAP, int(big.sum())))
        print("    %-16s %12s %12s %16s %10s" % ("label", "mean | big", "mean | rest", "UNDER if >0 / =0", "corr"))
        for k in ("pot_L", "pot_O", "pawn_diff", "passers_L", "passers_O", "protected_L", "outside_L", "outside_O", "unstop_L", "unstop_O",
                  "kdist", "kadv", "opposition", "majority", "mobile_majority", "blocked_frac"):
            x = np.array([r[k] for r in R], dtype=float)
            pos = x > 0 if k not in ("kdist",) else x < 0
            c = float(np.corrcoef(x, u)[0, 1]) if x.std() > 0 else float("nan")
            print("    %-16s %12.2f %12.2f %8.1f / %5.1f %+10.2f" % (k, x[big].mean() if big.any() else float("nan"),
                  x[~big].mean() if (~big).any() else float("nan"), u[pos].mean() if pos.any() else float("nan"),
                  u[~pos].mean() if (~pos).any() else float("nan"), c))
    print("\nC ▶ OWNER'S RULE — pawn endings where WE call a side ahead (|static| ≥ 0.5): does 'that side cannot create a passer'"
          " mean SF18 calls it drawish?  (SF18 win%% for OUR leader; 50 = level)")
    for pl_, po_ in ((True, False), (True, True), (False, False), (False, True)):
        S = [r for r in ruled if r[0] == pl_ and r[1] == po_]
        if not S:
            continue
        ours = [wp(abs(r[2])) for r in S]
        sf = [wp(r[3] * np.sign(r[2])) for r in S]
        d10 = [wp(r[4] * np.sign(r[2])) for r in S]
        print("  leader potential %-3s · other potential %-3s : %3d rows · ours static %.1f%% · our d10 %.1f%% · SF18 %.1f%% · "
              "SF18 within ±10 of 50: %d%%" % ("yes" if pl_ else "NO", "yes" if po_ else "no", len(S), np.mean(ours),
              np.mean(d10), np.mean(sf), 100 * np.mean([abs(x - 50) <= 10 for x in sf])))
    if KV.get("REVIEW"):
        n = int(KV["REVIEW"])
        Q = [r for r in pe if r["quiet"]]
        groups = [("UNDER-RATED, EQUAL PAWNS (quiet)", [r for r in Q if r["pawn_diff"] == 0], "under"),
                  ("UNDER-RATED, LEADER A PAWN+ UP (quiet)", [r for r in Q if r["pawn_diff"] > 0], "under"),
                  ("SF18 WINS DESPITE THE OTHER SIDE'S PASSER", [r for r in pe if r["passers_O"] > 0], "under"),
                  ("WE CALL A SIDE AHEAD THAT HAS NO PASSER POTENTIAL", None, None)]
        for title, G, key in groups:
            print("\n  ── %s" % title)
            if G is None:
                for pl_, po_, st, tg, dd, f in sorted([r for r in ruled if not r[0]], key=lambda r: -abs(r[2]))[:n]:
                    print("    %s   ours static %+.0f · d10 %+.0f · SF18 %+.0f" % (f, st, dd, tg))
                continue
            for r in sorted(G, key=lambda r: -r[key])[:n]:
                print("    %s   too drawish by %.0f pp (d10 %.0f) · pawns %+d · pot L/O %d/%d · passers L/O %d/%d · kdist %+d"
                      % (r["fen"], r["under"], r["under_d10"], r["pawn_diff"], r["pot_L"], r["pot_O"], r["passers_L"],
                         r["passers_O"], r["kdist"]))
    out = KV.get("OUT", "/mnt/e/chess_data/bench1007/pawn_ending_labels.csv")
    import csv
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(pe[0].keys()))
        w.writeheader(); w.writerows(pe)
    print("\nrows → %s   (kdist: the 'if >0' column means kdist < 0 = the leader's king is the closer raider)" % out)


if __name__ == "__main__":
    main()
