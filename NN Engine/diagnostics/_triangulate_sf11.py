# -*- coding: utf-8 -*-
"""TRIANGULATION, side by side with SF11's static eval (owner, 10-02): for each case in a triangulation note, SF11's
classical term table (MG + EG, blended with SF11's own phase) next to OUR term breakdown — "same subsystems, does SF11
get it and where?". SF11 is the achievability control (memory `static-eval-vs-search-move-has-a-42-percent-ceiling`).

Units: everything White-POV centipawns. SF11 prints terms in pawns of PawnValueEg (Trace::to_cp) ⇒ ×100.
SF11 phase (evaluate.cpp / material.cpp): npm (mg values N 781, B 825, R 1276, Q 2538, both sides), clamped to
[EndgameLimit 3915, MidgameLimit 15258] → ph ∈ [0, 128]; blended = (mg·ph + eg·(128 − ph))/128 (the scale factor and
initiative are already inside SF11's Total; per-term blends ignore the scale factor).

  pyrun diagnostics/_triangulate_sf11.py [NOTE=dev_notes/TRIANGULATION-2026-10-02.md]   (run with V2_PRESET=shipped)
"""
import os, sys, re, csv, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chess

KV = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
THIS = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(THIS)
sys.path.insert(0, ENGINE)
os.environ.setdefault("PRESET", "LONG_FORMAT"); os.environ.setdefault("USE_OPENING_BOOK", "0")
import ChessAI
SF11 = os.environ.get("SF11_BIN", "/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/stockfish_11_linux/"
                                  "stockfish-11-linux/Linux/stockfish_20011801_x64_bmi2")
NOTE = os.path.join(ENGINE, KV.get("NOTE", "dev_notes/TRIANGULATION-2026-10-02.md"))
OURS = ["pieces", "king_safety", "mobility", "pawn_struct", "v2_passers", "v2_placement", "v2_winnab"]


def sf11_eval(p, fen):
    p.stdin.write("position fen %s\neval\nisready\n" % fen); p.stdin.flush()
    total, terms = None, {}
    while True:
        ln = p.stdout.readline()
        if not ln or ln.strip() == "readyok":
            break
        m = re.search(r"Total evaluation:\s*([-+]?\d+\.\d+)", ln)
        if m:
            total = float(m.group(1))
        mm = re.match(r"\s*([A-Za-z ]+?)\s*\|.*\|.*\|\s*([-+]?\d+\.\d+|----)\s+([-+]?\d+\.\d+|----)\s*$", ln)
        if mm and mm.group(2) != "----":
            terms[mm.group(1).strip()] = (float(mm.group(2)), float(mm.group(3)))
    return total, terms


def sf11_phase(b):
    npm = sum(len(b.pieces(pt, c)) * v for pt, v in ((chess.KNIGHT, 781), (chess.BISHOP, 825), (chess.ROOK, 1276),
                                                     (chess.QUEEN, 2538)) for c in (chess.WHITE, chess.BLACK))
    npm = max(3915, min(15258, npm))
    return (npm - 3915) * 128 // (15258 - 3915)


def main():
    cases = re.findall(r"#(\d+)\s+(\S+ [wb] \S+ \S+ \d+ \d+)\n\s+SF18 d14\s+([-+]?\d+) cp.*?d10\s+([-+]?\d+)", open(NOTE).read())
    p = subprocess.Popen([SF11], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
    p.stdin.write("uci\n"); p.stdin.flush()
    while p.stdout.readline().strip() != "uciok":
        pass
    ai = ChessAI.ChessAI(None, None, chess.Board(), True)
    for num, fen, sf18, d10 in cases:
        b = chess.Board(fen)
        total, terms = sf11_eval(p, fen)
        ph = sf11_phase(b)
        bd = ai.ev_breakdown(b)
        ours_static = -float(bd["total"]) / 10.0
        print("\n#%s  %s" % (num, fen))
        print("  TOTALS (White cp):  SF18 d14 %+5d · SF11 static %+5.0f · ours static %+5.0f · ours d10 %+5d"
              % (int(sf18), 100 * total if total is not None else float("nan"), ours_static, int(d10)))
        sf = {k: (mg * ph + eg * (128 - ph)) / 128 * 100 for k, (mg, eg) in terms.items() if k != "Total"}
        print("  SF11 terms (blended, ph %d/128): " % ph + "  ".join("%s %+.0f" % (k, v) for k, v in sf.items() if abs(v) >= 5))
        print("  ours terms:                      " + "  ".join(
            "%s %+.0f" % (t, -float(bd[t]) / 10.0) for t in OURS if bd.get(t) is not None and abs(float(bd[t])) >= 50))
    p.stdin.write("quit\n"); p.stdin.flush()


if __name__ == "__main__" and KV.get("MODE") not in ("aggregate", "eg"):
    main()


def aggregate():
    """MODE=aggregate [GAP=8]: over ALL triangulation candidates (`_triangulate_cases.select`), per SF11 term vs our
    nearest term: how often the difference (≥ 30 cp) points the SAME way as SF18's disagreement with our d10 search
    (helps) vs the opposite (hurts), and the mean signed contribution toward closing the gap. Also: how often SF11's
    static total is closer to SF18 than ours."""
    import numpy as np
    import _triangulate_cases as TC
    cands, ai = TC.select(float(KV.get("GAP", 8)))
    PAIRS = [("Threats", None), ("King safety", "king_safety"), ("Passed", "v2_passers"), ("Pawns", "pawn_struct"),
             ("Mobility", "mobility"), ("Space", None), ("pieces(N+B+R+Q)", "v2_placement")]
    p = subprocess.Popen([SF11], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
    p.stdin.write("uci\n"); p.stdin.flush()
    while p.stdout.readline().strip() != "uciok":
        pass
    stats = {k: [0, 0, []] for k, _ in PAIRS}
    closer = n = 0
    for _, gap, f, sf18, d10, stat, _, _ in cands:
        b = chess.Board(f)
        total, terms = sf11_eval(p, f)
        if total is None:
            continue
        ph = sf11_phase(b)
        bl = lambda k: sum((terms[t][0] * ph + terms[t][1] * (128 - ph)) / 128 * 100 for t in k if t in terms)
        bd = ai.ev_breakdown(b)
        ours = lambda key: -float(bd.get(key) or 0) / 10.0 if key else 0.0
        n += 1
        closer += abs(100 * total - sf18) < abs(stat - sf18)
        s = 1.0 if gap > 0 else -1.0
        for name, key in PAIRS:
            sfv = bl(["Knights", "Bishops", "Rooks", "Queens"]) if name.startswith("pieces") else bl([name])
            d = sfv - ours(key)
            stats[name][2].append(s * d)
            if abs(d) >= 30:
                stats[name][0 if s * d > 0 else 1] += 1
    print("TRIANGULATION AGGREGATE  cases %d (GAP ≥ %s pp) · SF11 static closer to SF18 than ours: %d/%d (%.0f%%)"
          % (n, KV.get("GAP", 8), closer, n, 100.0 * closer / max(n, 1)))
    print("  %-18s %6s %6s %22s" % ("SF11 term vs ours", "helps", "hurts", "mean push toward SF18"))
    for name, _ in PAIRS:
        h, u, v = stats[name]
        print("  %-18s %6d %6d %+18.1f cp" % (name, h, u, float(np.mean(v)) if v else 0.0))


if KV.get("MODE") == "aggregate":
    aggregate()


# ═══ MODE=eg — ENDGAME DISAGREEMENTS, SF11's term table vs ours (owner, 2026-10-07) ═══════════════════════════════
# Rows: the DEPTH-target endgame rows (v2 phase256 < 128; SF18 d14 label; our d10 search of the 10-04 ship), typed by
# `_endgame_types.classify`. SELECTED = |win%(SF18 d14) − win%(our static)| ≥ GAP pp. Per endgame type and per term PAIR:
# where SF11's term differs from ours, does the difference point TOWARD SF18 (helps) or away (hurts), the counterfactual
# Δwin% error if this one pair took SF11's value (☠️ an UPPER BOUND on disagreement, never an expected gain — it assumes
# SF11 is right), and SILENCES (SF11 ≥ 30 cp, ours < 5 cp or absent: "look for zeros, not gaps").
#
# ☠️ WHAT EACH PAIR CONTAINS — names collide across engines, so the CONTENTS are the definition (read from the sources
# 2026-10-07: SF11 evaluate.cpp / pawns.cpp / material.cpp; ours eval_v2.cpp publication block). All White-POV cp, each
# term blended with its OWN engine's phase (SF11: npm in [3915, 15258] → 0..128; ours: phase256).
#   material   SF11 "Material" = piece values + ALL PSQT (incl. pawn & king PSQT — SF's king centralisation lives HERE)
#              ours `material` (piece values) + `pieces` (tapered PST only, incl. king eg PST)
#   imbalance  SF11 "Imbalance" = quadratic piece-pair table + bishop pair  ·  ours `kaufman_imbalance` (+ `pair_bonus`)
#   pawns      SF11 "Pawns" = isolated, backward, doubled, connected (phalanx/support × rank), weak-unopposed, weak lever.
#              NO passer VALUE and NO shelter (those are Passed / King rows). · ours `pawn_struct` (isolated, doubled,
#              backward, connected; same exclusions)
#   passed     SF11 "Passed" = rank bonus + BOTH kings' distance to the block square (eg) + free-path ladder + file
#              penalty · ours `v2_passers` (rank table + king distance eg; path ladder OFF in the ship) + `v2_pxpass` (PX, 0)
#   king       ☠️ SF11 "King safety" = king danger + shelter/storm + pawnless flank + flank attacks AND, in the
#              ENDGAME, −16 × distance(king, nearest OWN pawn) (pawns.cpp do_king_safety). · ours `king_safety`
#              (attack units) + `v2_shelter` (KS-B) + `v2_kflank` (0). ⇒ we have NO king-to-own-pawns term.
#   placement  SF11 Knights+Bishops+Rooks+Queens rows = outposts, minor behind pawn, KING PROTECTOR distance, bishop
#              pawns on its colour, long diagonal, rook (semi)open/queen file, trapped rook, weak queen — NOT PSQT, NOT
#              mobility · ours `v2_placement` (+ `v2_rookfile` 0, `v2_kprot` 0)
#   mobility   both: piece mobility area counts → table
#   threats    SF11 "Threats" · ours `threats` (closed 10-05, absent)
#   space      SF11 "Space" (zero once npm < 12222, so ~always 0 here) · ours `space` (absent)
#   initiative SF11 "Initiative" = ADDITIVE "complexity" (passers, pawn count, king outflanking, infiltration, both
#              flanks, pawnless board, almost-unwinnable), sign-preserving · ours: nothing additive
#   scaling    SF11: the eg multiplier sf/64 (opposite bishops, 36 + 7·strong-side pawns, 50-move decay) and the
#              MATERIAL SCALING FUNCTIONS (KPsK rook-file pawns, KPKP via the KPK bitbase, KBPsK, KQKRPs, pawnless
#              npm-difference draws) — no trace row; DERIVED here as final − tempo − unscaled blend. · ours `v2_winnab`
#              (the POT eg scale factor)
#   SHORTCUTS  SF11 SPECIALISED EVALS (KPK bitbase, KBNK, KRKP, KQKP, KXK …) and LAZY EXIT return before any term is
#              traced (Total row 0) → counted, excluded from term rows. Ours `draw_class` returns 0 with no terms → same.
#   tempo      SF11 adds Tempo (28 ≈ 0.13 pawn) for the side to move; ours none — removed from SF11's total here.
# CLOSURE: the mapped rows must sum to each engine's total (printed); a residual means an unmapped field.
#   pyrun diagnostics/_triangulate_sf11.py MODE=eg [GAP=15] [QUIET=1 QTOL=8] [EXAMPLES=3] [OURS=ours1004] [OUT=]
#   (run with V2_PRESET=shipped)
EG_PAIRS = [("material",   ["Material"],                              ["material", "pieces"]),
            ("imbalance",  ["Imbalance"],                             ["kaufman_imbalance", "pair_bonus"]),
            ("pawns",      ["Pawns"],                                 ["pawn_struct"]),
            ("passed",     ["Passed"],                                ["v2_passers", "v2_pxpass"]),
            ("king",       ["King safety"],                           ["king_safety", "v2_shelter", "v2_kflank"]),
            ("placement",  ["Knights", "Bishops", "Rooks", "Queens"], ["v2_placement", "v2_rookfile", "v2_kprot"]),
            ("mobility",   ["Mobility"],                              ["mobility"]),
            ("threats",    ["Threats"],                               ["threats"]),
            ("space",      ["Space"],                                 ["space"]),
            ("initiative", ["Initiative"],                            []),
            ("scaling",    None,                                      ["v2_winnab"])]
SF11_TEMPO_PAWNS = 28.0 / 213.0


def eg_mode():
    import numpy as np
    from collections import defaultdict
    import _revival_screen as RS
    from _endgame_types import classify
    GAP, EX = float(KV.get("GAP", 15)), int(KV.get("EXAMPLES", 3))
    wpf = lambda cp: 100.0 / (1.0 + np.exp(-0.00368208 * np.clip(cp, -1500, 1500)))
    fens, _, ph, sfl, base, _, _ = RS.load_rows()
    p = subprocess.Popen([SF11], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
    p.stdin.write("uci\n"); p.stdin.flush()
    while p.stdout.readline().strip() != "uciok":
        pass
    ai = ChessAI.ChessAI(None, None, chess.Board(), True)
    rows, short_sf, short_us, clo_sf, clo_us = [], defaultdict(int), defaultdict(int), [], []
    n_eg = defaultdict(int)
    for f, phase, tgt, d10 in zip(fens, ph, sfl, base):
        if phase >= 128:
            continue
        c = classify(f)
        if c is None:
            continue
        b = chess.Board(f)
        if b.is_check():
            continue
        n_eg[c[0]] += 1
        bd = ai.ev_breakdown(b)
        stat = -float(bd["total"]) / 10.0
        gap = wpf(tgt) - wpf(stat)
        if abs(gap) < GAP:
            continue
        # QUIET=1: keep rows where our d10 SEARCH agrees with our static (≤ QTOL pp) — the static error then PERSISTS at
        # depth, and the row is not a tactic a static eval cannot see (unquiet rows read 'threats' as the gap).
        if KV.get("QUIET") == "1" and abs(wpf(d10) - wpf(stat)) > float(KV.get("QTOL", 8)):
            continue
        total, terms = sf11_eval(p, f)
        if total is None:
            continue
        sph = sf11_phase(b)
        bl = lambda mgeg: (mgeg[0] * sph + mgeg[1] * (128 - sph)) / 128 * 100
        tempo = (SF11_TEMPO_PAWNS if b.turn == chess.WHITE else -SF11_TEMPO_PAWNS) * 100
        sf11_cp = 100 * total - tempo
        sf_short = terms.get("Total", (0.0, 0.0)) == (0.0, 0.0)
        us_short = bd.get("material") is None
        if sf_short:
            short_sf[c[0]] += 1
        if us_short:
            short_us[c[0]] += 1
        sfv, usv = {}, {}
        if not sf_short:
            unscaled = bl(terms["Total"])
            for name, sfk, _ in EG_PAIRS:
                sfv[name] = (sf11_cp - unscaled) if sfk is None else sum(bl(terms[k]) for k in sfk if k in terms)
            clo_sf.append(sum(sfv.values()) - sf11_cp)
        if not us_short:
            for name, _, uk in EG_PAIRS:
                vals = [-float(bd[k]) / 10.0 for k in uk if bd.get(k) is not None]
                usv[name] = sum(vals) if vals else None          # None = ABSENT (we carry nothing), never 0
            clo_us.append(sum(v for v in usv.values() if v is not None) - stat)
        rows.append(dict(fen=f, type=c[0], pawns=c[1], tgt=tgt, d10=d10, stat=stat, sf11=sf11_cp, gap=gap,
                         d10gap=wpf(tgt) - wpf(d10), sfv=sfv, usv=usv))
    p.stdin.write("quit\n"); p.stdin.flush()

    print("QUIET filter: %s" % ("our d10 within %s pp of our static" % KV.get("QTOL", 8) if KV.get("QUIET") == "1" else "off"))
    print("ENDGAME DISAGREEMENTS vs SF11's term table — rows |win%%(SF18 d14) − win%%(our static)| ≥ %.0f pp: %d of %d endgame rows"
          % (GAP, len(rows), sum(n_eg.values())))
    print("  endgame rows by type: %s" % dict(n_eg))
    print("  closure (mapped rows − total, cp): SF11 mean %+.2f max|.| %.2f · ours mean %+.2f max|.| %.2f"
          % (np.mean(clo_sf), np.max(np.abs(clo_sf)), np.mean(clo_us), np.max(np.abs(clo_us))))
    print("  shortcuts (no term table): SF11 specialised/lazy %s · ours draw_class %s" % (dict(short_sf), dict(short_us)))
    for t in sorted({r["type"] for r in rows}):
        R = [r for r in rows if r["type"] == t and r["sfv"] and r["usv"]]
        if not R:
            continue
        s = np.array([np.sign(r["gap"]) for r in R])
        err_us = np.array([abs(r["gap"]) for r in R])
        err_sf = np.array([abs(wpf(r["tgt"]) - wpf(r["sf11"])) for r in R])
        per = np.array([abs(r["d10gap"]) >= GAP for r in R])
        # side ahead = SF18's sign; we OVER-rate it when our static is further from 50% than SF18 on the same side
        ahead = [r for r in R if abs(r["tgt"]) > 25]
        over = np.mean([1.0 if r["gap"] * np.sign(r["tgt"]) < 0 else 0.0 for r in ahead]) if ahead else float("nan")
        print("\n▶ %s — %d rows (of %d) · SF11 static closer to SF18 than ours %d%% · gap persists at our d10 %d%% · we "
              "OVER-rate the side ahead in %d%% · mean |err| ours %.1f pp, SF11 %.1f pp"
              % (t, len(R), n_eg[t], 100 * np.mean(err_sf < err_us), 100 * np.mean(per), 100 * over, err_us.mean(), err_sf.mean()))
        print("    %-11s %6s %6s %8s %13s %10s %7s" % ("pair", "helps", "hurts", "silences", "mean push cp", "Δwin%err", "ours"))
        for name, _, _ in EG_PAIRS:
            d = np.array([r["sfv"][name] - (r["usv"][name] or 0.0) for r in R])
            sil = sum(1 for r in R if abs(r["sfv"][name]) >= 30 and (r["usv"][name] is None or abs(r["usv"][name]) < 5))
            dw = np.mean([abs(wpf(r["tgt"]) - wpf(r["stat"] + dd)) - abs(wpf(r["tgt"]) - wpf(r["stat"])) for r, dd in zip(R, d)])
            absent = all(r["usv"][name] is None for r in R)
            print("    %-11s %6d %6d %8d %+13.1f %+10.2f %7s" % (name, int(np.sum((s * d) >= 30)), int(np.sum((s * d) <= -30)),
                  sil, float(np.mean(s * d)), dw, "ABSENT" if absent else ""))
        R.sort(key=lambda r: -abs(r["gap"]))
        for r in R[:EX]:
            print("    · %s  SF18 %+.0f · ours static %+.0f / d10 %+.0f · SF11 %+.0f" % (r["fen"], r["tgt"], r["stat"], r["d10"], r["sf11"]))
            print("        SF11/ours: " + "  ".join("%s %+.0f/%s" % (n, r["sfv"][n], "—" if r["usv"][n] is None else "%+.0f" % r["usv"][n])
                                                for n, _, _ in EG_PAIRS if abs(r["sfv"][n]) >= 5 or (r["usv"][n] or 0) != 0))
    out = KV.get("OUT", "/mnt/e/chess_data/bench1007/eg_sf11_terms.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["fen", "type", "pawns", "sf18", "ours_static", "ours_d10", "sf11"] + ["sf11_" + n for n, _, _ in EG_PAIRS]
                   + ["ours_" + n for n, _, _ in EG_PAIRS])
        for r in rows:
            w.writerow([r["fen"], r["type"], r["pawns"], r["tgt"], r["stat"], r["d10"], r["sf11"]]
                       + ["%.1f" % r["sfv"][n] if r["sfv"] else "" for n, _, _ in EG_PAIRS]
                       + ["" if not r["usv"] or r["usv"][n] is None else "%.1f" % r["usv"][n] for n, _, _ in EG_PAIRS])
    print("\nrows → %s" % out)


if KV.get("MODE") == "eg":
    eg_mode()
