# -*- coding: utf-8 -*-
"""Phase-0.5 reconciliation: is the collapse over-read in our STATIC eval, or only in our depth-18 SEARCH?

The triage recorded our depth-18 SEARCH score (run_one) at each collapse decision-FEN and found it far above
SF. But the 2026-07-06 bias_profile found our STATIC ~= SF static at peaks. Those can't both hold on the
collapse FENs. This recomputes our STATIC (ChessAI.ev_breakdown, no search) + SF18 static per triage FEN and
prints, per position and in aggregate:
  our_search (from triage) | our_static | SF_static | static_gap | which static TERM drives it.
All in MOVER-POV centipawns (the collapsing side), so a positive gap = we (statically) over-read.

Outcome A: static_gap is large + KS/latent_threat drives it   -> static eval hole, KS lane confirmed.
Outcome B: static_gap ~= 0 (our static ~= SF) but search_gap large -> persistent d18 phantom, not a static hole.
Outcome C: static_gap large but a DIFFERENT term drives it     -> redirect to that term.

Run:  overnight_runner.sh pyrun diagnostics/verify_triage_static.py <triage.csv> [more.csv ...]
"""
import os, sys, csv
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

THIS = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(THIS)
sys.path.insert(0, ENGINE); sys.path.insert(0, THIS); sys.path.insert(0, os.path.join(ENGINE, "selfplay"))

import chess
from arbiter import Arbiter, find_stockfish
from ChessAI import ChessAI

UPP = 1000.0
def wp(ev):            # engine abs (Black-positive milli) -> White-POV pawns
    return -ev / UPP
TERMS = ["pieces", "capture_gains", "passed_pawn_support", "latent_threat", "king_safety", "central",
         "imbalance_white", "imbalance_black", "pair_bonus", "piece_value_boost", "advanced_endgame_white",
         "advanced_endgame_black"]
KS_TERMS = ["latent_threat", "king_safety"]
PVAL = {chess.KNIGHT: 305, chess.BISHOP: 333, chess.ROOK: 563, chess.QUEEN: 950}

def material_feats(b):
    """Mover-POV material/structure detectors (no engine call)."""
    us, opp = b.turn, not b.turn
    def npval(color):
        return sum(PVAL[pt] * len(b.pieces(pt, color)) for pt in PVAL)
    np_us, np_opp = npval(us), npval(opp)
    p_us, p_opp = len(b.pieces(chess.PAWN, us)), len(b.pieces(chess.PAWN, opp))
    wb, bb = b.pieces(chess.BISHOP, chess.WHITE), b.pieces(chess.BISHOP, chess.BLACK)
    oppb = int(len(wb) == 1 and len(bb) == 1 and
               (chess.square_rank(list(wb)[0]) + chess.square_file(list(wb)[0])) % 2 !=
               (chess.square_rank(list(bb)[0]) + chess.square_file(list(bb)[0])) % 2)
    return {"npedge": np_us - np_opp, "pawn_lead": p_us - p_opp, "totpawns": p_us + p_opp, "oppb": oppb}


def main():
    paths = sys.argv[1:]
    if not paths:
        print("usage: verify_triage_static.py <triage.csv|games_dir> [...]  (dirs sampled as CONTROL)"); return
    import glob, json
    # --dump <csv>: write a per-FEN detector corpus (label + detectors) for the convertibility fit.
    dump_path = None
    if "--dump" in paths:
        i = paths.index("--dump"); dump_path = paths[i + 1]; del paths[i:i + 2]
    # KEY=VAL args set engine env BEFORE ChessAI is constructed (env is read once at first construction),
    # so we can A/B any eval knob against the over-read single-core (e.g. ENABLE_ENDGAME_SCALE=1).
    knobs = [a for a in paths if ("=" in a and not a.endswith(".csv"))]
    paths = [a for a in paths if a not in knobs]
    for kv in knobs:
        k, v = kv.split("=", 1); os.environ[k] = v
    if knobs:
        print("[verify] engine knobs: " + " ".join(knobs))
    rows = []
    for p in paths:
        if p.endswith(".csv") and os.path.exists(p):
            for r in csv.DictReader(open(p)):
                r["_src"] = os.path.basename(os.path.dirname(p))
                rows.append(r)
        elif os.path.isdir(p):
            # CONTROL: sample midgame (>=14 pieces) FENs from this dir's game_*.jsonl, even-spread ~200.
            fens = []
            for gp in sorted(glob.glob(os.path.join(p, "game_*.jsonl")) + glob.glob(os.path.join(p, "*", "game_*.jsonl"))):
                for i, l in enumerate(open(gp)):
                    if i % 5 or not l.strip():
                        continue
                    try:
                        f = json.loads(l).get("fen")
                    except Exception:
                        continue
                    if f and len(chess.Board(f).piece_map()) >= 14:
                        fens.append(f)
            step = max(1, len(fens) // 200)
            for f in fens[::step][:200]:
                rows.append({"decision_fen": f, "class": "CONTROL", "_src": os.path.basename(p.rstrip("/")),
                             "our_cp": "nan", "gap_cp": "nan", "game": "?", "peak_eval": "0"})
        else:
            print("[skip] missing/unknown", p)
    if not rows:
        print("[verify] no rows"); return

    sf = find_stockfish()
    arb = Arbiter(sf, depth=None) if sf else None
    if arb is None:
        print("[verify] no Stockfish (set STOCKFISH_PATH)"); return
    seed = chess.Board(); ai = ChessAI(None, None, seed, seed.turn)
    # SF11 classical per-term tripwire (optional; degrade gracefully if the binary is unavailable/flaky).
    sf11 = None
    try:
        from eval_vs_sf11 import SF11Eval, SF11
        sf11 = SF11Eval(SF11)
    except Exception as e:
        print("[verify] SF11 tripwire unavailable (%s) — running SF18-only" % type(e).__name__)

    print("MOVER-POV cp (collapsing side). static_gap = our_static - SF18_static; +ve = our STATIC over-reads.")
    print(f"{'g':>3} {'src':>9} {'cls':>7} {'our_srch':>8} {'our_stat':>8} {'SF18':>8} {'SF11':>8} {'st_gap':>7}  driver(term:cp) (ourKS/sf11KS)")
    agg = {}          # class -> lists
    dumprows = []
    for r in rows:
        fen = r.get("decision_fen")
        cls = r.get("class", "?")
        if not fen:
            continue
        try:
            b = chess.Board(fen)
            bd = ai.ev_breakdown(b)
            if bd.get("checkmate"):
                continue
            sfc = arb.evaluate_static(b)      # White-POV cp
        except Exception as e:
            continue
        if sfc is None:
            continue
        mover = 1 if b.turn == chess.WHITE else -1
        our_stat_cp = wp(bd["total"]) * 100.0 * mover          # mover-POV cp
        sf_stat_cp = sfc * mover
        st_gap = our_stat_cp - sf_stat_cp
        # per-term contribution in mover-POV cp; find the largest (signed same way as st_gap)
        term_cp = {t: (wp(bd.get(t, 0)) * 100.0 * mover) for t in TERMS}
        driver = max(term_cp.items(), key=lambda kv: abs(kv[1]))
        ks_cp = sum(term_cp[t] for t in KS_TERMS)
        try:
            our_srch = float(r.get("our_cp"));  srch_gap = float(r.get("gap_cp"))
        except (TypeError, ValueError):
            our_srch = srch_gap = float("nan")
        # SF11 classical tripwire: total (mover-POV cp) + its King safety / Pawns / Passed terms (White-POV MG).
        sf11_tot_cp = float("nan"); sf11_ks = float("nan"); sf11_pawns = float("nan"); sf11_passed = float("nan")
        if sf11 is not None:
            try:
                t11, terms11 = sf11.eval(fen)
                if t11 is not None:
                    sf11_tot_cp = t11 * 100.0 * mover
                if "King safety" in terms11:
                    sf11_ks = terms11["King safety"] * 100.0     # White-POV cp
                if "Pawns" in terms11:
                    sf11_pawns = terms11["Pawns"] * 100.0
                if "Passed" in terms11:
                    sf11_passed = terms11["Passed"] * 100.0
            except Exception:
                pass
        print(f"{r.get('game','?'):>3} {r['_src']:>9} {cls:>7} {our_srch:>8.0f} {our_stat_cp:>8.0f} "
              f"{sf_stat_cp:>8.0f} {sf11_tot_cp:>8.0f} {st_gap:>7.0f}  {driver[0]}:{driver[1]:+.0f} "
              f"(ourKS {ks_cp:+.0f} / sf11KS {sf11_ks:+.0f})")
        if dump_path is not None:
            # mover-POV detectors (mover = side to move = collapsing side). det_ks_units_X = danger to X's king.
            our_w = (mover == 1)
            mf = material_feats(b)
            dumprows.append({
                "fen": fen, "src": r["_src"], "cls": cls,
                "is_collapse": 0 if cls == "CONTROL" else 1,
                "is_endgame": int(bool(bd.get("is_endgame"))),
                "over_read_cp": round(st_gap),
                "sf18_static_cp": round(sf_stat_cp),   # cached SF18 static (mover-POV) => fast knob-screen w/o SF
                "kdu_us":  bd["det_ks_units_w"] if our_w else bd["det_ks_units_b"],
                "kdu_opp": bd["det_ks_units_b"] if our_w else bd["det_ks_units_w"],
                "off_us":  bd["det_w_offense"] if our_w else bd["det_b_offense"],
                "def_us":  bd["det_w_defense"] if our_w else bd["det_b_defense"],
                "off_opp": bd["det_b_offense"] if our_w else bd["det_w_offense"],
                "def_opp": bd["det_b_defense"] if our_w else bd["det_w_defense"],
                "mob_us":  bd["det_w_mobility"] if our_w else bd["det_b_mobility"],
                "mob_opp": bd["det_b_mobility"] if our_w else bd["det_w_mobility"],
                "npedge": mf["npedge"], "pawn_lead": mf["pawn_lead"],
                "totpawns": mf["totpawns"], "oppb": mf["oppb"],
                "pieces_cp": round(term_cp["pieces"]), "capg_cp": round(term_cp["capture_gains"]),
                # king-danger comparison (both mover-POV net cp): our latent_threat+king_safety vs SF11's KS term.
                "our_ks_cp": round(ks_cp),
                "sf11_ks_cp": (round(sf11_ks * mover) if sf11_ks == sf11_ks else ""),
                # per-piece-type placement (mover-POV cp) for the pieces-over-read decomposition (Test B).
                "pt_pawns": round(wp(bd.get("pt_pawns", 0)) * 100 * mover),
                "pt_knights": round(wp(bd.get("pt_knights", 0)) * 100 * mover),
                "pt_bishops": round(wp(bd.get("pt_bishops", 0)) * 100 * mover),
                "pt_rooks": round(wp(bd.get("pt_rooks", 0)) * 100 * mover),
                "pt_queens": round(wp(bd.get("pt_queens", 0)) * 100 * mover),
                "pt_kings": round(wp(bd.get("pt_kings", 0)) * 100 * mover),
                # pawn-eval comparison: our pawn placement+passer vs SF11 Pawns+Passed (mover-POV cp) + advancement.
                "our_pawn_cp": round((wp(bd.get("pt_pawns", 0)) + wp(bd.get("passed_pawn_support", 0))) * 100 * mover),
                "sf11_pawns_cp": (round(sf11_pawns * mover) if sf11_pawns == sf11_pawns else ""),
                "sf11_passed_cp": (round(sf11_passed * mover) if sf11_passed == sf11_passed else ""),
                "pawn_adv": sum((chess.square_rank(sq) if b.turn == chess.WHITE else 7 - chess.square_rank(sq))
                                for sq in b.pieces(chess.PAWN, b.turn)),
            })
        clskey = cls if cls == "CONTROL" else (cls + ("-EG" if bd.get("is_endgame") else "-MG"))
        a = agg.setdefault(clskey, {"st_gap": [], "srch_gap": [], "ks": [], "sf11_tot": [],
                                    "drivers": {}, "terms": {t: [] for t in TERMS}})
        a["st_gap"].append(st_gap); a["srch_gap"].append(srch_gap if srch_gap == srch_gap else 0.0)
        a["ks"].append(ks_cp)
        if sf11_tot_cp == sf11_tot_cp:
            a["sf11_tot"].append(sf11_tot_cp)
        a["drivers"][driver[0]] = a["drivers"].get(driver[0], 0) + 1
        for t in TERMS:
            a["terms"][t].append(term_cp[t])
    arb.close()
    if sf11 is not None:
        sf11.close()

    if dump_path is not None and dumprows:
        cols = list(dumprows[0].keys())
        with open(dump_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(dumprows)
        nc = sum(1 for d in dumprows if d["is_collapse"])
        print(f"\n[dump] wrote {len(dumprows)} rows ({nc} collapse / {len(dumprows)-nc} control) -> {dump_path}")

    def mean(v): return sum(v) / len(v) if v else 0.0
    print("\n==== AGGREGATE (mover-POV cp) ====")
    for cls, a in sorted(agg.items()):
        drv = sorted(a["drivers"].items(), key=lambda kv: -kv[1])
        print(f"[{cls}] n={len(a['st_gap'])}  mean our_static_gap(vs SF18)={mean(a['st_gap']):+.0f}  "
              f"mean search_gap={mean(a['srch_gap']):+.0f}  mean ourKS_term={mean(a['ks']):+.0f}  "
              f"mean SF11_total={mean(a['sf11_tot']):+.0f} (n={len(a['sf11_tot'])})")
        print(f"        our largest-term histogram: " + ", ".join(f"{t}:{c}" for t, c in drv))
        tmeans = sorted(((t, mean(a["terms"][t])) for t in TERMS), key=lambda kv: -kv[1])
        print(f"        our per-term MEAN (mover-POV cp; SF total~0 so big +ve = over-reader):")
        print("          " + "  ".join(f"{t}={m:+.0f}" for t, m in tmeans if abs(m) >= 5))
    print("\nOutcome A: static_gap ~ search_gap (both large +) => static eval hole (NOT a d18 phantom).")
    print("Term: our-largest is a magnitude proxy, NOT proof of the error term (need per-term-vs-SF).")
    print("SF11 tripwire: if SF11_total (mover-POV) is also ~0 (<< our over-read), a CLASSICAL eval already")
    print("  separates the over-read => cheap detectors CAN fix it. If SF11_total also high => needs NNUE-class.")


if __name__ == "__main__":
    main()
