# -*- coding: utf-8 -*-
"""WHY did we pick the wrong move? — depth-1 move choice vs the three-way reference, in WIN%.

🚨 TERM NAMES ARE NOT 1-TO-1 ACROSS ENGINES. Our king-zone attacker-minus-defender was once
TRIPLE-counted; SF holds (mg,eg) pairs over a different decomposition. So "our king_safety vs SF's king
safety" is a category error — and it is a plausible reason that years of SF-ward KS work land flat: each
change may add signal we already count elsewhere. Only the TOTAL and the MOVE RANKING are comparable
across engines. This probe therefore uses SF as an ORACLE OVER MOVES and uses our breakdown only to
ATTRIBUTE our own choice. It never places one engine's term beside another's.

## Units: WIN%, and NO FILTER
Everything is scored in Lichess win% (k=0.00368208), not centipawns: 2 pawns of error at +8 barely
matters, 2 pawns at 0.0 flips the game, so cp regret silently over-weights won positions. Win% is
comparable across phases, which is what lets us keep EVERY position instead of filtering to a "quiet"
or "learnable" subset — filters throw away data and invite overfitting to the threshold.

Each position carries a signed WEIGHT instead of being kept or discarded:

    weight = our_winpct_error - sf11_winpct_error        (both measured against SF18-search)

  weight > 0  SF11-classical is closer to the truth than we are  -> STATICALLY FIXABLE, the signal
  weight < 0  WE are closer than SF11 is                         -> STABILIZATION: where we are ahead
  weight ~ 0  neither is close (tactics) or both are             -> self-suppressing, no filter needed

The tactical positions damp themselves out because both errors are large and the DIFFERENCE is small.
The old TRAIN 83.4% / GUARD 4.8% / discard 11.8% split becomes one continuous quantity, and the guard
set stops being held-out rows and becomes a live counterweight present in every position.

⚠️ This ranks WHERE and WHICH TERM. It cannot set a magnitude — that still needs games.
⚠️ SF's `eval` refuses in-check positions, so checking children are skipped and counted.

  pyrun diagnostics/_d1_move_attribution.py [POS=40] [DEPTH=13] [IN=ks_sets/diverse_corpus_wide.csv]
                                            [FENS=<file>] [OUT=/tmp/d1_attrib.csv]
"""
import os, sys, csv, math

for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
THIS = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(THIS)
sys.path.insert(0, ENGINE); sys.path.insert(0, THIS)

import chess
import chess.engine
from ChessAI import ChessAI
from eval_vs_sf11 import SF11Eval, SF11

SF18 = os.environ["STOCKFISH_PATH"]
POS = int(os.environ.get("POS", "40"))
DEPTH = int(os.environ.get("DEPTH", "13"))      # matches the banked corpus (SF18 d13)
OUT = os.environ.get("OUT", "/tmp/d1_attrib.csv")
FENS = os.environ.get("FENS", "")
IN = os.environ.get("IN", os.path.join(THIS, "ks_sets", "diverse_corpus_wide.csv"))
if not os.path.isabs(IN):
    IN = os.path.join(THIS, IN)

WIN_K = 0.00368208    # Lichess cp->win% sigmoid (per cp); matches cploss_frozen.py / probe_fens.py

# Our additive terms, Black-positive millipawns -> White-POV pawns = -v/1000.
TERMS = ["material", "pieces", "capture_gains", "passed_pawn_support", "latent_threat", "threats",
         "king_safety", "central", "imbalance_white", "imbalance_black", "pair_bonus",
         "piece_value_boost", "pawn_majority", "pawn_struct", "outpost", "mobility",
         "pt_pawns", "pt_knights", "pt_bishops", "pt_rooks", "pt_queens", "pt_kings"]


def winpct(pawns):
    cp = max(-1500.0, min(1500.0, pawns * 100.0))
    return 100.0 / (1.0 + math.exp(-WIN_K * cp))


def load_fens():
    if FENS:
        return [ln.strip() for ln in open(FENS) if ln.strip() and not ln.startswith("#")][:POS]
    rows = list(csv.DictReader(open(IN, newline="")))[:POS * 3]
    return [r["fen"] for r in rows][:POS]


def main():
    ai = ChessAI(None, None, chess.Board(), True)
    sf11 = SF11Eval(SF11)
    sf18 = chess.engine.SimpleEngine.popen_uci(SF18)

    rows, skipped, considered = [], 0, 0
    blame_fix, blame_guard = {}, {}          # summed signed win%-weighted term attribution
    n_fix = n_guard = n_both_right = n_neither = 0
    w_fix = w_guard = 0.0

    try:
        for fen in load_fens():
            try:
                b = chess.Board(fen)
            except Exception:
                continue
            if b.is_game_over(claim_draw=False):
                continue
            white_to_move = b.turn

            # --- the truth: SF18 search on the PARENT (one search per position, not per child) ---
            try:
                info = sf18.analyse(b, chess.engine.Limit(depth=DEPTH))
            except Exception:
                continue
            sc = info["score"].white().score(mate_score=100000)
            if sc is None or not info.get("pv"):
                continue
            target = sc / 100.0                       # White-POV pawns
            truth_move = info["pv"][0].uci()

            # --- position-level agreement, in win% ---
            ours_parent = -ai.ev_breakdown(b).get("total", 0) / 1000.0
            sf11_parent, _ = sf11.eval(b.fen())
            if sf11_parent is None:
                continue
            t_w = winpct(target)
            our_err = abs(winpct(ours_parent) - t_w)
            sf11_err = abs(winpct(sf11_parent) - t_w)
            weight = our_err - sf11_err               # >0 SF11 closer (fixable); <0 we are ahead

            # --- rank every legal child under both static evals ---
            kids = []
            for mv in b.legal_moves:
                b.push(mv)
                try:
                    if b.is_check():
                        skipped += 1
                        continue
                    bd = ai.ev_breakdown(b)
                    sv, _ = sf11.eval(b.fen())
                    if sv is None:
                        skipped += 1
                        continue
                    kids.append((mv.uci(), -bd.get("total", 0) / 1000.0, sv, bd))
                finally:
                    b.pop()
            if len(kids) < 2:
                continue
            considered += 1

            pick = (lambda key: (max if white_to_move else min)(kids, key=key))
            ours_pick = pick(lambda k: k[1])
            sf11_pick = pick(lambda k: k[2])
            ours_right = ours_pick[0] == truth_move
            sf11_right = sf11_pick[0] == truth_move

            if ours_right and sf11_right:
                n_both_right += 1
                continue
            if not ours_right and not sf11_right:
                n_neither += 1                      # neither static reaches it -> search's job
                continue

            order = sorted(kids, key=lambda k: k[1], reverse=white_to_move)
            truth_rank = 1 + [k[0] for k in order].index(truth_move) if truth_move in [k[0] for k in order] else -1

            # Attribute inside OUR eval only: how much did each term prefer the move we ranked top
            # over the truth move? Positive = that term pushed us away from the truth.
            tm = next((k for k in kids if k[0] == truth_move), None)
            if tm is None:
                continue
            attrib = {}
            for t in TERMS:
                a = -ours_pick[3].get(t, 0) / 1000.0
                c = -tm[3].get(t, 0) / 1000.0
                d = (a - c) if white_to_move else (c - a)
                if abs(d) >= 0.01:
                    attrib[t] = d

            if sf11_right and not ours_right:       # STATICALLY FIXABLE — the signal
                n_fix += 1; w_fix += max(weight, 0.0)
                for t, d in attrib.items():
                    blame_fix[t] = blame_fix.get(t, 0.0) + d * max(weight, 0.0)
                kind = "fixable"
            else:                                   # we found it and SF11 did not — stabilization
                n_guard += 1; w_guard += max(-weight, 0.0)
                for t, d in attrib.items():
                    blame_guard[t] = blame_guard.get(t, 0.0) + d * max(-weight, 0.0)
                kind = "guard"

            top = sorted(attrib.items(), key=lambda kv: -abs(kv[1]))[:3]
            rows.append({"fen": fen, "kind": kind, "our_move": ours_pick[0], "truth_move": truth_move,
                         "truth_rank_in_ours": truth_rank,
                         "weight_winpct": round(weight, 2),
                         "our_err_winpct": round(our_err, 2), "sf11_err_winpct": round(sf11_err, 2),
                         "top_terms": " | ".join("%s %+.2f" % (k, v) for k, v in top)})
    finally:
        sf11.close()
        try: sf18.quit()
        except Exception: pass

    # ⚠️ Print the summary BEFORE touching the filesystem. An earlier version wrote the CSV first and a
    # bad OUT path threw away a completed 400-position run — minutes of SF18 searches lost to an
    # unwritable directory. Results must survive a broken path.
    print("\n  positions scored            : %d   (children skipped, in check: %d)" % (considered, skipped))
    print("  both statics found SF18's move : %d" % n_both_right)
    print("  NEITHER did (search's job)     : %d" % n_neither)
    print("  ✅ SF11 found it, we did NOT    : %d   (statically fixable; win%% weight %.1f)" % (n_fix, w_fix))
    print("  🛡️  WE found it, SF11 did not    : %d   (stabilization; win%% weight %.1f)" % (n_guard, w_guard))

    def table(title, blame):
        if not blame:
            return
        print("\n  %s" % title)
        for name, v in sorted(blame.items(), key=lambda kv: -abs(kv[1]))[:10]:
            print("    %-22s %+9.2f" % (name, v))

    table("OUR TERMS THAT PUSHED US OFF THE TRUTH MOVE (win%%-weighted):", blame_fix)
    table("OUR TERMS CARRYING US WHERE SF11 FAILS (protect these):", blame_guard)

    # Every decisive case to stdout, so the run survives even if the CSV cannot be written at all.
    print("\n  DECISIVE CASES (kind | weight | truth rank in our ordering | our move -> truth move)")
    for r in rows:
        print("    %-8s %+6.2f  rank %-3s  %s -> %s   [%s]"
              % (r["kind"], r["weight_winpct"], r["truth_rank_in_ours"],
                 r["our_move"], r["truth_move"], r["top_terms"]))
        print("      %s" % r["fen"])

    try:
        with open(OUT, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["fen", "kind", "our_move", "truth_move",
                                              "truth_rank_in_ours", "weight_winpct",
                                              "our_err_winpct", "sf11_err_winpct", "top_terms"])
            w.writeheader(); w.writerows(rows)
        print("\n  csv -> %s" % OUT)
    except Exception as e:
        print("\n  ⚠️ CSV not written (%s) — the results above are complete regardless." % e)

    print("  🚨 Attribution is inside OUR eval only; no SF term was placed beside one of ours.")
    print("  ⚠️ Ranks WHERE and WHICH TERM — magnitude still needs games.\n")


main()
