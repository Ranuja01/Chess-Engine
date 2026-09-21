# -*- coding: utf-8 -*-
"""CAN ANY RETUNE HAVE PLAYED THE RIGHT MOVE? — the ceiling on tuning, measured before tuning.

WHY THIS EXISTS (2026-09-20). Eight consecutive eval CONCEPTS measured move-null (central, bishop pair,
space, threats, Kaufman, tier-2b, rook files, weak-unopposed), including the only 4/4-unanimous reference
term v2 lacked. Reference universality has therefore failed as a selection criterion on our distribution.
Before spending days on the full retune -- or on more auxiliary terms -- the question worth answering is
which KIND of failure we have:

    TUNING-REACHABLE   some reweighting of the terms we ALREADY compute ranks SF's move on top
    STRUCTURALLY MISSING   no reweighting does, so the distinction is not in our feature set at all

★ This is answerable exactly, not by opinion, because v2's breakdown is a COMPLETE PARTITION: the
published terms sum to `total` to the millipawn (verified 4000/4000 positions, 2026-09-20). So the eval is
literally  total = sum_i term_i , a candidate reweighting is  total(w) = sum_i w_i * term_i , and
"could any retune have played SF's move here?" is a LINEAR FEASIBILITY question over the sibling set:

    exists w in [W_LO, W_HI]^k  such that  (x_best - x_j) . w > 0  for every sibling j

Solved as an LP maximising the margin t. t* > 0 => separable => tuning could reach it.

☠️ READ THE BOUNDS THE RIGHT WAY ROUND -- they are asymmetric:
  * SEPARABLE is FIRM. Even coarse per-subsystem scaling reaches it, so a finer retune certainly can.
    The separable fraction is therefore a LOWER bound on what tuning can fix.
  * NOT SEPARABLE is WEAKER than "a term is missing". This reweights whole subsystems; a real retune can
    also reshape INSIDE a term (per-file tables, phase curves, gates), which this cannot express. So the
    non-separable count is an UPPER bound on structurally-missing, not proof of it. Where that bound turns
    out to be load-bearing, re-run those positions against finer breakdown fields.

⚠️ ONE-PLY STATIC, like _sibling_spread: the engine chooses by SEARCHING. What is bounded here is what the
leaf score and the static ordering can EXPRESS, not what a d7 search would play. A position we cannot
express may still be played correctly via depth, and vice versa.

⚠️ NOT a tuner. It never fits w to improve anything; it only asks whether a fixing w EXISTS. The jointly
optimal w it reports at the end is a by-product (a coarse, subsystem-granular retune) and is printed as a
STARTING POINT and a predicted ceiling for the real retune, never as something to ship.

  pyrun diagnostics/_term_separability.py [SET=ks_sets/game_regret_set_v2era.csv] [MAX_POS=800] [SEED=0]
        [W_LO=0.25] [W_HI=4.0] [MARGIN=10] [OUT=ks_sets/separability.csv] <ENGINE KNOBS...>

Every KEY=VAL argument is exported to the environment before ChessAI is imported, so the engine arm and
its knobs are set the same way as every other diagnostic here. One process per arm -- knobs latch at init.
"""
import os, sys, csv, math, random

for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
THIS = os.path.dirname(os.path.abspath(__file__))
ENG = os.path.dirname(THIS)
sys.path.insert(0, ENG); sys.path.insert(0, THIS)

import numpy as np
from scipy.optimize import linprog
import chess
from ChessAI import ChessAI

# ★★ SF11=1 -- THE ACHIEVABILITY CONTROL (owner's design, 2026-09-20). Without it this tool scores our
# STATIC one-ply ranking against a d14 SEARCH, and no static eval reproduces a deep search -- so a failure
# there is not evidence of anything. SF11's classical eval is the right yardstick because it is an HCE:
# whatever ITS static eval gets right is PROVABLY reachable by a hand-crafted eval, and whatever it also
# misses is the static-vs-search gap, which must not be charged to us.
#   SF11 static picks SF18's move  -> ACHIEVABLE. If we miss it, that is a REAL eval failure.
#   SF11 static misses it too      -> excluded; "everything is on the table" (owner's phrase).
USE_SF11 = os.environ.get("SF11", "0") == "1"
sf11 = None
if USE_SF11:
    from eval_vs_sf11 import SF11Eval, SF11 as SF11_DEFAULT
    sf11 = SF11Eval(os.environ.get("SF11_BIN", SF11_DEFAULT))

SET = os.environ.get("SET", os.path.join(THIS, "ks_sets", "game_regret_set_v2era.csv"))
if not os.path.isabs(SET):
    SET = os.path.join(THIS, SET)
OUT = os.environ.get("OUT", os.path.join(THIS, "ks_sets", "separability.csv"))
if not os.path.isabs(OUT):
    OUT = os.path.join(THIS, OUT)
MAX_POS = int(os.environ.get("MAX_POS", "800"))
SEED    = int(os.environ.get("SEED", "0"))
W_LO    = float(os.environ.get("W_LO", "0.25"))
W_HI    = float(os.environ.get("W_HI", "4.0"))
MARGIN  = float(os.environ.get("MARGIN", "10"))   # millipawns; 10 mp = 1 cp of separation demanded
# Minimum SF best-vs-2nd win% gap for a position to count as a real DECISION. 0 = keep everything.
MIN_A   = float(os.environ.get("MIN_A", "0"))
# ACC_ONLY=1: run ONLY the pure-eval accuracy comparison (root position, no children, no LPs). Cheap
# enough for the whole corpus, and it is the pass that answers whether our accuracy "win rate" is real
# or cosmetic. ACC_OUT= writes one row per position: fen, criticality, our error, SF11's, and the margin.
ACC_ONLY = os.environ.get("ACC_ONLY", "0") == "1"

# v2's published partition, in a FIXED order. ⚠️ Deliberately NOT bd.get(k, 0): under EVAL_ARM=1 an absent
# key means the term was never computed, and silently reading it as 0 is how four v1-era tools have already
# produced confident wrong answers (charter hygiene rule 8/9). A key that is absent for the WHOLE run is
# dropped from the vector once, loudly; a key that is absent for one position aborts that position.
V2_TERMS = ["material", "pieces", "king_safety", "mobility",
            "pawn_struct", "v2_passers", "v2_placement", "v2_rookfile"]


def winpct(cp):
    return 100.0 / (1.0 + math.exp(-0.00368208 * cp))


def sf_indices(moves_field, stm_white):
    """SF's cached multi-PV -> {uci: stm-POV win%}, plus the sharpness gap A."""
    mm = {}
    for pair in (moves_field or "").split(";"):
        if ":" in pair:
            u, c = pair.rsplit(":", 1)
            try:
                cp = float(c)
            except ValueError:
                continue
            mm[u] = winpct(cp) if stm_white else 100.0 - winpct(cp)
    if len(mm) < 2:
        return mm, None
    vals = sorted(mm.values(), reverse=True)
    return mm, vals[0] - vals[1]


def main():
    rows = list(csv.DictReader(open(SET, newline="")))
    random.Random(SEED).shuffle(rows)
    ai = ChessAI(None, None, chess.Board(), True)

    terms = list(V2_TERMS)
    dropped = []
    stats = dict(seen=0, scored=0, skip_fewmoves=0, skip_mate=0, skip_nolabel=0,
                 skip_partition=0, agree=0, disagree=0, sep=0, sep_margin=0, notsep=0)
    per_pos = []
    acc_rows = []
    # Accumulated constraint rows for the JOINT fit: one w for every position at once.
    joint_rows = []
    # ★ PRESERVATION SETS — what a retune must not break. `keep_both` = positions we ALREADY get right
    # that SF11 also gets right; `keep_adv` = ⭐ positions we get right and SF11 does NOT. A w fitted only
    # on our failures moves us toward SF11 and can silently destroy `keep_adv`, which is the one place we
    # are ahead. Measured as COLLATERAL DAMAGE after the fit rather than assumed away.
    keep_both, keep_adv = [], []

    for r in rows:
        if stats["scored"] >= MAX_POS:
            break
        stats["seen"] += 1
        fen = r.get("fen")
        best_uci = (r.get("best_uci") or "").strip()
        if not fen or not best_uci:
            stats["skip_nolabel"] += 1
            continue
        try:
            board = chess.Board(fen)
        except Exception:
            stats["skip_nolabel"] += 1
            continue
        if board.is_game_over(claim_draw=False):
            stats["skip_nolabel"] += 1
            continue

        # ★★ PURE-EVAL ACCURACY, THE SECOND ADVANTAGE CELL (owner, 2026-09-20): *"if our eval is just
        # correct to SF18 search over the SF11 eval (not just move search) then that's a win for us to
        # maintain."* Measured on the ROOT position only -- no children -- so it is nearly free and is
        # taken on EVERY row, before any decision filter narrows the sample.
        # ⚠️ Compared in WIN% space, not centipawns. Our piece values are deliberately non-standard, so an
        # absolute cp error would score our SCALE rather than our judgement; win% compresses the tails and
        # is the project's honest statistic. Both static evals are handicapped equally by the
        # static-vs-search gap, so the COMPARISON is fair even though neither absolute error is small.
        if USE_SF11:
            try:
                _our_mp = float(ai.ev_breakdown(board)["total"])
                _t11, _ = sf11.eval(fen)
                _tgt = float(r["best_cp"])
                _mm_acc, _A_pre_acc = sf_indices(r.get("moves"), fen.split()[1] == "w")
                if _t11 is not None:
                    _e_our = abs(winpct(-_our_mp / 10.0) - winpct(_tgt))   # our eval is Black-positive mp
                    _e_11 = abs(winpct(_t11 * 100.0) - winpct(_tgt))       # SF11 reports White-POV pawns
                    stats["acc_n"] = stats.get("acc_n", 0) + 1
                    stats["acc_our_sum"] = stats.get("acc_our_sum", 0.0) + _e_our
                    stats["acc_11_sum"] = stats.get("acc_11_sum", 0.0) + _e_11
                    if _e_our < _e_11:
                        stats["acc_we_win"] = stats.get("acc_we_win", 0) + 1
                    # ★★ A WIN RATE ALONE CANNOT TELL A REAL ADVANTAGE FROM A COSMETIC ONE (owner,
                    # 2026-09-20): *"its possible we are marginally closer on those 43% and on key
                    # situations SF11 still beats us."* Two things would hollow it out -- our wins being
                    # narrow while our losses are wide (which the worse MEAN already hints at), and SF11
                    # winning specifically where the decision is CRITICAL. Both need the per-position
                    # margin and the criticality alongside it, so dump them rather than only counting.
                    # ⚠️ Dump the RAW win% values, not only the absolute errors. Without the sign there is
                    # no way to test the alternative explanation for our critical-position advantage:
                    # that our eval simply reads HOTTER (the record has v1 reading ~8-9 pawns hotter than
                    # SF in won positions). In decisive positions SF18's value is extreme, so a
                    # larger-magnitude eval lands closer BY SCALE rather than by judgement -- and a scale
                    # advantage cannot buy Elo. Raw values let any rescale be applied offline.
                    acc_rows.append((fen, _A_pre_acc if _A_pre_acc is not None else "",
                                     round(_e_our, 4), round(_e_11, 4), round(_e_11 - _e_our, 4),
                                     round(winpct(-_our_mp / 10.0), 4), round(winpct(_t11 * 100.0), 4),
                                     round(winpct(_tgt), 4)))
            except Exception:
                pass
        if ACC_ONLY:
            continue        # accuracy needs no children: skip the expensive half entirely

        # ☠️ MIN_A -- DROP DECISIONS SF DOES NOT CARE ABOUT, BEFORE PAYING TO EVALUATE THEM.
        # Measured 2026-09-20: 60% of raw "disagreements" sit at a SF best-vs-2nd gap under 1 win%, i.e.
        # SF is indifferent and our differing argmax is a coin flip, not a failure. Including them
        # diluted the separable rate from 28% (A>=3) to 19.5% and the joint fit from its real value to
        # 5.2%. The gap is computable from the cached label alone, so filtering here also skips the
        # child-evaluation cost for ~80% of rows.
        _mm_pre, _A_pre = sf_indices(r.get("moves"), fen.split()[1] == "w")
        if MIN_A > 0 and (_A_pre is None or _A_pre < MIN_A):
            stats["skip_indifferent"] = stats.get("skip_indifferent", 0) + 1
            continue

        legal = list(board.legal_moves)
        # ⚠️ A position with a single legal move has no ranking to express and no gap to measure. Counting
        # it as "not separable" would manufacture structural failures out of forced replies.
        if len(legal) < 2:
            stats["skip_fewmoves"] += 1
            continue

        stm_white = (board.turn == chess.WHITE)
        # Engine eval is ABSOLUTE Black-positive; orient so higher = better for the side to move.
        sign = 1 if board.turn == chess.BLACK else -1

        vecs, ucis, totals, sf11s, mate_child = [], [], [], [], False
        for mv in legal:
            board.push(mv)
            try:
                if board.is_checkmate():
                    mate_child = True
                    break
                if board.is_game_over(claim_draw=False):
                    continue
                bd = ai.ev_breakdown(board)
                if bd.get("checkmate"):
                    mate_child = True
                    break
                missing = [k for k in terms if k not in bd]
                if missing:
                    if not vecs and not per_pos:
                        # First position: a term absent here is absent for the whole arm (e.g. rook files
                        # gated off). Drop it from the vector ONCE, visibly, rather than per position.
                        for k in missing:
                            terms.remove(k); dropped.append(k)
                    else:
                        vecs = None
                        break
                x = np.array([float(bd[k]) for k in terms], dtype=float)
                # The partition must hold, or the vector is not the eval and the LP answers about
                # something else. This is the tool's own self-check, per position.
                if abs(x.sum() - float(bd["total"])) > 0.5:
                    vecs = None
                    break
                vecs.append(sign * x)
                totals.append(sign * float(bd["total"]))
                ucis.append(mv.uci())
                if USE_SF11:
                    # SF11 reports WHITE-POV pawns; orient to the PARENT's side to move, exactly as
                    # `sign` does for our Black-positive engine, so both rankings mean "higher is better
                    # for the player choosing".
                    t11, _terms11 = sf11.eval(board.fen())
                    sf11s.append((t11 if stm_white else -t11) if t11 is not None else float("nan"))
            finally:
                board.pop()
        if mate_child:
            # A mate-in-1 available is decided by search, not by eval weights.
            stats["skip_mate"] += 1
            continue
        if vecs is None:
            stats["skip_partition"] += 1
            continue
        if len(vecs) < 2 or best_uci not in ucis:
            stats["skip_nolabel"] += 1
            continue

        bi = ucis.index(best_uci)
        oi = max(range(len(totals)), key=lambda i: totals[i])
        # ★ ACHIEVABILITY GATE. Score ourselves ONLY where a hand-crafted eval demonstrably can succeed.
        if USE_SF11:
            if any(s != s for s in sf11s):          # any NaN -> SF11 did not answer for some child
                stats["skip_sf11"] = stats.get("skip_sf11", 0) + 1
                continue
            si = max(range(len(sf11s)), key=lambda i: sf11s[i])
            achievable = (ucis[si] == best_uci)
            stats["achievable" if achievable else "unachievable"] = \
                stats.get("achievable" if achievable else "unachievable", 0) + 1
            # ★★ THE 2x2, NOT A GATE (owner, 2026-09-20): *"in a small but existing amount of the time our
            # eval actually beat SF11 in being correct to 18 search. If we could maintain our small
            # advantage and tune the others towards SF11, that would be ideal."*
            #   SF11 hits + we miss  -> the tune TARGET (analysed below)
            #   SF11 MISSES + we hit -> ⭐ OUR ADVANTAGE. ☠️ The first version of this gate `continue`d
            #     here and DISCARDED it, which also meant the fitted w was optimised purely to move us
            #     TOWARD SF11 with no account of what that costs where we are already better --
            #     corpus-fit-is-anti-correlated-with-elo in miniature.
            #   both miss            -> static-vs-search gap, excluded from the separability analysis.
            if not achievable:
                if ucis[max(range(len(totals)), key=lambda i: totals[i])] == best_uci:
                    stats["our_advantage"] = stats.get("our_advantage", 0) + 1
                    _ai2 = ucis.index(best_uci)
                    keep_adv.append(np.array([vecs[_ai2] - vecs[j]
                                              for j in range(len(vecs)) if j != _ai2], dtype=float))
                    per_pos.append(dict(fen=fen, verdict="our_advantage", sf_gap_A="",
                                        our_confusability_mp="", t_star_mp="", n_children=len(vecs),
                                        our_move=best_uci, sf_move=best_uci))
                else:
                    stats["both_miss"] = stats.get("both_miss", 0) + 1
                continue                            # not a failure we can be charged with
        # QUIET=1 -- the instrument is only FAIR where a static eval could plausibly be the judge.
        # ☠️ Measured 2026-09-20 on the unfiltered set: SF's best was separable 10.2% and a RANDOM
        # sibling 10.0% -- i.e. no information at all, because most disagreements with a d14 search are
        # TACTICAL and no reweighting of a static eval expresses a tactic. Restricting to positions where
        # BOTH the contested moves are quiet (no capture, promotion or check) asks the question the eval
        # is actually responsible for. Keep the control: if SF still matches the random rate here, the
        # conclusion is about our term set, not about the filter.
        if os.environ.get("QUIET", "0") == "1":
            def _quiet(u):
                mv_ = chess.Move.from_uci(u)
                if board.is_capture(mv_) or mv_.promotion:
                    return False
                board.push(mv_)
                chk = board.is_check()
                board.pop()
                return not chk
            if not (_quiet(ucis[bi]) and _quiet(ucis[oi])):
                stats["skip_tactical"] = stats.get("skip_tactical", 0) + 1
                continue
        stats["scored"] += 1
        mm, A = sf_indices(r.get("moves"), stm_white)
        # OUR CONFUSABILITY: how tightly our eval packs SF's top three. Small spread = we cannot tell
        # SF's candidates apart, which is where a better eval would actually change the move.
        top3 = sorted(mm.items(), key=lambda kv: -kv[1])[:3]
        ours_on_top3 = [totals[ucis.index(u)] for u, _ in top3 if u in ucis]
        confus = (max(ours_on_top3) - min(ours_on_top3)) if len(ours_on_top3) > 1 else float("nan")

        # ☠️ THE NULL IS NOT ZERO UNTIL MEASURED. "x% of SF's moves are reachable by reweighting" means
        # nothing until we know what fraction of ARBITRARY moves are reachable: if any move can be made
        # top by some w in the bounds, the statistic is about the bounds, not about SF.
        # CONTROL = a uniformly random sibling that is neither our argmax nor SF's best.
        # SELF-CHECK = our own argmax, which MUST be separable (w = 1 already ranks it top). If the
        # self-check ever fails, the vector orientation or the partition is wrong, not the eval.
        def _sep(target_idx):
            D_ = np.array([vecs[target_idx] - vecs[j] for j in range(len(vecs)) if j != target_idx],
                          dtype=float)
            k_ = len(terms)
            r_ = linprog(c=np.concatenate([np.zeros(k_), [-1.0]]),
                         A_ub=np.hstack([-D_, np.ones((D_.shape[0], 1))]),
                         b_ub=np.zeros(D_.shape[0]),
                         bounds=[(W_LO, W_HI)] * k_ + [(None, None)], method="highs")
            return (float(r_.x[-1]) if r_.success else float("nan")), D_

        # ⚠️ TIES ARE NOT FAILURES. Two children can carry the SAME term vector (mirror-ish moves, or a
        # move that changes nothing any term measures). Strict separation is then impossible for ANY w,
        # so t* = 0. That is an INDIFFERENCE of the feature set, not a mis-weighting -- and for the SF
        # target it is the strongest possible form of "missing signal": no eval built on these features
        # can ever prefer one over the other. Counted separately from both.
        def _twin(idx):
            return any(np.allclose(vecs[idx], vecs[j], atol=0.5)
                       for j in range(len(vecs)) if j != idx)

        t_self, _ = _sep(oi)
        if not (t_self > 0) and not _twin(oi):
            stats["selfcheck_fail"] = stats.get("selfcheck_fail", 0) + 1
        elif not (t_self > 0):
            stats["selfcheck_tie"] = stats.get("selfcheck_tie", 0) + 1
        pool = [j for j in range(len(vecs)) if j != oi and j != bi]
        if pool:
            t_ctrl, _ = _sep(random.Random(SEED + stats["scored"]).choice(pool))
            stats["ctrl_n"] = stats.get("ctrl_n", 0) + 1
            if t_ctrl > 0:
                stats["ctrl_sep"] = stats.get("ctrl_sep", 0) + 1

        if bi == oi:
            stats["agree"] += 1
            verdict = "agree"
            t_star = float("nan")
            keep_both.append(np.array([vecs[bi] - vecs[j] for j in range(len(vecs)) if j != bi],
                                      dtype=float))
        else:
            stats["disagree"] += 1
            D = np.array([vecs[bi] - vecs[j] for j in range(len(vecs)) if j != bi], dtype=float)
            joint_rows.append(D)
            k = len(terms)
            # maximise t s.t. D.w >= t, W_LO <= w <= W_HI  ->  minimise -t with [-D, 1].w' <= 0
            A_ub = np.hstack([-D, np.ones((D.shape[0], 1))])
            res = linprog(c=np.concatenate([np.zeros(k), [-1.0]]),
                          A_ub=A_ub, b_ub=np.zeros(D.shape[0]),
                          bounds=[(W_LO, W_HI)] * k + [(None, None)],
                          method="highs")
            t_star = float(res.x[-1]) if res.success else float("nan")
            if res.success and t_star > 0:
                stats["sep"] += 1
                verdict = "separable"
                if t_star >= MARGIN:
                    stats["sep_margin"] += 1
            elif _twin(bi):
                # SF's move is REPRESENTATIONALLY identical to a sibling under our terms.
                stats["twin"] = stats.get("twin", 0) + 1
                verdict = "indistinguishable"
            else:
                stats["notsep"] += 1
                verdict = "not_separable"

        per_pos.append(dict(fen=fen, verdict=verdict, sf_gap_A=("" if A is None else round(A, 3)),
                            our_confusability_mp=("" if confus != confus else round(confus, 1)),
                            t_star_mp=("" if t_star != t_star else round(t_star, 2)),
                            n_children=len(vecs), our_move=ucis[oi], sf_move=best_uci))

    # ★ VEC_OUT -- dump the raw difference rows so a FIT can be iterated offline without re-evaluating
    # every child. One row per constraint: (pos_id, class, d_1..d_k) where d = x_target - x_sibling and
    # the fit's requirement is d·w > 0. `class` is what the fit must do with it:
    #   fix   = our failure on an ACHIEVABLE position (SF11 gets it, we do not) -- the objective
    #   keep  = a position we already get right and SF11 does too       -- must not break
    #   adv   = ⭐ we get it right and SF11 does NOT                      -- must not break, the advantage
    # ⇒ the honest objective is MAXIMISE fixed SUBJECT TO keeping `keep` and `adv`, not "minimise
    # distance to SF11". Fitted the naive way this set scores net −17 (fixed 11, broke 28).
    if os.environ.get("VEC_OUT"):
        vp = os.environ["VEC_OUT"]
        if not os.path.isabs(vp):
            vp = os.path.join(THIS, vp)
        with open(vp, "w", newline="") as fh:
            vw = csv.writer(fh)
            vw.writerow(["pos_id", "class"] + ["d_" + t for t in terms])
            pid = 0
            for cls, mats in (("fix", joint_rows), ("keep", keep_both), ("adv", keep_adv)):
                for Di in mats:
                    pid += 1
                    for row in Di:
                        vw.writerow([pid, cls] + ["%.4f" % v for v in row])
        print("  wrote constraint rows -> %s" % vp)

    if os.environ.get("ACC_OUT") and acc_rows:
        ap = os.environ["ACC_OUT"]
        if not os.path.isabs(ap):
            ap = os.path.join(THIS, ap)
        with open(ap, "w", newline="") as fh:
            aw = csv.writer(fh)
            aw.writerow(["fen", "sf_gap_A", "err_ours_pp", "err_sf11_pp", "margin_pp_ours_better",
                         "wp_ours", "wp_sf11", "wp_target"])
            aw.writerows(acc_rows)
        print("  wrote per-position accuracy -> %s  (%d rows)" % (ap, len(acc_rows)))

    if ACC_ONLY:
        an = stats.get("acc_n", 0)
        if an:
            print("\n  ★ PURE-EVAL ACCURACY (ACC_ONLY, whole corpus, no decision filter):")
            print("    mean |error|: OURS %.2fpp · SF11 %.2fpp" % (stats["acc_our_sum"] / an,
                                                                  stats["acc_11_sum"] / an))
            print("    ⭐ our eval closer: %d / %d (%.1f%%)"
                  % (stats.get("acc_we_win", 0), an, 100.0 * stats.get("acc_we_win", 0) / an))
        return

    if not stats["scored"]:
        sys.exit("no positions scored from %s" % SET)

    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_pos[0].keys()))
        w.writeheader(); w.writerows(per_pos)

    n = stats["scored"]; dis = stats["disagree"]
    print("\nTERM SEPARABILITY — can any reweighting of v2's subsystems play SF's move?")
    print("  set=%s  scored=%d  terms=%s" % (os.path.basename(SET), n, ",".join(terms)))
    if dropped:
        print("  ⚠️ dropped (never published by this arm, NOT zero): %s" % ",".join(dropped))
    print("  skipped: <2 legal %d · mate-in-1 available %d · no label %d · partition mismatch %d"
          % (stats["skip_fewmoves"], stats["skip_mate"], stats["skip_nolabel"], stats["skip_partition"]))
    if USE_SF11:
        ach, unach = stats.get("achievable", 0), stats.get("unachievable", 0)
        tot_ach = ach + unach
        print("\n  ★ ACHIEVABILITY (SF11's classical eval as the HCE yardstick):")
        print("    SF11 STATIC also picks SF18's move : %d / %d (%.1f%%)  <- the only fair target set"
              % (ach, tot_ach, 100.0 * ach / max(1, tot_ach)))
        print("    SF11 static misses it too          : %d (%.1f%%)  <- static-vs-search gap"
              % (unach, 100.0 * unach / max(1, tot_ach)))
        adv, bm = stats.get("our_advantage", 0), stats.get("both_miss", 0)
        print("       ⭐ of those, WE pick SF18's move and SF11 does NOT : %d (%.1f%% of all decisions)"
              % (adv, 100.0 * adv / max(1, tot_ach)))
        print("          both miss: %d. ⇒ our advantage cell is REAL and must be PRESERVED by any retune"
              % bm)
        print("          that moves us toward SF11 — see the collateral rows below.")
    if stats.get("acc_n"):
        an = stats["acc_n"]
        print("\n  ★ PURE-EVAL ACCURACY vs SF18's search value (win%% error, ALL rows, no decision filter):")
        print("    mean |error|: OURS %.2fpp · SF11 %.2fpp   (lower is better)"
              % (stats["acc_our_sum"] / an, stats["acc_11_sum"] / an))
        print("    ⭐ positions where OUR eval is CLOSER than SF11's: %d / %d (%.1f%%)"
              % (stats.get("acc_we_win", 0), an, 100.0 * stats.get("acc_we_win", 0) / an))
        print("       ⇒ a real advantage cell to PRESERVE, and it matters independently of Elo:"
              " the roadmap is HCE → the owner's own NN, and eval accuracy IS the teacher's label quality.")
    print("\n  our static argmax == SF best : %d (%.1f%%)" % (stats["agree"], 100.0 * stats["agree"] / n))
    print("  disagreements                : %d (%.1f%%)" % (dis, 100.0 * dis / n))
    if dis:
        print("    SEPARABLE (some w in [%.2f,%.2f] fixes it) : %d (%.1f%% of disagreements)"
              % (W_LO, W_HI, stats["sep"], 100.0 * stats["sep"] / dis))
        print("      of which with >= %.0f mp margin           : %d (%.1f%%)"
              % (MARGIN, stats["sep_margin"], 100.0 * stats["sep_margin"] / dis))
        print("    NOT separable by ANY subsystem reweighting : %d (%.1f%% of disagreements)"
              % (stats["notsep"], 100.0 * stats["notsep"] / dis))
        cn, cs = stats.get("ctrl_n", 0), stats.get("ctrl_sep", 0)
        print("\n  ☠️ CONTROLS — read the row above ONLY against these:")
        print("    NULL: a RANDOM sibling, same test : %d/%d (%.1f%%)  <- if this matches the SF rate,"
              % (cs, cn, 100.0 * cs / max(1, cn)))
        print("          the statistic is about the weight BOUNDS, not about SF's move.")
        print("    SELF-CHECK: our own argmax separable: %d failures (must be 0; a failure means the"
              % stats.get("selfcheck_fail", 0))
        print("          vector orientation or the partition is wrong, not the eval). ties=%d"
              % stats.get("selfcheck_tie", 0))
        print("    INDISTINGUISHABLE: SF's move carries the SAME term vector as a sibling: %d (%.1f%%)"
              % (stats.get("twin", 0), 100.0 * stats.get("twin", 0) / dis))
        print("          ⇒ no eval built on these features can EVER prefer it. The hardest missing-signal"
              " class, and distinct from mis-weighting.")
        print("\n  ⇒ TUNING CEILING (subsystem-granular, static, one ply): at most %.1f%% of our move"
              % (100.0 * stats["sep"] / dis))
        print("    disagreements are reachable by reweighting; the remaining %.1f%% need a signal v2's"
              % (100.0 * stats["notsep"] / dis))
        print("    current term set cannot express AT THIS GRANULARITY (see the bounds note in the docstring).")

    # JOINT fit: one w for ALL disagreements at once. This is the realistic bound -- per-position
    # separability is optimistic because each position may want a different w.
    if joint_rows:
        D = np.vstack(joint_rows)
        k = len(terms)
        A_ub = np.hstack([-D, np.ones((D.shape[0], 1))])
        res = linprog(c=np.concatenate([np.zeros(k), [-1.0]]),
                      A_ub=A_ub, b_ub=np.zeros(D.shape[0]),
                      bounds=[(W_LO, W_HI)] * k + [(None, None)], method="highs")
        if res.success:
            w = res.x[:k]
            # ⚠️ Count POSITIONS, not constraint rows. A position is fixed only if its move beats EVERY
            # sibling, so every row of its own block must be positive; summing rows across the stacked
            # matrix would report a much larger, meaningless number.
            fixed = sum(1 for Di in joint_rows if bool((Di @ w > 0).all()))
            print("\n  JOINT single-w fit over all %d disagreements (the realistic bound):" % dis)
            print("    positions a SINGLE reweighting fixes: %d (%.1f%% of disagreements)"
                  % (fixed, 100.0 * fixed / max(1, len(joint_rows))))
            print("    w = " + "  ".join("%s=%.2f" % (t, v) for t, v in zip(terms, w)))
            # ★★ COLLATERAL DAMAGE. A w fitted only on our failures drags us toward SF11; these two rows
            # say what that costs where we are ALREADY right -- and especially where we are right and
            # SF11 is WRONG, which is the advantage the owner wants preserved.
            kb = sum(1 for Di in keep_both if bool((Di @ w > 0).all()))
            ka = sum(1 for Di in keep_adv if bool((Di @ w > 0).all()))
            print("    collateral: keeps %d/%d positions we already get right (%.1f%% BROKEN)"
                  % (kb, len(keep_both), 100.0 * (len(keep_both) - kb) / max(1, len(keep_both))))
            print("    ⭐ collateral on OUR ADVANTAGE (we right, SF11 wrong): keeps %d/%d (%.1f%% BROKEN)"
                  % (ka, len(keep_adv), 100.0 * (len(keep_adv) - ka) / max(1, len(keep_adv))))
            print("    ⇒ net positions moved = %+d (fixed %d − broken %d), the only figure that matters."
                  % (fixed - (len(keep_both) - kb) - (len(keep_adv) - ka),
                     fixed, (len(keep_both) - kb) + (len(keep_adv) - ka)))
            print("    ⚠️ a STARTING POINT and a ceiling estimate, not a shippable config: it is fitted to"
                  " static one-ply rankings, and every prior corpus-fitted config lost in games.")
    print("\n  wrote %s" % OUT)


if __name__ == "__main__":
    main()
