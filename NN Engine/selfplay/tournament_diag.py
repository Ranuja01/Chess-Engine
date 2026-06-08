# -*- coding: utf-8 -*-
"""
Tournament diagnostics — the per-tag color / depth / calibration readout.

Re-runnable version of the 2026-06-05 ad-hoc analysis that surfaced the depth-dependent color
instability (identical-engine ship_lightning went Black-65% while ship_blitz went White-80% from
SF-equal openings). Point it at any games/<tag>/ and it prints:

  - result tally + WHITE vs BLACK win-rate, resign-by-color, termination-reason breakdown
  - per-opening outcomes (both color-runs), with the SF seed eval of each opening
  - our search depth vs Stockfish depth, split midgame / endgame
  - eval calibration: our search eval vs SF search eval (overall + split by who SF says is winning),
    and, when the newer annotations carry it, our static vs SF static (NNUE)

COLOR CAVEAT: white/black win-rate is only a clean ENGINE color signal for identical-engine
tournaments (e.g. ship vs ship2, same config). For A/B runs (nmp_on vs nmp_off, stackON vs stackOFF)
the color columns conflate color with the config difference — read the depth/calibration sections
instead. The script prints a heads-up when the two configs differ.

Run in WSL from NN Engine/ (no rebuild needed — pure data analysis):

    python selfplay/tournament_diag.py --tag ship_lightning
    python selfplay/tournament_diag.py --tag nmp_standard
    python selfplay/tournament_diag.py --tag ship_lightning ship_blitz   # compare several
"""

import os
import sys
import glob
import json
import argparse
import statistics as st

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
GAMES_DIR = os.path.join(THIS_DIR, "games")
OPENINGS_TXT = os.path.join(THIS_DIR, "openings.txt")

# Engine search eval is recorded White-POV in milli-pawns (pawn = 1000); SF cp is centipawns.
# eval_breakdown totals are ABSOLUTE (Black-positive) milli-pawns; White-POV cp = -total / 10.
ENDGAME_PIECE_COUNT = 14   # <= this many pieces on the board => "endgame" bucket
EQUAL_SEED_CP = 60         # |SF seed cp| below this => the opening is "equal"
WINNING_CP = 100           # |SF cp| above this => one side is "winning" (for the calibration split)


def load_openings():
    out = []
    if not os.path.exists(OPENINGS_TXT):
        return out
    with open(OPENINGS_TXT) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                out.append(line)
    return out


def load_games(tag):
    """Return [(game_dir, [records])], preferring annotated JSONL when present."""
    out = []
    for gd in sorted(glob.glob(os.path.join(GAMES_DIR, tag, "game_*"))):
        ap = os.path.join(gd, "game.annotated.jsonl")
        p = ap if os.path.exists(ap) else os.path.join(gd, "game.jsonl")
        if not os.path.exists(p):
            continue
        with open(p) as f:
            recs = [json.loads(line) for line in f if line.strip()]
        if recs:
            out.append((gd, recs))
    return out


def meta_of(recs):
    return recs[0] if recs and recs[0].get("type") == "meta" else {}


def result_of(recs):
    for r in recs:
        if r.get("type") == "result":
            return r
    return {}


def opening_index(meta, openings):
    op = meta.get("opening", "")
    for i, o in enumerate(openings):
        if o == op:
            return i
    return None


def piece_count(fen):
    return sum(1 for c in fen.split(" ")[0] if c.isalpha()) if fen else 32


def seed_eval(recs):
    """SF search eval (White-POV cp) at the last booked opening ply."""
    last = None
    for r in recs:
        if r.get("type") == "move" and r.get("opening") and r.get("sf_cp") is not None:
            last = r["sf_cp"]
    return last


def outcome_letter(res, reason):
    if "crash" in (reason or ""):
        return "C"
    if res == "1-0":
        return "W"
    if res == "0-1":
        return "B"
    if res and "1/2" in res:
        return "D"
    return "?"


def analyze(tag, openings):
    games = load_games(tag)
    print("\n" + "=" * 78)
    print("%s  (%d games)" % (tag, len(games)))
    print("=" * 78)
    if not games:
        print("  no games found under %s" % os.path.join(GAMES_DIR, tag))
        return

    meta = meta_of(games[0][1])
    cfg_w, cfg_b = meta.get("config_white", ""), meta.get("config_black", "")
    print("  white=%s  black=%s" % (meta.get("white"), meta.get("black")))
    identical = cfg_w == cfg_b
    print("  config: %s" % cfg_w)
    if not identical:
        print("  !! configs differ (white=%r black=%r) -- this is an A/B run, NOT a clean color test;"
              " treat WHITE/BLACK win-rate with care." % (cfg_w, cfg_b))

    ww = bw = dr = cr = 0
    resign_color = {"white": 0, "black": 0}
    reasons = {}
    opening_outcomes = {}      # idx -> [outcome letters]
    opening_seed = {}          # idx -> SF seed cp

    # depth + calibration accumulators
    od_mid, od_end, sd = [], [], []
    dn = {"white": {"d": [], "n": []}, "black": {"d": [], "n": []}}
    diffs, sf_for_diff = [], []          # our_search - sf_search (cp), and the sf value
    static_diffs = {"white": [], "black": []}   # our_static - sf_static (cp) by winning side

    for gd, recs in games:
        gmeta, res = meta_of(recs), result_of(recs)
        reason, result = res.get("reason", "") or "", res.get("result")
        idx = opening_index(gmeta, openings)
        let = outcome_letter(result, reason)
        if let == "W":
            ww += 1
        elif let == "B":
            bw += 1
        elif let == "D":
            dr += 1
        elif let == "C":
            cr += 1
        opening_outcomes.setdefault(idx, []).append(let)
        if idx not in opening_seed:
            opening_seed[idx] = seed_eval(recs)

        rk = "crash" if "crash" in reason else reason.split(" at ")[0].split("  [")[0]
        reasons[rk] = reasons.get(rk, 0) + 1

        if "resigned" in reason:
            lbl = reason.split(" resigned")[0]
            loser = "white" if lbl == gmeta.get("white") else "black"
            resign_color[loser] += 1

        for r in recs:
            if r.get("type") != "move" or r.get("opening"):
                continue
            col, d, n = r.get("color"), r.get("depth"), r.get("nodes")
            if d is not None:
                (od_end if piece_count(r.get("fen", "")) <= ENDGAME_PIECE_COUNT else od_mid).append(d)
                if col in dn:
                    dn[col]["d"].append(d)
            if n is not None and col in dn:
                dn[col]["n"].append(n)
            if r.get("sf_depth") is not None:
                sd.append(r["sf_depth"])
            ew, cp = r.get("eval_white_pov"), r.get("sf_cp")
            if ew is not None and cp is not None and abs(cp) < 1500:
                diffs.append(ew / 10.0 - cp)   # both -> cp, White-POV
                sf_for_diff.append(cp)
            eb, ss = r.get("eval_breakdown"), r.get("sf_static_cp")
            if isinstance(eb, dict) and eb.get("total") is not None and ss is not None:
                gap = (-eb["total"] / 10.0) - ss   # our_static (White-POV cp) - SF_static cp
                if ss > WINNING_CP:
                    static_diffs["white"].append(gap)
                elif ss < -WINNING_CP:
                    static_diffs["black"].append(gap)

    n = len(games)
    print("\n  RESULTS: White %d (%.0f%%)  Black %d (%.0f%%)  draws %d  crashes %d"
          % (ww, 100 * ww / n, bw, 100 * bw / n, dr, cr))
    print("  resigned by color: white %d  black %d" % (resign_color["white"], resign_color["black"]))
    print("  termination reasons: %s" % reasons)

    # per-opening
    print("\n  per-opening (SF seed eval, White-POV cp; outcomes are both color-runs):")
    print("    %4s %11s %8s  outcomes" % ("op", "SF_seed_cp", "verdict"))
    for idx in sorted(opening_outcomes, key=lambda x: (x is None, x)):
        se = opening_seed.get(idx)
        verdict = "?" if se is None else ("EQUAL" if abs(se) < EQUAL_SEED_CP
                                          else ("W-fav" if se > 0 else "B-fav"))
        print("    %4s %11s %8s  %s" % (idx, se, verdict, opening_outcomes[idx]))

    # depth
    def m(x):
        return "%.1f" % st.mean(x) if x else "-"

    def md(x):
        return "%.0f" % st.median(x) if x else "-"

    print("\n  DEPTH:")
    print("    our midgame(>%dpc): mean %s med %s | endgame(<=%dpc): mean %s med %s"
          % (ENDGAME_PIECE_COUNT, m(od_mid), md(od_mid), ENDGAME_PIECE_COUNT, m(od_end), md(od_end)))
    print("    SF depth (at its movetime): mean %s med %s" % (m(sd), md(sd)))
    if dn["white"]["d"] and dn["black"]["d"]:
        print("    by side-to-move: white d%.2f / %s nodes  |  black d%.2f / %s nodes"
              % (st.mean(dn["white"]["d"]), "%.0f" % st.mean(dn["white"]["n"]) if dn["white"]["n"] else "-",
                 st.mean(dn["black"]["d"]), "%.0f" % st.mean(dn["black"]["n"]) if dn["black"]["n"] else "-"))

    # calibration (search)
    if diffs:
        wpos = [d for d, s in zip(diffs, sf_for_diff) if s > WINNING_CP]
        bpos = [d for d, s in zip(diffs, sf_for_diff) if s < -WINNING_CP]
        eqp = [d for d, s in zip(diffs, sf_for_diff) if abs(s) <= WINNING_CP]
        print("\n  EVAL CALIBRATION (our_search - SF_search, White-POV cp):")
        print("    overall: n=%d mean %+.0f  median %+.0f" % (len(diffs), st.mean(diffs), st.median(diffs)))
        print("    White-winning: %+.0f (n%d) | Black-winning: %+.0f (n%d) | equal: %+.0f (n%d)"
              % (st.mean(wpos) if wpos else 0, len(wpos),
                 st.mean(bpos) if bpos else 0, len(bpos),
                 st.mean(eqp) if eqp else 0, len(eqp)))
    if static_diffs["white"] or static_diffs["black"]:
        sw, sb = static_diffs["white"], static_diffs["black"]
        print("  STATIC CALIBRATION (our_static - SF_static NNUE, White-POV cp):")
        print("    White-winning: %s (n%d) | Black-winning: %s (n%d)"
              % (("%+.0f" % st.mean(sw)) if sw else "-", len(sw),
                 ("%+.0f" % st.mean(sb)) if sb else "-", len(sb)))


def main():
    ap = argparse.ArgumentParser(description="Per-tag tournament color / depth / calibration diagnostics.")
    ap.add_argument("--tag", nargs="+", required=True, help="one or more games/<tag>/ directories")
    args = ap.parse_args()
    openings = load_openings()
    for tag in args.tag:
        analyze(tag, openings)


if __name__ == "__main__":
    main()
