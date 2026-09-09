# -*- coding: utf-8 -*-
"""Footprint-filtered D7 regret: the SENSITIVITY instrument for a subsystem change the general set is too
blunt to resolve. On the game-representative multi-PV set, run our fixed-depth move for a BASE arm and one
or more CANDIDATE arms; then compare SF18 win%-regret ONLY on the positions where the candidate CHANGES our
move vs base. A KS change is move-neutral on most positions (memory most-eval-error-is-move-neutral), so its
aggregate held-regret delta drowns in noise; but on the positions it actually moves, the mean regret of the
new move vs the old is directly resolvable and is the honest "when it changes our move, is it better?" test.

Not overfit: it is just a filter over real game positions, adjudicated by SF18 ground-truth labels.

  pyrun diagnostics/_ks_footprint_regret.py [SET=ks_sets/game_regret_set.csv] [DEPTH=7] [JOBS=4]
        [SPLIT=all] [SPLIT_FRAC=0.7] [MAXN=0]

⚠️ FIXED depth (deterministic), a PROXY for game depth. A positive footprint delta that replicates on the
v2 cross-set is a real (usually small) structural gain; games still decide the Elo.
"""
import os, sys, csv, math, subprocess

for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v


def _winpct(cp):
    return 50.0 + 50.0 * (2.0 / (1.0 + math.exp(-0.00368208 * cp)) - 1.0)


# ---------------- WORKER: one config, one slice -> per-position (fen, our_move, regret) ----------------
if os.environ.get("WORKER") == "1":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['PRESET'] = 'LONG_FORMAT'
    os.environ['MAX_DEPTH'] = os.environ.get('DEPTH', '7')
    os.environ['USE_OPENING_BOOK'] = '0'
    THIS = os.path.dirname(os.path.abspath(__file__)); ENGINE = os.path.dirname(THIS)
    sys.path.insert(0, ENGINE); sys.path.insert(0, THIS); sys.path.insert(0, os.path.join(ENGINE, "selfplay"))
    import chess
    from tactical_test import run_one
    SET = os.environ["SET"]; MISS = float(os.environ.get("MISS", "30"))
    si, sn = (int(x) for x in os.environ["SLICE"].split("/"))
    frac = float(os.environ.get("SPLIT_FRAC", "0.7"))
    which = os.environ.get("SPLIT", "all")
    MAXN = int(os.environ.get("MAXN", "0"))
    rows = list(csv.DictReader(open(SET, newline="")))
    import random as _r
    _r.Random(1234).shuffle(rows)
    if MAXN:
        rows = rows[:MAXN]
    cut = int(frac * len(rows))
    rows = rows[:cut] if which == "tune" else (rows[cut:] if which == "held" else rows)
    rows = rows[si::sn]
    out = open(os.environ["OUT"], "w", newline="")
    w = csv.writer(out)
    for r in rows:
        fen = r.get("fen")
        try:
            best_cp = float(r["best_cp"])
            mm = {}
            for pair in (r.get("moves") or "").split(";"):
                if ":" in pair:
                    u, c = pair.rsplit(":", 1); mm[u] = float(c)
        except Exception:
            continue
        if not mm:
            continue
        try:
            our = run_one(fen, set())["uci"]
        except Exception:
            continue
        stm_white = (fen.split()[1] == 'w')
        our_cp = mm.get(our)
        if our_cp is None:
            worst = min(mm.values()) if stm_white else max(mm.values())
            our_cp = (worst - MISS) if stm_white else (worst + MISS)
        bw = _winpct(best_cp) if stm_white else (100.0 - _winpct(best_cp))
        ow = _winpct(our_cp) if stm_white else (100.0 - _winpct(our_cp))
        reg = max(0.0, bw - ow)
        # OUR engine's phase variable (cpp_bitboard.cpp:7337-7342), so strata line up with the isEndGame
        # branch. The corpus `phase_bucket` is a raw PIECE COUNT (>=26 / >=14, _build_game_regret_set.py:58)
        # which puts our mid/end boundary INSIDE its "midgame" bucket -- it structurally cannot see the
        # phase cliff.
        _bd = fen.split()[0]
        _ph = min(24, 4 * (_bd.count('q') + _bd.count('Q'))
                    + 2 * (_bd.count('r') + _bd.count('R'))
                    + sum(_bd.count(_c) for _c in 'nbNB'))
        _ps = 128 * (24 - _ph) // 24
        # CRITICALITY = win%(SF best) - win%(SF 2nd best). 75.5% of this corpus is "benign" (<3%), where
        # choosing differently costs almost nothing. An arm's win% edge is only Elo-relevant to the extent
        # it lands on the critical tail, so the aggregate can be large and irrelevant, or small and decisive.
        _vals = sorted(mm.values(), reverse=stm_white)
        _crit = abs(_winpct(_vals[0]) - _winpct(_vals[1])) if len(_vals) >= 2 else 0.0
        w.writerow([fen, our, "%.6f" % reg, r.get("phase_bucket", "?"), _ps, "%.4f" % _crit])
    out.close()
    sys.exit(0)

# ---------------- DRIVER ----------------
THIS = os.path.dirname(os.path.abspath(__file__)); ENGINE = os.path.dirname(THIS)
PY = sys.executable
SET = os.environ.get("SET", "ks_sets/game_regret_set.csv")
if not os.path.isabs(SET):
    SET = os.path.join(THIS, SET)
DEPTH = os.environ.get("DEPTH", "7")
JOBS = int(os.environ.get("JOBS", "4"))
SPLIT = os.environ.get("SPLIT", "all")
SPLIT_FRAC = os.environ.get("SPLIT_FRAC", "0.7")
MAXN = int(os.environ.get("MAXN", "0"))

BUNDLE = {"OVD_BOUNDED_MODE": 2, "OVD_CAP": 300, "OVD_KNEE": 40,
          "CENTRAL_BOUNDED_MODE": 1, "CENTRAL_CAP": 150, "CENTRAL_KNEE": 200}


def cfg(**kw):
    d = dict(BUNDLE); d.update(kw); return d


BASE = ("bundle+defaware1 (the ship)", cfg(KS_DEFAWARE_MODE=1))
CANDS = [
    # KS move-decisiveness UNDER SEARCH: turn KS OFF and see whether our D7 move changes vs base, and on the
    # positions it changes, whether the KS-ON move was better (positive delta) or worse (negative delta) per
    # SF18. MAG=0 zeroes the KS contribution (verified: sibling_spread king_safety -> 0.0% flip at MAG=0).
    # The MAG=3000 arm == base default => a determinism/noise-floor CONTROL: it must report ~0 changed moves,
    # else the instrument itself is noisy and the KS-off delta is untrustworthy.
    ("KS OFF (KING_SAFETY_MAG=0)", cfg(KING_SAFETY_MAG=0)),
]


# Optional override so this instrument can be pointed at ANY arm pair, not only the KS arms it was written
# for. `CAND_KNOBS` (and optionally `BASE_KNOBS`) are space-separated KEY=VAL strings; unlike cfg() they do
# NOT merge the KS BUNDLE, so the base is the SHIPPED defaults. Absent => everything above runs unchanged,
# byte-identical for every existing caller.
# Motivation: "when it changes our move, is it better?" is the right question for ANY eval change, not just
# a KS one -- and it is the question a raw move-FLIP RATE cannot answer (measured 2026-09-06: the SF15c
# oracle flips 63.5% of positions, but so does a taper arm we know is not an improvement, at 40%, against
# a ~20.8% aspiration-noise floor. Flip rate does not rank evals; regret on the flipped set does).
def _parse_knobs(s):
    d = {}
    for kv in s.split():
        if "=" in kv:
            k, v = kv.split("=", 1)
            d[k] = v
    return d


_ck = os.environ.get("CAND_KNOBS", "").strip()
if _ck:
    BASE = ("shipped defaults", _parse_knobs(os.environ.get("BASE_KNOBS", "")))
    CANDS = [(os.environ.get("CAND_NAME", "candidate"), _parse_knobs(_ck))]


def collect(config, tag):
    """Run JOBS slice-workers; return {fen: (our_move, regret)}."""
    env = dict(os.environ, SET=SET, DEPTH=DEPTH, SPLIT=SPLIT, SPLIT_FRAC=SPLIT_FRAC, MAXN=str(MAXN))
    knob_args = ["%s=%s" % (k, v) for k, v in config.items()]
    procs = []
    for i in range(JOBS):
        # PID-qualified: the path used to be /tmp/_fp_<tag>_<i>.csv, which is IDENTICAL across concurrent
        # invocations -- two arms launched at once clobber each other's worker output and silently return
        # the same numbers for different knobs (observed 2026-09-06: two arms byte-identical, and a
        # truncated position count). Keep this unique or arms must be run strictly sequentially.
        outp = "/tmp/_fp_%d_%s_%d.csv" % (os.getpid(), tag, i)
        e = dict(env, WORKER="1", SLICE="%d/%d" % (i, JOBS), OUT=outp)
        procs.append((subprocess.Popen([PY, "-u", os.path.abspath(__file__)] + knob_args,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                       text=True, cwd=ENGINE, env=e), outp))
    d = {}
    for p, outp in procs:
        p.communicate()
        try:
            for row in csv.reader(open(outp, newline="")):
                if len(row) >= 3:
                    d[row[0]] = (row[1], float(row[2]), row[3] if len(row) > 3 else "?",
                                 int(row[4]) if len(row) > 4 else -1,
                                 float(row[5]) if len(row) > 5 else -1.0)
        except Exception:
            pass
    return d


base = collect(BASE[1], "base")
print("SET=%s DEPTH=%s SPLIT=%s  base=%s  positions=%d\n" % (os.path.basename(SET), DEPTH, SPLIT, BASE[0], len(base)), flush=True)
print("  %-38s %7s %8s   %10s %10s %9s  %6s/%-6s %5s"
      % ("candidate", "changed", "%chg", "reg_base", "reg_cand", "delta", "bett", "wors", "win%"), flush=True)
for name, c in CANDS:
    cand = collect(c, "cand")
    shared = [f for f in base if f in cand]
    changed = [f for f in shared if base[f][0] != cand[f][0]]
    if not changed:
        print("  %-38s %7d %8s   %10s %10s %9s" % (name, 0, "-", "-", "-", "-")); continue
    rb = sum(base[f][1] for f in changed) / len(changed)
    rc = sum(cand[f][1] for f in changed) / len(changed)
    # better/worse COUNTS, not just the mean. A mean delta averages away the thing that decides whether a
    # fix ships: our eval error is TWO-SIDED, so a change that is right about its target population is
    # usually wrong about a comparable population elsewhere. If bett/wors is ~50/50 for every arm, that IS
    # the explanation for why detectable, patterned collapses cannot be plugged without regressing elsewhere.
    def _bw(fs):
        return (sum(1 for f in fs if cand[f][1] < base[f][1] - 1e-9),
                sum(1 for f in fs if cand[f][1] > base[f][1] + 1e-9))
    _b, _w = _bw(changed)
    print("  %-38s %7d %7.1f%%   %10.4f %10.4f %+9.4f  %6d/%-6d %5.1f%%"
          % (name, len(changed), 100.0 * len(changed) / len(shared), rb, rc, rc - rb,
             _b, _w, 100.0 * _b / max(1, _b + _w)), flush=True)
    # BY_PHASE=0 suppresses. Splits the SAME changed set by the corpus `phase_bucket` column, so a gain
    # concentrated in one phase is visible instead of averaged away. Each phase still needs its OWN null
    # arm -- the selection bias documented below is per-population, not a single global offset.
    if os.environ.get("BY_PHASE", "1") != "0":
        buckets = {}
        for f in changed:
            buckets.setdefault(base[f][2] if len(base[f]) > 2 else "?", []).append(f)
        for ph in sorted(buckets):
            fs = buckets[ph]
            _rb = sum(base[f][1] for f in fs) / len(fs)
            _rc = sum(cand[f][1] for f in fs) / len(fs)
            _tot = sum(1 for f in shared if (base[f][2] if len(base[f]) > 2 else "?") == ph)
            _pb, _pw = _bw(fs)
            print("      %-34s %7d %7.1f%%   %10.4f %10.4f %+9.4f  %6d/%-6d %5.1f%%"
                  % ("." + ph, len(fs), 100.0 * len(fs) / max(1, _tot), _rb, _rc, _rc - _rb,
                     _pb, _pw, 100.0 * _pb / max(1, _pb + _pw)), flush=True)
    # BY_PS=0 suppresses. Buckets straddle the engine's OWN mid/end step (isEndGame = phase_score > 64;
    # only 25 phase_score values are reachable, so 64 -> 69 is a single minor trade). Every mapped eval
    # cliff -- KS to zero, advanced_endgame_eval switching on, the N/B/Q evaluator swap, central and ovd
    # vanishing -- fires between buckets 2 and 3. A gain that JUMPS there implicates the cliff; a flat
    # profile across it means the discontinuities are cosmetic.
    if os.environ.get("BY_PS", "1") != "0":
        def _psb(v):
            if v < 0:   return "?"
            if v <= 53: return "ps1_mid_far   (<=53)"
            if v <= 64: return "ps2_mid_EDGE  (58-64)"
            if v <= 74: return "ps3_end_EDGE  (69-74)"
            return "ps4_end_far   (>=80)"
        pb = {}
        for f in changed:
            pb.setdefault(_psb(base[f][3] if len(base[f]) > 3 else -1), []).append(f)
        for ph in sorted(pb):
            fs = pb[ph]
            _rb = sum(base[f][1] for f in fs) / len(fs)
            _rc = sum(cand[f][1] for f in fs) / len(fs)
            _tot = sum(1 for f in shared if _psb(base[f][3] if len(base[f]) > 3 else -1) == ph)
            _pb, _pw = _bw(fs)
            print("      %-34s %7d %7.1f%%   %10.4f %10.4f %+9.4f  %6d/%-6d %5.1f%%"
                  % (ph, len(fs), 100.0 * len(fs) / max(1, _tot), _rb, _rc, _rc - _rb,
                     _pb, _pw, 100.0 * _pb / max(1, _pb + _pw)), flush=True)
    # BY_CRIT=0 suppresses. Splits by how much the position's best move beats its second (SF18 win%).
    # ~75% of the corpus is benign (<3%) where being right costs almost nothing; only the tail decides games.
    # An arm whose edge is flat across criticality is winning mostly-irrelevant decisions.
    if os.environ.get("BY_CRIT", "1") != "0":
        def _cb(v):
            if v < 0:    return "?"
            if v < 3.0:  return "cr1_benign    (<3%)"
            if v < 8.0:  return "cr2_minor     (3-8%)"
            if v < 20.0: return "cr3_moderate  (8-20%)"
            return "cr4_CRITICAL  (>20%)"
        cbk = {}
        for f in changed:
            cbk.setdefault(_cb(base[f][4] if len(base[f]) > 4 else -1.0), []).append(f)
        for ph in sorted(cbk):
            fs = cbk[ph]
            _rb = sum(base[f][1] for f in fs) / len(fs)
            _rc = sum(cand[f][1] for f in fs) / len(fs)
            _tot = sum(1 for f in shared if _cb(base[f][4] if len(base[f]) > 4 else -1.0) == ph)
            _pb, _pw = _bw(fs)
            print("      %-34s %7d %7.1f%%   %10.4f %10.4f %+9.4f  %6d/%-6d %5.1f%%"
                  % (ph, len(fs), 100.0 * len(fs) / max(1, _tot), _rb, _rc, _rc - _rb,
                     _pb, _pw, 100.0 * _pb / max(1, _pb + _pw)), flush=True)
    # DUMP=<path>: per-position changed rows. delta = reg(cand=KS-off) - reg(base=KS-on):
    #   delta > 0  => KS-off worse => KS-on move better => KS HELPED here
    #   delta < 0  => KS-off better => KS-on move worse => KS HURT here (the damp target)
    if os.environ.get("DUMP"):
        with open(os.environ["DUMP"], "w", newline="") as _f:
            _w = csv.writer(_f)
            _w.writerow(["fen", "ks_on_move", "ks_off_move", "reg_ks_on", "reg_ks_off", "delta"])
            for _fen in changed:
                _w.writerow([_fen, base[_fen][0], cand[_fen][0], base[_fen][1], cand[_fen][1],
                             cand[_fen][1] - base[_fen][1]])
print("\n  read: on the positions the candidate CHANGES our move, reg_cand < reg_base (negative delta) means\n"
      "  the new move scores better per SF18. Positive = the change makes our move worse.\n"
      "\n"
      "  !! THE NULL IS NOT ZERO. This statistic conditions on positions where the move CHANGED, which\n"
      "  selects positions where the base eval was MARGINAL -- they carry above-average base regret (see\n"
      "  reg_base, which is recomputed per arm on its own changed subset) and regress toward better under\n"
      "  ANY perturbation. Measured 2026-09-06: two knobs from the Elo-NEUTRAL aspiration lane read\n"
      "  ASPIRATION_DELTA=300 -> -0.1389 (34.7% changed) and ASPIRATION_DELTA=800 -> -0.0619 (33.2%),\n"
      "  a 2.2x spread at the SAME flip rate with no eval change at all. The bias is arm-specific and does\n"
      "  NOT scale with flip rate, so it cannot be divided out.\n"
      "\n"
      "  => Run a NEUTRAL arm (e.g. CAND_KNOBS='ASPIRATION_DELTA=300') in the same session and read the\n"
      "  candidate against THAT, never against zero. Two neutral points give the band a candidate must\n"
      "  clear; one gives only a number. A delta below ~0.2 is not resolvable by this tool.\n"
      "  Then confirm the sign replicates on the v2 cross-set before folding in.", flush=True)
