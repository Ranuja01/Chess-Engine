# -*- coding: utf-8 -*-
"""Is the KS over-read concentrated in ENDGAMES (a phase-gate fix) or spread across phases (deeper over-attack)?
Runs base (ship) vs a candidate KS config, and buckets the footprint move-regret drift by the regret set's
phase_bucket (opening/midgame/endgame). A big positive delta ONLY in endgame => our phase taper under-gates
(flank/breadth fires on endgame activity) => SF/Ethereal-style phase gate likely fixes it. Positive across all
phases => the over-attack is not just a phase artifact.

  pyrun diagnostics/_ks_phase_split.py [SET=ks_sets/game_regret_set_v2.csv] [DEPTH=7] [JOBS=4]
"""
import os, sys, csv, math, subprocess
from collections import defaultdict

for _a in sys.argv[1:]:
    if '=' in _a:
        k, v = _a.split('=', 1); os.environ[k] = v


def _winpct(cp):
    return 50.0 + 50.0 * (2.0 / (1.0 + math.exp(-0.00368208 * cp)) - 1.0)


if os.environ.get("WORKER") == "1":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'; os.environ['PRESET'] = 'LONG_FORMAT'
    os.environ['MAX_DEPTH'] = os.environ.get('DEPTH', '7'); os.environ['USE_OPENING_BOOK'] = '0'
    THIS = os.path.dirname(os.path.abspath(__file__)); ENGINE = os.path.dirname(THIS)
    sys.path.insert(0, ENGINE); sys.path.insert(0, THIS); sys.path.insert(0, os.path.join(ENGINE, "selfplay"))
    import chess
    from tactical_test import run_one
    SET = os.environ["SET"]; MISS = float(os.environ.get("MISS", "30"))
    si, sn = (int(x) for x in os.environ["SLICE"].split("/"))
    rows = list(csv.DictReader(open(SET, newline="")))[si::sn]
    out = open(os.environ["OUT"], "w", newline=""); w = csv.writer(out)
    for r in rows:
        fen = r.get("fen")
        try:
            best_cp = float(r["best_cp"]); mm = {}
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
        oc = mm.get(our)
        if oc is None:
            worst = min(mm.values()) if stm_white else max(mm.values()); oc = (worst - MISS) if stm_white else (worst + MISS)
        bw = _winpct(best_cp) if stm_white else (100.0 - _winpct(best_cp))
        ow = _winpct(oc) if stm_white else (100.0 - _winpct(oc))
        # Non-pawn material (both sides) from the FEN board field — SF's KS phase variable, robust to whacky.
        board = fen.split()[0]
        _npv = {'n': 3, 'b': 3, 'r': 5, 'q': 9}
        npm = sum(_npv.get(c.lower(), 0) for c in board if c.lower() in _npv)
        # CRITICALITY (2026-08-13): how much SF's best move beats its 2nd-best, in the mover's win% — HIGH = one
        # clearly-best move (critical, getting it wrong is a real failure), LOW = several reasonable moves (benign,
        # a move-swap costs ~nothing). Lets us separate real failures from benign reshuffles (the cross-set noise).
        _wps = sorted(((_winpct(c) if stm_white else 100.0 - _winpct(c)) for c in mm.values()), reverse=True)
        crit = (_wps[0] - _wps[1]) if len(_wps) >= 2 else 0.0
        qc = sum(1 for c in board if c in "Qq")   # queens on the board (both sides) — coarse count
        qcls = (1 if "Q" in board else 0) + (1 if "q" in board else 0)   # per-SIDE queen PRESENCE (0/1/2) = the KS_NO_QUEEN gate semantics (!(queens & enemy) is per-king)
        w.writerow([fen, our, "%.4f" % max(0.0, bw - ow), r.get("phase_bucket", "?"), npm, "%.4f" % crit, r.get("best_uci", "?"), qc, qcls])
    out.close(); sys.exit(0)

THIS = os.path.dirname(os.path.abspath(__file__)); ENGINE = os.path.dirname(THIS); PY = sys.executable
SET = os.environ.get("SET", "ks_sets/game_regret_set_v2.csv")
if not os.path.isabs(SET): SET = os.path.join(THIS, SET)
DEPTH = os.environ.get("DEPTH", "7"); JOBS = int(os.environ.get("JOBS", "4"))
BUNDLE = {"OVD_BOUNDED_MODE": 2, "OVD_CAP": 300, "OVD_KNEE": 40, "CENTRAL_BOUNDED_MODE": 1, "CENTRAL_CAP": 150, "CENTRAL_KNEE": 200}


def cfg(**kw):
    d = dict(BUNDLE); d.update(kw); return d


# Stage-1 coordination-gate ISOLATION (2026-08-12): base = bundle with defaware OFF + flat attacker term;
# cand = same + count*weight PRODUCT attacker term at a swept divisor. Isolates "flat sum -> product" ONLY.
# COORD_DIV drives the sweep (2/3/4/6, and a large value = near-inert control arm). COORD_DIV=0 => the old
# detector-stack experiment (legacy). Set COORD_DIV to run the coordination-gate isolation.
_cdiv = int(os.environ.get("COORD_DIV", "0"))
_accum = int(os.environ.get("ACCUM", "0"))
_proxctl = int(os.environ.get("PROXCTL", "0"))
_ksmag = int(os.environ.get("KSMAG_TEST", "0"))
_nqtax = int(os.environ.get("NQ_TAX", "0"))
_phasez = int(os.environ.get("PHASEZ", "0"))
_detonly = int(os.environ.get("DETONLY", "0"))
_egext = int(os.environ.get("EGEXT", "0"))
_egmat = int(os.environ.get("EGMAT", "0"))
_capgs = int(os.environ.get("CAPG_SCALE", "0"))
if _detonly:
    # Detectors ALONE (no curve changes), criticality lens: do the discrimination-validated detectors help the
    # CRITICAL bands (where the compound+curve HURT)? Isolates detectors from the known-bad KS_FLOOR/KNEE/DIVISOR curve.
    BASE = cfg(KS_DEFAWARE_MODE=1)
    CAND = cfg(KS_DEFAWARE_MODE=1, KS_SQC_MODE=1, KS_PIN_MODE=1, KS_WEAK_VAL_MODE=1, KS_FLANK_MODE=2)
elif _egext:
    # Deep-endgame KS EXTENSION test (2026-08-13, fable-verified): base = defaware bundle (KS cliffs to 0 at ps=65);
    # cand = same + KS_EXTEND_EG=1 (KS runs in the endgame branch, smooth taper to ps=104). First check: does it
    # CHANGE MOVES at all (prove-it-executes)? Then read where — expect the effect in the labeler-"endgame" bucket
    # and low-material bands (the ps 65-104 positions that previously had zero king-danger model).
    BASE = cfg(KS_DEFAWARE_MODE=1)
    CAND = cfg(KS_DEFAWARE_MODE=1, KS_EXTEND_EG=1)
elif _egmat:
    # Deep-endgame material taper test (2026-08-13): cross-set-validated finding = KS hurts at low non-pawn material
    # (<=12). Scale KS down below KS_EG_MAT_HI toward KS_EG_MAT_FLOOR%. base = defaware bundle; cand = +the taper.
    # Should improve the low-material bands on BOTH sets w/o touching high material (npm >= HI = full KS, untouched).
    # EG_PZ (optional): raise KS_PHASE_ZERO in BOTH arms so KS fires through the deep endgame (disable the upstream
    # phase cliff) — lets the material taper actually have KS to scale, and tests the cliff-vs-taper hypothesis.
    _egpz = int(os.environ.get("EG_PZ", "0"))
    _bcfg = dict(KS_DEFAWARE_MODE=1)
    _ccfg = dict(KS_DEFAWARE_MODE=1, KS_EG_MAT_GATE=1,
               KS_EG_MAT_LO=int(os.environ.get("EG_LO", "12")),
               KS_EG_MAT_HI=int(os.environ.get("EG_HI", "20")),
               KS_EG_MAT_FLOOR=int(os.environ.get("EG_FLOOR", "25")))
    if _egpz:
        _bcfg["KS_PHASE_ZERO"] = _egpz; _ccfg["KS_PHASE_ZERO"] = _egpz
    BASE = cfg(**_bcfg)
    CAND = cfg(**_ccfg)
elif _phasez:
    # Phase-taper test (2026-08-13): the endgame is where KS is weakest (neutral-to-harmful). Gate KS down EARLIER
    # via KS_PHASE_ZERO (default 104 = zero only in deep EG). Lowering it zeros KS for more endgame positions —
    # PHASE-localized, so opening/midgame (below the taper) are untouched: no material collateral (vs no-queen).
    BASE = cfg(KS_DEFAWARE_MODE=1)
    CAND = cfg(KS_DEFAWARE_MODE=1, KS_PHASE_ZERO=_phasez)
elif _nqtax:
    # No-queen tax test (2026-08-13): the incumbent KS is net-positive in opening/midgame but weak (neutral-to-
    # harmful) in the endgame — and queenless positions are disproportionately endgames. Crank the no-queen
    # suppressor (KS_NO_QUEEN, default 6) so KS speaks LESS when there's no enemy queen (SF's -873 mechanism,
    # material-based ⇒ robust to whacky). Subtractive. Does it de-harm the endgame WITHOUT costing opening/midgame?
    BASE = cfg(KS_DEFAWARE_MODE=1)
    CAND = cfg(KS_DEFAWARE_MODE=1, KS_NO_QUEEN=_nqtax)
elif _ksmag:
    # RULER TEST (2026-08-13): can D7 move-regret even SEE KS? base = KS fully OFF (KING_SAFETY_MAG=0),
    # cand = KS ON (default). The 'changed' count = fraction of positions where KS is DECISIVE for the d7 move;
    # reg_base(KS off) vs reg_cand(KS on) on those = whether KS actually helps. Tiny changed fraction => the
    # instrument is largely blind to KS => the 13 KS nulls are uninformative, games are the only arbiter.
    BASE = cfg(KS_DEFAWARE_MODE=1, KING_SAFETY_MAG=0)
    CAND = cfg(KS_DEFAWARE_MODE=1)
elif int(os.environ.get("NQ_SUPP", "0")):
    # No-queen suppressor sweep (2026-08-14, CLEAN harness): base = shipped default (KS_NO_QUEEN=6); cand = + a bigger
    # KS_NO_QUEEN. Fable-derived: floor 13 > knee 12 so firing = linear seg; KS_NO_QUEEN subtracts BEFORE the floor, so
    # it raises the fire bar to 13+KS_NO_QUEEN (default 6 = bar 19 = haircut). Try 20 (bar 33) / 35 (SF-faithful zero).
    # Read on the queen x material split (QSPLIT=1): expect Qless mid-high over-read (+0.25..0.37) -> <=0 (delta<0 = cand
    # better), Qon UNTOUCHED (the gate is !(queens & enemy)). delta<0 = MORE suppression better.
    BASE = cfg(KS_DEFAWARE_MODE=1)
    CAND = cfg(KS_DEFAWARE_MODE=1, KS_NO_QUEEN=int(os.environ["NQ_SUPP"]))
elif _proxctl:
    # CONTROL (2026-08-13): proximity demotion ALONE (KS_ATTACK_COUNT=A_ATTCOUNT, default 0), NO accum machinery,
    # vs the defaware bundle. Isolates whether the accum's opening win is just proximity removal or the
    # suppressor+threshold "when" system earning its keep. If this ~= the ACCUM result, the machinery adds nothing yet.
    BASE = cfg(KS_DEFAWARE_MODE=1)
    CAND = cfg(KS_DEFAWARE_MODE=1, KS_ATTACK_COUNT=int(os.environ.get("A_ATTCOUNT", "0")))
elif _accum:
    # Step-1 signed-accumulator test (2026-08-13): base = the DEPLOYMENT defaware bundle; cand = same + the
    # rebalanced accum object (proximity DEMOTED via A_ATTCOUNT, no-queen suppressor A_NQ, threshold A_THRESH,
    # linear map A_LIN). The four king-credit channels stay FROZEN (only these KS knobs move) per the channel law.
    # Deployment-relevant comparison (vs defaware bundle, NOT flat). Magnitudes are trace-derived STARTING points.
    BASE = cfg(KS_DEFAWARE_MODE=1)
    _cand = dict(KS_DEFAWARE_MODE=1, KS_ACCUM_MODE=1,
               KS_ATTACK_COUNT=int(os.environ.get("A_ATTCOUNT", "0")),
               KS_NQ_SUP=int(os.environ.get("A_NQ", "10")),
               KS_ACCUM_THRESH=int(os.environ.get("A_THRESH", "10")),
               KS_ACCUM_LIN=int(os.environ.get("A_LIN", "96")))
    # Step-2 positive-side rebalance: ELEVATE the discriminating signal via DETECTORS (not magnitude cranks, which
    # the failure map shows invert across sets). A_WEAKVAL = value-coupled weak (redistributive); A_SQC = value-aware
    # contest into defaware; A_WEAK = weak weight (held unless testing). Off by default = pure step-1 accum.
    if int(os.environ.get("A_WEAKVAL", "0")): _cand["KS_WEAK_VAL_MODE"] = 1
    if int(os.environ.get("A_SQC", "0")):     _cand["KS_SQC_MODE"] = 1
    if os.environ.get("A_WEAK"):              _cand["KS_WEAK"] = int(os.environ["A_WEAK"])
    CAND = cfg(**_cand)
elif _cdiv > 0:
    # NOTE: cfg() does not set KS_DEFAWARE_MODE, so both arms use the ENGINE DEFAULT (=1, the shipped bundle);
    # defaware is ON here, not off (the older "defaware OFF" comment was stale). Empirically confirmed: forcing
    # KS_DEFAWARE_MODE=1 in both arms is byte-identical to this.
    BASE = cfg()                                              # shipped default (defaware on) + flat attacker term
    CAND = cfg(KS_COORD_GATE_MODE=1, KS_COORD_DIVISOR=_cdiv)  # + product attacker term at this divisor
elif _capgs:
    # Redirect test (2026-08-15): pvb flags capture_gains as the top culprit dragging us onto materialistic
    # quiet moves over SF's sacrifices. Does DAMPING it (SCALE_CAPTURE_GAINS<100) reduce SEARCHED regret, or does
    # the D7 search already fix the static over-read (=> search-bound, not an eval win)? base=default vs cand=damped.
    BASE = cfg(KS_DEFAWARE_MODE=1)
    CAND = cfg(KS_DEFAWARE_MODE=1, SCALE_CAPTURE_GAINS=_capgs)
else:
    BASE = cfg(KS_DEFAWARE_MODE=1)
    CAND = cfg(KS_DEFAWARE_MODE=1, KS_SQC_MODE=1, KS_PIN_MODE=1, KS_WEAK_VAL_MODE=1, KS_FLANK_MODE=2, KS_FLOOR=15, KS_KNEE=40, KS_DIVISOR=8)


def collect(config, tag):
    env = dict(os.environ, SET=SET, DEPTH=DEPTH)
    ka = ["%s=%s" % (k, v) for k, v in config.items()]
    procs = []
    for i in range(JOBS):
        outp = "/tmp/_ps_%d_%s_%d.csv" % (os.getpid(), tag, i)   # PID-unique: concurrent runs must not share /tmp files
        procs.append((subprocess.Popen([PY, "-u", os.path.abspath(__file__)] + ka,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True, cwd=ENGINE,
                     env=dict(env, WORKER="1", SLICE="%d/%d" % (i, JOBS), OUT=outp)), outp))
    d = {}
    for p, outp in procs:
        p.communicate()
        for row in csv.reader(open(outp, newline="")):
            if len(row) >= 4:
                npm = int(row[4]) if len(row) >= 5 else -1
                crit = float(row[5]) if len(row) >= 6 else -1.0
                best = row[6] if len(row) >= 7 else "?"
                qc = int(row[7]) if len(row) >= 8 else 0
                qcls = int(row[8]) if len(row) >= 9 else (2 if qc else 0)
                d[row[0]] = (row[1], float(row[2]), row[3], npm, crit, best, qc, qcls)
    return d


base = collect(BASE, "base"); cand = collect(CAND, "cand")
shared = [f for f in base if f in cand]
changed = [f for f in shared if base[f][0] != cand[f][0]]
byph = defaultdict(lambda: [0, 0.0, 0.0])   # phase -> [n, sum_base, sum_cand]
allph = defaultdict(int)
for f in shared:
    allph[base[f][2]] += 1
for f in changed:
    ph = base[f][2]; byph[ph][0] += 1; byph[ph][1] += base[f][1]; byph[ph][2] += cand[f][1]

print("PHASE SPLIT of the KS over-read  (base=ship vs compound OURS)  set=%s\n" % os.path.basename(SET))
print("  %-10s %8s %8s   %9s %9s %9s" % ("phase", "positions", "changed", "reg_base", "reg_cand", "delta"))
for ph in ("opening", "midgame", "endgame"):
    n, sb, sc = byph[ph]
    if n:
        print("  %-10s %8d %8d   %9.4f %9.4f %+9.4f" % (ph, allph[ph], n, sb/n, sc/n, (sc-sb)/n))
tot_n = sum(byph[p][0] for p in byph); tot_b = sum(byph[p][1] for p in byph); tot_c = sum(byph[p][2] for p in byph)
print("  %-10s %8d %8d   %9.4f %9.4f %+9.4f" % ("ALL", len(shared), tot_n, tot_b/max(1,tot_n), tot_c/max(1,tot_n), (tot_c-tot_b)/max(1,tot_n)))
print("\n  read: if the positive delta is CONCENTRATED in endgame, it's a phase-gate problem (our taper under-gates\n"
      "  R/Q endgames that SF/Ethereal phase out). If it's positive across ALL phases, the over-attack is deeper.", flush=True)

# NON-PAWN MATERIAL map (2026-08-13): KS move-value as a continuous function of non-pawn material (SF's blend
# variable). Buckets over CHANGED positions; delta<0 = cand better in that material band. This is the substrate
# for a SMOOTH phase/material blend (see where KS helps vs hurts across the material spectrum), instead of guessing.
if base and next(iter(base.values()))[3] >= 0:
    MB = [(0, 6, "0-6  bare/pawn EG"), (7, 12, "7-12 minor EG"), (13, 19, "13-19 R/RR EG"),
          (20, 27, "20-27 late mid"), (28, 40, "28-40 midgame"), (41, 99, "41+  opening")]
    bym = {lo: [0, 0.0, 0.0] for lo, hi, lab in MB}
    for f in changed:
        npm = base[f][3]
        for lo, hi, lab in MB:
            if lo <= npm <= hi:
                bym[lo][0] += 1; bym[lo][1] += base[f][1]; bym[lo][2] += cand[f][1]; break
    print("\n  NON-PAWN MATERIAL map (KS move-value by material band; delta<0 = cand better):")
    print("  %-20s %8s   %9s %9s %9s" % ("material band", "changed", "reg_base", "reg_cand", "delta"))
    for lo, hi, lab in MB:
        n, sb, sc = bym[lo]
        if n:
            print("  %-20s %8d   %9.4f %9.4f %+9.4f" % (lab, n, sb/n, sc/n, (sc-sb)/n))
    print(flush=True)

# CRITICALITY split (2026-08-13, owner insight): bucket the changed positions by how critical they are — how much
# SF's best move beats its 2nd-best (mover's win%). BENIGN = several reasonable moves (a swap costs ~0, pure noise);
# CRITICAL = one clearly-best move (getting it wrong is a REAL failure). If our regressions concentrate in BENIGN,
# they're the cross-set noise; if in CRITICAL, they're real. Improving the CRITICAL column is the actual goal.
if base and next(iter(base.values()))[4] >= 0:
    CB = [(0.0, 3.0, "benign   <3%"), (3.0, 8.0, "minor  3-8%"), (8.0, 20.0, "moderate 8-20%"), (20.0, 1e9, "CRITICAL >20%")]
    byc = {lo: [0, 0.0, 0.0] for lo, hi, lab in CB}
    for f in changed:
        cr = base[f][4]
        for lo, hi, lab in CB:
            if lo <= cr < hi:
                byc[lo][0] += 1; byc[lo][1] += base[f][1]; byc[lo][2] += cand[f][1]; break
    print("  CRITICALITY split (SF best-vs-2nd win%% gap; delta<0 = cand better where it MATTERS):")
    print("  %-16s %8s   %9s %9s %9s" % ("criticality", "changed", "reg_base", "reg_cand", "delta"))
    for lo, hi, lab in CB:
        n, sb, sc = byc[lo]
        if n:
            print("  %-16s %8d   %9.4f %9.4f %+9.4f" % (lab, n, sb/n, sc/n, (sc-sb)/n))
    print(flush=True)

# QUEEN × MATERIAL split (2026-08-14): disentangle "endgame problem" from "no-queens problem". If the KS hurt (delta>0)
# is confined to LOW material regardless of queens => it's a material/endgame problem (a smooth material taper fixes it and
# AUTO-handles promotion/multi-queen, since those just change material). If QUEENLESS positions hurt even at HIGH material,
# OR queen-present LOW-material positions are FINE => it's a no-queens problem (needs a live queen-keyed suppressor). Gated QSPLIT.
if int(os.environ.get("QSPLIT", "0")) and base and len(next(iter(base.values()))) >= 8:
    QMB = [(0, 12, "0-12 low"), (13, 27, "13-27 mid-low"), (28, 99, "28+  high")]
    # Per-SIDE queen presence (qcls): 0 = neither side has a queen (KS_NO_QUEEN fires for BOTH kings),
    # 1 = one side has a queen (fires for the queenless side's enemy king), 2 = both have queens (suppressor INERT
    # = the clean control that a KS_NO_QUEEN sweep must leave ~0). cell[(matlabel, qcls)] = [n, sum_base, sum_cand].
    cellq = {(lab, q): [0, 0.0, 0.0] for _, _, lab in QMB for q in (0, 1, 2)}
    for f in changed:
        npm = base[f][3]; q = base[f][7]
        for lo, hi, lab in QMB:
            if lo <= npm <= hi:
                cellq[(lab, q)][0] += 1; cellq[(lab, q)][1] += base[f][1]; cellq[(lab, q)][2] += cand[f][1]; break
    print("  QUEEN(per-side presence) x MATERIAL split (delta<0 = cand better; 2Q = suppressor-INERT control):")
    print("  %-14s %6s %8s   %6s %8s   %6s %8s" % ("material band", "0Q n", "0Q d", "1Q n", "1Q d", "2Q n", "2Q d"))
    for _, _, lab in QMB:
        vals = []
        for q in (0, 1, 2):
            n, sb, sc = cellq[(lab, q)]
            vals += [n, (sc - sb) / n if n else 0.0]
        print("  %-14s %6d %+8.4f   %6d %+8.4f   %6d %+8.4f" % (lab, vals[0], vals[1], vals[2], vals[3], vals[4], vals[5]))
    print("  read: a clean KS_NO_QUEEN sweep should HELP 0Q/1Q (delta<0) and leave 2Q ~0 (its gate is per-side !enemy-queen).", flush=True)

# WORST-HURT DUMP (2026-08-14): list the individual positions where cand HURTS most inside a target material band,
# so the pattern is eyeball-able (what KS over-reads). Gated by DUMP_NPM (max non-pawn material to include; 0 = off).
# Columns: delta (cand-base regret; +ve = KS-on WORSE), npm, crit, SFbest, off_mv (base move), on_mv (cand move),
# on_reg (cand regret in win%). off_mv==SFbest with on_mv different is the signature of KS pulling us off the best move.
_dump_npm = int(os.environ.get("DUMP_NPM", "0"))
if _dump_npm > 0 and base and next(iter(base.values()))[3] >= 0:
    _dn = int(os.environ.get("DUMP_N", "30"))
    _dmin = int(os.environ.get("DUMP_NPM_MIN", "0"))
    _dqmax = int(os.environ.get("DUMP_QMAX", "9"))   # max per-side queen presence (0 = queenless only, 1 = <=one side has a queen)
    hurt = []
    for f in changed:
        if _dmin <= base[f][3] <= _dump_npm and base[f][7] <= _dqmax:
            hurt.append((cand[f][1] - base[f][1], base[f][3], base[f][7], base[f][4], base[f][5], base[f][0], cand[f][0], cand[f][1], f))
    hurt.sort(reverse=True)   # biggest positive delta first = most hurt by the cand config
    print("  WORST-HURT dump (npm %d..%d, per-side-queens<=%d, most-hurt first; delta=cand-base regret win%%):" % (_dmin, _dump_npm, _dqmax))
    print("  %8s %4s %2s %6s  %-6s %-6s %-6s %8s  %s" % ("delta", "npm", "Q", "crit", "SFbest", "off_mv", "on_mv", "on_reg", "fen"))
    for delta, npm, q, crit, sfbest, offm, onm, onr, f in hurt[:_dn]:
        print("  %+8.3f %4d %2d %6.1f  %-6s %-6s %-6s %8.3f  %s" % (delta, npm, q, crit, sfbest, offm, onm, onr, f))
    print("  (n=%d in band; top %d)" % (len(hurt), min(_dn, len(hurt))), flush=True)
