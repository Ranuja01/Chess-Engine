# -*- coding: utf-8 -*-
"""Build a DIVERSE, SF18-anchored joint KS+OvD fit corpus in the fit_corpus.csv schema, fusing situation
families so the fit cannot buy one by breaking another (the blind spot that hid this session's STS regression):
  - BANK families (self-play, already SF18-labeled): target (KS under-read) / working / crowded_safe / calm.
  - STS-15-theme "prone-to-break" quiet-positional positions as a single `sts_guard` tier (SF18-labeled here).

Target = SF18-search truth in win%-space, but CAPPED at the static-achievable ceiling (SF11-static): we aspire
toward SF18 yet do not punish our static eval where even SF11-static cannot reach SF18 (search-only magnitude).
Validity: drop rows where SF11-static and SF18 disagree in sign AND both are large (lost cause / search-only).
Stratified train/val split within (tier x phase) so no family is silently untested.

  pyrun diagnostics/build_diverse_corpus.py [STS_PER_THEME=13] [DEPTH=18] [CEIL_SLACK=0.5]
Writes ks_sets/diverse_corpus.csv. Reuses: position_bank.csv (SF18 col), results/sts_results_ksoff.csv (STS
fens+themes), eval_vs_sf11.SF11Eval (SF11-static), selfplay/arbiter.find_stockfish (SF18).
"""
import os, sys, csv
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'; os.environ.setdefault('KS_SAFE_CHECK_DEF', '5')
for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1); os.environ.setdefault(_k, _v)
STS_PER_THEME = int(os.environ.get('STS_PER_THEME', '13'))
DEPTH = int(os.environ.get('DEPTH', '18'))
CEIL_SLACK = float(os.environ.get('CEIL_SLACK', '0.5'))

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
sys.path.insert(0, os.path.join(os.path.dirname(THIS), "selfplay"))
import chess, chess.engine
from collections import defaultdict, Counter
from eval_vs_sf11 import SF11Eval, SF11
from arbiter import find_stockfish
from ChessAI import ChessAI

BANK = os.path.join(THIS, "ks_sets", "position_bank.csv")
STS = os.path.join(THIS, "results", "sts_results_ksoff.csv")
OUT = os.path.join(THIS, "ks_sets", "diverse_corpus.csv")


def sgn(x):
    return (x > 0) - (x < 0)


def phase_bucket(ps):
    ps = float(ps)
    return "opening" if ps < 24 else "midgame" if ps < 64 else "endgame" if ps < 104 else "adveg"


def ceiling_target(sf18, sf11_total):
    """Aspire to SF18 but cap magnitude at the static-achievable SF11-static ceiling (+slack), same sign."""
    if sf11_total is None:
        return sf18
    if sgn(sf18) == sgn(sf11_total) and abs(sf18) > abs(sf11_total) + CEIL_SLACK:
        return sgn(sf18) * (abs(sf11_total) + CEIL_SLACK)
    return sf18


def valid(sf11_total, sf18):
    # SF11-static completely off from SF18-search -> exclude (search-only / lost cause)
    return not (sf11_total is not None and sgn(sf11_total) != sgn(sf18)
                and abs(sf11_total) >= 1.0 and abs(sf18) >= 1.0)


rows_out = []

# ---- PART A: bank families (already SF18-labeled) ----
for r in csv.DictReader(open(BANK)):
    if r.get("sf18", "") in ("", None):
        continue
    try:
        our_total = float(r["our_total"]); our_ks = float(r["our_ks"])
        sf11_total = float(r["sf11_total"]); sf11_ks = float(r["sf11_ks"]); sf18 = float(r["sf18"])
        kz = max(int(r["kzone_w"]), int(r["kzone_b"]))
    except Exception:
        continue
    if abs(sf18) >= 98:                      # forced mate = search-bound, exclude from static fit
        continue
    if not valid(sf11_total, sf18):
        continue
    tt = ceiling_target(sf18, sf11_total)
    if abs(sf11_ks) >= 1.0 and abs(our_ks) < 0.3:
        tier = "target"
    elif abs(sf11_ks) >= 0.5 and abs(our_ks - sf11_ks) < 0.5:
        tier = "working"
    elif kz >= 3 and abs(sf18) < 0.75:
        tier = "crowded_safe"
    elif abs(sf11_ks) < 0.3:
        tier = "calm"
    else:
        continue                              # non-golden, non-control -> drop
    rows_out.append({"fen": r["fen"], "target_ks": round(sf11_ks if tier == "target" else our_ks, 3),
                     "target_total": round(tt, 3), "our_total_base": round(our_total, 3),
                     "our_ks_base": round(our_ks, 3), "tier": tier,
                     "phase_bucket": phase_bucket(r["phase_score"]), "split": "train"})

nbank = len(rows_out)

# ---- PART B: STS "prone-to-break" family (SF18-label here) ----
sts_rows = list(csv.DictReader(open(STS)))
by_theme = defaultdict(list)
for r in sts_rows:
    by_theme[r["theme"]].append(r)
picks = []
for theme in sorted(by_theme):
    picks.extend(by_theme[theme][:STS_PER_THEME])          # first N/theme (deterministic)

print("STS to SF18-label: %d (%d themes x %d)" % (len(picks), len(by_theme), STS_PER_THEME))
ai = ChessAI(None, None, chess.Board(), True)
sf11 = SF11Eval(SF11)
sf18e = chess.engine.SimpleEngine.popen_uci(find_stockfish())


def s18(fen):
    b = chess.Board(fen); i = sf18e.analyse(b, chess.engine.Limit(depth=DEPTH)); s = i["score"].white()
    return 99.0 if (s.is_mate() and s.mate() > 0) else (-99.0 if s.is_mate() else s.score() / 100.0)


done = 0
for r in picks:
    fen = r["fen"]
    try:
        bd = ai.ev_breakdown(chess.Board(fen))
        if bd.get("checkmate"):
            continue
        our_total = -bd.get("total", 0.0) / 1000.0
        our_ks = -bd.get("king_safety", 0.0) / 1000.0
        sf11_total, _ = sf11.eval(fen)
        sf18v = s18(fen)
    except Exception:
        continue
    done += 1
    if done % 40 == 0:
        print("  ...%d" % done)
    if abs(sf18v) >= 98 or not valid(sf11_total, sf18v):
        continue
    tt = ceiling_target(sf18v, sf11_total)
    ps = bd.get("phase_score", 64)
    rows_out.append({"fen": fen, "target_ks": round(our_ks, 3), "target_total": round(tt, 3),
                     "our_total_base": round(our_total, 3), "our_ks_base": round(our_ks, 3),
                     "tier": "sts_guard", "phase_bucket": phase_bucket(ps), "split": "train"})
sf18e.quit()

# ---- stratified split: ~20% val within (tier x phase) ----
groups = defaultdict(list)
for i, r in enumerate(rows_out):
    groups[(r["tier"], r["phase_bucket"])].append(i)
for key, idxs in groups.items():
    for j, i in enumerate(idxs):
        if j % 5 == 0:                        # deterministic every-5th -> val, but stratified per group
            rows_out[i]["split"] = "val"

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["fen", "target_ks", "target_total", "our_total_base",
                                      "our_ks_base", "tier", "phase_bucket", "split"])
    w.writeheader(); w.writerows(rows_out)

print("\ndiverse corpus: %d rows (%d bank + %d sts) -> %s" % (len(rows_out), nbank, len(rows_out) - nbank, OUT))
print("tiers:", dict(Counter(r["tier"] for r in rows_out)))
print("phases:", dict(Counter(r["phase_bucket"] for r in rows_out)))
print("splits:", dict(Counter(r["split"] for r in rows_out)))
