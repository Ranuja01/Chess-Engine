# -*- coding: utf-8 -*-
"""Build the WIDE, multi-family, SF18-anchored whole-system fit corpus (superset of build_diverse_corpus.py).
Diversity so the fit cannot buy one family by breaking another (the STS-regression blind spot):
  A) BANK families (SF18-labeled broad mine): KS tiers target/working/crowded_safe/calm + a `diverse` CATCH-ALL
     that keeps every other labeled (mostly quiet) position -- general play the fit must not wreck.
  B) STS-theme "prone-to-break" quiet-positional -> `sts_guard` tier (SF18-labeled live) = the bench proxy.
  C) PASSER family from passer_fit.csv -> `passer_*` tiers (cross-subsystem guard).

TWO-REFERENCE static-achievability gate (per user): a SF18 target is fit-worthy only if BOTH classical
non-NNUE evals -- SF11-static (`sf11_total`) AND SF15.1-static (`sf15_static`) -- agree-ish with SF18. Where a
classical witness strongly disagrees in sign, SF18 is seeing a search-only tactic -> DROP. Target magnitude is
CAPPED at the largest same-sign classical witness (+slack): we never aspire past what a classical eval reaches.
Stratified ~20% val within (tier x phase). Writes ks_sets/diverse_corpus_wide.csv.
  pyrun diagnostics/build_diverse_wide.py [STS_PER_THEME=13] [DEPTH=18] [CEIL_SLACK=0.5]
"""
import os, sys, csv
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'; os.environ.setdefault('KS_SAFE_CHECK_DEF', '5')
for _a in sys.argv[1:]:
    if '=' in _a: _k, _v = _a.split('=', 1); os.environ.setdefault(_k, _v)
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
PASSER = os.path.join(THIS, "ks_sets", "passer_fit.csv")
OUT = os.path.join(THIS, "ks_sets", "diverse_corpus_wide.csv")


def sgn(x): return (x > 0) - (x < 0)
def phase_bucket(ps):
    ps = float(ps)
    return "opening" if ps < 24 else "midgame" if ps < 64 else "endgame" if ps < 104 else "adveg"


def _fnum(v):
    try: return float(v)
    except (TypeError, ValueError): return None


def valid(sf18, witnesses):
    """Invalid if ANY classical witness strongly disagrees in SIGN with SF18 (search-only tactic)."""
    for w in witnesses:
        if w is not None and sgn(w) != sgn(sf18) and abs(w) >= 1.0 and abs(sf18) >= 1.0:
            return False
    return True


def ceiling_target(sf18, witnesses):
    """Cap |target| at the LARGEST same-sign classical witness (+slack): never aspire past classical reach."""
    same = [abs(w) for w in witnesses if w is not None and sgn(w) == sgn(sf18)]
    if not same:
        return sf18
    ceil = max(same) + CEIL_SLACK
    return sgn(sf18) * min(abs(sf18), ceil)


rows_out = []

# ---- PART A: bank families (SF18-labeled), two-reference gate, diverse catch-all ----
nbank_drop_valid = 0
for r in csv.DictReader(open(BANK)):
    if r.get("sf18", "") in ("", None):
        continue
    our_total = _fnum(r.get("our_total")); our_ks = _fnum(r.get("our_ks"))
    sf11_total = _fnum(r.get("sf11_total")); sf11_ks = _fnum(r.get("sf11_ks"))
    sf15 = _fnum(r.get("sf15_static")); sf18 = _fnum(r.get("sf18"))
    if None in (our_total, our_ks, sf11_ks, sf18):
        continue
    try: kz = max(int(r["kzone_w"]), int(r["kzone_b"]))
    except Exception: kz = 0
    if abs(sf18) >= 98:                                   # forced mate = search-bound
        continue
    witnesses = [sf11_total, sf15]
    if not valid(sf18, witnesses):
        nbank_drop_valid += 1
        continue
    tt = ceiling_target(sf18, witnesses)
    if abs(sf11_ks) >= 1.0 and abs(our_ks) < 0.3:
        tier = "target"
    elif abs(sf11_ks) >= 0.5 and abs(our_ks - sf11_ks) < 0.5:
        tier = "working"
    elif kz >= 3 and abs(sf18) < 0.75:
        tier = "crowded_safe"
    elif abs(sf11_ks) < 0.3:
        tier = "calm"
    else:
        tier = "diverse"                                  # keep every other labeled (quiet/general) row as a guard
    rows_out.append({"fen": r["fen"], "target_ks": round(sf11_ks if tier == "target" else our_ks, 3),
                     "target_total": round(tt, 3), "our_total_base": round(our_total, 3),
                     "our_ks_base": round(our_ks, 3), "tier": tier,
                     "phase_bucket": phase_bucket(r["phase_score"]), "split": "train"})
nbank = len(rows_out)

# ---- PART B: STS "prone-to-break" family (SF18 + SF11 live; single-reference OK for quiet-positional) ----
nsts = 0
if os.path.exists(STS):
    by_theme = defaultdict(list)
    for r in csv.DictReader(open(STS)):
        by_theme[r["theme"]].append(r)
    picks = []
    for theme in sorted(by_theme):
        picks.extend(by_theme[theme][:STS_PER_THEME])
    print("STS to SF18-label: %d (%d themes x %d)" % (len(picks), len(by_theme), STS_PER_THEME))
    ai = ChessAI(None, None, chess.Board(), True)
    sf11 = SF11Eval(SF11)
    sf18e = chess.engine.SimpleEngine.popen_uci(find_stockfish())
    try: sf18e.configure({"Threads": 4})
    except Exception: pass

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
        if done % 40 == 0: print("  ...%d" % done)
        if abs(sf18v) >= 98 or not valid(sf18v, [sf11_total]):
            continue
        tt = ceiling_target(sf18v, [sf11_total])
        ps = bd.get("phase_score", 64)
        rows_out.append({"fen": fen, "target_ks": round(our_ks, 3), "target_total": round(tt, 3),
                         "our_total_base": round(our_total, 3), "our_ks_base": round(our_ks, 3),
                         "tier": "sts_guard", "phase_bucket": phase_bucket(ps), "split": "train"})
        nsts += 1
    sf18e.quit()

# ---- PART C: passer family (already sf18-targeted in passer_fit.csv) ----
npass = 0
if os.path.exists(PASSER):
    for r in csv.DictReader(open(PASSER)):
        tt = _fnum(r.get("target_total"))
        if tt is None or abs(tt) >= 98:
            continue
        rows_out.append({"fen": r["fen"], "target_ks": 0.0, "target_total": round(tt, 3),
                         "our_total_base": _fnum(r.get("our_total_base")) or 0.0, "our_ks_base": 0.0,
                         "tier": r.get("tier", "passer_x"), "phase_bucket": r.get("phase_bucket", "endgame"),
                         "split": r.get("split", "train")})
        npass += 1

# ---- stratified ~20% val within (tier x phase) ----
groups = defaultdict(list)
for i, r in enumerate(rows_out):
    groups[(r["tier"], r["phase_bucket"])].append(i)
for key, idxs in groups.items():
    for j, i in enumerate(idxs):
        if j % 5 == 0:
            rows_out[i]["split"] = "val"

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["fen", "target_ks", "target_total", "our_total_base",
                                      "our_ks_base", "tier", "phase_bucket", "split"])
    w.writeheader(); w.writerows(rows_out)

print("\ndiverse WIDE corpus: %d rows (%d bank + %d sts + %d passer; %d dropped by 2-ref gate) -> %s"
      % (len(rows_out), nbank, nsts, npass, nbank_drop_valid, OUT))
print("tiers:", dict(Counter(r["tier"] for r in rows_out)))
print("phases:", dict(Counter(r["phase_bucket"] for r in rows_out)))
print("splits:", dict(Counter(r["split"] for r in rows_out)))
