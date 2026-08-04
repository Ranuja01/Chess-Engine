# -*- coding: utf-8 -*-
"""Break down WHAT eval terms actually drive a collapse CLASS (esp. the residual 'positional' bucket, which is
defined by EXCLUSION and has repeatedly turned out to be mislabeled KS / passed-pawn). Attributes each collapse's
over-read to the OUR term(s) that most EXCEED the classical (non-NNUE) references SF11 + SF15.1 -- turning the
residual label into a ranked term list so we target the true top contributors.

For each collapse decision_fen: our clean-partition ev_breakdown (WHITE-POV pawns; fields sum to total) vs SF11
and SF15 classical per-term tables. Per OUR term, excess = our_term - avg(SF11,SF15 counterpart) (ours-specific
terms like capture_gains/latent_threat have no SF counterpart -> full value is 'excess'). Ranks terms by mean
excess in the over-read direction + counts how often each is the TOP contributor. Optional CONTROL set (bank
quiet rows) shows which terms are ANOMALOUSLY hot in collapses vs normal play.
  pyrun diagnostics/collapse_term_attribution.py [CLASS=positional] [N=250] [CONTROL=1]
"""
import os, sys, csv, re, subprocess
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'; os.environ.setdefault('KS_SAFE_CHECK_DEF', '5')
for a in sys.argv[1:]:
    if '=' in a: k, v = a.split('=', 1); os.environ.setdefault(k, v)
CLASS = os.environ.get('CLASS', 'positional'); N = int(os.environ.get('N', '250'))
CONTROL = int(os.environ.get('CONTROL', '1'))
THIS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
import chess
from collections import defaultdict
from eval_vs_sf11 import SF11Eval, SF11
from ChessAI import ChessAI
SF15 = os.environ.get('SF15_BIN', "/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/stockfish_15_linux/stockfish_15.1_linux_x64/stockfish-ubuntu-20.04-x86-64")


class SF15Eval:
    """SF15.1 classical (NNUE off) -> (total_white_pov_pawns, {term: total_mg}). Same table shape as SF11."""
    def __init__(self, path):
        self.p = subprocess.Popen([path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self._send('uci'); self._drain('uciok')
        self._send('setoption name Use NNUE value false'); self._send('isready'); self._drain('readyok')
    def _send(self, s): self.p.stdin.write(s + '\n'); self.p.stdin.flush()
    def _drain(self, tok):
        while True:
            ln = self.p.stdout.readline()
            if not ln or ln.strip().startswith(tok): break
    def eval(self, fen):
        self._send('position fen %s' % fen); self._send('eval'); self._send('isready')
        total, terms = None, {}
        while True:
            ln = self.p.stdout.readline()
            if not ln or ln.startswith('readyok'): break
            m = re.search(r'Classical evaluation\s+([-+]?\d+\.\d+)', ln)
            if m: total = float(m.group(1))
            mm = re.match(r'\|\s*([A-Za-z ]+?)\s*\|[^|]*\|[^|]*\|\s*([-+]?\d+\.\d+|----)\s+([-+]?\d+\.\d+|----)', ln)
            if mm:
                try: terms[mm.group(1).strip()] = float(mm.group(2))
                except ValueError: pass
        return total, terms
    def close(self):
        try: self._send('quit'); self.p.wait(timeout=2)
        except Exception:
            try: self.p.kill()
            except Exception: pass


# OUR ev_breakdown CLEAN-PARTITION terms only (these sum to total; pt_*/det_*/flags are sub-components -> excluded
# to avoid double-count). Value = SF classical label(s); empty list = OURS-SPECIFIC (no SF counterpart -> the full
# value is 'excess over classical', which is exactly what flags an over-read SF doesn't share -- e.g. OvD, capg).
# ⚠️ `material` is DELIBERATELY ABSENT. It is a DIAGNOSTIC (blackPieceVal - whitePieceVal), not part of the
# additive sum -- verified by breakdown_partition_check.py (residual 0 without it). Including it double-counted
# material, since `pieces` is the sum of the pt_* sub-views and those bundle each piece's VALUE with its
# placement. For the same reason `pieces` must map to SF's Material AND Pawns AND the piece terms: SF reports
# material separately from activity, we do not. Mapping `pieces` to the activity terms alone made our piece
# aggregate look 3-5 pawns "over-read" in every position where we were simply ahead on material.
OUR2SF = {
    "kaufman_imbalance": ["Imbalance"], "pair_bonus": ["Imbalance"],
    "imbalance_white": [], "imbalance_black": [],          # OvD offense/defense -- ours-specific (SF has no OvD)
    "pieces": ["Material", "Pawns", "Knights", "Bishops", "Rooks", "Queens"], "king_safety": ["King safety"],
    "passed_pawn_support": ["Passed"], "latent_threat": [], "threats": ["Threats"],
    "central": ["Space"], "capture_gains": [], "piece_value_boost": [],
}
PARTITION = list(OUR2SF.keys())                            # only attribute the clean-partition terms


def our_terms(bd):
    """ev_breakdown clean-partition -> {term: WHITE-POV pawns}. Black-positive milli-pawns -> -v/1000."""
    return {k: -float(bd[k]) / 1000.0 for k in PARTITION
            if k in bd and isinstance(bd[k], (int, float))}


def sf_counterpart(term, sfterms):
    labels = OUR2SF.get(term, None)
    if not labels:
        return None                                   # ours-specific
    return sum(sfterms.get(l, 0.0) for l in labels)


ai = ChessAI(None, None, chess.Board(), True)
sf11 = SF11Eval(SF11); sf15 = SF15Eval(SF15)


def profile(fens, label):
    our_sum = defaultdict(float); exc_sum = defaultdict(float); topcnt = defaultdict(int); n = 0
    for fen in fens:
        try:
            bd = ai.ev_breakdown(chess.Board(fen))
            if bd.get("checkmate"): continue
            ot = our_terms(bd)
            _, s11 = sf11.eval(fen); _, s15 = sf15.eval(fen)
        except Exception:
            continue
        n += 1
        best_t, best_e = None, 0.0
        for t, ov in ot.items():
            our_sum[t] += ov
            cp = sf_counterpart(t, {k: (s11.get(k, 0.0) + s15.get(k, 0.0)) / 2.0
                                    for k in set(s11) | set(s15)})
            exc = ov if cp is None else (ov - cp)      # excess over classical (ours-specific = full value)
            exc_sum[t] += exc
            if abs(exc) > abs(best_e): best_t, best_e = t, exc
        if best_t: topcnt[best_t] += 1
    print("\n=== %s  (n=%d) ===" % (label, n))
    print("  %-22s %8s %8s %8s" % ("term", "mean_our", "mean_exc", "top%"))
    for t in sorted(exc_sum, key=lambda x: -abs(exc_sum[x])):
        print("  %-22s %+8.2f %+8.2f %7.0f%%" % (t, our_sum[t] / n, exc_sum[t] / n, 100.0 * topcnt[t] / n))
    return n


def per_fen(path):
    """PER-POSITION mode (FENS=<file> with 'label<TAB>fen'): print our clean-partition terms beside BOTH
    classical references, so a single position can be reasoned about term by term. The aggregate mode says
    WHICH term is hot across a class; this says WHY for one board. Ours-specific terms (capture_gains,
    piece_value_boost, latent_threat, OvD imbalance) are marked '--' on the SF side -- they have no
    counterpart, so their whole value is excess over classical."""
    items = []
    for ln in open(path):
        ln = ln.rstrip("\n")
        if not ln.strip() or ln.lstrip().startswith("#"):
            continue
        lbl, fen = (ln.split("\t", 1) if "\t" in ln else ("", ln))
        items.append((lbl.strip(), fen.strip()))
    for lbl, fen in items:
        try:
            bd = ai.ev_breakdown(chess.Board(fen))
            ot = our_terms(bd)
            t11, s11 = sf11.eval(fen)
            t15, s15 = sf15.eval(fen)
        except Exception as exc:
            print("%s  ERROR %s" % (lbl, exc))
            continue
        print("=" * 104)
        print("%s   %s" % (lbl, fen))
        print("  totals: OURS %+6.2f | SF11 %s | SF15c %s"
              % (-bd.get("total", 0) / 1000.0,
                 "n/a" if t11 is None else "%+6.2f" % t11,
                 "n/a" if t15 is None else "%+6.2f" % t15))
        print("  %-22s %9s %9s %9s %9s   %s"
              % ("our term", "OURS", "SF11", "SF15c", "excess", "SF label(s)"))
        rows = []
        for t, ov in ot.items():
            labels = OUR2SF.get(t, None)
            c11 = None if not labels else sum(s11.get(l, 0.0) for l in labels)
            c15 = None if not labels else sum(s15.get(l, 0.0) for l in labels)
            avg = None if (c11 is None and c15 is None) else \
                ((c11 or 0.0) + (c15 or 0.0)) / (int(c11 is not None) + int(c15 is not None))
            exc = ov if avg is None else ov - avg
            rows.append((abs(exc), t, ov, c11, c15, exc, ",".join(labels) if labels else "(ours-only)"))
        for _, t, ov, c11, c15, exc, lab in sorted(rows, reverse=True):
            if abs(ov) < 0.05 and abs(exc) < 0.05:
                continue
            f = lambda v: "  --  " if v is None else "%+6.2f" % v
            print("  %-22s %9.2f %9s %9s %+9.2f   %s" % (t, ov, f(c11), f(c15), exc, lab))


if os.environ.get("FENS"):
    per_fen(os.environ["FENS"])
    sf11.close(); sf15.close()
    raise SystemExit

# ---- collapse positions of the target class ----
CLS = os.path.join(THIS, "ks_sets", "collapse_dataset_classified.csv")
coll = [r["decision_fen"] for r in csv.DictReader(open(CLS))
        if r.get("ks_class") == CLASS and r.get("decision_fen")][:N]
print("attributing %d '%s' collapses (SF11+SF15 classical references)" % (len(coll), CLASS))
profile(coll, "COLLAPSE:" + CLASS)

# ---- control: bank quiet positions ----
if CONTROL:
    BANK = os.path.join(THIS, "ks_sets", "position_bank.csv")
    ctrl = [r["fen"] for r in csv.DictReader(open(BANK)) if r.get("geo_class") == "quiet"][:N]
    profile(ctrl, "CONTROL:quiet")

sf11.close(); sf15.close()
