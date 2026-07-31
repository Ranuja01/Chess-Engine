# -*- coding: utf-8 -*-
"""Isolate WHERE MOD_KS_REALIZ's MSE regression comes from.

KSR=128 leaves MAE flat (12.546 -> 12.543) while MSE rises (347.0 -> 350.3). MSE is quadratic, so that
combination means the TYPICAL position is untouched and a SMALL NUMBER got much worse -- a tail, not broad
degradation. This dumps per-position error plus the features that would explain it, so the regressed tail
can be characterised and fixed surgically (e.g. via KS_REALIZ_FLOOR, which caps how far KS may be damped)
rather than by abandoning a +117 STS lever.

Mechanism hypothesis to test: KSR damps when threat_side_edge < 0 (attacker materially BEHIND). That is
usually a fantasy attack -- but an attacker who is behind BECAUSE THEY SACRIFICED is exactly when the
attack is real. Small population, large errors, which is the signature we see.

  pyrun diagnostics/_ksr_tail.py <KEY=VAL...>      # dump for this config
  pyrun diagnostics/_ksr_tail.py --compare         # rank regressions and characterise
"""
import os, sys, csv, math
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v
THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS))
os.chdir(os.path.dirname(THIS))

import chess
WIN_K = 0.00368208
def winpct(cp): return 50.0 + 50.0 * (2.0 / (1.0 + math.exp(-WIN_K * cp)) - 1.0)

BANK = os.path.join(THIS, "ks_sets", "position_bank.csv")

def load():
    out = []
    for r in csv.DictReader(open(BANK)):
        if not r.get("sf18"):
            continue
        try:
            s = float(r["sf18"])
        except Exception:
            continue
        if abs(s) > 20.0:
            continue
        out.append(r)
    return out

if "--compare" not in sys.argv:
    from ChessAI import ChessAI
    rows = load()
    seed = chess.Board(rows[0]["fen"])
    ai = ChessAI(None, None, seed, seed.turn)
    ksr = os.environ.get("MOD_KS_REALIZ", "0")
    outp = os.path.join(THIS, f"_ksr_tail_{ksr}.csv")
    with open(outp, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["fen", "err", "ours", "sf18", "ks", "mat_edge", "phase", "geo"])
        for r in rows:
            b = chess.Board(r["fen"])
            bd = ai.ev_breakdown(b)
            if bd.get("checkmate"):
                continue
            ours = -bd["total"] / 1000.0
            sf = float(r["sf18"])
            err = winpct(ours * 100.0) - winpct(sf * 100.0)
            # material edge WHITE-POV from the (now corrected) accumulators
            mat = -bd.get("material", 0) / 1000.0
            w.writerow([r["fen"], f"{err:.3f}", f"{ours:.3f}", f"{sf:.3f}",
                        f"{-bd.get('king_safety',0)/1000.0:.3f}", f"{mat:.3f}",
                        r.get("phase_score", ""), r.get("geo_class", "")])
    print(f"wrote {outp} ({len(rows)} rows, MOD_KS_REALIZ={ksr})")
    sys.exit(0)

a = {r["fen"]: r for r in csv.DictReader(open(os.path.join(THIS, "_ksr_tail_0.csv")))}
b_ = {r["fen"]: r for r in csv.DictReader(open(os.path.join(THIS, "_ksr_tail_128.csv")))}
common = [f for f in a if f in b_]
recs = []
for f in common:
    e0, e1 = float(a[f]["err"]), float(b_[f]["err"])
    recs.append((abs(e1) - abs(e0), e0, e1, a[f], b_[f]))
recs.sort(key=lambda x: -x[0])

worse = [r for r in recs if r[0] > 0.5]
better = [r for r in recs if r[0] < -0.5]
same = len(recs) - len(worse) - len(better)
print(f"positions: {len(recs)}   worse(> .5pp): {len(worse)}   better(< -.5pp): {len(better)}   ~unchanged: {same}")
print(f"total MSE delta contribution: worse {sum(r[2]**2-r[1]**2 for r in worse):+.0f}   "
      f"better {sum(r[2]**2-r[1]**2 for r in better):+.0f}")

def prof(name, rs):
    if not rs:
        print(f"  {name}: none"); return
    n = len(rs)
    over = sum(1 for r in rs if r[2] > 0)     # our eval too HIGH for White
    matneg = sum(1 for r in rs if float(r[3]["mat_edge"]) * (1 if r[2] > 0 else -1) < 0)
    ph = sum(float(r[3]["phase"] or 0) for r in rs) / n
    ksm = sum(abs(float(r[3]["ks"])) for r in rs) / n
    ksm1 = sum(abs(float(r[4]["ks"])) for r in rs) / n
    geo = {}
    for r in rs:
        geo[r[3]["geo"]] = geo.get(r[3]["geo"], 0) + 1
    print(f"  {name}: n={n}  mean|KS| {ksm:.2f}->{ksm1:.2f}  mean phase {ph:.0f}  "
          f"over-read after: {over}/{n}  geo {geo}")

print("\nprofile of the REGRESSED tail (worst 40) vs the improved set:")
prof("worst40", recs[:40])
prof("all worse", worse)
prof("all better", better)
print("\ntop 12 regressions:")
for d, e0, e1, r0, r1 in recs[:12]:
    print(f"  d|err|={d:+6.2f}  err {e0:+7.2f}->{e1:+7.2f}  KS {float(r0['ks']):+6.2f}->{float(r1['ks']):+6.2f}  "
          f"mat {float(r0['mat_edge']):+5.2f}  ph {r0['phase']:>4}  {r0['geo']:<6} {r0['fen'][:38]}")
