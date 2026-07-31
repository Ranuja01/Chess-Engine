# -*- coding: utf-8 -*-
"""KS firing-profile harness — the ANTI-REGRESSION gate. For each candidate knob set, report our king_safety
term (our POV, pawns) on the three sets:
  danger      : KS SHOULD fire (large negative = our king in danger).
  control_calm: KS MUST stay ~0 (safe midgame king).
  control_eg  : KS MUST stay ~0 (endgame exposed king; catches phase-taper over-fire = the prior regression).
Deterministic, no games. A good set = high danger fire-rate + ~0 false-fire on BOTH controls.

Env->Config is parsed ONCE per process (initialize_engine static guard), so each candidate runs in a FRESH
worker subprocess with its knobs in the environment.
Run: pyrun diagnostics/ks_firing_profile.py
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import sys
import subprocess

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SETS = os.path.join(THIS_DIR, "ks_sets")
PY = sys.executable

BASE_DEFAULT = dict(KS_ATT_KNIGHT="2", KS_ATT_BISHOP="2", KS_ATT_ROOK="3", KS_ATT_QUEEN="5",
                    KS_ATTACK_COUNT="1", KS_WEAK="2", KS_SAFE_CHECK="3", KS_FLOOR="0", KS_DYN="0")
SAFE_DOMINANT = dict(KS_ATT_KNIGHT="1", KS_ATT_BISHOP="1", KS_ATT_ROOK="1", KS_ATT_QUEEN="1",
                     KS_ATTACK_COUNT="0", KS_WEAK="1", KS_SAFE_CHECK="18", KS_FLOOR="0", KS_DYN="0")
DEF0 = {**BASE_DEFAULT, "KS_DEFENDER": "0"}
SFDEF = {**DEF0, "ENABLE_KS_SF_WEAK": "1", "ENABLE_KS_SF_SAFECHECK": "1"}
F13BASE = {**SFDEF, "KS_FLOOR": "13", "KS_NO_QUEEN": "6"}
CANDIDATES = [
    ("R0 def=0 no-SF (reference)", dict(KING_SAFETY_MAG="4000", **DEF0)),
    ("F13 MAG=3000 (ship)", dict(KING_SAFETY_MAG="3000", **F13BASE)),
    ("F13 MAG=3000 + ZONE2", dict(KING_SAFETY_MAG="3000", **{**F13BASE, "KS_ZONE2": "1"})),
    ("F13 MAG=3000 + ZONE2 FLOOR=16", dict(KING_SAFETY_MAG="3000", **{**F13BASE, "KS_ZONE2": "1", "KS_FLOOR": "16"})),
]


def load(name):
    out = []
    for ln in open(os.path.join(SETS, name + ".txt")):
        ks, fen = ln.rstrip("\n").split("\t", 1)
        out.append(fen)
    return out


def worker():
    import chess
    sys.path.insert(0, os.path.dirname(THIS_DIR))
    from ChessAI import ChessAI
    seed = chess.Board()
    ai = ChessAI(None, None, seed, seed.turn)

    def our_ks(fen):
        b = chess.Board(fen)
        bd = ai.ev_breakdown(b)
        if bd.get("checkmate"):
            return None
        povsign = 1.0 if b.turn == chess.WHITE else -1.0
        return (-bd.get("king_safety", 0.0) / 1000.0) * povsign

    import statistics
    res = {}
    for name in ("danger", "control_calm", "control_eg"):
        vals = [our_ks(f) for f in load(name)]
        vals = [v for v in vals if v is not None]
        mean = statistics.mean(vals) if vals else 0.0
        fire = 100.0 * sum(1 for v in vals if v <= -1.0) / max(1, len(vals))
        ff = sum(1 for v in vals if abs(v) > 0.5)
        res[name] = (mean, fire, ff)
    print("RESULT\t%s" % "\t".join("%s:%.2f,%.0f,%d" % (k, v[0], v[1], v[2]) for k, v in res.items()))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        worker(); return
    print("%-32s | %-20s | %-16s | %-16s" % ("candidate", "DANGER mean/fire%", "CALM mean/#ff", "EG mean/#ff"))
    print("-" * 92)
    for label, knobs in CANDIDATES:
        env = os.environ.copy()
        env.update(knobs)
        p = subprocess.run([PY, os.path.abspath(__file__), "worker"], env=env, capture_output=True, text=True)
        row = next((l for l in p.stdout.splitlines() if l.startswith("RESULT")), None)
        if not row:
            print("%-32s | (worker failed) %s" % (label, p.stderr.strip()[-80:])); continue
        d = {kv.split(":")[0]: kv.split(":")[1] for kv in row.split("\t")[1:]}
        dm, df, _ = d["danger"].split(",")
        cm, _, cff = d["control_calm"].split(",")
        em, _, eff = d["control_eg"].split(",")
        print("%-32s | %+6.2f / %4.0f%%     | %+6.2f / %-3s   | %+6.2f / %-3s"
              % (label, float(dm), float(df), float(cm), cff, float(em), eff))


if __name__ == "__main__":
    main()
