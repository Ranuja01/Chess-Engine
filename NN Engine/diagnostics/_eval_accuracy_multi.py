# -*- coding: utf-8 -*-
"""EVAL ACCURACY across EVERY corpus at once — the cross-set guard, made structural.

WHY THIS EXISTS (2026-09-11). `_eval_accuracy_arms.py` scores one corpus. Tuning eval-v2's king safety on
`lichess_ks_labelled` alone produced a config that improved KS-critical accuracy by 10.95% and DEGRADED
general accuracy by 20.09% — a perfect anti-correlation, and it was one build away from going to selfplay
as a regression. `corpus-fit-is-anti-correlated-with-elo` is on the books; single-corpus tuning walks into
it every time.

▶️ THIS scores every arm on every corpus in one run and reports:
  - per-corpus % against THAT CORPUS'S OWN baseline arm (arm #1), because baselines differ enormously
    (general ~377 MSE vs KS-critical ~1908) and pooling raw MSE lets the biggest-error corpus dominate —
    which is precisely the mistake that produced the regression above
  - mean% across corpora
  - ☠️ WORST% — the number that should decide. A config that helps four corpora and wrecks one is not a
    candidate; the ladder needs terms that are safe everywhere, not terms that are excellent somewhere.

★ One child process per ARM (not per arm x corpus): knobs latch once at ChessAI construction, so the arm
must be a separate process, but a single child can walk every corpus. That is 6x fewer engine loads.

  pyrun diagnostics/_eval_accuracy_multi.py N=2500 \
      CORPORA="ks_sets/game_regret_set.csv,ks_sets/game_regret_set_v2.csv,..." \
      ARMS="base:EVAL_ARM=1|ks:EVAL_ARM=1 KS_V2_MAX=4000"

⚠️ Closeness to SF18 is NOT Elo. This RANKS arms and REJECTS unsafe ones; games still decide.
"""
import os, sys, csv, math, subprocess

for _a in sys.argv[1:]:
    if _a.startswith("ARMS=") or _a.startswith("CORPORA="):
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v
    elif '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v

THIS = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(THIS)
N = int(os.environ.get("N", "2500"))

DEFAULT_CORPORA = ",".join([
    "ks_sets/game_regret_set.csv",        # general self-play, primary
    "ks_sets/game_regret_set_v2.csv",     # general, DISJOINT from primary
    "ks_sets/game_regret_set_x4.csv",     # general, newest
    "ks_sets/game_regret_set_uho.csv",    # UHO openings — different opening distribution
    "ks_sets/variant_regret_set.csv",     # piece-replacement / 960 — structure-INDEPENDENT
    "ks_sets/lichess_ks_labelled.csv",    # KS-critical — the tail this rung targets
])


def resolve(p):
    for cand in (os.path.join(ENGINE, p), os.path.join(THIS, p), os.path.abspath(p)):
        if os.path.exists(cand):
            return cand
    return None


CORPORA = []
for c in os.environ.get("CORPORA", DEFAULT_CORPORA).split(','):
    c = c.strip()
    if not c:
        continue
    r = resolve(c)
    if r:
        CORPORA.append((os.path.basename(r).replace(".csv", ""), r))
    else:
        sys.stderr.write("⚠️ skipping missing corpus: %s\n" % c)


def winpct(cp):
    return 50.0 + 50.0 * (2.0 / (1.0 + math.exp(-0.00368208 * cp)) - 1.0)


# ---- child: walk EVERY corpus under this process's knobs, emit "corpus,ours_cp,target_cp" ------------
if os.environ.get("_MULTI_CHILD"):
    sys.path.insert(0, ENGINE)
    import chess
    import ChessAI
    ai = ChessAI.ChessAI(None, None, chess.Board(), True)
    out = []
    for cname, cpath in CORPORA:
        with open(cpath, newline="") as f:
            for i, row in enumerate(csv.DictReader(f)):
                if i >= N:
                    break
                try:
                    tgt = float(row["best_cp"])
                except (KeyError, ValueError):
                    continue
                if abs(tgt) > 9000:          # mates: winpct saturates, no gradient about accuracy
                    continue
                b = chess.Board(row["fen"])
                if b.is_checkmate() or b.is_stalemate():
                    continue
                # ☠️ ev() is ABSOLUTE BLACK-POSITIVE MILLIPAWNS; best_cp is WHITE-POV CENTIPAWNS.
                out.append("%s,%.4f,%.4f" % (cname, -ai.ev(b) / 10.0, tgt))
    sys.stdout.write("\n".join(out))
    sys.exit(0)


def run_arm(knobs):
    env = dict(os.environ)
    env["_MULTI_CHILD"] = "1"
    env["PRESET"] = "LONG_FORMAT"
    env["USE_OPENING_BOOK"] = "0"
    for kv in knobs.split():
        if '=' in kv:
            k, v = kv.split('=', 1)
            env[k] = v
    p = subprocess.run([sys.executable, "-u", os.path.abspath(__file__)],
                       cwd=ENGINE, env=env, capture_output=True, text=True, timeout=7200)
    if p.returncode != 0:
        sys.stderr.write("child failed (%s):\n%s\n" % (knobs, p.stderr[-1500:]))
        sys.exit(2)
    per = {}
    for ln in p.stdout.split("\n"):
        parts = ln.split(',')
        if len(parts) != 3:
            continue
        cname, ours, tgt = parts[0], float(parts[1]), float(parts[2])
        d = winpct(ours) - winpct(tgt)
        s, n = per.get(cname, (0.0, 0))
        per[cname] = (s + d * d, n + 1)
    return {k: (v[0] / v[1] if v[1] else float("nan")) for k, v in per.items()}


arms = []
for ent in os.environ.get("ARMS", "base:EVAL_ARM=1").split('|'):
    name, _, knobs = ent.partition(':')
    arms.append((name.strip(), knobs.strip()))

names = [c[0] for c in CORPORA]
print("EVAL ACCURACY — %d arms x %d corpora, first %d rows each" % (len(arms), len(CORPORA), N))
print("Per-corpus %% vs arm #1 (that corpus's own baseline). NEGATIVE = better.")
print("☠️ WORST is the deciding column: a term must be safe on EVERY position type, not excellent on one.\n")

hdr = "  %-22s" % "arm"
for nm in names:
    hdr += " %>14s" % nm[:14] if False else " %14s" % nm[:14]
print(hdr + " %8s %8s" % ("mean%", "WORST%"))

base = None
for name, knobs in arms:
    res = run_arm(knobs)
    if base is None:
        base = res
        line = "  %-22s" % name
        for nm in names:
            line += " %14.2f" % res.get(nm, float("nan"))
        print(line + " %8s %8s" % ("(base)", "(base)"))
        continue
    pcts, line = [], "  %-22s" % name
    for nm in names:
        b, v = base.get(nm), res.get(nm)
        if b and v and b == b and v == v:
            pc = 100.0 * (v - b) / b
            pcts.append(pc)
            line += " %13.2f%%" % pc
        else:
            line += " %14s" % "-"
    if pcts:
        print(line + " %7.2f%% %7.2f%%" % (sum(pcts) / len(pcts), max(pcts)))
    else:
        print(line + " %8s %8s" % ("-", "-"))

print("\n  ⚠️ Closeness to SF18 is NOT Elo. This RANKS arms and REJECTS unsafe ones; games decide.")
