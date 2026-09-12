# -*- coding: utf-8 -*-
"""EVAL ACCURACY per arm: how close is our STATIC eval to SF18's assessment? No search, no move choice.

WHY THIS EXISTS (2026-09-11). Everything used on eval v2 so far measures MOVE CHOICE -- STS picks a move,
the d7 regret gate picks a move. Neither measures how ACCURATE the eval is. A change can make the eval
substantially truer and still not flip the move in most positions, because the move only changes when the
top two candidates cross. That dilution is exactly why the gate reads king safety as null while KS is
plainly doing work, and it made an 8-config KS shape screen come back mutually indistinguishable on STS.

▶️ THIS measures the thing itself: |win%(ours) - win%(SF18)| squared, over a corpus, per arm.
  - no search      ⇒ no chaotic sensitivity (INSTRUMENT-MAP §H), no tie-breaking artifacts
  - no move choice ⇒ no population confound (§F2), no null band needed
  - deterministic  ⇒ a difference is a difference
★ It is also the RIGHT question for a rebuild: the ladder's premise is that a truer eval is the goal and
pruning/move quality follow. Measure the premise directly instead of through two layers of proxy.

⚠️ WHAT IT CANNOT TELL YOU. Closeness to SF18 is not Elo, and `corpus-fit-is-anti-correlated-with-elo` is
on the books. This RANKS arms by accuracy; it does not promote one. Games still decide.
⚠️ And SF18 is a SEARCHING engine, so no static eval can reach 0 -- see `_reference_ceiling.py` for the
achievable floor. Only DIFFERENCES between arms on the same corpus are meaningful here.

  pyrun diagnostics/_eval_accuracy_arms.py CORPUS=ks_sets/lichess_ks_labelled.csv N=5000 \
        ARMS="base:EVAL_ARM=1|ks:EVAL_ARM=1 KS_V2_MAX=4000"

ARMS is `name:KNOB=V KNOB=V` entries separated by `|`. Each runs in its OWN process, because knobs latch
once per process at ChessAI construction.
"""
import os, sys, csv, math, subprocess

for _a in sys.argv[1:]:
    if '=' in _a and not _a.startswith("ARMS="):
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v
for _a in sys.argv[1:]:
    if _a.startswith("ARMS="):
        os.environ["ARMS"] = _a.split('=', 1)[1]

THIS = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(THIS)
N = int(os.environ.get("N", "5000"))
CORPUS = os.environ.get("CORPUS", "ks_sets/lichess_ks_labelled.csv")


def resolve(p):
    for cand in (os.path.join(ENGINE, p), os.path.join(THIS, p), os.path.abspath(p)):
        if os.path.exists(cand):
            return cand
    sys.stderr.write("☠️ corpus not found: %s\n" % p)
    sys.exit(2)


CORPUS_PATH = resolve(CORPUS)

# Lichess sigmoid, the same constant the fit objective and the regret gate use, so numbers stay comparable
# across every instrument we own.
def winpct(cp):
    return 50.0 + 50.0 * (2.0 / (1.0 + math.exp(-0.00368208 * cp)) - 1.0)


# ---- child: evaluate the corpus under this process's knobs -------------------------------------------
if os.environ.get("_ACC_CHILD"):
    sys.path.insert(0, ENGINE)
    import chess
    import ChessAI
    ai = ChessAI.ChessAI(None, None, chess.Board(), True)
    out = []
    with open(CORPUS_PATH, newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i >= N:
                break
            try:
                tgt = float(row["best_cp"])
            except (KeyError, ValueError):
                continue
            # ⚠️ Skip mate scores: winpct saturates and they carry no gradient about eval accuracy.
            if abs(tgt) > 9000:
                continue
            b = chess.Board(row["fen"])
            if b.is_checkmate() or b.is_stalemate():
                continue
            # ☠️ UNITS AND SIGN. ev() is ABSOLUTE BLACK-POSITIVE MILLIPAWNS; best_cp is WHITE-POV
            # CENTIPAWNS. Getting either wrong produces a plausible-looking loss that means nothing --
            # exactly the failure mode this file exists to avoid elsewhere.
            ours_cp = -ai.ev(b) / 10.0
            out.append("%.4f,%.4f" % (ours_cp, tgt))
    sys.stdout.write("\n".join(out))
    sys.exit(0)


def run_arm(knobs):
    env = dict(os.environ)
    env["_ACC_CHILD"] = "1"
    env["PRESET"] = "LONG_FORMAT"
    env["USE_OPENING_BOOK"] = "0"
    for kv in knobs.split():
        if '=' in kv:
            k, v = kv.split('=', 1)
            env[k] = v
    p = subprocess.run([sys.executable, "-u", os.path.abspath(__file__)],
                       cwd=ENGINE, env=env, capture_output=True, text=True, timeout=3600)
    if p.returncode != 0:
        sys.stderr.write("child failed (%s):\n%s\n" % (knobs, p.stderr[-1500:]))
        sys.exit(2)
    rows = []
    for ln in p.stdout.split("\n"):
        if ',' in ln:
            a, b = ln.split(',')
            rows.append((float(a), float(b)))
    return rows


arms = []
for ent in os.environ.get("ARMS", "base:EVAL_ARM=1").split('|'):
    name, _, knobs = ent.partition(':')
    arms.append((name.strip(), knobs.strip()))

print("EVAL ACCURACY vs SF18  —  %s, first %d rows" % (os.path.basename(CORPUS_PATH), N))
print("  win%%-space MSE, LOWER IS BETTER. Differences between arms are the signal; the absolute")
print("  value is bounded below by what any static eval can reach against a searching engine.\n")
print("  %-28s %8s %12s %12s" % ("arm", "n", "win%_MSE", "mean|err|"))

base_mse = None
for name, knobs in arms:
    rows = run_arm(knobs)
    if not rows:
        print("  %-28s %8s  ☠️ no rows" % (name, "-"))
        continue
    sse = 0.0
    abserr = 0.0
    for ours, tgt in rows:
        d = winpct(ours) - winpct(tgt)
        sse += d * d
        abserr += abs(d)
    mse = sse / len(rows)
    if base_mse is None:
        base_mse = mse
        tag = "(baseline)"
    else:
        tag = "%+.2f%%" % (100.0 * (mse - base_mse) / base_mse)
    print("  %-28s %8d %12.2f %12.3f  %s" % (name, len(rows), mse, abserr / len(rows), tag))

print("\n  ⚠️ Closeness to SF18 is NOT Elo — `corpus-fit-is-anti-correlated-with-elo`. This RANKS arms;")
print("     it does not promote one. Games decide.")
