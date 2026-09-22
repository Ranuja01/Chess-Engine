# -*- coding: utf-8 -*-
"""FIRE-RATE + MAGNITUDE for one eval knob: how often does it change the score, and by how much?

WHY THIS EXISTS (2026-09-21). Before spending a move-channel screen -- let alone a games night -- on a new
term, two numbers decide whether any instrument we own can resolve it at all:

    FIRE RATE   the fraction of positions whose static eval MOVES when the knob is switched on
    MAGNITUDE   how far it moves them

A term that shifts 50% of positions by 10 mp is invisible to everything: it sits under the d7 regret null,
under the +/-150 STS floor, and under a night of games. A term that shifts 5% by 300 mp is resolvable. The
twelve straight move-nulls of 2026-09 were, almost without exception, the first kind -- a constant on a
detector -- and NONE of them was checked this way first.
★ This is also the "prove the knob is LIVE" check that CLAUDE.md demands: a CHANGED-RATE, not byte-identity.
A knob that moves 0 positions is inert, and every null measured on it afterwards is worthless.
See a-detector-gate-passes-vacuously-unless-the-term-is-proved-to-fire.

▶️ HOW IT WORKS. Two child processes evaluate the SAME corpus through ChessAI.ev under two knob settings
and the value streams are diffed. It reimplements nothing, so it cannot disagree with the C++ for reasons
of its own -- the usual failure of a Python mirror of an eval term. No search, no clock: a pure function of
the position, so a single differing value is real rather than run-to-run noise.

⚠️ Knobs latch once per process (initialize_engine via ChessAI.__cinit__), so the two settings MUST be
separate processes. That is what the fork is for, not speed.
⚠️ STATIC eval only. A large static swing that never changes a MOVE is still move-null -- this tool is a
NECESSARY, not a sufficient, condition. It tells you whether an instrument can see the term, not whether
the term is right. Pair it with the d7 regret gate (INSTRUMENT-MAP.md §B).

  pyrun diagnostics/_eval_knob_delta.py A=<knobs> B=<knobs> [SET=ks_sets/game_regret_set.csv] [N=4000] \\
        [<shared knobs, e.g. the whole EVAL_ARM=1 v2 block>]

A and B are space-separated KEY=VALUE strings applied ON TOP of the shared knobs; A is the control.
Example -- the passer path-safety ladder against its own absence:

  ... A='PASSER_V2_PATH_PCT=0' B='PASSER_V2_PATH_PCT=100' EVAL_ARM=1 PASSER_V2_MAG=60 ...

Exit 0 always (this is a measurement, not a gate); exit 2 on a harness failure.
"""
import os, sys, csv, subprocess

for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v

THIS = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(THIS)
SET = os.environ.get("SET", "ks_sets/game_regret_set.csv")
N = int(os.environ.get("N", "4000"))


def resolve_set(p):
    # Corpora live in diagnostics/ks_sets/, but callers write SET= both ways depending on whether they
    # think of themselves as running from the engine root or from diagnostics/. Accept either.
    for cand in (os.path.join(ENGINE, p), os.path.join(THIS, p), os.path.abspath(p)):
        if os.path.exists(cand):
            return cand
    sys.stderr.write("corpus not found: %s (tried engine root, diagnostics/, cwd)\n" % p)
    sys.exit(2)


SET_PATH = resolve_set(SET)


# ---- child mode: evaluate the corpus under whatever knobs this process was started with -------------
if os.environ.get("_KNOB_CHILD"):
    # Running this file by absolute path puts diagnostics/ on sys.path, not the engine root where the
    # built extension lives, so the import has to be told where to look.
    sys.path.insert(0, ENGINE)
    import chess
    import ChessAI
    ai = ChessAI.ChessAI(None, None, chess.Board(), True)
    out = []
    with open(SET_PATH, newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i >= N:
                break
            b = chess.Board(row["fen"])
            # ev() assumes a non-terminal position, exactly as the engine does.
            out.append("0" if (b.is_checkmate() or b.is_stalemate()) else str(ai.ev(b)))
    sys.stdout.write("\n".join(out))
    sys.exit(0)


def run_side(label, knobs):
    env = dict(os.environ)
    env["_KNOB_CHILD"] = "1"
    env["PRESET"] = "LONG_FORMAT"
    env["USE_OPENING_BOOK"] = "0"
    for tok in knobs.split():
        if '=' in tok:
            k, v = tok.split('=', 1)
            env[k] = v
    p = subprocess.run([sys.executable, "-u", os.path.abspath(__file__)],
                       cwd=ENGINE, env=env, capture_output=True, text=True, timeout=3600)
    if p.returncode != 0:
        sys.stderr.write("child %s failed (exit %d):\n%s\n" % (label, p.returncode, p.stderr[-2000:]))
        sys.exit(2)
    # ⚠️ Echo the child's toggles banner. A run where the knob is ABSENT from it never reached the child
    # and the whole measurement proved nothing (env-knob-name-verify).
    for line in p.stderr.splitlines():
        if line.startswith("[toggles]"):
            for tok in line.split():
                if tok.split('=', 1)[0] in {kk.split('=', 1)[0] for kk in knobs.split() if '=' in kk}:
                    sys.stderr.write("  %s child sees %s\n" % (label, tok))
    return [int(x) for x in p.stdout.split("\n") if x.strip() != ""]


A = os.environ.get("A", "")
B = os.environ.get("B", "")
if not B:
    sys.stderr.write("B= is required (the arm under test); A= defaults to the shared config alone\n")
    sys.exit(2)

va = run_side("A", A)
vb = run_side("B", B)
if len(va) != len(vb) or not va:
    sys.stderr.write("value streams differ in length (%d vs %d) -- a child died early\n" % (len(va), len(vb)))
    sys.exit(2)

deltas = [b - a for a, b in zip(va, vb)]
nz = [d for d in deltas if d != 0]
n = len(deltas)


def pct(vals, q):
    if not vals:
        return 0
    s = sorted(vals)
    return s[min(len(s) - 1, int(q * len(s)))]


print("corpus            %s" % SET_PATH)
print("positions         %d" % n)
print("A (control)       %s" % (A or "(shared config only)"))
print("B (under test)    %s" % B)
print("")
print("FIRE RATE         %d / %d  = %.1f%%" % (len(nz), n, 100.0 * len(nz) / n))
if not nz:
    print("")
    print("INERT: the knob changed no position. Any null measured on this arm is worthless --")
    print("find out why it does not fire before spending another instrument on it.")
    sys.exit(0)

mags = [abs(d) for d in nz]
print("|delta| mean      %.1f mp   (over the positions that MOVED)" % (sum(mags) / len(mags)))
print("|delta| median    %d mp" % pct(mags, 0.50))
print("|delta| p90       %d mp" % pct(mags, 0.90))
print("|delta| max       %d mp" % max(mags))
# Signed mean over ALL positions. A large value means the term mostly shifts the MEAN rather than adding
# signal, which is a retune cost rather than a gain (every-eval-term-error-is-bidirectional).
print("signed mean       %+.1f mp  (over all positions)" % (sum(deltas) / float(n)))
print("")
# The resolvability verdict. The bars are the measured floors of the instruments we actually own.
mean_all = sum(mags) / n
print("mean |delta| over ALL positions: %.1f mp" % mean_all)
if len(nz) / n < 0.02:
    print("  -> fires rarely; a corpus-wide mean will HIDE it. Screen on the firing subset only.")
if pct(mags, 0.50) < 20:
    print("  -> median swing is small; expect the d7 regret gate to read null even if the term is right.")
if max(mags) > 200:
    print("  -> the tail is large enough for the move channel to see. Worth a regret gate.")
