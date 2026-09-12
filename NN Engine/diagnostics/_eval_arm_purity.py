# -*- coding: utf-8 -*-
"""TIMING-FREE purity test: does running eval v2 perturb eval v1's own result?

WHY THIS EXISTS (2026-09-11). The shadow arm (EVAL_ARM=2) runs BOTH evals in the same node and returns
v1's value, so "arm 2 reproduces the baseline node count" was supposed to prove v2 writes no global that v1
reads -- the compiler cannot enforce that, because cpp_bitboard.h exposes every one of v1's globals.

☠️ THAT TEST WAS BADLY DESIGNED, and it failed on its first use for the wrong reason. A `wac` bench
measures a SEARCH, and a search can depend on elapsed time (time checks, aborts). Arm 2 runs two evals per
node, so it is ~2x slower by construction and can diverge from the baseline for reasons that have nothing
to do with purity. The test therefore CONFLATES "v2 is impure" with "arm 2 is slow", which are the two
hypotheses it was built to separate. A test that cannot fail for only one reason is not a test.
⇒ `byte-identity-does-not-imply-speed-identity`, inverted: speed DIFFERENCE does not imply value difference,
and a search bench cannot tell you which one you have.

▶️ THIS test compares VALUES, never a search. For each FEN it takes the static eval through ChessAI.ev
under arm 0 and under arm 2. Arm 2's dispatcher returns v1's value, so:

    ev(fen, EVAL_ARM=0) != ev(fen, EVAL_ARM=2)   <=>   running v2 changed what v1 computed

No search, no clock, no node counts, no cache pressure -- a pure function of the position. Deterministic by
construction, so a single disagreement is a real defect rather than a run-to-run artifact.

⚠️ Knobs latch once per process (initialize_engine, via ChessAI.__cinit__), so the two arms MUST be
separate processes. This script forks one child per arm and diffs the two value streams.

  pyrun diagnostics/_eval_arm_purity.py [SET=ks_sets/game_regret_set.csv] [N=2000]

Exit code 0 = pure (every value identical). 1 = a divergence was found; the first 10 are printed with their
FENs so the offending position can be replayed under a debugger.
"""
import os, sys, csv, subprocess

for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v

THIS = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(THIS)
SET = os.environ.get("SET", "ks_sets/game_regret_set.csv")
N = int(os.environ.get("N", "2000"))


def resolve_set(p):
    # The corpora live in diagnostics/ks_sets/, but callers write SET= both ways depending on whether they
    # think of themselves as running from the engine root or from diagnostics/. Accept either.
    for cand in (os.path.join(ENGINE, p), os.path.join(THIS, p), os.path.abspath(p)):
        if os.path.exists(cand):
            return cand
    sys.stderr.write("☠️ corpus not found: %s (tried engine root, diagnostics/, and cwd)\n" % p)
    sys.exit(2)


SET_PATH = resolve_set(SET)


# ---- child mode: evaluate the corpus under whatever EVAL_ARM this process was started with ----------
if os.environ.get("_ARM_CHILD"):
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


def run_arm(arm):
    env = dict(os.environ)
    env["_ARM_CHILD"] = "1"
    env["EVAL_ARM"] = str(arm)
    env["PRESET"] = "LONG_FORMAT"
    env["USE_OPENING_BOOK"] = "0"
    p = subprocess.run([sys.executable, "-u", os.path.abspath(__file__)],
                       cwd=ENGINE, env=env, capture_output=True, text=True, timeout=1800)
    if p.returncode != 0:
        sys.stderr.write("child EVAL_ARM=%d failed (exit %d):\n%s\n" % (arm, p.returncode, p.stderr[-2000:]))
        sys.exit(2)
    # ⚠️ The arm banner goes to STDERR by design, so stdout stays a clean value stream. Echo it -- a run
    # where arm 2's banner is ABSENT means the knob never reached the child and the test proved nothing.
    for ln in (p.stderr or "").splitlines():
        if "EVAL_ARM" in ln:
            sys.stderr.write("  [arm %d banner] %s\n" % (arm, ln.strip()))
    return p.stdout.split("\n")


print("purity: %s, first %d positions, ChessAI.ev under EVAL_ARM 0 vs 2" % (os.path.basename(SET_PATH), N))
a0 = run_arm(0)
a2 = run_arm(2)

if len(a0) != len(a2):
    print("☠️ length mismatch: arm0=%d arm2=%d -- the corpora did not line up" % (len(a0), len(a2)))
    sys.exit(2)

bad = [i for i in range(len(a0)) if a0[i] != a2[i]]
print("compared %d positions, %d divergent" % (len(a0), len(bad)))

if not bad:
    print("✅ PURE: running v2 did not change a single v1 value.")
    print("   ⇒ v2 writes no global that v1 reads.")
    print("   ⚠️ This does NOT license 'any arm-2 bench divergence is therefore a speed effect'. A bench")
    print("      can diverge with every eval VALUE identical, because the eval's global side effects are")
    print("      load-bearing and a cache HIT skips them -- so anything that changes the cache hit/miss")
    print("      pattern changes when those globals are refreshed. That is a real behavioural difference")
    print("      with no value difference, and it is what bit us on 2026-09-11. See eval_cache_key().")
    sys.exit(0)

print("☠️ IMPURE: v2 perturbs v1. First %d divergences:" % min(10, len(bad)))
fens = []
with open(SET_PATH, newline="") as f:
    for i, row in enumerate(csv.DictReader(f)):
        if i >= N:
            break
        fens.append(row["fen"])
for i in bad[:10]:
    print("   [%5d] arm0=%-10s arm2=%-10s  %s" % (i, a0[i], a2[i], fens[i]))
sys.exit(1)
