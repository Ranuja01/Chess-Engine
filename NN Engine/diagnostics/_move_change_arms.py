# -*- coding: utf-8 -*-
"""Does a GATED knob change which move we play? — the two-arm move-change rate.

`_sibling_spread.py` answers this by DELETING a breakdown term group, which is the right instrument for
a term that is already live. It cannot be pointed at a mechanism that is gated OFF: there is nothing to
delete, and a gated term is usually not a breakdown field at all. This is the knob-arm form.

Why this metric rather than cp error or STS: most eval error is MOVE-NEUTRAL (memory
`most-eval-error-is-move-neutral` -- 52% of large eval errors still produce a fine move, only 18% a bad
one). A change can move 81% of evals and reorder nothing, and a term applied as a scale to the SUMMED
TOTAL is the extreme case -- a strictly monotone transform of the total cannot change an argmax at all.
Eval movement is therefore an upper bound on move change, often a very loose one, and games can only
adjudicate what actually reaches the move.

For N real corpus positions, evaluate EVERY legal child in each arm (one process per arm -- knobs latch
at engine init) and compare the argmax. Reports the flip rate and the REGRET the flip costs in the
baseline arm's own units, so a flip between two near-equal moves is not counted like a real change.

🚨 The reading is COMPARATIVE. Pass a control arm that is known to matter: `ENABLE_THREATS=0` against
defaults measures the capped-threats change, which shipped and won +45 Elo. A candidate whose flip rate
is far BELOW that control cannot be resolved by a night of games no matter what the bench says.

⚠️ ONE-PLY STATIC proxy, same caveat as _sibling_spread: the engine chooses by searching, so a term that
cannot reorder static children could still act through deeper lines. What this bounds is the leaf score
and the static ordering.

  pyrun diagnostics/_move_change_arms.py ARM=ENABLE_WINNABILITY=1 [N=400] [QUIET_ONLY=0]
"""
import os, sys, csv, subprocess

# ⚠️ Set EVERY KEY=VAL, not a whitelist of the control keys. A whitelist silently drops the arm's own
# knobs on the way into the worker, both arms then run identical, and the tool reports a confident 0%
# flip rate for a change that was never applied -- the same defect knob_liveness_audit.py records
# having shipped with. The control arm caught it immediately, which is what controls are for.
for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
THIS = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(THIS)
sys.path.insert(0, ENGINE); sys.path.insert(0, THIS)

N = int(os.environ.get("N", "400"))
QUIET_ONLY = os.environ.get("QUIET_ONLY", "0") == "1"
IN = os.environ.get("IN", os.path.join(THIS, "ks_sets", "diverse_corpus_wide.csv"))
if not os.path.isabs(IN):
    IN = os.path.join(THIS, IN)


def run_worker(out_path):
    """Pick the static-argmax child for each position and record it with its regret margin."""
    import chess
    from ChessAI import ChessAI

    ai = ChessAI(None, None, chess.Board(), True)
    rows = list(csv.DictReader(open(IN, newline="")))[:N]
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fen", "best", "regret"])
        for r in rows:
            try:
                b = chess.Board(r["fen"])
                if b.is_game_over(claim_draw=False):
                    continue
                # Absolute eval is Black-positive, so White picks the MINIMUM child.
                white_to_move = b.turn
                scored = []
                for mv in b.legal_moves:
                    if QUIET_ONLY and (b.is_capture(mv) or b.piece_type_at(mv.from_square) == chess.PAWN):
                        continue
                    b.push(mv)
                    try:
                        scored.append((ai.ev_breakdown(b).get("total", 0), mv.uci()))
                    finally:
                        b.pop()
                if len(scored) < 2:
                    continue
                scored.sort(key=lambda t: t[0], reverse=not white_to_move)
                w.writerow([r["fen"], scored[0][1], abs(scored[1][0] - scored[0][0])])
            except Exception:
                continue


if os.environ.get("WORKER") == "1":
    run_worker(os.environ["OUT"])
    sys.exit(0)

ARM = os.environ.get("ARM", "")
if not ARM:
    sys.exit("ARM=<KNOB=VAL[,KNOB=VAL...]> is required")

# One process per arm: knobs latch at engine init, so there is no way to flip a gate inside a run.
# Both files are written under this process's lifetime because /tmp does not survive between calls.
base_out, arm_out = "/tmp/_mc_base.csv", "/tmp/_mc_arm.csv"
for out_path, knobs in ((base_out, []), (arm_out, ARM.split(","))):
    cmd = [sys.executable, "-u", os.path.abspath(__file__), "WORKER=1", "OUT=" + out_path,
           "N=" + str(N), "IN=" + IN, "QUIET_ONLY=" + ("1" if QUIET_ONLY else "0")] + knobs
    subprocess.run(cmd, cwd=ENGINE, check=True)

base = {r["fen"]: r for r in csv.DictReader(open(base_out, newline=""))}
arm = {r["fen"]: r for r in csv.DictReader(open(arm_out, newline=""))}
shared = [f for f in base if f in arm]

flips = [f for f in shared if base[f]["best"] != arm[f]["best"]]
# Regret is read in the BASELINE arm's units: what the baseline thought it was giving up by taking the
# second-best move. A flip between two moves the baseline scored equally is not a behavioural change.
regrets = sorted(float(base[f]["regret"]) for f in flips)
meaningful = [r for r in regrets if r >= 100]     # >= 10 cp of baseline regret

print("\n  arm: %s%s" % (ARM, "   [QUIET_ONLY]" if QUIET_ONLY else ""))
print("  positions compared %d" % len(shared))
if shared:
    print("  move CHANGED       %d  (%.1f%%)" % (len(flips), 100.0 * len(flips) / len(shared)))
    print("  of those, regret >= 10 cp   %d  (%.1f%% of all positions)"
          % (len(meaningful), 100.0 * len(meaningful) / len(shared)))
if regrets:
    print("  baseline regret at the flips: median %.0f mp   max %.0f mp"
          % (regrets[len(regrets) // 2], regrets[-1]))
print("  🚨 comparative only -- read against a control arm run the same way "
      "(ENABLE_THREATS=0 = the +45 Elo shipped change).\n")
