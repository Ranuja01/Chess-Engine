# -*- coding: utf-8 -*-
"""Does a candidate move our CHOSEN MOVE toward SF's best move -- on the target set AND a holdout?

The validator the eval lane actually needs, and a replacement for `move_proxy`, which is opaque (both
engine passes are redirected to /tmp, so nothing is visible until the final summary) and whose row parser
silently discards every line whose tag does not match `game_\\d+`.

★ Why MOVE match and not centipawn distance: a two-set mean-|gap|-vs-SF11 screen was built first and
FAILED ITS OWN CONTROL -- `PV_BOOST_MAG=0`, a uniform shrink, beat every real candidate on it (target
−12.5%, holdout −20.0%). Our eval is systematically larger than SF11's, so ANY shrink improves a cp
distance. Shrinking an eval does NOT move its argmax toward SF's move, so match/plugged/broke is not
gameable that way.

Reports, per set:
  match   how often our move == SF's best   (base vs arm)
  plugged moves the arm FIXES  (base wrong -> arm right)
  broke   moves the arm BREAKS (base right -> arm wrong)

✅ WANT: plugged > broke on the TARGET and broke not inflated on the HOLDOUT.
☠️ The historical failure: plugged > broke on target, broke > plugged on holdout.

⚠️ Knobs latch at engine init => the arm runs in its OWN process (one process per setting).
⚠️ Progress is printed per FEN and FLUSHED: a stall must be distinguishable from slowness.

  pyrun diagnostics/_move_match_arms.py ARM=KS_MIN_ATTACKERS=8 SET=diagnostics/_mp_target.csv [N=79] [MT=0.3]
"""
import os, sys, csv, subprocess

for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
THIS = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(THIS)
sys.path.insert(0, ENGINE); sys.path.insert(0, THIS)
sys.path.insert(0, os.path.join(ENGINE, "selfplay"))

SET = os.environ.get("SET", os.path.join(THIS, "_mp_target.csv"))
if not os.path.isabs(SET):
    SET = os.path.join(ENGINE, SET)
N = int(os.environ.get("N", "79"))
MT = float(os.environ.get("MT", "0.3"))


def fens():
    return [r["fen_start"].strip() for r in csv.DictReader(open(SET)) if r.get("fen_start", "").strip()][:N]


def run_worker(out_path):
    import time
    import chess
    from tactical_test import run_one
    from arbiter import Arbiter, find_stockfish

    arb = Arbiter(find_stockfish(), movetime=MT)
    rows = fens()
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["fen", "our", "sf"])
        for i, fen in enumerate(rows, 1):
            t0 = time.time()
            try:
                r = run_one(fen, set())
                _, sf_best, _ = arb.evaluate(chess.Board(fen))
            except Exception as e:
                print("  [%3d/%3d] SKIP %s" % (i, len(rows), type(e).__name__), flush=True)
                continue
            w.writerow([fen, r["uci"], sf_best])
            print("  [%3d/%3d] %5.1fs  ours=%-6s sf=%-6s %s"
                  % (i, len(rows), time.time() - t0, r["uci"], sf_best,
                     "OK" if r["uci"] == sf_best else ""), flush=True)
    try:
        arb.close()
    except Exception:
        pass


if os.environ.get("WORKER") == "1":
    run_worker(os.environ["OUT"])
    sys.exit(0)

ARM = os.environ.get("ARM", "")
if not ARM:
    sys.exit("ARM=<KNOB=VAL[,KNOB=VAL...]> is required")

base_out = os.path.join(THIS, "_mm_base.csv")
arm_out = os.path.join(THIS, "_mm_arm.csv")
for out_path, knobs in ((base_out, []), (arm_out, ARM.split(","))):
    print("\n  === %s ===  set=%s N=%d" % ("BASELINE" if not knobs else ARM, os.path.basename(SET), N), flush=True)
    cmd = [sys.executable, "-u", os.path.abspath(__file__), "WORKER=1", "OUT=" + out_path,
           "SET=" + SET, "N=%d" % N, "MT=%s" % MT] + knobs
    # 🚨 FIXED DEPTH, not LIGHTNING. A time-based preset makes the search non-deterministic, so the
    # BASELINE itself wanders between runs (measured: 27 then 26 SF-best matches on the same 79 FENs at
    # defaults). That puts a +-1..2 noise band on plugged/broke -- exactly the size of the effects we are
    # trying to resolve. Fixed depth makes both arms reproducible, so every plug/break is real signal.
    env = dict(os.environ, PRESET="LONG_FORMAT", MAX_DEPTH=os.environ.get("DEPTH", "10"),
               USE_OPENING_BOOK="0", OMP_NUM_THREADS="1")
    subprocess.run(cmd, cwd=ENGINE, check=True, env=env)


def load(p):
    return {r["fen"]: (r["our"], r["sf"]) for r in csv.DictReader(open(p, newline=""))}


b, a = load(base_out), load(arm_out)
shared = [f for f in b if f in a]
bm = sum(1 for f in shared if b[f][0] == b[f][1])
am = sum(1 for f in shared if a[f][0] == a[f][1])
plugged = sum(1 for f in shared if b[f][0] != b[f][1] and a[f][0] == a[f][1])
broke = sum(1 for f in shared if b[f][0] == b[f][1] and a[f][0] != a[f][1])
changed = sum(1 for f in shared if b[f][0] != a[f][0])

print("\n  arm: %s   set: %s   N=%d" % (ARM, os.path.basename(SET), len(shared)))
print("  SF-best match   base %d (%.1f%%)   arm %d (%.1f%%)   delta %+d"
      % (bm, 100.0 * bm / max(1, len(shared)), am, 100.0 * am / max(1, len(shared)), am - bm))
print("  plugged %d   broke %d   net %+d   (moves changed at all: %d)"
      % (plugged, broke, plugged - broke, changed))
print()
