# -*- coding: utf-8 -*-
"""TEXEL FIT A2, stage 1: engine passes for a grid of PHASE definitions.

v2's phase runs linearly from EVAL_V2_EG_LIMIT to EVAL_V2_MG_LIMIT of total non-pawn material (search_engine.h);
neither limit has ever been fitted. Material decides phase, and phase decides how much of every mg and eg leg
applies -- the owner's "material count as an input feature" (2026-09-26). Changing the limits changes the
FIXED part of every position, so each setting needs its own engine pass; the Windows-side fitter
(_texel_pst_fit.py PASS=<dir>) then refits the PST per setting and the settings are compared on held-out loss.

Runs the passes sequentially in ONE process tree (one engine at a time), so it is a single prompt-free call:
  pyrun diagnostics/_texel_phase_grid.py [MG=50000,61700,72000] [EG=10000,15800,22000] [ROOT=/mnt/e/chess_data/texel]
The default setting (61700 / 15800) is skipped when its pass already exists (fit A's `pass/`).
"""
import os, sys, subprocess, time

KV = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
MGS = [int(x) for x in KV.get("MG", "50000,61700,72000").split(",")]
EGS = [int(x) for x in KV.get("EG", "10000,15800,22000").split(",")]
ROOT = KV.get("ROOT", "/mnt/e/chess_data/texel")
THIS = os.path.dirname(os.path.abspath(__file__))
PASS = os.path.join(THIS, "_texel_engine_pass.py")

t0 = time.time()
for mg in MGS:
    for eg in EGS:
        if mg <= eg:
            continue
        out = os.path.join(ROOT, "pass" if (mg, eg) == (61700, 15800) else "pass_mg%d_eg%d" % (mg, eg))
        for mode, zero in (("zero", "1"), ("full", "0")):
            target = os.path.join(out, "%s_0_of_1.csv" % mode)
            if os.path.exists(target):
                print("[grid] have %s" % target, flush=True)
                continue
            env = dict(os.environ, V2_PRESET="shipped", PST_V2_TAPERED="1", PST_V2_ZERO=zero,
                       EVAL_V2_MG_LIMIT=str(mg), EVAL_V2_EG_LIMIT=str(eg))
            r = subprocess.run([sys.executable, "-u", PASS, "MODE=" + mode, "SHARD=0", "NSHARD=1", "OUT_DIR=" + out],
                               env=env, capture_output=True, text=True)
            tail = [l for l in r.stderr.splitlines() if l.startswith("[pass") or "☠" in l][-1:]
            ok = r.returncode == 0 and os.path.exists(target)
            print("[grid %5.0fs] mg=%d eg=%d %s %s %s" % (time.time() - t0, mg, eg, mode, "OK" if ok else "FAILED",
                                                          tail[0] if tail else r.stderr[-300:]), flush=True)
            # Prove the limits reached the engine (the toggles banner does not print them): a non-default
            # setting must change the phase of some positions relative to the default pass.
            default_zero = os.path.join(ROOT, "pass", "zero_0_of_1.csv")
            if ok and mode == "zero" and (mg, eg) != (61700, 15800) and os.path.exists(default_zero):
                with open(target) as a, open(default_zero) as b:
                    pa = [l.rsplit(",", 1)[-1] for _, l in zip(range(20001), a)][1:]
                    pb = [l.rsplit(",", 1)[-1] for _, l in zip(range(20001), b)][1:]
                diff = sum(x != y for x, y in zip(pa, pb))
                print("[grid] liveness: phase differs from default on %d / %d rows%s"
                      % (diff, len(pa), "" if diff else "  ☠️ KNOB DID NOT REACH THE ENGINE"), flush=True)
print("[grid] DONE %.0fs" % (time.time() - t0), flush=True)
