# -*- coding: utf-8 -*-
"""Does each knob actually MOVE THE ENGINE? — the liveness audit.

A knob can be dead in three different ways, and all three look identical from the outside:
  1. declared in Config but never READ by the eval/search   (EG_CLAMP_* — registered, echoed in the
     toggles dump, and appearing NOWHERE in cpp_bitboard.cpp)
  2. read, but only inside a branch that never executes at defaults
     (ENABLE_BISHOP_FWD_RANK_FIX — behind ENABLE_CHEAP_BISHOP_COMPLEX, which returns earlier)
  3. read and reached, but with values that make the change a no-op
     (ENABLE_CLOSEDNESS with all-zero tables was byte-identical to gate-off)

☠️ Why this matters more than it sounds: a coordinate descent that sweeps a DEAD knob records a null,
and that null is a statement about the WIRING, not about the mechanism. Both dead knobs above were
found by ACCIDENT during other work. This makes it systematic.

For each knob, evaluates N corpus positions at the DEFAULT and at a probe value in one process each
(knobs latch at engine init, so it must be one process per setting), then diffs. Reports the fraction of
positions that moved and the largest change.

  pyrun diagnostics/knob_liveness_audit.py [N=600] [KNOBS=A=1,B=0,...] [OUT=/tmp/audit.csv]

KNOBS is a comma-separated list of `NAME=probe_value`. With no list it audits the recently-added set.
⚠️ Interpreting a DEAD result: it means "this probe value changed nothing", not "the mechanism is
worthless". Try a second probe value before concluding — a gate whose weights are all zero is dead at
every gate setting but alive once the weights move.
"""
import os, sys, csv, subprocess

# ⚠️ BASE must be in this list. It was omitted once, so `BASE=ENABLE_WINNABILITY=1` was silently
# dropped and the gated weights were probed with their gate still OFF -- producing five false corpses.
# An audit tool that silently ignores an argument is worse than no audit.
for _a in sys.argv[1:]:
    if '=' in _a and _a.split('=', 1)[0] in ("N", "KNOBS", "OUT", "IN", "BASE"):
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v

THIS = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(THIS)
N = os.environ.get("N", "600")
OUT = os.environ.get("OUT", "/tmp/knob_audit.csv")

# Recently-added / never-formally-tested knobs. Probe values are chosen to be clearly OFF the default
# so a live knob cannot fail to move; a knob that does not move on THIS probe is reported DEAD and
# should get a second probe before it is believed.
DEFAULT_KNOBS = [
    # endgame clamps -- suspected entirely unwired
    ("EG_CLAMP_KNIGHT", "2000"), ("EG_CLAMP_BISHOP", "2000"),
    ("EG_CLAMP_ROOK", "2000"),   ("EG_CLAMP_QUEEN", "2000"),
    # endgame existence boosts (previously hardcoded literals, knob-ised recently)
    ("EG_EXIST_KNIGHT", "0"), ("EG_EXIST_BISHOP", "0"),
    ("EG_EXIST_ROOK", "0"),   ("EG_EXIST_QUEEN", "0"),
    # midgame clamps
    ("MG_CLAMP_KNIGHT", "1000"), ("MG_CLAMP_BISHOP_A", "1000"), ("MG_CLAMP_BISHOP_B", "1000"),
    # phase blend
    ("PHASE_BLEND_LO", "34"), ("PHASE_BLEND_RANGE", "24"),
    # the three never-game-tested mechanisms and their weights
    ("ENABLE_WINNABILITY", "1"), ("ENABLE_CLOSEDNESS", "1"), ("ENABLE_ENDGAME_SCALE", "1"),
    ("WINNAB_BASE", "300"), ("WINNAB_SCALE", "40"), ("WINNAB_TENSION", "50"),
    ("CLOSED_N4", "-40"), ("CLOSED_R4", "-120"), ("CLOSED_B4", "60"),
    # structural / passer knobs added in the same wave
    ("STRUCT_OPPOSED_MG_PCT", "40"), ("STRUCT_OPPOSED_EG_PCT", "40"),
    ("ENABLE_PASSER_DEFER_ON_FLAG", "1"), ("PASSER_R_MAX", "128"),
    # symmetry knobs from 2026-08-08 (expected LIVE; included as positive controls)
    ("KING_ZONE_SYM_MODE", "2"), ("ENABLE_CAPG_LVA_STATIC", "1"),
    ("ENABLE_CAPG_EVADE_POLARITY_FIX", "1"), ("ENABLE_BISHOP_FWD_RANK_FIX", "1"),
]


def parse_knobs():
    spec = os.environ.get("KNOBS")
    if not spec:
        return DEFAULT_KNOBS
    out = []
    for part in spec.split(","):
        part = part.strip()
        if part and "=" in part:
            k, v = part.split("=", 1)
            out.append((k.strip(), v.strip()))
    return out


# BASE is applied to BOTH arms. Essential for auditing a GATED weight: while its gate is off the weight
# is inert, so it reads DEAD for a reason that has nothing to do with wiring. That is the documented
# coordinate-descent trap, and an audit that ignores it manufactures false corpses.
#   pyrun ... BASE=ENABLE_WINNABILITY=1 KNOBS=WINNAB_BASE=300,WINNAB_SCALE=40
BASE = [tuple(p.split("=", 1)) for p in os.environ.get("BASE", "").split(",") if "=" in p]


def dump(path, extra_env):
    """One process per setting -- knobs latch at engine init, so this cannot be done in-process."""
    cmd = [sys.executable, "-u", os.path.join(THIS, "_eval_dump_simple.py"),
           "OUT=" + path, "N=" + N] + ["%s=%s" % kv for kv in BASE + list(extra_env)]
    subprocess.run(cmd, cwd=ENGINE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    try:
        return {r["fen"]: int(r["total"]) for r in csv.DictReader(open(path, newline=""))}
    except Exception:
        return {}


def main():
    knobs = parse_knobs()
    base = dump("/tmp/_audit_base.csv", [])
    if not base:
        print("baseline dump failed -- is the build current?")
        return
    print("baseline: %d positions\n" % len(base))
    print("  %-34s %-10s %8s %9s %10s   %s" % ("knob", "probe", "changed", "of", "max |d|", "verdict"))

    rows = []
    for name, val in knobs:
        cur = dump("/tmp/_audit_probe.csv", [(name, val)])
        if not cur:
            print("  %-34s %-10s %8s %9s %10s   %s" % (name, val, "-", "-", "-", "DUMP FAILED"))
            continue
        d = [abs(base[k] - cur[k]) for k in base if k in cur]
        nz = [x for x in d if x]
        verdict = "live" if nz else "☠️ DEAD on this probe"
        print("  %-34s %-10s %8d %9d %10d   %s"
              % (name, val, len(nz), len(d), max(nz) if nz else 0, verdict))
        rows.append({"knob": name, "probe": val, "changed": len(nz), "n": len(d),
                     "max_delta": max(nz) if nz else 0, "live": int(bool(nz))})

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["knob", "probe", "changed", "n", "max_delta", "live"])
        w.writeheader(); w.writerows(rows)

    dead = [r["knob"] for r in rows if not r["live"]]
    print("\n%d/%d live -> %s" % (sum(r["live"] for r in rows), len(rows), OUT))
    if dead:
        print("☠️ DEAD on their probe value: " + ", ".join(dead))
        print("   Before believing any of these, try a SECOND probe value -- a gate whose weights are")
        print("   all zero is dead at every gate setting but alive once the weights move.")


if __name__ == "__main__":
    main()
