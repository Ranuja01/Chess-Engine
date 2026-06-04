# -*- coding: utf-8 -*-
"""
Ablation sweep — localize WHICH pruning/reduction mechanism drops winning lines.

The resolution experiment (eval_at_resolution.py) exonerated the static eval and
pointed at the search: the winning lines are forcing sacrifices that LMR / futility
/ razoring / null-move prune before they resolve. This driver flips each mechanism
off (via the Config::ENABLE_* env-var toggles wired into the engine) and re-runs a
failing suite, so we can read off a position x combo matrix of who-solves-what.

Because the engine reads the toggles ONCE per process (a fresh ChessAI is built per
position in tactical_test), each combo must run in its OWN subprocess — this driver
spawns `tactical_test.py` per combo with the right environment, then reads the
per-position CSV it writes. The engine echoes the active toggles to stderr
("[toggles] ..."); we verify that line matches the intended combo so a typo'd var
name can't silently run baseline.

The sweep inherits whatever fixed depth the build is set to (currently 10) — there
is no depth argument here.

Combo sets:
  --quick   baseline + all_off + each single-off            (7 runs)
  (default) major: baseline + all_off + single-off + leave-one-in   (12 runs)
  --full    every subset of the 5 toggles                   (32 runs)

The all_off / leave-one-in combos blow up node counts; run the fast 4-probe suite
first, then point it at the bigger fail set.

Run (WSL, from NN Engine/):
    python ablation_sweep.py                       # 4 resolution probes, major combos
    python ablation_sweep.py --quick
    python ablation_sweep.py tactical_fails_deep12.epd
    python ablation_sweep.py tactical_fails_deep12.epd --full
"""

import os
import re
import csv
import sys
import subprocess
from timeit import default_timer as timer

import chess

# diagnostics layout: this tool + tactical_test.py live in diagnostics/, inputs in
# suites/, outputs in results/.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SUITES_DIR = os.path.join(THIS_DIR, 'suites')
RESULTS_DIR = os.path.join(THIS_DIR, 'results')

TOGGLES = ["LMR", "FUTILITY", "RAZORING", "NULLMOVE", "QDELTA"]

# The deep-miseval cluster + the trivial control (id, FEN, best move UCI), same set
# as eval_at_resolution / line_probe. Used to write a small default suite.
BASE_PROBES = [
    ("WAC.213", "3r1r1k/1b4pp/ppn1p3/4Pp1R/Pn5P/3P4/4QP2/1qB1NKR1 w - - 0 1", "h5h7"),
    ("WAC.204", "r1b1qrk1/1p3ppp/p1p5/3Nb3/5N2/P7/1P4PQ/K1R1R3 w - - 0 1", "e1e5"),
    ("WAC.283", "3q1rk1/4bp1p/1n2P2Q/3p1p2/6r1/Pp2R2N/1B4PP/7K w - - 0 1", "h3g5"),
    ("WAC.018", "R7/P4k2/8/8/8/8/r7/6K1 w - - 0 1", "a8h8"),
]
DEFAULT_SUITE = "resolution_probes.epd"


def write_probe_suite(path):
    """Write the 4 resolution probes as a proper EPD (bm in SAN, with id)."""
    lines = []
    for label, fen, uci in BASE_PROBES:
        board = chess.Board(fen)
        move = chess.Move.from_uci(uci)
        lines.append(board.epd(bm=[move], id=label))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def combo_env(off_set):
    """Environment for a combo: the named toggles OFF ("0"), the rest ON ("1")."""
    env = dict(os.environ)
    for t in TOGGLES:
        env[f"ENABLE_{t}"] = "0" if t in off_set else "1"
    return env


def build_combos(mode):
    """Return [(name, frozenset_of_OFF_toggles)] for the requested mode."""
    combos = [("baseline", frozenset())]
    if mode == "full":
        # every subset of the 5 toggles, by popcount then name
        from itertools import combinations
        seen = {frozenset()}
        for k in range(1, len(TOGGLES) + 1):
            for c in combinations(TOGGLES, k):
                fs = frozenset(c)
                if fs not in seen:
                    seen.add(fs)
                    name = "all_off" if len(fs) == len(TOGGLES) else "off_" + "_".join(sorted(fs))
                    combos.append((name, fs))
        return combos
    combos.append(("all_off", frozenset(TOGGLES)))
    for t in TOGGLES:                       # single-off
        combos.append((f"no_{t}", frozenset({t})))
    if mode == "major":
        for t in TOGGLES:                   # leave-one-in (all off except t)
            combos.append((f"only_{t}", frozenset(set(TOGGLES) - {t})))
    return combos


_TOGGLE_RE = re.compile(
    r"\[toggles\] LMR=(\d) FUTILITY=(\d) RAZORING=(\d) NULLMOVE=(\d) QDELTA=(\d)")


def verify_toggles(stderr, off_set):
    """Confirm the engine's echoed toggle line matches the intended combo."""
    m = _TOGGLE_RE.search(stderr or "")
    if not m:
        return None  # couldn't find it — engine may predate the toggles
    got = {t: (m.group(i + 1) == "1") for i, t in enumerate(TOGGLES)}
    want = {t: (t not in off_set) for t in TOGGLES}
    return got == want


def run_combo(epd, name, off_set):
    """Run tactical_test.py for one combo in a subprocess; return per-position dict."""
    tag = f"abl_{name}"
    t0 = timer()
    proc = subprocess.run(
        [sys.executable, os.path.join(THIS_DIR, "tactical_test.py"), epd, tag],
        env=combo_env(off_set), capture_output=True, text=True)
    dt = timer() - t0

    ok = verify_toggles(proc.stderr, off_set)
    if ok is False:
        sys.stderr.write(f"  !! {name}: engine toggle echo did NOT match the intended "
                         f"combo — results suspect.\n    stderr tail: {proc.stderr[-300:]}\n")
    elif ok is None:
        sys.stderr.write(f"  !! {name}: no [toggles] line from the engine — is the "
                         f"toggle build compiled?\n")

    csv_path = os.path.join(RESULTS_DIR, f"tactical_results_{tag}.csv")
    results = {}
    if not os.path.exists(csv_path):
        sys.stderr.write(f"  !! {name}: {csv_path} not written. stdout tail:\n"
                         f"{proc.stdout[-500:]}\n")
        return results, 0, 0, dt

    solved = total = 0
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            key = row.get("id") or f"#{row['idx']}"
            res = row["result"]
            results[key] = res
            if res != "BOOK":
                total += 1
                solved += int(res == "PASS")
    return results, solved, total, dt


def main():
    args = [a for a in sys.argv[1:]]
    mode = "major"
    if "--full" in args:
        mode = "full"; args.remove("--full")
    if "--quick" in args:
        mode = "quick"; args.remove("--quick")
    epd = None
    if args:
        if os.path.exists(args[0]):
            epd = args[0]
        elif os.path.exists(os.path.join(SUITES_DIR, args[0])):
            epd = os.path.join(SUITES_DIR, args[0])

    if epd is None:
        os.makedirs(SUITES_DIR, exist_ok=True)
        epd = os.path.join(SUITES_DIR, DEFAULT_SUITE)
        write_probe_suite(epd)
        print(f"No suite given — wrote {len(BASE_PROBES)} resolution probes to {epd}")

    combos = build_combos(mode)
    print(f"Ablation sweep over {epd}  |  {len(combos)} combos ({mode})  |  "
          f"inherits the build's fixed depth")
    print("NOTE: all_off / leave-one-in explode node counts — slower than baseline.\n")

    # position (row) x combo (column) matrix
    all_results = {}   # combo_name -> {pos_key: result}
    order = []         # preserve position order from the baseline run
    summary = []       # (name, solved, total, seconds)

    for name, off_set in combos:
        print(f"  running {name:<16} ({'all on' if not off_set else 'OFF: ' + ','.join(sorted(off_set))}) ...",
              flush=True)
        results, solved, total, dt = run_combo(epd, name, off_set)
        all_results[name] = results
        summary.append((name, solved, total, dt))
        for k in results:
            if k not in order:
                order.append(k)
        print(f"     -> {solved}/{total} solved  ({dt:.1f}s)")

    # ---- print the matrix (P = pass, . = fail, BK = book) ----
    def cell(res):
        return {"PASS": "P", "fail": ".", "BOOK": "BK"}.get(res, "?")

    combo_names = [c[0] for c in combos]
    w = max((len(k) for k in order), default=8)
    print("\n=== position x combo (P=pass .=fail BK=book) ===")
    header = " " * (w + 2) + "  ".join(f"{n[:9]:>9}" for n in combo_names)
    print(header)
    for k in order:
        row = "  ".join(f"{cell(all_results[n].get(k, '?')):>9}" for n in combo_names)
        print(f"{k:<{w}}  {row}")

    print("\n=== solved / total per combo ===")
    base = summary[0]
    for name, solved, total, dt in summary:
        delta = ""
        if name != "baseline":
            delta = f"  ({solved - base[1]:+d} vs baseline)"
        print(f"  {name:<16} {solved:>3}/{total:<3} {dt:>6.1f}s{delta}")

    # ---- which positions flipped fail->pass, and under which combo ----
    base_res = all_results["baseline"]
    flips = []
    for k in order:
        if base_res.get(k) == "fail":
            fixers = [n for n in combo_names if n != "baseline" and all_results[n].get(k) == "PASS"]
            if fixers:
                flips.append((k, fixers))
    print("\n=== baseline fails that some combo SOLVES (the localization signal) ===")
    if flips:
        for k, fixers in flips:
            print(f"  {k:<{w}}  solved by: {', '.join(fixers)}")
    else:
        print("  (none — no combo rescued a baseline failure in this suite)")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, f"ablation_sweep_{os.path.splitext(os.path.basename(epd))[0]}.csv")
    with open(out, "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["position"] + combo_names)
        for k in order:
            wtr.writerow([k] + [all_results[n].get(k, "") for n in combo_names])
        wtr.writerow([])
        wtr.writerow(["SOLVED"] + [f"{s}/{t}" for (_, s, t, _) in summary])
    print(f"\n-> {out}")
    print("Reading it: the all_off column splits pruning-bug (non-mate fails flip to "
          "PASS) from depth-bug (they don't). A single-off column that flips a fail "
          "names the responsible mechanism.")


if __name__ == "__main__":
    main()
