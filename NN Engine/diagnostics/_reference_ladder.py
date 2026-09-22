# -*- coding: utf-8 -*-
"""Generate the REFERENCE BENCH LADDER: every local Stockfish scored on OUR sts300 and wac suites.

WHY THIS EXISTS (2026-09-21, the owner's request). The project has quoted numbers like "STS 54.9% vs SF18
79.3% at equal depth" as its absolute anchor for months, but they were scattered across four generations of
measurement taken at DIFFERENT hash sizes and with different binaries (Windows .exe vs native ELF), and no
single document held them. Consequences on the record:
  - SF18 at d10 on sts300 reads 77.4% / 77.5% / 80.5% in three places and nobody knows which factor explains it.
  - WAC was only ever measured for SF18, once (2026-06-15). Our own docs claim v1 "beats SF11 on WAC" with
    NO number behind it anywhere.
  - SF16, SF17 and SF19 have never been benched at all.
This driver fixes the findability problem by running every engine under ONE pinned condition in one job.

☠️ READ THE NODES COLUMN OR DO NOT READ THE TABLE. At a fixed nominal DEPTH the engines do wildly different
amounts of work -- SF11 reaches depth 10 in ~26k nodes where we need ~249k. So the depth columns flatter
whoever prunes least, which is US, and "we beat SF at d10 on WAC" is mostly a statement about tree size.
The `--nodes` regime is the equal-WORK reading and it is the harsher, more honest one.

⚠️ SF1.1 / SF16 / SF17 ship here only as Windows .exe. Running those through WSL binfmt is recorded as
flaky ("Exec format error, whole runs void"), so their rows are marked UNTRUSTED and may simply fail.

  pyrun diagnostics/_reference_ladder.py [--depth 10] [--nodes N] [--limit N]
        [--engines sf11,sf18] [--suites sts300.epd,wac.epd] [--out ../dev_notes/REFERENCE-BENCH-LADDER.md]
"""
import os, sys, re, subprocess, argparse, datetime

THIS = os.path.dirname(os.path.abspath(__file__))
ENGINE_ROOT = os.path.dirname(THIS)
REF = os.path.join(THIS, "_sts_reference.py")

ALL_ENGINES = ["sf1", "sf11", "sf15c", "sf15n", "sf16", "sf17", "sf18", "sf19"]
NATIVE = {"sf11", "sf15c", "sf15n", "sf18", "sf19"}
PRETTY = {"sf1": "SF1.1 (2008)", "sf11": "SF11 (last full classical)", "sf15c": "SF15.1 classical",
          "sf15n": "SF15.1 NNUE", "sf16": "SF16", "sf17": "SF17", "sf18": "SF18", "sf19": "SF19"}

SCORE_RE = re.compile(r'^(STS|WAC) score:\s*(\d+)/(\d+)\s*\(([\d.]+)%\)')
NODES_RE = re.compile(r'^nodes/position:\s*([\d.]+)')


def run_one(engine, suite, depth, nodes, limit):
    cmd = [sys.executable, "-u", REF, "--engine", engine, "--epd", suite]
    if nodes:
        cmd += ["--nodes", str(nodes)]
    else:
        cmd += ["--depth", str(depth)]
    if limit:
        cmd += ["--limit", str(limit)]
    try:
        p = subprocess.run(cmd, cwd=ENGINE_ROOT, capture_output=True, text=True, timeout=7200)
    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT"}
    out = p.stdout or ""
    score = npos = None
    for line in out.splitlines():
        m = SCORE_RE.match(line.strip())
        if m:
            score = (int(m.group(2)), int(m.group(3)), float(m.group(4)))
        m = NODES_RE.match(line.strip())
        if m:
            npos = float(m.group(1))
    if score is None:
        # Keep the reason: a silent blank cell is the failure mode this table exists to prevent.
        tail = (p.stderr or out).strip().splitlines()
        return {"status": "FAILED", "why": (tail[-1][:120] if tail else "no output")}
    return {"status": "ok", "pts": score[0], "max": score[1], "pct": score[2], "nodes": npos}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--nodes", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--engines", default=",".join(ALL_ENGINES))
    ap.add_argument("--suites", default="sts300.epd,wac.epd")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    suites = [s.strip() for s in args.suites.split(",") if s.strip()]
    regime = ("nodes=%d" % args.nodes) if args.nodes else ("depth=%d" % args.depth)

    results = {}
    for eng in engines:
        for suite in suites:
            key = (eng, suite)
            sys.stderr.write("running %s on %s (%s) ...\n" % (eng, suite, regime))
            sys.stderr.flush()
            results[key] = run_one(eng, suite, args.depth, args.nodes, args.limit)
            r = results[key]
            sys.stderr.write("  -> %s\n" % (("%d/%d (%.1f%%) @ %.0f n/pos"
                                             % (r["pts"], r["max"], r["pct"], r["nodes"]))
                                            if r["status"] == "ok" else r["status"]))
            sys.stderr.flush()

    def cell(eng, suite):
        r = results.get((eng, suite))
        if not r:
            return "—"
        if r["status"] != "ok":
            return "☠️ %s" % r["status"]
        return "**%d/%d** (%.1f%%) · %s n/pos" % (r["pts"], r["max"], r["pct"], fmt_n(r["nodes"]))

    def fmt_n(n):
        return ("%.0fk" % (n / 1000.0)) if n >= 1000 else ("%.0f" % n)

    lines = []
    lines.append("| engine | native | %s | %s |" % tuple(s.replace(".epd", "") for s in (suites + ["—"])[:2]))
    lines.append("|---|---|---|---|")
    for eng in engines:
        lines.append("| %s | %s | %s | %s |"
                     % (PRETTY.get(eng, eng),
                        "yes" if eng in NATIVE else "⚠️ .exe",
                        cell(eng, suites[0]),
                        cell(eng, suites[1]) if len(suites) > 1 else "—"))
    table = "\n".join(lines)

    print("\n=== REFERENCE BENCH LADDER  (%s, hash=128, threads=1, no book/TB) ===\n" % regime)
    print(table)
    print("\n☠️ Read the n/pos column with every score: at fixed DEPTH the engines do different amounts of")
    print("   work, so the score alone flatters whoever prunes least.")

    if args.out:
        out = args.out if os.path.isabs(args.out) else os.path.join(ENGINE_ROOT, args.out)
        stamp = datetime.date.today().isoformat()
        with open(out, "a") as f:
            f.write("\n## %s @ %s (hash 128, 1 thread, no book/TB)\n\n" % (stamp, regime))
            f.write(table + "\n")
        print("\nappended -> %s" % out)


if __name__ == "__main__":
    main()
