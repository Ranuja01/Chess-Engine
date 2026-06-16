# -*- coding: utf-8 -*-
"""Themed move-match harness for eval tuning — the inner-loop proxy.

Move-match (does the engine pick the suite's scored move after a shallow search) tracks Elo far better
than static eval-match, because it forces the tuned parameters through the engine's search and pruning.
This wraps the STS suite (15 themes, partial-credit c8/c9 scoring) with three things the tuning loop needs:
  - a THEME SUBSET so a hypothesis is tested only on its relevant themes (~100-300 positions, minutes),
  - a per-position CSV per run (candidate vs baseline are separate tags),
  - a DIFF that scores candidate-vs-baseline per theme and lists the positions whose move changed.

Candidate knobs are passed through the environment (the engine reads SCALE_*/PRESET/MAX_DEPTH/etc. in
initialize_engine), so the caller sets them; this script only runs and scores. Reuses sts_test.load_sts_epd
and tactical_test.run_one so the engine is invoked identically to the existing benches.

Run (from NN Engine/):
    PRESET=LIGHTNING MAX_DEPTH=10 USE_OPENING_BOOK=0 python diagnostics/movematch.py run base
    ... SCALE_LATENT_THREAT=144 python diagnostics/movematch.py run cand_lt --themes "Center Control,Advancement"
    python diagnostics/movematch.py diff base cand_lt
    python diagnostics/movematch.py themes
"""

import os
import csv
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SUITES_DIR = os.path.join(THIS_DIR, 'suites')
RESULTS_DIR = os.path.join(THIS_DIR, 'results')
DEFAULT_EPD = 'STS1-STS15_LAN_v3.epd'

from sts_test import load_sts_epd, _theme_of  # noqa: E402
from tactical_test import run_one             # noqa: E402

FIELDS = ["idx", "id", "theme", "engine", "score", "max", "eval", "depth", "time", "fen"]


def suite_path():
    p = os.path.join(SUITES_DIR, DEFAULT_EPD)
    return p if os.path.exists(p) else DEFAULT_EPD


def select_themes(positions, themes):
    """Keep positions whose theme contains any of the comma-separated needles (case-insensitive)."""
    if not themes:
        return positions
    needles = [t.strip().lower() for t in themes.split(",") if t.strip()]
    return [p for p in positions if any(n in p[3].lower() for n in needles)]


def results_csv(tag):
    return os.path.join(RESULTS_DIR, f"movematch_{tag}.csv")


def cmd_run(tag, themes):
    positions = load_sts_epd(suite_path())
    positions = select_themes(positions, themes)
    if not positions:
        print("no positions matched (themes=%r)" % themes)
        return 1

    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("movematch run tag=%s  positions=%d  themes=%s" % (tag, len(positions), themes or "ALL"))

    rows = []
    total = max_total = booked = 0
    theme_pts, theme_max = {}, {}
    for idx, (fen, score_map, mx, theme, epd_id) in enumerate(positions):
        r = run_one(fen, set())
        if r['booked']:
            booked += 1
            pts = None
        else:
            pts = score_map.get(r['uci'], 0)
            total += pts
            max_total += mx
            theme_pts[theme] = theme_pts.get(theme, 0) + pts
            theme_max[theme] = theme_max.get(theme, 0) + mx
        rows.append({"idx": idx, "id": epd_id, "theme": theme, "engine": r['uci'],
                     "score": ('' if pts is None else pts), "max": mx, "eval": r['eval'],
                     "depth": r['depth'], "time": round(r['time'], 3), "fen": fen})
        if (idx + 1) % 100 == 0:
            print("  %d/%d" % (idx + 1, len(positions)))

    with open(results_csv(tag), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    pct = (100.0 * total / max_total) if max_total else 0.0
    print("\nTOTAL %d/%d (%.1f%%)%s" % (total, max_total, pct,
          ("  [%d book]" % booked) if booked else ""))
    for theme in sorted(theme_pts):
        tm, tp = theme_max[theme], theme_pts[theme]
        print("  %-26s %4d/%-4d (%.0f%%)" % (theme, tp, tm, (100.0 * tp / tm) if tm else 0))
    print("results -> %s" % results_csv(tag))
    return 0


def _load_rows(tag):
    path = tag if os.path.exists(tag) else results_csv(tag)
    with open(path) as f:
        return {row["id"]: row for row in csv.DictReader(f)}


def cmd_diff(base_tag, cand_tag):
    base, cand = _load_rows(base_tag), _load_rows(cand_tag)
    ids = [i for i in base if i in cand]
    if not ids:
        print("no shared ids between %s and %s" % (base_tag, cand_tag))
        return 1

    theme_b, theme_c, theme_mx = {}, {}, {}
    changed = []
    bt = ct = mt = 0
    for i in ids:
        b, c = base[i], cand[i]
        if b["score"] == '' or c["score"] == '':  # book hit on either side
            continue
        theme = b["theme"]
        sb, sc, mx = int(b["score"]), int(c["score"]), int(b["max"])
        theme_b[theme] = theme_b.get(theme, 0) + sb
        theme_c[theme] = theme_c.get(theme, 0) + sc
        theme_mx[theme] = theme_mx.get(theme, 0) + mx
        bt += sb; ct += sc; mt += mx
        if b["engine"] != c["engine"]:
            changed.append((theme, i, b["engine"], sb, c["engine"], sc))

    print("movematch diff  base=%s  cand=%s  (n=%d)" % (base_tag, cand_tag, len(ids)))
    print("\nper-theme  (base -> cand / max,  delta):")
    for theme in sorted(theme_mx):
        db, dc, mx = theme_b[theme], theme_c[theme], theme_mx[theme]
        flag = "  <== REGRESS" if dc < db else ("  <== gain" if dc > db else "")
        print("  %-26s %4d -> %4d / %-4d  %+d%s" % (theme, db, dc, mx, dc - db, flag))
    print("\nTOTAL %d -> %d / %d  (%+d pts, %+.2f%%)" % (bt, ct, mt, ct - bt,
          100.0 * (ct - bt) / mt if mt else 0))
    print("moves changed: %d" % len(changed))
    for theme, i, bm, sb, cm, sc in changed[:40]:
        print("  %-22s %-28s %s(%d) -> %s(%d)  %+d" % (theme, i, bm, sb, cm, sc, sc - sb))
    if len(changed) > 40:
        print("  ... %d more" % (len(changed) - 40))
    return 0


def cmd_themes():
    positions = load_sts_epd(suite_path())
    counts = {}
    for _, _, _, theme, _ in positions:
        counts[theme] = counts.get(theme, 0) + 1
    print("STS themes (%d positions):" % len(positions))
    for theme in sorted(counts):
        print("  %-26s %d" % (theme, counts[theme]))
    return 0


def main():
    args = sys.argv[1:]
    cmd = args[0] if args else "themes"
    if cmd == "run":
        tag = args[1] if len(args) > 1 else "mm"
        themes = None
        if "--themes" in args:
            themes = args[args.index("--themes") + 1]
        return cmd_run(tag, themes)
    if cmd == "diff":
        if len(args) < 3:
            print("usage: movematch.py diff <base_tag> <cand_tag>")
            return 2
        return cmd_diff(args[1], args[2])
    if cmd == "themes":
        return cmd_themes()
    print("usage: movematch.py {run <tag> [--themes \"A,B\"] | diff <base> <cand> | themes}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
