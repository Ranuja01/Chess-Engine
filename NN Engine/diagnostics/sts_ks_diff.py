# -*- coding: utf-8 -*-
"""Per-position diff of two STS result CSVs (diagnostics/results/sts_results_<tag>.csv).

Purpose: localize WHY a KS (or any eval) config regresses the positional STS bench even though it
improves eval accuracy on the attack corpus. Joins base vs test on position idx, and reports:
  - net score delta + per-theme breakdown (where the regression concentrates)
  - every position whose move CHANGED, with score before/after and eval before/after
  - the REGRESSORS (score dropped) separated from the GAINERS (score rose)

Run (from NN Engine/):
  pyrun diagnostics/sts_ks_diff.py <base_tag> <test_tag>
e.g. pyrun diagnostics/sts_ks_diff.py ksoff kson
"""
import os, sys, csv

THIS = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(THIS, "results")


def load(tag):
    p = os.path.join(RES, f"sts_results_{tag}.csv")
    rows = {}
    with open(p) as f:
        for r in csv.DictReader(f):
            rows[int(r["idx"])] = r
    return rows


def as_int(s):
    try:
        return int(s)
    except (ValueError, TypeError):
        return None


def main():
    if len(sys.argv) < 3:
        print("usage: sts_ks_diff.py <base_tag> <test_tag>")
        return
    base_tag, test_tag = sys.argv[1], sys.argv[2]
    base, test = load(base_tag), load(test_tag)

    common = sorted(set(base) & set(test))
    theme_delta = {}
    changed, regressors, gainers = [], [], []
    base_total = test_total = 0

    for idx in common:
        b, t = base[idx], test[idx]
        bs, ts = as_int(b["score"]), as_int(t["score"])
        if bs is None or ts is None:      # book hit in one run -> skip
            continue
        base_total += bs
        test_total += ts
        theme = b["theme"]
        theme_delta[theme] = theme_delta.get(theme, 0) + (ts - bs)
        if b["engine"] != t["engine"]:
            row = {"idx": idx, "theme": theme, "id": b["id"],
                   "base_mv": b["engine"], "base_pts": bs,
                   "test_mv": t["engine"], "test_pts": ts,
                   "base_eval": b["eval"], "test_eval": t["eval"],
                   "max": b["max"], "fen": b["fen"]}
            changed.append(row)
            if ts < bs:
                regressors.append(row)
            elif ts > bs:
                gainers.append(row)

    print(f"=== STS diff: base={base_tag} ({base_total})  test={test_tag} ({test_total})  "
          f"delta={test_total - base_total} over {len(common)} positions ===\n")

    print("Per-theme delta (test - base), most-negative first:")
    for theme, d in sorted(theme_delta.items(), key=lambda kv: kv[1]):
        if d != 0:
            print(f"  {theme:<20} {d:+d}")
    print()

    print(f"Move CHANGED on {len(changed)} positions: "
          f"{len(regressors)} regressed, {len(gainers)} gained, "
          f"{len(changed) - len(regressors) - len(gainers)} score-neutral\n")

    regressors.sort(key=lambda r: r["test_pts"] - r["base_pts"])
    print("--- REGRESSORS (move changed, points LOST) ---")
    for r in regressors:
        print(f"[{r['idx']:>4}] {r['theme']:<16} {r['base_mv']}({r['base_pts']}/{r['max']}) "
              f"-> {r['test_mv']}({r['test_pts']})  eval {r['base_eval']}->{r['test_eval']}")
        print(f"        {r['fen']}")

    print("\n--- GAINERS (move changed, points GAINED) ---")
    gainers.sort(key=lambda r: r["base_pts"] - r["test_pts"])
    for r in gainers:
        print(f"[{r['idx']:>4}] {r['theme']:<16} {r['base_mv']}({r['base_pts']}/{r['max']}) "
              f"-> {r['test_mv']}({r['test_pts']})  eval {r['base_eval']}->{r['test_eval']}")


if __name__ == "__main__":
    main()
