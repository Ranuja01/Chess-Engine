# -*- coding: utf-8 -*-
"""Merge the schema-compatible fit corpora into one, with the hygiene checks that make a blend trustworthy.

Four corpora already share the fit schema exactly
(`fen,target_ks,target_total,our_total_base,our_ks_base,tier,phase_bucket,split`) so they concatenate with
no conversion. Concatenating them NAIVELY would be wrong in four separate ways, each silent:

  1. STALE BASELINES. `our_total_base` is engine-dependent. `passer_fit` / `ks_sts_corpus` / `fit_corpus`
     predate the +45 Elo threats ship, so their baseline column describes an engine that no longer exists.
     This script cannot fix that -- it DETECTS and reports it, and refuses unless FORCE=1. The fix is to
     re-run refresh_bank_ours.py (or rebuild) so every row's baseline is the current engine.
  2. DUPLICATE FENs across sources, which silently weight those positions two or three times.
  3. SPLIT LEAKAGE. The same FEN landing in `train` in one corpus and `val` in another leaks the answer.
     Split is therefore RE-ASSIGNED deterministically by hashing the FEN, so a position is always on the
     same side no matter which corpus it came from.
  4. ACCIDENTAL COMPOSITION. diverse_corpus_wide is ~73% of the merged rows and would dominate the loss by
     size alone. Tier weighting is a DECISION -- `WEIGHTS=` makes it an explicit one. An optimum belongs to
     its corpus: a 29%->56% general-play shift once inverted a knob ranking outright.

⚠️ Blending changes the corpus, so every existing optimum becomes stale and `val` stops being comparable
with anything measured before. SNAPSHOT first (SNAPSHOT=1 writes .bak copies).

  pyrun diagnostics/blend_corpora.py [OUT=ks_sets/blended_corpus.csv] [VAL_PCT=22] [SNAPSHOT=1]
                                     [WEIGHTS=passer_fit:2,ks_sts_corpus:2] [FORCE=0] [DRY=1]
"""
import os, sys, csv, hashlib, shutil
from collections import defaultdict, Counter

for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v

THIS = os.path.dirname(os.path.abspath(__file__))
KS = os.path.join(THIS, "ks_sets")
OUT = os.environ.get("OUT", os.path.join(KS, "blended_corpus.csv"))
if not os.path.isabs(OUT):
    OUT = os.path.join(THIS, OUT)
VAL_PCT = int(os.environ.get("VAL_PCT", "22"))
SNAPSHOT = os.environ.get("SNAPSHOT", "1") == "1"
FORCE = os.environ.get("FORCE", "0") == "1"
DRY = os.environ.get("DRY", "0") == "1"

FIELDS = ["fen", "target_ks", "target_total", "our_total_base", "our_ks_base",
          "tier", "phase_bucket", "split"]

# name -> (path, pre-ship?) . The staleness flag is a FACT about when the file was built, not a guess.
SOURCES = [
    ("diverse_corpus_wide", os.path.join(KS, "diverse_corpus_wide.csv"), False),
    ("fit_corpus",          os.path.join(KS, "fit_corpus.csv"),          True),
    ("ks_sts_corpus",       os.path.join(KS, "ks_sts_corpus.csv"),       True),
    ("passer_fit",          os.path.join(KS, "passer_fit.csv"),          True),
]

WEIGHTS = {}
for kv in filter(None, os.environ.get("WEIGHTS", "").split(",")):
    k, v = kv.split(":")
    WEIGHTS[k] = int(v)


def split_of(fen):
    """Deterministic per-FEN split: identical FENs always land on the same side, whatever their source."""
    h = int(hashlib.sha1(fen.encode("utf-8")).hexdigest()[:8], 16)
    return "val" if (h % 100) < VAL_PCT else "train"


def main():
    seen, rows, per_src, dupes = {}, [], Counter(), 0
    stale = []
    for name, path, pre_ship in SOURCES:
        if not os.path.exists(path):
            print("  MISSING %-22s %s" % (name, path)); continue
        if pre_ship:
            stale.append(name)
        w = WEIGHTS.get(name, 1)
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                fen = (r.get("fen") or "").strip()
                if not fen or any(f not in r for f in FIELDS):
                    continue
                if fen in seen:
                    dupes += 1
                    continue                       # first source wins; never double-weight by accident
                seen[fen] = name
                r["split"] = split_of(fen)          # re-assigned, never inherited
                r["tier"] = r["tier"] or name
                for _ in range(w):
                    rows.append({k: r[k] for k in FIELDS})
                per_src[name] += 1

    print("\nBLEND SUMMARY")
    for name, _, _ in SOURCES:
        if per_src[name]:
            print("  %-22s %5d unique rows%s" % (name, per_src[name],
                                                 "  (x%d weight)" % WEIGHTS[name] if name in WEIGHTS else ""))
    print("  %-22s %5d" % ("duplicates dropped", dupes))
    print("  %-22s %5d rows total (after weighting)" % ("MERGED", len(rows)))
    tc = Counter(r["tier"] for r in rows); pc = Counter(r["phase_bucket"] for r in rows)
    sc = Counter(r["split"] for r in rows)
    print("  tiers:  %s" % dict(tc.most_common()))
    print("  phases: %s" % dict(pc.most_common()))
    print("  split:  %s" % dict(sc))
    top = tc.most_common(1)[0]
    print("  largest tier = %s at %.0f%% of rows%s"
          % (top[0], 100.0 * top[1] / len(rows),
             "  ⚠️ DOMINATES -- weight deliberately" if top[1] / len(rows) > 0.6 else ""))

    if stale:
        print("\n🚨 STALE BASELINES: %s predate the shipped engine, so `our_total_base` describes an engine"
              % ", ".join(stale))
        print("   that no longer exists. Re-run refresh_bank_ours.py before fitting, or pass FORCE=1.")
        if not FORCE:
            sys.exit("refusing to write a blend with stale baselines (FORCE=1 to override)")

    if DRY:
        print("\n(dry run — nothing written)"); return
    if SNAPSHOT:
        for _, path, _ in SOURCES:
            if os.path.exists(path) and not os.path.exists(path + ".bak"):
                shutil.copy2(path, path + ".bak")
        print("\n  snapshots written (.bak)")
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    print("  wrote %s" % OUT)
    print("\n⚠️ val is NOT comparable with any number measured on a different corpus. Re-derive optima.")


if __name__ == "__main__":
    main()
