# -*- coding: utf-8 -*-
"""Score several of our eval arms with _reference_ceiling.py's loss (win% MSE vs SF18 d14 search, train/val split)
on the SAME rows the reference engines are scored on, one process per arm (knobs latch at engine init).

_eval_accuracy_arms.py reads a `best_cp` column; the ceiling corpora (playdist_ceiling, diverse_corpus_wide) carry
`target_total` in pawns, so that tool reports "no rows" on them. This wrapper runs _reference_ceiling.py REFS=0 per
arm instead, so every number is directly comparable with the reference ladder rows.

  pyrun diagnostics/_accuracy_arm_grid.py CORPUS=diagnostics/ks_sets/playdist_ceiling.csv N=3000 \
        ARMS='v1:EVAL_ARM=0|shipped:V2_PRESET=shipped|...'
"""
import os, sys, subprocess, re

ARGS = {}
for a in sys.argv[1:]:
    k, _, v = a.partition("=")
    ARGS[k] = v
THIS = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(THIS)
corpus = ARGS.get("CORPUS", "diagnostics/ks_sets/diverse_corpus_wide.csv")
n = ARGS.get("N", "3000")
print("ARM ACCURACY vs SF18 search -- %s, %s rows (win%% MSE, lower is better)" % (os.path.basename(corpus), n))
print("  %-12s %10s %10s %8s" % ("arm", "train", "val", "n"))
for ent in ARGS.get("ARMS", "").split("|"):
    name, _, knobs = ent.partition(":")
    env = dict(os.environ, REFS="0", CORPUS=corpus, N=n, PRESET="LONG_FORMAT", USE_OPENING_BOOK="0")
    for kv in knobs.split():
        k, _, v = kv.partition("=")
        env[k] = v
    r = subprocess.run([sys.executable, "-u", os.path.join(THIS, "_reference_ceiling.py")], cwd=ENGINE, env=env,
                       capture_output=True, text=True)
    m = re.search(r"OURS \(current build\)\s+([\d.nan]+)\s+([\d.nan]+)\s+(\d+)", r.stdout)
    if m:
        print("  %-12s %10s %10s %8s" % (name.strip(), m.group(1), m.group(2), m.group(3)), flush=True)
    else:
        print("  %-12s FAILED: %s" % (name.strip(), (r.stdout + r.stderr)[-400:]), flush=True)
