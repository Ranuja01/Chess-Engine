# -*- coding: utf-8 -*-
"""Snapshot ks_sets/collapse_dataset_classified.csv under a new suffix before re-pooling.

Why this exists: a second game run that REUSES a tag overwrites selfplay/games/<tag>/collapses.csv, so the
next collect_collapses.py silently REPLACES that family's rows in the pooled dataset. The first sample is
then unrecoverable (its PGNs are gone too). Take a snapshot first; the leverage/profile scripts can be
pointed at the copy by passing DATA=.

  pyrun diagnostics/_snapshot_classified.py SUFFIX=s1
"""
import os, sys, shutil

for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v

THIS = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(THIS, "ks_sets", "collapse_dataset_classified.csv")
SUFFIX = os.environ.get("SUFFIX", "s1")
DST = SRC.replace(".csv", "_%s.csv" % SUFFIX)

if not os.path.exists(SRC):
    sys.exit("missing: %s" % SRC)
if os.path.exists(DST):
    sys.exit("refusing to overwrite existing snapshot: %s" % DST)
shutil.copy2(SRC, DST)
print("snapshot %.1f KB -> %s" % (os.path.getsize(DST) / 1024.0, os.path.basename(DST)))
