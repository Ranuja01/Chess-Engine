# -*- coding: utf-8 -*-
"""Snapshot the vs_sf corpus before a re-run overwrites it.

The `vs_sf` tag is hardcoded `vssf_<elo>`, so a second run at the same strength REPLACES the games, the
PGNs and the pooled collapse dataset. Every earlier sample is then unrecoverable, which also destroys the
ability to compare a fresh corpus against the build it was mined on. Copy (not move): the live paths must
stay where the harness expects them, and the copy is the thing that survives.

  pyrun diagnostics/_preserve_vssf.py LABEL=<name> [TAG=vssf_2400]
"""
import os, sys, shutil

for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v

THIS = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(THIS)

TAG = os.environ.get("TAG", "vssf_2400")
LABEL = os.environ.get("LABEL", "")
if not LABEL:
    sys.exit("LABEL=<name> is required (it names the snapshot directory)")

GAMES = os.path.join(ENGINE, "selfplay", "games", TAG)
KS = os.path.join(THIS, "ks_sets")
DEST = os.path.join(ENGINE, "selfplay", "games", "_archive", LABEL)

# Refuse to write into an existing snapshot: a silent merge of two corpora would be worse than no
# snapshot at all, because nothing downstream could tell which games came from which build.
if os.path.exists(DEST):
    sys.exit("snapshot already exists, refusing to overwrite: %s" % DEST)

if not os.path.isdir(GAMES):
    print("  !! no games dir at %s -- nothing to preserve there" % GAMES)
os.makedirs(DEST, exist_ok=True)

copied = 0
if os.path.isdir(GAMES):
    shutil.copytree(GAMES, os.path.join(DEST, TAG))
    copied += sum(len(f) for _, _, f in os.walk(GAMES))

for name in ("collapse_dataset.csv", "collapse_dataset_classified.csv"):
    src = os.path.join(KS, name)
    if os.path.isfile(src):
        shutil.copy2(src, os.path.join(DEST, name))
        copied += 1
    else:
        print("  !! missing %s" % src)

print("\n  snapshot: %s" % DEST)
print("  files copied: %d\n" % copied)
