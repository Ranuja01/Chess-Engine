# -*- coding: utf-8 -*-
"""Validity check for the ledger runs: did --our-config actually reach the engine?

The engine prints a [toggles] line per process. If ENABLE_PASSER_V3 reads 0 in the V3 arm, the arms were
identical and the whole comparison is seed noise rather than a result.

Run: pyrun diagnostics/_ledger_validity.py
"""
import os, glob, re

THIS = os.path.dirname(os.path.abspath(__file__))
GAMES = os.path.join(os.path.dirname(THIS), "selfplay", "games")

for tag in ("ledger_base_s0", "ledger_v3_s0", "ledger_base_s1", "ledger_v3_s1"):
    d = os.path.join(GAMES, tag)
    errs = sorted(glob.glob(os.path.join(d, "*.stderr")))
    found = {}
    for fn in errs[:6]:
        try:
            txt = open(fn, errors="ignore").read()
        except Exception:
            continue
        for key in ("ENABLE_PASSER_V3", "ENABLE_PASSER_V2", "USE_OPENING_BOOK", "PRESET"):
            m = re.search(key + r"=(\S+)", txt)
            if m:
                found.setdefault(key, set()).add(m.group(1))
    if not errs:
        print(f"{tag:<18} (no .stderr files)")
        continue
    desc = "  ".join(f"{k}={'/'.join(sorted(v))}" for k, v in sorted(found.items()))
    print(f"{tag:<18} files={len(errs):>4}  {desc if desc else '(no toggles line found)'}")
