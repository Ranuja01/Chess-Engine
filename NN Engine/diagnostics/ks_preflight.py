# -*- coding: utf-8 -*-
"""Pre-flight knob screen for the overnight KS levers: run ks_separation.py (FIRENEW up / SUPPRESS flat, the
noise-free discrimination objective) for each candidate config as a SEQUENTIAL subprocess (one core at a time),
so we pick the KS_WEAK value and confirm gentle-aim BEFORE spending any 76-min game-run. Kaufman is irrelevant to
the king_safety term itself, so separation is measured without it; the STS/WAC bench-guard (run separately, WITH
Kaufman on) is the shipped-context check. Prints a compact table."""
import os, sys, subprocess, re

THIS = os.path.dirname(os.path.abspath(__file__))
SEP = os.path.join(THIS, "ks_separation.py")
PY = sys.executable

CONFIGS = [
    ("baseline (KS_WEAK=2)",        []),
    ("KS_WEAK=3",                   ["KS_WEAK=3"]),
    ("KS_WEAK=4",                   ["KS_WEAK=4"]),
    ("aim gentle 1/2/3",            ["ENABLE_KS_AIM=1"]),
    ("aim 1/2/3 + KS_WEAK=3",       ["ENABLE_KS_AIM=1", "KS_WEAK=3"]),
]

pat = re.compile(r"fireold=([\d.]+) firenew=([\d.]+) suppress=([\d.]+) score=([-\d.]+)")
print("%-26s %8s %8s %9s %8s" % ("config", "fireold", "firenew", "suppress", "score"))
for name, knobs in CONFIGS:
    env = dict(os.environ)
    for kv in knobs:
        k, v = kv.split("=", 1); env[k] = v
    out = subprocess.run([PY, SEP] + knobs, capture_output=True, text=True, env=env).stdout
    m = pat.search(out)
    if m:
        fo, fn, su, sc = m.groups()
        print("%-26s %8s %8s %9s %8s" % (name, fo, fn, su, sc))
    else:
        print("%-26s  (no RESULT line -- stderr tail: %s)" % (name, out.strip()[-120:]))
