# -*- coding: utf-8 -*-
"""Emit a moves_dump-compatible corpus (fen,stratum) from a category slice of collapse_classified.csv.
  python diagnostics/make_slice_corpus.py diagnostics/collapse_classified.csv overpush diagnostics/overpush_corpus.csv
"""
import sys, csv
src, cat, out = sys.argv[1], sys.argv[2], sys.argv[3]
rows = [r for r in csv.DictReader(open(src)) if r.get("category") == cat]
with open(out, "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["fen", "stratum"])
    for r in rows:
        w.writerow([r["fen"], cat])
print(f"wrote {len(rows)} {cat} rows -> {out}")
