"""Remap passer_corpus.csv (cols fen,sf18,our_total,wp,bp,phase,tier,src,over_read) into the _ks_fit_eval.py
schema (fen,target_ks,target_total,tier,phase_bucket,split) so the win%-MSE worker can read passer-themed rows
in a blended fit. target_total = sf18 (WHITE-POV pawns, the SF18 truth); target_ks = 0 (no per-term KS target on
passer rows -> they don't drive the KS DIRECTION metric, only the whole-eval win%-MSE, which is the point).
Deterministic 80/20 train/val split by FEN hash (no RNG). Writes ks_sets/passer_fit.csv.
Run: pyrun diagnostics/relabel_passer_fit.py
"""
import os, csv, zlib
THIS = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(THIS, "ks_sets", "passer_corpus.csv")
OUT = os.path.join(THIS, "ks_sets", "passer_fit.csv")
rows_in = list(csv.DictReader(open(SRC)))
cols = ["fen", "target_ks", "target_total", "our_total_base", "our_ks_base", "tier", "phase_bucket", "split"]
n = 0
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
    for r in rows_in:
        fen = r["fen"]
        try:
            tgt = float(r["sf18"])
        except (KeyError, ValueError):
            continue
        split = "val" if (zlib.crc32(fen.encode()) % 5 == 0) else "train"   # deterministic ~20% val
        w.writerow({"fen": fen, "target_ks": 0.0, "target_total": tgt, "our_total_base": r.get("our_total", 0.0),
                    "our_ks_base": 0.0, "tier": "passer_" + r.get("tier", "x"),
                    "phase_bucket": r.get("phase", "x"), "split": split})
        n += 1
print("wrote %d passer rows -> %s" % (n, OUT))
