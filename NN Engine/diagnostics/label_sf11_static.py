# -*- coding: utf-8 -*-
"""Label the fit corpus with SF11's STATIC eval — the retargeting pass.

Why: our fit objective is `(winpct(ours) - winpct(SF18_SEARCH))^2`, and that target is unreachable BY
CONSTRUCTION — SF18's own static eval misses it by 68.9, because the gap is search. Descending an
unreachable objective makes least-squares shrink toward the conditional mean, which is exactly the
flattening that cost 86 Elo in games (`sprt_noblend`).

SF11-classical is a hand-written STATIC function, so a hand-written static function can in principle
match it: the floor becomes 0 rather than 95, and the unrepresentable search component leaves the
gradient. This script produces the labels that retargeting needs.

It also emits SF18-static, because the pair supports the FILTER that makes the retarget trustworthy:
keep rows where SF11-static agrees directionally with the SF18-SEARCH target (proof the position's
truth is findable STATICALLY), and hold out rows where WE agree with SF18-search and SF11 does not
(our positional edge — a GUARD that must not regress, never a training label, since selecting rows for
already being right and then fitting to our own output is self-confirming).

  pyrun diagnostics/label_sf11_static.py [IN=ks_sets/diverse_corpus_wide.csv]
                                         [OUT=ks_sets/sf11_static_labels.csv] [N=0] [CKPT=200]

Resumable: re-running skips FENs already present in OUT, so a kill costs at most CKPT rows.
"""
import os, sys, csv, atexit

for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
THIS = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(THIS)
sys.path.insert(0, ENGINE); sys.path.insert(0, THIS)

IN = os.environ.get("IN", "ks_sets/diverse_corpus_wide.csv")
OUT = os.environ.get("OUT", "ks_sets/sf11_static_labels.csv")
if not os.path.isabs(IN):
    IN = os.path.join(THIS, IN)
if not os.path.isabs(OUT):
    OUT = os.path.join(THIS, OUT)
N = int(os.environ.get("N", "0"))
CKPT = int(os.environ.get("CKPT", "200"))

FIELDS = ["fen", "sf11_static", "sf18_static"]


def load_done():
    """Resume support: a kill mid-run must cost at most CKPT rows, not the whole pass."""
    if not os.path.exists(OUT):
        return {}
    done = {}
    try:
        for r in csv.DictReader(open(OUT, newline="")):
            if r.get("fen"):
                done[r["fen"]] = r
    except Exception:
        return {}
    return done


def main():
    rows = list(csv.DictReader(open(IN, newline="")))
    if N:
        rows = rows[:N]
    done = load_done()
    print("corpus %d rows, %d already labelled" % (len(rows), len(done)), flush=True)

    from eval_vs_sf11 import SF11Eval, SF11
    sf11 = SF11Eval(SF11)

    sys.path.insert(0, THIS)
    from _reference_ceiling import RawStatic
    sf18 = RawStatic(os.environ["STOCKFISH_PATH"], None)

    out_rows = list(done.values())

    def flush():
        tmp = OUT + ".tmp"
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            for r in out_rows:
                w.writerow({k: r.get(k, "") for k in FIELDS})
        os.replace(tmp, OUT)     # atomic: a kill during the write cannot corrupt the file

    atexit.register(flush)

    n_new = 0
    for i, r in enumerate(rows):
        fen = r.get("fen")
        if not fen or fen in done:
            continue
        try:
            a = sf11.eval(fen)
            a = a[0] if isinstance(a, (tuple, list)) else a
        except Exception:
            a = None
        try:
            b = sf18.ev(fen)
        except Exception:
            b = None
        out_rows.append({"fen": fen, "sf11_static": "" if a is None else a,
                         "sf18_static": "" if b is None else b})
        n_new += 1
        if n_new % CKPT == 0:
            flush()
            print("  labelled %d new (%d/%d scanned)" % (n_new, i + 1, len(rows)), flush=True)

    flush()
    print("DONE: %d new labels, %d total -> %s" % (n_new, len(out_rows), OUT), flush=True)


if __name__ == "__main__":
    main()
