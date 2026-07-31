"""Correction-history SIGNAL gate (Phase 0).

Reads [CORRLOG] records emitted by the C++ logger (ENABLE_CORRHIST_LOG) — one per update-eligible search node:
    [CORRLOG] <pawn_key> <maxbit> <static_eval> <best_score> <remaining_depth>
and answers the go/no-go question BEFORE we build the correction-history table:

    Does a per-(pawn-structure)-key correction, learned on a TRAIN split, actually reduce the static-eval error
    |staticEval - searchBestScore| on a HELD-OUT split?

If held-out MAE drops meaningfully, correction history has signal for us (build it). If it does not (because our
search AGREES with the static over-read — the eval-hole prediction), corrhist is structurally blocked and we
shelve it. Held-out (not train) is the honest metric: any per-key mean trivially reduces TRAIN error (overfit).

Correction learned per key = shrunk mean residual: sum(err)/(count + LAMBDA)  (shrinks low-count keys toward 0).
Usage:  python corrhist_signal.py <corrlog_file> [--lambda 16] [--seed 0]
        (accepts a file that may contain other lines; only [CORRLOG] lines are parsed)
"""
import sys, argparse, random

def parse(path):
    rows = []
    with open(path, errors="ignore") as f:
        for ln in f:
            if "[CORRLOG]" not in ln:
                continue
            try:
                _, rest = ln.split("[CORRLOG]", 1)
                pk, mb, se, bs, rd = rest.split()
                rows.append((int(pk), int(mb), int(se), int(bs), int(rd)))
            except ValueError:
                continue
    return rows

def mae(vals):
    return sum(abs(v) for v in vals) / len(vals) if vals else 0.0

def evaluate(rows, key_fn, lam, seed, label):
    rnd = random.Random(seed)
    idx = list(range(len(rows)))
    rnd.shuffle(idx)
    half = len(idx) // 2
    train_ids, hold_ids = idx[:half], idx[half:]

    # learn shrunk per-key correction on TRAIN: correction[key] = sum(err) / (count + lambda)
    agg = {}
    for i in train_ids:
        pk, mb, se, bs, rd = rows[i]
        err = bs - se
        k = key_fn(pk, mb, rd)
        s, c = agg.get(k, (0, 0))
        agg[k] = (s + err, c + 1)
    corr = {k: s / (c + lam) for k, (s, c) in agg.items()}

    base_err, corr_err, covered = [], [], 0
    for i in hold_ids:
        pk, mb, se, bs, rd = rows[i]
        err = bs - se
        base_err.append(err)
        k = key_fn(pk, mb, rd)
        c = corr.get(k, 0.0)
        if k in corr:
            covered += 1
        corr_err.append(err - c)
    b, a = mae(base_err), mae(corr_err)
    red = 100.0 * (b - a) / b if b else 0.0
    cov = 100.0 * covered / len(hold_ids) if hold_ids else 0.0
    print(f"  {label:<26} held-out MAE  base={b:8.1f}  corrected={a:8.1f}  reduction={red:+6.2f}%  "
          f"key-coverage={cov:5.1f}%  keys={len(corr)}")
    return red

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--lambda", dest="lam", type=float, default=16.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rows = parse(a.file)
    print(f"[corrhist_signal] parsed {len(rows)} CORRLOG rows from {a.file}  (lambda={a.lam}, seed={a.seed})")
    if len(rows) < 200:
        print("  too few rows for a meaningful split"); return
    def run_set(subset, label):
        if len(subset) < 400:
            print(f"  [{label}] too few rows ({len(subset)})"); return
        raw = mae([bs - se for _, _, se, bs, _ in subset])
        print(f"  [{label}]  n={len(subset)}  raw |best-static| MAE={raw:.1f}")
        pk_red = evaluate(subset, lambda pk, mb, rd: (mb, pk),      a.lam, a.seed, "  pawn x maxbit")
        ctl    = evaluate(subset, lambda pk, mb, rd: hash((pk*2654435761)&0xffff), a.lam, a.seed, "  control(random)")
        print(f"    -> NET signal (pawn x maxbit MINUS control) = {pk_red - ctl:+.2f}%  "
              f"({'SIGNAL' if pk_red - ctl >= 3 else 'weak/none'})")

    print("  FULL SET (all logged nodes):")
    evaluate(rows, lambda pk, mb, rd: pk,                          a.lam, a.seed, "pawn-only")
    r_pm = evaluate(rows, lambda pk, mb, rd: (mb, pk),             a.lam, a.seed, "pawn x maxbit")
    r_pmd= evaluate(rows, lambda pk, mb, rd: (mb, pk, min(rd, 8)), a.lam, a.seed, "pawn x maxbit x depth")
    r_ctl= evaluate(rows, lambda pk, mb, rd: hash((pk*2654435761)&0xffff), a.lam, a.seed, "control(random)")
    print(f"  -> FULL-SET NET (pawn x maxbit MINUS control) = {r_pm - r_ctl:+.2f}%")
    # QUIET subset: where corrhist actually operates (static eval trusted, disagreement structural not tactical).
    for thr in (300, 600):
        run_set([r for r in rows if abs(r[3] - r[2]) < thr], f"QUIET |err|<{thr}")
    # SHALLOW-RD subset: nodes near leaves where RFP/futility actually read the static eval.
    run_set([r for r in rows if r[4] <= 5], "RFP-range rd<=5")

if __name__ == "__main__":
    main()
