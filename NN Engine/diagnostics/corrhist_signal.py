"""Correction-history SIGNAL gate (Phase 0).

Reads [CORRLOG] records emitted by the C++ logger (ENABLE_CORRHIST_LOG) — one per update-eligible search node.

    NEW format (10 fields):
        [CORRLOG] pawn minor major nonPawnW nonPawnB maxbit seval best rd failtype
    OLD format (5 fields, still parsed):
        [CORRLOG] pawn maxbit seval best rd

and answers the go/no-go question BEFORE we spend a games slot on correction history.

Two DIFFERENT questions are asked, and the distinction is the whole point:

  1. ORACLE  — does a per-key correction, learned on a TRAIN split, reduce |staticEval - searchBestScore|
     on a HELD-OUT split? This is an upper bound: it uses a converged, shrunk per-key mean.
  2. REPLAY  — does the ACTUAL table, with the ACTUAL update rule (integer EMA at CORR_SHIFT, clamped to
     CORR_MAX, applied at CORR_W/CORR_DIV, folded into CORR_SIZE entries), reduce that error when driven
     through the log IN ORDER, scoring each record against the table state BEFORE its own update?

The oracle can look healthy while the replay is flat — that gap is exactly the "our constants are wrong"
hypothesis, and it is the number the live mechanism can actually achieve. Decide on the REPLAY.

☠️ Run the engine with CORRHIST_LOG_STRIDE=1 for replay work. The replay sees only logged records, so a
stride >1 under-trains the simulated table relative to the real one and understates achievable reduction.

Usage:  python corrhist_signal.py <corrlog_file> [--lambda 16] [--seed 0]
"""
import sys, argparse, random

CORR_SIZE = 16384  # must match cache_management.h

# Column layout of the new record, by name -> index into the key tuple we keep.
KEY_NAMES = ["pawn", "minor", "major", "nonPawnW", "nonPawnB"]


def parse(path):
    """Returns rows of (keys_tuple, maxbit, seval, best, rd, failtype).

    keys_tuple always has 5 entries; the old 5-field format fills pawn and leaves the rest 0 (which the
    per-family comparison then reports as degenerate, rather than silently mixing formats)."""
    rows, old_fmt = [], False
    with open(path, errors="ignore") as f:
        for ln in f:
            if "[CORRLOG]" not in ln:
                continue
            parts = ln.split("[CORRLOG]", 1)[1].split()
            try:
                if len(parts) >= 10:
                    k = tuple(int(x) for x in parts[:5])
                    mb, se, bs, rd, ft = (int(x) for x in parts[5:10])
                elif len(parts) == 5:
                    old_fmt = True
                    pk, mb, se, bs, rd = (int(x) for x in parts)
                    k, ft = (pk, 0, 0, 0, 0), 0
                else:
                    continue
            except ValueError:
                continue
            rows.append((k, mb, se, bs, rd, ft))
    return rows, old_fmt


def mae(vals):
    return sum(abs(v) for v in vals) / len(vals) if vals else 0.0


# ---------------------------------------------------------------- oracle (held-out per-key mean)

def evaluate(rows, key_fn, lam, seed, label, keys=None):
    """keys, when given, supplies a precomputed per-record bucket (used for the position-INDEPENDENT
    random control, which cannot be expressed as a function of the position)."""
    rnd = random.Random(seed)
    idx = list(range(len(rows)))
    rnd.shuffle(idx)
    half = len(idx) // 2
    train_ids, hold_ids = idx[:half], idx[half:]

    agg = {}
    for i in train_ids:
        k_, mb, se, bs, rd, ft = rows[i]
        key = keys[i] if keys is not None else key_fn(k_, mb, rd)
        s, c = agg.get(key, (0, 0))
        agg[key] = (s + (bs - se), c + 1)
    corr = {k: s / (c + lam) for k, (s, c) in agg.items()}

    base_err, corr_err, covered = [], [], 0
    for i in hold_ids:
        k_, mb, se, bs, rd, ft = rows[i]
        err = bs - se
        base_err.append(err)
        key = keys[i] if keys is not None else key_fn(k_, mb, rd)
        c = corr.get(key, 0.0)
        if key in corr:
            covered += 1
        corr_err.append(err - c)
    b, a = mae(base_err), mae(corr_err)
    red = 100.0 * (b - a) / b if b else 0.0
    cov = 100.0 * covered / len(hold_ids) if hold_ids else 0.0
    print(f"  {label:<28} held-out MAE  base={b:8.1f}  corrected={a:8.1f}  reduction={red:+6.2f}%  "
          f"coverage={cov:5.1f}%  keys={len(corr)}")
    return red


# ---------------------------------------------------------------- replay (the real table dynamics)

def replay(rows, key_i, shift, cap, w, div, depth_weighted=False, guards=False, single_slot=False):
    """Drive the ACTUAL table through the log in order. Each record is scored against the table state
    BEFORE its own update (no self-prediction), which is what the live search sees."""
    table = [[0] * CORR_SIZE, [0] * CORR_SIZE]
    base, corrected, applied = [], [], 0
    for k_, mb, se, bs, rd, ft in rows:
        idx = 0 if single_slot else (k_[key_i] & (CORR_SIZE - 1))
        e = table[mb][idx]
        pred = (e * w) // div
        err = bs - se
        base.append(err)
        corrected.append(err - pred)
        if pred:
            applied += 1
        # Weiss-style exclusion of uninformative bound observations: a fail-high whose score never
        # exceeded the static eval says nothing about the eval's error.
        if guards and ft == 2 and bs <= se:
            continue
        r = err - e
        if depth_weighted:
            e += (r * min(rd, 8)) >> (shift + 3)   # == plain EMA at rd==8, gentler below
        else:
            e += r >> shift
        table[mb][idx] = max(-cap, min(cap, e))
    b, a = mae(base), mae(corrected)
    red = 100.0 * (b - a) / b if b else 0.0
    nz = sum(1 for t in table for v in t if v)
    return red, 100.0 * applied / len(rows), nz


def occupancy(rows, key_i):
    counts = {}
    for k_, mb, se, bs, rd, ft in rows:
        idx = (mb, k_[key_i] & (CORR_SIZE - 1))
        counts[idx] = counts.get(idx, 0) + 1
    v = sorted(counts.values())
    if not v:
        return
    def pct(p):
        return v[min(len(v) - 1, int(len(v) * p))]
    print(f"    distinct slots={len(v):6d} ({100.0*len(v)/(2*CORR_SIZE):4.1f}% of table)  "
          f"updates/slot p50={pct(0.50)} p90={pct(0.90)} p99={pct(0.99)} max={v[-1]}")
    conv = sum(1 for c in v if c >= 64)
    print(f"    slots with >=64 updates (EMA at shift 6 roughly converged): {conv} "
          f"({100.0*conv/len(v):.1f}% of touched slots)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--lambda", dest="lam", type=float, default=16.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rows, old_fmt = parse(a.file)
    print(f"[corrhist_signal] parsed {len(rows)} CORRLOG rows from {a.file}  (lambda={a.lam}, seed={a.seed})")
    if old_fmt:
        print("  ⚠️ OLD 5-field records present: only the pawn key is available; per-family results are void.")
    if len(rows) < 200:
        print("  too few rows for a meaningful split")
        return

    raw = mae([bs - se for _, _, se, bs, _, _ in rows])
    print(f"  raw |best-static| MAE = {raw:.1f}   "
          f"(fail-high {sum(1 for r in rows if r[5]==2)}, fail-low {sum(1 for r in rows if r[5]==1)}, "
          f"exact {sum(1 for r in rows if r[5]==0)})")

    # ---- 1. ORACLE: is there ANY predictable per-key structure? (upper bound)
    #
    # ☠️ THE BASELINE IS THE WHOLE BALLGAME. A shrunk per-key mean absorbs the GLOBAL residual bias, so on
    # skewed data ANY partition "reduces MAE" without using structure at all. Two honest baselines:
    #   GLOBAL  — one bucket for everything. This is what a single constant offset buys. Per-key keying is
    #             only worth building for the margin ABOVE this.
    #   RANDOM  — buckets assigned independently of the position, matched in count to the real key. Measures
    #             the shrunk-mean noise floor.
    # (The old control hashed the pawn key itself, so it was a COARSENED pawn key and retained real signal —
    # it flattered nothing and understated the net. Do not reintroduce it.)
    print("\n  [1] ORACLE — held-out per-key mean (upper bound on any scheme using this key)")
    gmean = sum(bs - se for _, _, se, bs, _, _ in rows) / len(rows)
    print(f"    global mean residual = {gmean:+.1f} mp  (a constant offset of this size is the thing to beat)")
    r_glob = evaluate(rows, lambda k, mb, rd: 0, a.lam, a.seed, "GLOBAL (1 bucket)")
    nkeys = len({(r[1], r[0][0]) for r in rows})
    rndb = random.Random(a.seed + 1)
    rand_keys = [rndb.randrange(nkeys) for _ in rows]
    r_rnd = evaluate(rows, None, a.lam, a.seed, f"RANDOM buckets (n={nkeys})", keys=rand_keys)
    evaluate(rows, lambda k, mb, rd: k[0], a.lam, a.seed, "pawn-only")
    r_pm = evaluate(rows, lambda k, mb, rd: (mb, k[0]), a.lam, a.seed, "pawn x maxbit")
    evaluate(rows, lambda k, mb, rd: (mb, k[0], min(rd, 8)), a.lam, a.seed, "pawn x maxbit x depth")
    print(f"  -> ORACLE NET vs GLOBAL = {r_pm - r_glob:+.2f}%   vs RANDOM = {r_pm - r_rnd:+.2f}%")
    print("     (NET vs GLOBAL is the number that justifies a TABLE instead of a constant.)")

    # ---- 2. Which KEY FAMILY carries the signal? (prices multi-keying with zero implementation)
    if not old_fmt:
        print("\n  [2] KEY FAMILY — oracle per structural key (answers multi-keying before building it)")
        for i, nm in enumerate(KEY_NAMES):
            evaluate(rows, lambda k, mb, rd, i=i: (mb, k[i]), a.lam, a.seed, nm)
        evaluate(rows, lambda k, mb, rd: (mb, k[0], k[1]), a.lam, a.seed, "pawn+minor")
        evaluate(rows, lambda k, mb, rd: (mb, k[0], k[1], k[2]), a.lam, a.seed, "pawn+minor+major")

    # ---- 3. OCCUPANCY: is any slot seeing enough updates for the EMA to converge?
    print("\n  [3] OCCUPANCY / SKEW (pawn key)")
    occupancy(rows, 0)

    # ---- 4. REPLAY: what the LIVE mechanism achieves with the real update rule and constants.
    print("\n  [4] REPLAY — actual table dynamics, scored before each record's own update")
    cur, ap_pct, nz = replay(rows, 0, 6, 2000, 192, 256)
    print(f"    CURRENT DEFAULTS (shift 6, cap 2000, w 192/256): reduction={cur:+6.2f}%  "
          f"correction applied on {ap_pct:.1f}% of records  nonzero slots={nz}")
    # Same replay, ONE slot: a running global offset with no keying at all. The keyed table has to beat this.
    print("    weight sweep (shift 6)   [GLOBAL = single-slot replay, i.e. no keying]:")
    for w in (32, 64, 96, 128, 192, 256):
        red, apct, _ = replay(rows, 0, 6, 2000, w, 256)
        gred, _, _ = replay(rows, 0, 6, 2000, w, 256, single_slot=True)
        print(f"      w={w:3d}/256 ({w/256:.3f})  keyed={red:+6.2f}%  GLOBAL={gred:+6.2f}%  "
              f"NET={red-gred:+6.2f}%  applied={apct:5.1f}%")
    print("    shift sweep (w 64/256):")
    for sh in (4, 5, 6, 7, 8):
        red, _, _ = replay(rows, 0, sh, 2000, 64, 256)
        print(f"      shift={sh}  reduction={red:+6.2f}%")
    print("    variants (shift 6, w 64/256):")
    red_d, _, _ = replay(rows, 0, 6, 2000, 64, 256, depth_weighted=True)
    print(f"      depth-weighted update      reduction={red_d:+6.2f}%")
    if not old_fmt:
        red_g, _, _ = replay(rows, 0, 6, 2000, 64, 256, guards=True)
        print(f"      + Weiss bound guards       reduction={red_g:+6.2f}%")
        red_dg, _, _ = replay(rows, 0, 6, 2000, 64, 256, depth_weighted=True, guards=True)
        print(f"      depth-weighted + guards    reduction={red_dg:+6.2f}%")
        print("    per-family replay (shift 6, w 64/256):")
        for i, nm in enumerate(KEY_NAMES):
            red, _, _ = replay(rows, i, 6, 2000, 64, 256)
            print(f"      {nm:<10} reduction={red:+6.2f}%")

    # ---- 5. Subsets where the correction actually gets read.
    print("\n  [5] SUBSETS (replay, shift 6, w 64/256)")
    for label, sub in (("QUIET |err|<300", [r for r in rows if abs(r[3] - r[2]) < 300]),
                       ("QUIET |err|<600", [r for r in rows if abs(r[3] - r[2]) < 600]),
                       ("RFP-range rd<=5", [r for r in rows if r[4] <= 5])):
        if len(sub) < 400:
            print(f"    [{label}] too few rows ({len(sub)})")
            continue
        red, apct, _ = replay(sub, 0, 6, 2000, 64, 256)
        print(f"    [{label}]  n={len(sub):7d}  replay reduction={red:+6.2f}%  applied={apct:5.1f}%")

    print("\n  ☠️ KILL CRITERIA (any one => lane dead, do NOT spend a games slot):")
    print("     - best REPLAY *NET vs GLOBAL* over the weight sweep < 5%")
    print("       (a keyed table that only matches the single-slot global offset is a CONSTANT, not a")
    print("        correction history — ship the constant, or nothing, but do not build the table)")
    print("     - ORACLE NET vs GLOBAL < 10%  [no predictable per-key structure beyond a constant]")
    print("     - if no family in [2] clears GLOBAL by a meaningful margin, multi-keying dies too and the")
    print("       whole corrhist family closes, not just the pawn table.")


if __name__ == "__main__":
    main()
