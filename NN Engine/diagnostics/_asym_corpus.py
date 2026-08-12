# -*- coding: utf-8 -*-
"""Item 1 of the retune: build the SIGN-GATED, ASYMMETRIC, WIN%-IMPACT-WEIGHTED fit corpus.

Joins `diverse_corpus_wide.csv` (SF18-SEARCH truth in `target_total`, our current eval in `our_total_base`,
white-POV pawns, train/val split) with `sf11_static_labels.csv` (`sf11_static`, `sf18_static`) on FEN, then
classifies every row by the OWNER-SPEC gate + objective:

  sf18  = target_total            (SF18-SEARCH truth, already ceiling-capped to a static-achievable value)
  flip(a,b) = clear opposite sign beyond a NEAR_ZERO dead-zone  (so "directional agreement incl. near-0" KEEPS)

  LEARNABLE  (NOT flip(sf11, sf18))  -> a static eval CAN be right here; TRAIN/VAL row.
      asymmetric, strength-preserving target:
        |sf11-sf18| < |ours-sf18|  -> fit_target = sf11        (SF11 closer -> LEARN toward SF11)
        else                       -> fit_target = our_base    (WE are closer -> ANCHOR to current, preserve)
  GUARD      (flip(sf11,sf18) AND NOT flip(ours,sf18)) -> OUR positional edge (SF11 wrong, we are right).
      split := 'guard', fit_target = our_base. Held out, NEVER trained (fitting rows for already being right
      and then targeting our own output is self-confirming); a rising guard MSE = we regressed our edge.
  DROP       (both flipped) -> tactical: even SF11 is wrong-signed -> the truth is SEARCH, not static -> drop.

The asymmetric objective is delivered as DATA: `fit_target` is written into the `target_total` column the fit
worker (`_ks_fit_eval.py`) already descends, so the existing symmetric win%-MSE BECOMES the asymmetric target.
The only remaining worker change (item 2) is to multiply each row's loss by `weight`.

  weight = |winpct(our_base*100) - winpct(sf18*100)|   (win%-impact of our CURRENT error vs reachable truth;
           steep near 0 by the sigmoid, so a +3->0 error outweighs +10->+7, per spec)

⚠️ `our_total_base` was computed at DEFAULTS (bounded modes OFF = byte-identical), which IS the current
shipping eval and the correct anchor -- the retune runs with the modes ON, so preserving our_base on anchor/
guard rows is exactly "the re-shape must not change our good reads."

  pyrun diagnostics/_asym_corpus.py [WIDE=ks_sets/diverse_corpus_wide.csv]
        [LABELS=ks_sets/sf11_static_labels.csv] [OUT=ks_sets/diverse_corpus_asym.csv] [NEAR_ZERO=0.5]
"""
import os, sys, csv, math

for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v

THIS = os.path.dirname(os.path.abspath(__file__))


def _p(name, default):
    v = os.environ.get(name, default)
    return v if os.path.isabs(v) else os.path.join(THIS, v)


WIDE = _p("WIDE", "ks_sets/diverse_corpus_wide.csv")
LABELS = _p("LABELS", "ks_sets/sf11_static_labels.csv")
OUT = _p("OUT", "ks_sets/diverse_corpus_asym.csv")
Z = float(os.environ.get("NEAR_ZERO", "0.5"))


def winpct(cp):
    return 50.0 + 50.0 * (2.0 / (1.0 + math.exp(-0.00368208 * cp)) - 1.0)


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def flip(a, b):
    """Clear opposite sign beyond the dead-zone -> a genuine directional disagreement."""
    return (a > Z and b < -Z) or (a < -Z and b > Z)


# Join labels on FEN.
lab = {}
for r in csv.DictReader(open(LABELS, newline="")):
    f = (r.get("fen") or "").strip()
    if f:
        lab[f] = r

OUT_FIELDS = ["fen", "target_ks", "target_total", "sf18_truth", "our_total_base",
              "sf11_static", "sf18_static", "tier", "phase_bucket", "split", "row_class", "weight"]

rows_in = kept = 0
no_sf11 = bad_num = 0
cls_cnt = {"learn": 0, "anchor": 0, "guard": 0, "drop_tactical": 0}
split_cnt = {}
per_phase = {}
w_learn = []
out = []

for r in csv.DictReader(open(WIDE, newline="")):
    rows_in += 1
    fen = (r.get("fen") or "").strip()
    lr = lab.get(fen)
    sf11 = fnum(lr.get("sf11_static")) if lr else None
    sf18s = fnum(lr.get("sf18_static")) if lr else None
    sf18 = fnum(r.get("target_total"))          # SF18-SEARCH truth (ceiling-capped)
    ours = fnum(r.get("our_total_base"))
    if sf11 is None:                            # SF11 could not score (e.g. in-check) -> not gateable
        no_sf11 += 1
        continue
    if sf18 is None or ours is None:
        bad_num += 1
        continue

    if not flip(sf11, sf18):
        # LEARNABLE: choose the asymmetric target.
        if abs(sf11 - sf18) < abs(ours - sf18):
            row_class, fit_target = "learn", sf11
        else:
            row_class, fit_target = "anchor", ours
        split = r.get("split", "train")
    elif not flip(ours, sf18):
        # GUARD: our edge (SF11 flipped, we did not). Never trained.
        row_class, fit_target = "guard", ours
        split = "guard"
    else:
        cls_cnt["drop_tactical"] += 1
        continue

    weight = abs(winpct(ours * 100.0) - winpct(sf18 * 100.0))
    cls_cnt[row_class] += 1
    split_cnt[split] = split_cnt.get(split, 0) + 1
    ph = r.get("phase_bucket", "?")
    d = per_phase.setdefault(ph, {"learn": 0, "anchor": 0, "guard": 0})
    d[row_class] = d.get(row_class, 0) + 1
    if row_class == "learn":
        w_learn.append(weight)
    kept += 1
    out.append({
        "fen": fen, "target_ks": r.get("target_ks", "0.0"),
        "target_total": "%.4f" % fit_target,        # <-- the asymmetric target the worker descends
        "sf18_truth": "%.4f" % sf18, "our_total_base": "%.4f" % ours,
        "sf11_static": "%.4f" % sf11, "sf18_static": ("" if sf18s is None else "%.4f" % sf18s),
        "tier": r.get("tier", "diverse"), "phase_bucket": ph, "split": split,
        "row_class": row_class, "weight": "%.4f" % weight,
    })

with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=OUT_FIELDS)
    w.writeheader()
    w.writerows(out)

print("\n  WIDE=%s  (%d rows)" % (os.path.basename(WIDE), rows_in))
print("  joined labels: %d   dropped no-SF11(in-check/blank): %d   bad-number: %d" % (len(lab), no_sf11, bad_num))
print("  NEAR_ZERO dead-zone Z = %.2f pawn\n" % Z)
print("  kept %d rows -> %s\n" % (kept, os.path.basename(OUT)))
print("  row classes:")
print("    learn  (fit toward SF11)      : %5d" % cls_cnt["learn"])
print("    anchor (preserve our current) : %5d" % cls_cnt["anchor"])
print("    guard  (our edge, val-only)   : %5d" % cls_cnt["guard"])
print("    DROPPED tactical (both flip)  : %5d" % cls_cnt["drop_tactical"])
print("\n  split (what the worker will fit/report):")
for s in sorted(split_cnt):
    tag = "  <- TRAINED" if s == "train" else ("  <- held-out" if s == "val" else "  <- never trained")
    print("    %-6s %5d%s" % (s, split_cnt[s], tag))
if w_learn:
    ws = sorted(w_learn)
    print("\n  learn-row win%%-impact weight: median %.1f  p90 %.1f  max %.1f  (0..100 win%% pts)"
          % (ws[len(ws)//2], ws[int(0.9*len(ws))], ws[-1]))
print("\n  by phase        %6s %6s %6s" % ("learn", "anchor", "guard"))
for ph in sorted(per_phase):
    d = per_phase[ph]
    print("    %-12s %6d %6d %6d" % (ph, d.get("learn", 0), d.get("anchor", 0), d.get("guard", 0)))
print()
