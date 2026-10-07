# -*- coding: utf-8 -*-
"""Build the POSITION-TYPE bench corpora for `_reference_ceiling.py` (owner, 2026-10-06: break the static-accuracy ladder
down by type — 960/variant vs classic, odds, K+P stress — not only phase). Output format = the ceiling corpora's:
fen, target_total (White-POV pawns, SF search), split (all `val`: pure measurement, nothing is fitted on these), type.
  bench_variant.csv  ← ks_sets/variant_regret_set.csv (960-style + piece-replacement arrays, SF d14 best_cp), N sampled
  bench_kp.csv       ← ks_sets/kp_stress_sf18.csv (K+pawns stress, SF18 d14; dense ≥ 14 pawns / pure)
  bench_odds.csv     ← selfplay/openings_odds.txt starts, LABELLED here with SF18 d14 (family kept as `type`)
  pyrun diagnostics/_bench_type_corpora.py [N=3000] [DEPTH=14]
"""
import os, sys, csv, random
import chess, chess.engine

KV = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
THIS = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(THIS)
KS = os.path.join(THIS, "ks_sets")
SF18 = os.environ.get("STOCKFISH_PATH", "/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/stockfish_18_linux/"
                                        "stockfish-ubuntu-x86-64-avx2")
N, DEPTH = int(KV.get("N", 3000)), int(KV.get("DEPTH", 14))
clip = lambda cp: max(-50.0, min(50.0, cp / 100.0))


def write(name, rows):
    with open(os.path.join(KS, name), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["fen", "target_total", "split", "type"]); w.writerows(rows)
    print("BENCH CORPUS %-18s %5d rows" % (name, len(rows)))


def main():
    rng = random.Random(1007)
    v = [r for r in csv.DictReader(open(os.path.join(KS, "variant_regret_set.csv"), newline="")) if r.get("best_cp")]
    v = rng.sample(v, min(N, len(v)))
    write("bench_variant.csv", [[r["fen"], "%.3f" % clip(float(r["best_cp"])), "val", "variant:" + r["phase_bucket"]] for r in v])
    k = [r for r in csv.DictReader(open(os.path.join(KS, "kp_stress_sf18.csv"), newline="")) if r.get("best_cp")]
    write("bench_kp.csv", [[r["fen"], "%.3f" % clip(float(r["best_cp"])), "val",
                            "kp:dense" if int(r["pawns"]) >= 14 else "kp:pure"] for r in k])
    starts = []
    for line in open(os.path.join(ENGINE, "selfplay", "openings_odds.txt")):
        if line.strip() and not line.startswith("#"):
            fen, _, fam = line.partition(";")
            starts.append((fen.strip(), fam.strip() or "odds"))
    e = chess.engine.SimpleEngine.popen_uci(SF18, timeout=120)
    rows = []
    for fen, fam in starts:
        b = chess.Board(fen)
        if b.is_game_over():
            continue
        cp = e.analyse(b, chess.engine.Limit(depth=DEPTH))["score"].white().score(mate_score=100000)
        rows.append([fen, "%.3f" % clip(cp), "val", "odds:" + fam])
    e.quit()
    write("bench_odds.csv", rows)


if __name__ == "__main__":
    main()
