# -*- coding: utf-8 -*-
"""PAWN-ENDING SEARCH-RULE ARMS (2026-10-15, POT brief §5.1) — inputs and read-out for queue #69.

  pyrun diagnostics/_pe_arms.py MODE=prep
      → /mnt/e/chess_data/pe/pe_in.csv     (fen, best_uci, best_cp, stratum): the 313 own-game pawn-only rows
        (ks_sets/pe_rows_sf18.csv, stratum "own") + EGX_SAMPLE pawn endings from EGX_sf18 (stratum "egx"); all SF18 d14.
      → /mnt/e/chess_data/pe/light_src_s{0,1,2}.csv (fen, phase_bucket): LIGHT endgames (pawns, no rooks / queens) from EGX for
        the SF18 multi-PV critical-point labels (_label_positions.py), split in three for three parallel labellers.
  pyrun diagnostics/_pe_arms.py MODE=read ARMS=base,g1,g2,gate,ext,all
      reads /mnt/e/chess_data/pe/d10_<arm>_s*of3.csv (our d10 search per arm, _depth_residual_pass.py) and prints per arm and
      stratum: win% MSE and side-ahead BIAS vs SF18 d14 (bias over |SF18| > 25 cp; − = too drawish), CHANGED-rate vs base (a
      rule that changes 0% of rows did not fire — the liveness check), and total nodes vs base.
  pyrun diagnostics/_pe_arms.py MODE=sf11 DEPTHS=6,8   — SF11 searches in the same file format (see sf11()).
"""
import os, sys, csv, glob, random
import numpy as np
import chess

KV = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
THIS = os.path.dirname(os.path.abspath(__file__))
D = "/mnt/e/chess_data/pe"
EGX_SAMPLE = int(KV.get("EGX_SAMPLE", "2000"))
LIGHT_N = int(KV.get("LIGHT_N", "6000"))
wp = lambda cp: 100.0 / (1.0 + np.exp(-0.00368208 * np.clip(np.asarray(cp, float), -1500, 1500)))


def pawn_only(b):
    return not (b.occupied & ~(b.pawns | b.kings)) and bool(b.pawns) and not b.is_check()


def prep():
    os.makedirs(D, exist_ok=True)
    out = []
    for r in csv.DictReader(open(os.path.join(THIS, "ks_sets/pe_rows_sf18.csv"), newline="")):
        out.append((r["fen"], "", r["best_cp"], "own"))
    own = {f for f, _, _, _ in out}
    egx = []
    light = []
    for r in csv.DictReader(open("/mnt/e/chess_data/retune/EGX_sf18.csv", newline="")):
        if not r.get("best_cp") or abs(float(r["best_cp"])) >= 20000:
            continue
        b = chess.Board(r["fen"])
        if b.is_game_over():
            continue
        if pawn_only(b) and r["fen"] not in own:
            egx.append((r["fen"], r["best_uci"], r["best_cp"], "egx"))
        if b.pawns and not (b.rooks | b.queens):
            light.append(r["fen"])
    random.seed(1015)
    random.shuffle(egx); random.shuffle(light)
    out += egx[:EGX_SAMPLE]
    with open(os.path.join(D, "pe_in.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["fen", "best_uci", "best_cp", "stratum"]); w.writerows(out)
    light = light[:LIGHT_N]
    for s in range(3):
        with open(os.path.join(D, "light_src_s%d.csv" % s), "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["fen", "phase_bucket"])
            w.writerows((f, "endgame") for f in light[s::3])
    print("PREP: pe_in.csv %d rows (own %d + egx %d) · light endgames %d → 3 shard files in %s"
          % (len(out), len(own), min(EGX_SAMPLE, len(egx)), len(light), D))


def read():
    tgt, strat = {}, {}
    for r in csv.DictReader(open(os.path.join(D, "pe_in.csv"), newline="")):
        tgt[r["fen"]] = float(r["best_cp"]); strat[r["fen"]] = r["stratum"]
    arms = KV.get("ARMS", "base,g1,g2,gate,ext,all").split(",")
    res = {}
    for a in arms:
        d = {}
        for p in glob.glob(os.path.join(D, "d10_%s_s*of3.csv" % a)):
            for r in csv.DictReader(open(p, newline="")):
                d[r["fen"]] = (float(r["ours_cp_white"]), int(r["nodes"]))
        res[a] = d
    base = res.get(arms[0], {})
    print("PAWN-ENDING ARMS — our d10 search vs SF18 d14 (bias over |SF18| > 25 cp; − = too drawish)")
    print("%-6s %-5s %6s %9s %8s %9s %10s" % ("arm", "set", "n", "win%MSE", "bias", "changed", "nodes/base"))
    for a in arms:
        for st in ("own", "egx", "all"):
            fens = [f for f in res[a] if f in tgt and (st == "all" or strat[f] == st) and f in base]
            if not fens:
                continue
            p = np.array([res[a][f][0] for f in fens]); t = np.array([tgt[f] for f in fens])
            mse = float(np.mean((wp(p) - wp(t)) ** 2))
            m = np.abs(t) > 25
            bias = float(np.mean((wp(p[m]) - wp(t[m])) * np.sign(t[m]))) if m.any() else float("nan")
            ch = np.mean([res[a][f][0] != base[f][0] for f in fens])
            nr = sum(res[a][f][1] for f in fens) / max(1, sum(base[f][1] for f in fens))
            print("%-6s %-5s %6d %9.1f %+8.2f %8.1f%% %10.3f" % (a, st, len(fens), mse, bias, 100 * ch, nr))


def sf11():
    """MODE=sf11 DEPTHS=6,8: SF11 (classical) search on pe_in.csv at each depth → d10_sf11d<N>_s0of3.csv in the depth-pass
    format (nodes 0), so MODE=read ARMS=base,sf11d6,... compares it with our arms (horizon vs constant-offset split)."""
    import chess.engine
    from _triangulate_sf11 import SF11
    rows = list(csv.DictReader(open(os.path.join(D, "pe_in.csv"), newline="")))
    eng = chess.engine.SimpleEngine.popen_uci(SF11, timeout=120)
    for dd in [int(x) for x in KV.get("DEPTHS", "10").split(",")]:
        out = os.path.join(D, "d10_sf11d%d_s0of3.csv" % dd)
        with open(out, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["fen", "ours_cp_white", "depth", "nodes"])
            for r in rows:
                sc = eng.analyse(chess.Board(r["fen"]), chess.engine.Limit(depth=dd))["score"].white().score(mate_score=5000)
                w.writerow([r["fen"], "%.1f" % max(-5000, min(5000, sc)), dd, 0])
        print("SF11 d%d: %d rows → %s" % (dd, len(rows), out), flush=True)
    eng.quit()


if __name__ == "__main__":
    {"prep": prep, "sf11": sf11}.get(KV.get("MODE"), read)()
