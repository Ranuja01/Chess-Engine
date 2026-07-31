# -*- coding: utf-8 -*-
"""Causal test: on the ACTUAL collapse positions, does V3 move our eval toward or away from SF18?

Class-by-presence is base-rate contaminated (61% of BASE collapses already contain a passer), so it cannot
say whether passer evaluation CAUSED anything. This evaluates every collapse decision FEN under the current
knob config and dumps it; run twice (V3 off / V3 on) then compare with --compare, which adds SF18 static as
the reference and splits by passer presence.

  pyrun diagnostics/_collapse_v3_causal.py                      # dump for current config
  pyrun diagnostics/_collapse_v3_causal.py ENABLE_PASSER_V3=1   # dump for V3
  pyrun diagnostics/_collapse_v3_causal.py --compare            # report

Reads collapses from ALL four ledger dirs (union, deduped) so both arms' failures are judged by the same
yardstick -- a fix must help the positions it is blamed for AND not wreck the ones it already handled.
"""
import os, sys, csv, subprocess, re
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v
THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS))
os.chdir(os.path.dirname(THIS))
GAMES = os.path.join("selfplay", "games")
TAGS = ["ledger_base_s0", "ledger_v3_s0", "ledger_base_s1", "ledger_v3_s1"]

import chess

def collect():
    seen, out = set(), []
    for t in TAGS:
        p = os.path.join(GAMES, t, "collapses.csv")
        if not os.path.isfile(p):
            continue
        for r in csv.DictReader(open(p)):
            f = r.get("decision_fen")
            if f and f not in seen:
                seen.add(f)
                out.append((f, t))
    return out

def has_passer(b):
    for color in (chess.WHITE, chess.BLACK):
        them = not color
        for sq in b.pieces(chess.PAWN, color):
            f, r = chess.square_file(sq), chess.square_rank(sq)
            blocked = False
            for df in (-1, 0, 1):
                nf = f + df
                if not 0 <= nf <= 7: continue
                for esq in b.pieces(chess.PAWN, them):
                    if chess.square_file(esq) != nf: continue
                    er = chess.square_rank(esq)
                    if (er > r) if color == chess.WHITE else (er < r):
                        blocked = True; break
                if blocked: break
            if not blocked:
                return True
    return False

if "--compare" not in sys.argv:
    from ChessAI import ChessAI
    fens = collect()
    seed = chess.Board(fens[0][0])
    ai = ChessAI(None, None, seed, seed.turn)
    v3 = os.environ.get("ENABLE_PASSER_V3", "0")
    outp = os.path.join(THIS, f"_collapse_evals_v3_{v3}.csv")
    with open(outp, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["fen", "arm", "our_white_pawns", "has_passer"])
        for fen, arm in fens:
            b = chess.Board(fen)
            bd = ai.ev_breakdown(b)
            if bd.get("checkmate"): continue
            w.writerow([fen, arm, f"{-bd['total']/1000.0:.3f}", int(has_passer(b))])
    print(f"wrote {outp}  ({len(fens)} collapse FENs, ENABLE_PASSER_V3={v3})")
    sys.exit(0)

# ---- compare ----
SF = os.environ.get("STOCKFISH_PATH",
    "/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/stockfish_18_linux/stockfish-ubuntu-x86-64-avx2")
a = {r["fen"]: r for r in csv.DictReader(open(os.path.join(THIS, "_collapse_evals_v3_0.csv")))}
b_ = {r["fen"]: r for r in csv.DictReader(open(os.path.join(THIS, "_collapse_evals_v3_1.csv")))}
common = [f for f in a if f in b_]
print(f"collapse FENs evaluated under both configs: {len(common)}")

p = subprocess.Popen([SF], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=subprocess.STDOUT, text=True, bufsize=1)
def send(s): p.stdin.write(s + "\n"); p.stdin.flush()
send("uci"); send("isready")
while True:
    ln = p.stdout.readline()
    if not ln or ln.startswith("readyok"): break

def sf_eval(fen, depth=14):
    send("position fen " + fen); send(f"go depth {depth}")
    best = None
    while True:
        ln = p.stdout.readline()
        if not ln: break
        m = re.search(r"score cp (-?\d+)", ln)
        if m: best = int(m.group(1)) / 100.0
        if re.search(r"score mate (-?\d+)", ln):
            mm = int(re.search(r"score mate (-?\d+)", ln).group(1))
            best = 30.0 if mm > 0 else -30.0
        if ln.startswith("bestmove"): break
    stm_white = fen.split()[1] == "w"
    return best if stm_white else (-best if best is not None else None)

buckets = {(True,): [], (False,): []}
allrec = []
for fen in common:
    sf = sf_eval(fen)
    if sf is None: continue
    e0 = float(a[fen]["our_white_pawns"]); e1 = float(b_[fen]["our_white_pawns"])
    hp = a[fen]["has_passer"] == "1"
    err0, err1 = abs(e0 - sf), abs(e1 - sf)
    allrec.append((hp, err0, err1, e0, e1, sf, a[fen]["arm"]))
send("quit")

def rep(name, rec):
    if not rec:
        print(f"  {name:<22}(none)"); return
    m0 = sum(r[1] for r in rec) / len(rec); m1 = sum(r[2] for r in rec) / len(rec)
    better = sum(1 for r in rec if r[2] < r[1] - 1e-9)
    worse = sum(1 for r in rec if r[2] > r[1] + 1e-9)
    print(f"  {name:<22} n={len(rec):>4}  |err| {m0:>6.2f} -> {m1:>6.2f} ({m1-m0:+.2f})   "
          f"V3 closer: {better:>3}   V3 further: {worse:>3}   unchanged: {len(rec)-better-worse:>3}")

print("\nabs eval error vs SF18 d14 (pawns, White-POV) on COLLAPSE positions:")
rep("ALL", allrec)
rep("has passed pawn", [r for r in allrec if r[0]])
rep("no passed pawn", [r for r in allrec if not r[0]])
print("\n  'V3 further' on passer positions = V3 is CAUSING error where it acts.")
