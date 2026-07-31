# -*- coding: utf-8 -*-
"""SF18-validate the KS 'control' sets. The control_calm/eg positions were the OLD detector's must-suppress set,
but they are NOT ground truth. Run SF18-search on each: keep only the GENUINELY-safe ones (|SF18| small) as the
real must-not-over-fire guard; separately flag control positions SF18 says are ACTUALLY dangerous (there, our
new aim/coffin firing is CORRECT, not noise). This is the methodology: SF18-search is truth; SF11-static/old
controls only help us find where WE are absolutely wrong. Writes control_sf18safe.txt."""
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ.setdefault('KS_SAFE_CHECK_DEF', '5')
THIS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
import chess, chess.engine
sys.path.insert(0, os.path.join(os.path.dirname(THIS), "selfplay"))
from arbiter import find_stockfish

def load(name):
    p = os.path.join(THIS, "ks_sets", name); out = []
    if not os.path.exists(p): return out
    for ln in open(p):
        ln = ln.strip()
        if not ln or ln.startswith("#"): continue
        parts = ln.split(None, 1)
        try:
            float(parts[0]); out.append(parts[1] if len(parts) == 2 else ln)
        except ValueError:
            out.append(ln)
    return out

fens = load("control_calm.txt") + load("control_eg.txt")
sf18 = chess.engine.SimpleEngine.popen_uci(find_stockfish())
def s18(fen, d=18):
    b = chess.Board(fen); i = sf18.analyse(b, chess.engine.Limit(depth=d)); s = i["score"].white()
    return 99.0 if (s.is_mate() and s.mate() > 0) else (-99.0 if s.is_mate() else s.score() / 100.0)

safe, dangerous, mid = [], [], 0
for fen in fens:
    try: e = s18(fen)
    except Exception: continue
    if abs(e) < 0.75: safe.append(fen)
    elif abs(e) >= 1.5: dangerous.append((fen, e))
    else: mid += 1
sf18.quit()

out = os.path.join(THIS, "ks_sets", "control_sf18safe.txt")
with open(out, "w") as f:
    for fen in safe: f.write(fen + "\n")
print("control positions: %d   SF18-genuinely-safe (|e|<0.75): %d   SF18-actually-dangerous (|e|>=1.5): %d   mid: %d" % (
    len(fens), len(safe), len(dangerous), mid))
print("-> %s (the real must-not-over-fire guard)" % out)
print("\nControl positions SF18 says are ACTUALLY DANGEROUS (our KS firing here would be CORRECT, not noise):")
for fen, e in sorted(dangerous, key=lambda x: -abs(x[1]))[:12]:
    print("  SF18=%+.2f  %s" % (e, fen))
