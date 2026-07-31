# -*- coding: utf-8 -*-
"""Decisive fork for the KS/STS regression: on the STS positions where OUR recalibrated KS
(FLOOR=6 SAFE_CHECK=8 ATTACK_COUNT=2) FIRES, does SF11-static ALSO fire, or does SF11 correctly
stay quiet? If SF11 stays quiet where we fire, SF's DETECTION discriminates apply-vs-not better than
ours (architecture gap, not tuning). If SF11 also fires (or the divergence is some other term), it's
a different problem.

Runs with the ON config baked in (set before import). For each STS300 position: our ON-config KS,
our OFF-config KS (from ks_band_off.csv), SF11-static KS, and whether the STS move regressed
(sts_results_ksoff vs kson). Buckets the FIRING positions by what SF11 says.

  pyrun diagnostics/ks_overfire_vs_sf11.py
"""
import os, sys, csv, re, subprocess
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KS_FLOOR'] = '6'; os.environ['KS_SAFE_CHECK'] = '8'; os.environ['KS_ATTACK_COUNT'] = '2'

THIS = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(THIS, "results")
sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
SF11_PATH = "/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/stockfish_11/stockfish-11-win/Windows/stockfish_20011801_x64_bmi2.exe"
import chess
from ChessAI import ChessAI
ai = ChessAI(None, None, chess.Board(), True)


class SF11:
    def __init__(self, path):
        self.p = subprocess.Popen([path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.p.stdin.write("uci\n"); self.p.stdin.flush()
        for _ in range(200):
            if (self.p.stdout.readline() or "").strip() == "uciok":
                break

    def ks(self, fen):
        self.p.stdin.write("position fen %s\neval\nisready\n" % fen); self.p.stdin.flush()
        val = None
        while True:
            ln = self.p.stdout.readline()
            if not ln or ln.strip() == "readyok":
                break
            mm = re.match(r"\s*([A-Za-z ]+?)\s*\|.*\|.*\|\s*([-+]?\d+\.\d+|----)\s+([-+]?\d+\.\d+|----)\s*$", ln)
            if mm and mm.group(1).strip() == "King safety":
                try:
                    val = float(mm.group(2))
                except ValueError:
                    pass
        return val

    def close(self):
        try:
            self.p.stdin.write("quit\n"); self.p.stdin.flush()
        except Exception:
            pass


def load(name):
    d = {}
    with open(os.path.join(RES, name)) as f:
        for r in csv.DictReader(f):
            d[int(r["idx"])] = r
    return d


def si(x):
    try:
        return int(x)
    except (ValueError, TypeError):
        return None


band_off = load("ks_band_off.csv")
sts_off = load("sts_results_ksoff.csv")
sts_on = load("sts_results_kson.csv")
sf = SF11(SF11_PATH)

FIRE = 0.30      # |KS| we count as "fires"
QUIET = 0.30     # |SF11 KS| we count as "SF stays quiet"
rows = []
for idx in sorted(sts_off):
    fen = sts_off[idx]["fen"]
    bd = ai.ev_breakdown(chess.Board(fen))
    ks_on = -bd.get("king_safety", 0.0) / 1000.0
    ks_off = float(band_off[idx]["ks"]) if idx in band_off else 0.0
    sf_ks = sf.ks(fen)
    bs, ts = si(sts_off[idx]["score"]), si(sts_on[idx]["score"])
    dpts = (ts - bs) if (bs is not None and ts is not None) else None
    rows.append((idx, ks_off, ks_on, sf_ks, dpts, fen))
sf.close()

# Focus on positions where WE fire under the ON config.
fired = [r for r in rows if abs(r[2]) >= FIRE and r[3] is not None]
sf_quiet = [r for r in fired if abs(r[3]) < QUIET]
sf_fires_same = [r for r in fired if abs(r[3]) >= QUIET and (r[3] > 0) == (r[2] > 0)]
sf_fires_opp = [r for r in fired if abs(r[3]) >= QUIET and (r[3] > 0) != (r[2] > 0)]

print("=== Positions where OUR ON-config KS fires (|KS|>=%.2f): %d ===\n" % (FIRE, len(fired)))
print("SF11 buckets among those we fire on:")
print("  SF11 QUIET (|SF11_KS|<%.2f)      : %3d   <- SF discriminates 'do NOT apply KS' where we DO" % (QUIET, len(sf_quiet)))
print("  SF11 FIRES, SAME sign            : %3d   <- both see danger (agree)" % len(sf_fires_same))
print("  SF11 FIRES, OPPOSITE sign        : %3d   <- disagree on which king" % len(sf_fires_opp))


def netpts(group):
    return sum(r[4] for r in group if r[4] is not None)


print("\nNet STS points (on - off) contributed by each bucket:")
print("  SF11 QUIET      : %+d  (over-firing where SF stays silent)" % netpts(sf_quiet))
print("  SF11 SAME sign  : %+d" % netpts(sf_fires_same))
print("  SF11 OPP sign   : %+d" % netpts(sf_fires_opp))

print("\n--- WE FIRE, SF11 QUIET, and STS REGRESSED (the 'SF detection superior' set) ---")
print("%5s %8s %8s %8s %6s  fen" % ("idx", "KS_off", "KS_on", "SF11_KS", "dpts"))
for idx, ko, kon, sfk, dp, fen in sorted([r for r in sf_quiet if r[4] is not None and r[4] < 0],
                                         key=lambda r: r[4]):
    print("%5d %8.2f %8.2f %8.2f %6d  %s" % (idx, ko, kon, sfk, dp, fen))
