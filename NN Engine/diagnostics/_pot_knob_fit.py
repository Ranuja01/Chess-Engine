# -*- coding: utf-8 -*-
"""POT knob fit, STATIC stage (2026-10-16; POT brief §5b) — class cells + race weights by exact EMULATION of the engine.

  MODE=dump   V2_PRESET=shipped pyrun diagnostics/_pot_knob_fit.py [OUT=/mnt/e/chess_data/pe/potfit_dump.csv]
  MODE=fit    pyrun diagnostics/_pot_knob_fit.py [IN=...dump] [KNOBS=PAWNEND,ONEMINOR,WIN_RACE,W_BEST,W_PROMO,W_TEMPI,W_DECISIVE]
  MODE=verify V2_PRESET=shipped <the knob env> pyrun diagnostics/_pot_knob_fit.py KV="PAWNEND=..,..."  (emulation == engine?)

dump: EGX rows (SF18 d14 best_cp) in the classes the knobs can touch — pure pawn endings and one-minor endings (no rook /
  queen, each side ≤ 1 minor) — every other row colour-mirrored. One engine pass with POT_V2_RACE=1 and zero weights (the eval
  is unchanged; the race features are published): total, winnability adjustment, phase, race features, class.
EMULATION of eval_v2 for any knob setting (pot_race_mp, then win_scale_adjust): T1 = (total − winnab) − Σ w·feature (pure pawn
  endings only); the shipped scale factor is RECOMPUTED unclamped from the board (64 + BASE + SP·leader pawns + OCB, gen-3
  values), the leader taken from T1's sign as the engine does; then f += class cell + WIN_RACE × the race margin in the
  leader's favour, clamped to [0, 64 + positive cell / race raise]; total = T1 + trunc(T1·(256 − phase)·(f − 64) / (256·64)).
  MODE=verify measures the residual against real engine evals (must be ~0).
fit: least squares on win% (SF18) through k = 0.00368208, 5-fold CV BY GAME, Nelder-Mead from 0, knobs rounded; prints the
  in-sample and held-out MSE per class vs the shipped engine, and the env string for the chosen setting.
"""
import os, sys, csv, math, zlib, random
for _a in sys.argv[1:]:
    if "=" in _a:
        _k, _v = _a.split("=", 1)
        os.environ[_k] = _v
THIS = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(THIS)
sys.path.insert(0, ENGINE); sys.path.insert(0, THIS)
import numpy as np

MODE = os.environ.get("MODE", "fit")
DUMP = os.environ.get("OUT", os.environ.get("IN", "/mnt/e/chess_data/pe/potfit_dump.csv"))
K = 0.00368208
RACE = ["best", "second", "promo", "break", "decisive", "over", "tempi", "stm", "contested"]
KNOBS = os.environ.get("KNOBS", "PAWNEND,ONEMINOR,WIN_RACE,W_BEST,W_PROMO,W_TEMPI,W_DECISIVE").split(",")


def cls_of(b):
    import chess
    if b.rooks | b.queens:
        return None
    wm = chess.popcount((b.knights | b.bishops) & b.occupied_co[chess.WHITE])
    bm = chess.popcount((b.knights | b.bishops) & b.occupied_co[chess.BLACK])
    if wm == 0 and bm == 0:
        return "pawn" if b.pawns else None
    return "minor" if wm <= 1 and bm <= 1 else None


def dump():
    os.environ.setdefault("PRESET", "LONG_FORMAT"); os.environ.setdefault("USE_OPENING_BOOK", "0")
    os.environ["POT_V2_RACE"] = "1"
    import chess, ChessAI
    ai = ChessAI.ChessAI(None, None, chess.Board(), True)
    games = {r["fen"]: r["game"] for r in csv.DictReader(open("/mnt/e/chess_data/retune/EGX_fens.csv", newline=""))}
    n = 0
    with open(DUMP, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["fen", "game", "cls", "sf_cp", "total", "winnab", "phase", "wpawns", "bpawns", "ocb", *RACE])
        for r in csv.DictReader(open("/mnt/e/chess_data/retune/EGX_sf18.csv", newline="")):
            if not r.get("best_cp") or abs(float(r["best_cp"])) >= 20000:
                continue
            b = chess.Board(r["fen"])
            if b.is_game_over():
                continue
            c = cls_of(b)
            if c is None:
                continue
            cp = float(r["best_cp"])
            if n % 2 == 1:
                b = b.mirror(); cp = -cp
            n += 1
            eb = ai.ev_breakdown(b)
            feats = [eb.get("pot_race_" + k, 0) for k in RACE]
            wb_, bb_ = b.bishops & b.occupied_co[chess.WHITE], b.bishops & b.occupied_co[chess.BLACK]
            ocb = 0
            if chess.popcount(wb_) == 1 and chess.popcount(bb_) == 1 and not (b.knights | b.rooks | b.queens):
                sw, sb = chess.lsb(wb_), chess.lsb(bb_)
                ocb = int((chess.square_file(sw) + chess.square_rank(sw)) % 2 != (chess.square_file(sb) + chess.square_rank(sb)) % 2)
            w.writerow([b.fen(), games.get(r["fen"], r["fen"]), c, "%.0f" % cp, eb["total"], eb.get("v2_winnab", 0),
                        eb.get("v2_phase256", 0), chess.popcount(b.pawns & b.occupied_co[chess.WHITE]),
                        chess.popcount(b.pawns & b.occupied_co[chess.BLACK]), ocb, *feats])
    print("DUMP: %d rows → %s" % (n, DUMP))


def load():
    R = list(csv.DictReader(open(DUMP, newline="")))
    D = {
        "game": np.array([r["game"] for r in R]),
        "pawn": np.array([r["cls"] == "pawn" for r in R]),
        "minor": np.array([r["cls"] == "minor" for r in R]),
        "race_on": np.array([any(int(r[k]) != 0 for k in RACE) or r["cls"] == "pawn" for r in R]),
        "y": np.array([100.0 / (1.0 + math.exp(-K * max(-1500, min(1500, float(r["sf_cp"]))))) for r in R]),
        "total": np.array([float(r["total"]) for r in R]),
        "winnab": np.array([float(r["winnab"]) for r in R]),
        "phase": np.array([float(r["phase"]) for r in R]),
        "F": np.array([[float(r[k]) for k in RACE] for r in R]),
        "wp": np.array([float(r["wpawns"]) for r in R]),
        "bp": np.array([float(r["bpawns"]) for r in R]),
        "ocb": np.array([float(r["ocb"]) for r in R]),
    }
    D["tpre"] = D["total"] - D["winnab"]
    return D


# The SHIPPED scale factor before its clamp (eval_v2 win_scale_adjust, gen 3: BASE −37, SP +34 per leader pawn, OCB −80; the
# other base cells and caps are 0). Recomputed from the board, not recovered from the published adjustment: the engine adds the
# new cells to the UNCLAMPED f (10-16 verify: recovering a clamped f0 left a 2,000 mp tail).
WIN_BASE, WIN_SP, WIN_OCB = -37.0, 34.0, -80.0


def emulate(D, kv, exact=True):
    g = lambda k: kv.get(k, 0.0)
    F = D["F"]
    wmap = {"W_BEST": 0, "W_SECOND": 1, "W_PROMO": 2, "W_BREAK": 3, "W_DECISIVE": 4, "W_OVER": 5, "W_TEMPI": 6, "W_STM": 7}
    race_w = sum(g(k) * F[:, i] for k, i in wmap.items())
    t1 = D["tpre"] - np.where(D["pawn"], race_w, 0.0)                      # Black-positive mp
    sp = np.where(t1 > 0, D["bp"], D["wp"])                                 # the leader's pawns (Black leads when t1 > 0)
    f = 64.0 + WIN_BASE + WIN_SP * sp + WIN_OCB * D["ocb"]
    fmax = np.full(len(f), 64.0)
    pe, om, wr = g("PAWNEND"), g("ONEMINOR"), g("WIN_RACE")
    f = f + np.where(D["pawn"], pe, 0.0) + np.where(D["minor"], om, 0.0)
    fmax = np.where(D["pawn"] & (pe > 0), 64 + pe, fmax)
    fmax = np.where(D["minor"] & (om > 0), 64 + om, fmax)
    lead = np.clip(np.where(t1 > 0, -F[:, 0], F[:, 0]), -3, 3)            # Black leads when t1 > 0
    raise_ = np.where(D["pawn"], wr * lead, 0.0)
    f = f + raise_
    fmax = np.where((raise_ > 0) & (64 + 3 * wr > fmax), 64 + 3 * wr, fmax)
    f = np.clip(f, 0, fmax)
    d = t1 * (256.0 - D["phase"]) * (f - 64.0) / (256.0 * 64.0)
    if exact:                                                              # the engine truncates; the FIT uses the smooth form
        d = np.trunc(d)
    d = np.where(t1 == 0, 0.0, d)
    return t1 + d


def wp(total_bp):
    cp = np.clip(-total_bp / 10.0, -1500, 1500)                            # Black-positive mp → White cp
    return 100.0 / (1.0 + np.exp(-K * cp))


def mse(D, kv, m, exact=True):
    return float(np.mean((wp(emulate(D, kv, exact))[m] - D["y"][m]) ** 2))


def fit():
    from scipy.optimize import minimize
    D = load()
    n = len(D["y"])
    fold = np.array([zlib.crc32(g.encode()) % 5 for g in D["game"]])
    base = {}
    print("POT knob fit — %d rows (pawn %d, one-minor %d) · knobs %s" % (n, D["pawn"].sum(), D["minor"].sum(), ",".join(KNOBS)))
    print("emulation at all-zero vs engine total: max |diff| %.1f mp (must be ~0)" % np.max(np.abs(emulate(D, base) - D["total"])))

    def solve(mask):
        obj = lambda x: mse(D, dict(zip(KNOBS, x)), mask, exact=False)
        sim = np.vstack([np.zeros(len(KNOBS)), 10.0 * np.eye(len(KNOBS))])     # whole-number knobs: start with steps of 10
        r = minimize(obj, np.zeros(len(KNOBS)), method="Nelder-Mead",
                     options={"maxiter": 4000, "xatol": 0.5, "fatol": 1e-3, "initial_simplex": sim})
        return dict(zip(KNOBS, np.round(r.x)))
    held = np.zeros(n); held0 = np.zeros(n)
    for k in range(5):
        tr, te = fold != k, fold == k
        kv = solve(tr)
        held[te] = (wp(emulate(D, kv))[te] - D["y"][te]) ** 2
        held0[te] = (wp(emulate(D, base))[te] - D["y"][te]) ** 2
    kv = solve(np.ones(n, bool))
    for name, m in (("all", np.ones(n, bool)), ("pawn", D["pawn"]), ("one-minor", D["minor"])):
        print("  %-9s held-out MSE shipped %.1f → fitted %.1f (%+.1f%%)   in-sample with final knobs %.1f"
              % (name, held0[m].mean(), held[m].mean(), 100 * (held[m].mean() / held0[m].mean() - 1), mse(D, kv, m)))
    env = {"PAWNEND": "POT_V2_WIN_PAWNEND", "ONEMINOR": "POT_V2_WIN_ONEMINOR", "WIN_RACE": "POT_V2_WIN_RACE"}
    print("FITTED:", " ".join("%s=%d" % (env.get(k, "POT_V2_RACE_" + k), v) for k, v in kv.items()), "POT_V2_RACE=1")
    print("KV for verify:", ",".join("%s:%d" % (k, v) for k, v in kv.items()))


def verify():
    os.environ.setdefault("PRESET", "LONG_FORMAT"); os.environ.setdefault("USE_OPENING_BOOK", "0")
    import chess, ChessAI
    D = load()
    kv = {k: float(v) for k, v in (x.split(":") for x in os.environ["KV"].split(","))}
    emu = emulate(D, kv)
    ai = ChessAI.ChessAI(None, None, chess.Board(), True)
    R = list(csv.DictReader(open(DUMP, newline="")))
    idx = random.Random(1016).sample(range(len(R)), min(3000, len(R)))
    diff = np.array([ai.ev_breakdown(chess.Board(R[i]["fen"]))["total"] - emu[i] for i in idx])
    print("VERIFY %s: %d rows · exact %.1f%% · |diff| median %.0f · 99th %.0f · max %.0f mp"
          % (kv, len(idx), 100 * np.mean(diff == 0), np.median(np.abs(diff)), np.percentile(np.abs(diff), 99), np.max(np.abs(diff))))


if __name__ == "__main__":
    {"dump": dump, "fit": fit, "verify": verify}[MODE]()
