# -*- coding: utf-8 -*-
"""POT TRAJECTORY SET (owner, 2026-10-14): potential is value that is not yet KINETIC but becomes kinetic under good play ⇒
label a start position by what it TURNS INTO along best play, not only by its value now.

  V2_PRESET=shipped pyrun diagnostics/_build_trajectories.py [N=40000] [SHARD=i/n] [PLIES=8,16,24] [MOVE_NODES=50000]
        [LABEL_DEPTH=14] [CONTROL=0.3] [IN=/mnt/e/chess_data/retune/corpus_v2.csv] [OUT=/mnt/e/chess_data/traj/traj.csv]

START positions: own-play corpus rows (A / A2 / V1 / GAUNT, not mate-scored), middlegame → early endgame; ~70% chosen because a
system is UNRESOLVED (pawn tension · lever available · candidate majority with no passer yet · material imbalance · a king
not castled with queens on), ~30% uniformly at random (CONTROL — selection must not manufacture the signal).
ROLL-OUT: SF18 plays BOTH sides from the start at MOVE_NODES per move ("best engine play") up to max(PLIES) plies.
At t = 0 and every checkpoint: our static eval (ev_breakdown total → White cp), SF18 d LABEL_DEPTH score (White cp), and
resolution markers (material vs start, passers per side, pawn count, both kings' ranks, in-check, game over).
Rows: one per (start, checkpoint) — start_fen, game, start_stage, markers_at_start, k, fen_k, ours_static_k, sf_k, … so a fit
can target static(t+k) − static(t) (TRANSFORMATION our eval later sees) or sf(t+k) (the realised OUTCOME), then the depth target.
Resumable (starts already in OUT are skipped); SHARD splits the starts across parallel processes (one SF18 + our engine each).

MODE=eg (2026-10-15, POT brief §5.3/§5.5): the ENDGAME slice the mg set lacks. Starts = EGX endgame rows
(IN=/mnt/e/chess_data/retune/EGX_fens.csv, phase_bucket == endgame), chosen by ENDGAME unresolved markers (contested
opposition · a pawn race within one tempo · a breakthrough clash · unequal spare tempi) + the CONTROL share, and BALANCED on
side to move (half White, half Black to move) so a fit cannot learn a disguised tempo bonus. Same roll-out and row layout;
the marker columns are the eg ones. OUT default /mnt/e/chess_data/traj/traj_eg.csv.
"""
import os, sys, csv, random, time
for _a in sys.argv[1:]:
    if "=" in _a:
        _k, _v = _a.split("=", 1)
        os.environ[_k] = _v
THIS = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(THIS)
sys.path.insert(0, ENGINE); sys.path.insert(0, THIS); sys.path.insert(0, os.path.join(ENGINE, "selfplay"))
os.environ.setdefault("PRESET", "LONG_FORMAT"); os.environ.setdefault("USE_OPENING_BOOK", "0")
import chess
import chess.engine
import ChessAI
from arbiter import find_stockfish

N = int(os.environ.get("N", "40000"))
PLIES = [int(x) for x in os.environ.get("PLIES", "8,16,24").split(",")]
MOVE_NODES = int(os.environ.get("MOVE_NODES", "50000"))
LABEL_DEPTH = int(os.environ.get("LABEL_DEPTH", "14"))
CONTROL = float(os.environ.get("CONTROL", "0.3"))
MODE = os.environ.get("MODE", "mg")
IN = os.environ.get("IN", "/mnt/e/chess_data/retune/EGX_fens.csv" if MODE == "eg" else "/mnt/e/chess_data/retune/corpus_v2.csv")
OUT = os.environ.get("OUT", "/mnt/e/chess_data/traj/traj_eg.csv" if MODE == "eg" else "/mnt/e/chess_data/traj/traj.csv")
si, sn = (int(x) for x in os.environ.get("SHARD", "0/1").split("/"))
if sn > 1:
    OUT = OUT.replace(".csv", "_s%dof%d.csv" % (si, sn))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
MAT = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def material(b):
    return sum(v * (len(b.pieces(t, True)) - len(b.pieces(t, False))) for t, v in MAT.items())


def passers(b, c):
    n = 0
    for sq in b.pieces(chess.PAWN, c):
        f, r = chess.square_file(sq), chess.square_rank(sq)
        ahead = range(r + 1, 8) if c else range(0, r)
        if not any(b.piece_at(chess.square(ff, rr)) == chess.Piece(chess.PAWN, not c)
                   for ff in (f - 1, f, f + 1) if 0 <= ff < 8 for rr in ahead):
            n += 1
    return n


def unresolved(b):
    """Markers of systems that are not yet kinetic (each 0/1)."""
    tension = int(any(b.piece_at(t) == chess.Piece(chess.PAWN, not c)
                      for c in (True, False) for s in b.pieces(chess.PAWN, c) for t in b.attacks(s)))
    lever = 0
    for m in b.legal_moves:
        p = b.piece_at(m.from_square)
        if p and p.piece_type == chess.PAWN and not b.is_capture(m):
            b.push(m)
            if any(b.piece_at(t) == chess.Piece(chess.PAWN, not p.color) for t in b.attacks(m.to_square)):
                lever = 1
            b.pop()
            if lever:
                break
    files = {c: [chess.square_file(s) for s in b.pieces(chess.PAWN, c)] for c in (True, False)}
    major = 0
    for c in (True, False):
        for wing in (range(0, 4), range(4, 8)):
            if sum(f in wing for f in files[c]) > sum(f in wing for f in files[not c]) and passers(b, c) == 0:
                major = 1
    sets = [tuple(len(b.pieces(t, c)) for t in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)) for c in (True, False)]
    imbal = int(sets[0] != sets[1])
    king = int(bool(b.pieces(chess.QUEEN, True) or b.pieces(chess.QUEEN, False)) and
               any(b.king(c) is not None and chess.square_rank(b.king(c)) == (0 if c else 7) and
                   chess.square_file(b.king(c)) in (3, 4) for c in (True, False)))
    return [tension, lever, major, imbal, king]


MG_MARKERS = ["tension", "lever", "majority", "imbalance", "king_center"]
EG_MARKERS = ["opposition", "race", "breakthrough", "tempi_unequal"]


def spare_tempi(b, c):
    """Pawn moves side c could make that neither capture nor walk into an enemy pawn's attack (a cheap reserve-tempo
    count: the moves that pass the turn in a zugzwang)."""
    n, occ = 0, b.occupied
    for s in b.pieces(chess.PAWN, c):
        t = s + 8 if c else s - 8
        if not 0 <= t < 64 or occ & (1 << t):
            continue
        if not (b.attackers_mask(not c, t) & b.pieces_mask(chess.PAWN, not c)):
            n += 1
    return n


def unresolved_eg(b):
    """ENDGAME markers of systems that are not yet kinetic (each 0/1). Cheap structure reads, not verdicts."""
    wk, bk = b.king(True), b.king(False)
    pawns = b.pieces_mask(chess.PAWN, True) | b.pieces_mask(chess.PAWN, False)
    if wk is None or bk is None:
        return [0, 0, 0, 0]
    df, dr = abs(chess.square_file(wk) - chess.square_file(bk)), abs(chess.square_rank(wk) - chess.square_rank(bk))
    # Opposition and reserve tempi decide only where zugzwang can: no queens or rooks (pawn / minor endings).
    light = not (b.pieces_mask(chess.QUEEN, True) | b.pieces_mask(chess.QUEEN, False) |
                 b.pieces_mask(chess.ROOK, True) | b.pieces_mask(chess.ROOK, False))
    # direct / distant opposition: kings on a line (file, rank or diagonal) an EVEN number of squares apart
    on_line = df == 0 or dr == 0 or df == dr
    opposition = int(light and bool(pawns) and on_line and max(df, dr) in (2, 4, 6))
    # a race within one tempo: a TRUE passed pawn's distance to promotion vs the enemy king's distance to that square
    race = 0
    for c in (True, False):
        ek = bk if c else wk
        for s in b.pieces(chess.PAWN, c):
            f, r = chess.square_file(s), chess.square_rank(s)
            ahead = range(r + 1, 8) if c else range(0, r)
            if any(b.piece_at(chess.square(ff, rr)) == chess.Piece(chess.PAWN, not c)
                   for ff in (f - 1, f, f + 1) if 0 <= ff < 8 for rr in ahead):
                continue
            q = chess.square(f, 7 if c else 0)
            steps = (7 - r) if c else r
            if abs(chess.square_distance(ek, q) - steps) <= 1:
                race = 1
    # breakthrough clash: >= 2 blocked pawn pairs (a pawn directly facing an enemy pawn) on adjacent files
    blocked = sorted(chess.square_file(s) for s in b.pieces(chess.PAWN, True)
                     if s + 8 < 64 and b.piece_at(s + 8) == chess.Piece(chess.PAWN, False))
    breakthrough = int(any(blocked[i + 1] - blocked[i] == 1 for i in range(len(blocked) - 1)))
    tempi_unequal = int(light and spare_tempi(b, True) != spare_tempi(b, False))
    return [opposition, race, breakthrough, tempi_unequal]


def marks(b, start_mat):
    return [material(b) - start_mat, passers(b, True), passers(b, False), len(b.pieces(chess.PAWN, True)) + len(b.pieces(chess.PAWN, False)),
            chess.square_rank(b.king(True)) if b.king(True) is not None else -1,
            chess.square_rank(b.king(False)) if b.king(False) is not None else -1, int(b.is_check()), int(b.is_game_over())]


def main():
    rows = []
    if MODE == "eg":
        # EGX carries no split: the split is assigned by game at corpus assembly, so record the game id only.
        for r in csv.DictReader(open(IN, newline="")):
            if r["phase_bucket"] == "endgame":
                rows.append((r["fen"], r["game"], "eg", ""))
    else:
        for r in csv.DictReader(open(IN, newline="")):
            if r["slice"] in ("A", "A2", "V1", "GAUNT") and r["mate"] != "1" and r["stage"] in ("mg", "mgl", "egl") and r["split"] != "excluded":
                rows.append((r["fen"], r["group"], r["stage"], r["split"]))
    random.seed(1014)
    random.shuffle(rows)
    markers_fn = unresolved_eg if MODE == "eg" else unresolved
    # choose starts: unresolved-system positions + a uniform CONTROL share (deterministic across shards)
    chosen, n_ctl = [], int(N * CONTROL)
    if MODE == "eg":
        # Side-to-move BALANCE by exact COLOUR MIRROR: EGX is ~96% White to move, so every other chosen start is
        # replaced by board.mirror() (same position, colours swapped; our eval and SF are colour-symmetric). The
        # mover-vs-opponent balance of WHOSE potential is realised is only known after the roll-out ⇒ fit weighting.
        n_sel = 0
        for fen, g, st, sp in rows:
            if len(chosen) >= N:
                break
            b = chess.Board(fen)
            if b.is_game_over():
                continue
            ctl = 1 if len(chosen) < n_ctl else 0
            if not ctl and sum(markers_fn(b)) < 1:
                continue
            if n_sel % 2 == 1:
                b = b.mirror()
            n_sel += 1
            chosen.append((b.fen(), g, st, sp, ctl))
    else:
        for fen, g, st, sp in rows:
            if len(chosen) >= N:
                break
            if len(chosen) < n_ctl:
                chosen.append((fen, g, st, sp, 1))
                continue
            if sum(unresolved(chess.Board(fen))) >= 1:
                chosen.append((fen, g, st, sp, 0))
    chosen = chosen[si::sn]
    done = set()
    if os.path.exists(OUT):
        for r in csv.DictReader(open(OUT, newline="")):
            done.add(r["start_fen"])
    new = not os.path.exists(OUT)
    fh = open(OUT, "a", newline="")
    w = csv.writer(fh)
    if new:
        w.writerow(["start_fen", "game", "start_stage", "split", "control", *(EG_MARKERS if MODE == "eg" else MG_MARKERS),
                    "k", "fen_k", "ours_static_k", "sf_k", "mat_change", "passers_w", "passers_b", "pawns", "wk_rank", "bk_rank",
                    "in_check", "game_over"])
    ai = ChessAI.ChessAI(None, None, chess.Board(), True)
    sf = chess.engine.SimpleEngine.popen_uci(os.environ.get("STOCKFISH_PATH") or find_stockfish())
    try:
        sf.configure({"Threads": 1, "Hash": 64})
    except Exception:
        pass

    def label(b):
        info = sf.analyse(b, chess.engine.Limit(depth=LABEL_DEPTH))
        return info["score"].pov(chess.WHITE).score(mate_score=100000)

    def ours(b):
        return -int(ai.ev_breakdown(b)["total"]) / 10.0           # Black-positive mp → White cp

    t0, n_new = time.time(), 0
    for fen, g, st, sp, ctl in chosen:
        if fen in done:
            continue
        b = chess.Board(fen)
        um = markers_fn(b)
        m0 = material(b)
        out = [(0, b.fen(), ours(b), label(b), marks(b, m0))]
        ply = 0
        while ply < max(PLIES) and not b.is_game_over():
            mv = sf.play(b, chess.engine.Limit(nodes=MOVE_NODES)).move
            if mv is None:
                break
            b.push(mv); ply += 1
            if ply in PLIES:
                out.append((ply, b.fen(), ours(b) if not b.is_game_over() else float("nan"),
                            label(b) if not b.is_game_over() else float("nan"), marks(b, m0)))
        for k, fk, ok, sk, mk in out:
            w.writerow([fen, g, st, sp, ctl, *um, k, fk, "%.1f" % ok if ok == ok else "", sk if sk == sk else "", *mk])
        n_new += 1
        if n_new % 20 == 0:
            fh.flush()
            print("[traj %d/%d] %d new · %.2f/s" % (si, sn, n_new, n_new / (time.time() - t0)), flush=True)
    fh.close()
    sf.quit()
    print("TRAJ shard %d/%d DONE: %d new trajectories → %s" % (si, sn, n_new, OUT), flush=True)


if __name__ == "__main__":
    main()
