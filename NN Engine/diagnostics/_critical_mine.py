# -*- coding: utf-8 -*-
"""Stage-2 data: positions where the right move matters and is not obvious, so the eval has to judge danger or opportunity.

  MODE=mine      Lichess puzzles (quiet, defensive, sacrifice and zugzwang themes, no mate themes, rating window, capped per eval
                 stage) and the collapses recorded in our own games (the position before and the position after our eval fell).
                 Never a position of a bench, a regret set or corpus_v3. → CR_fens.csv (fen, stage, split, group, source, rating, themes)
  MODE=classify  labels (SF18 multi-PV, `moves` = uci:cp;… best-first, White-POV cp) → class per position → OUT, and a table.
                 GLOB=<labels glob, under D>  OUT=<csv under D>  META=<fens csv with source / split / group, optional>
                 D10=<glob of d10 passes, optional, adds how often our search misses the best move>

Critical = missing the best move costs the mover at least GAP win%, and the move is not obvious:
  quiet           no capture, promotion or check
  sacrifice       the move gives up material by static exchange (a capture or a quiet move)
  check_choice    a check that is not the only way: the gap stays under CHECK_MAX (above it the check simply has to be played)
  evasion_choice  in check, at least EV_MIN legal replies, one clearly best (gap >= EV_GAP), not simply winning material
Not critical: benign (gap under GAP), obvious captures (static exchange >= 0: recaptures, winning captures), queen promotions, forced
checks, single or double replies to check, positions where the best move is a forced mate.
"""
import os, sys, csv, glob, math, zlib
import chess

KV = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
THIS = os.path.dirname(os.path.abspath(__file__))
D = KV.get("D", "/mnt/e/chess_data/retune")
GAP = float(KV.get("GAP", 3.0))
CHECK_MAX = float(KV.get("CHECK_MAX", 20.0))
EV_GAP = float(KV.get("EV_GAP", 8.0))
EV_MIN = int(KV.get("EV_MIN", 3))
MATE = 90000              # labels score mates at 100000
K = 0.00368208
PV = {chess.KNIGHT: 3250, chess.BISHOP: 3450, chess.ROOK: 5000, chess.QUEEN: 10000}
NPM_LO, NPM_HI = 10000, 40000
VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100}


def wp(cp):
    return 100.0 / (1.0 + math.exp(-K * max(-1500.0, min(1500.0, cp))))


def key_of(fen):
    return " ".join(fen.split()[:4])


def stage_of(board):
    npm = sum(PV[t] * len(board.pieces(t, c)) for t in PV for c in (chess.WHITE, chess.BLACK))
    if npm <= NPM_LO:
        return "eg"
    if npm >= NPM_HI:
        return "mg"
    return "mgl" if (256 * (npm - NPM_LO)) // (NPM_HI - NPM_LO) >= 128 else "egl"


def see(board, move):
    """Static exchange value of a move for the mover, in pawns, by the swap algorithm. Pins are ignored, as usual for SEE."""
    b = board.copy(stack=False)
    to, us = move.to_square, b.turn
    if b.is_en_passant(move):
        b.remove_piece_at(chess.square(chess.square_file(to), chess.square_rank(move.from_square)))
        gain = [VAL[chess.PAWN]]
    else:
        victim = b.piece_type_at(to)
        gain = [VAL[victim] if victim else 0]
    ptype = move.promotion or b.piece_type_at(move.from_square)
    b.remove_piece_at(move.from_square)
    b.set_piece_at(to, chess.Piece(ptype, us))
    last, side = VAL[ptype], not us
    while True:
        atk = b.attackers(side, to)
        if not atk:
            break
        sq = min(atk, key=lambda s: VAL[b.piece_type_at(s)])
        t = b.piece_type_at(sq)
        if t == chess.KING and b.attackers(not side, to):
            break                 # the king cannot capture into a defended square
        gain.append(last - gain[-1])
        b.remove_piece_at(sq)
        b.set_piece_at(to, chess.Piece(t, side))
        last, side = VAL[t], not side
    for i in range(len(gain) - 1, 0, -1):
        gain[i - 1] = -max(-gain[i - 1], gain[i])
    return gain[0]


def classify(fen, moves):
    """→ (class, gap). `moves` = [(uci, White-POV cp), …] best first."""
    if len(moves) < 2:
        return "forced", 0.0
    b = chess.Board(fen)
    s = 1.0 if b.turn == chess.WHITE else -1.0
    best, second = s * moves[0][1], s * moves[1][1]
    if abs(best) >= MATE:
        return "mate", 0.0
    gap = wp(best) - wp(second)
    if gap < GAP:
        return "benign", gap
    m = chess.Move.from_uci(moves[0][0])
    v = see(b, m)
    capture = b.is_capture(m)
    if b.is_check():
        if b.legal_moves.count() < EV_MIN:
            return "evasion_forced", gap
        if capture and v > 0:
            return "capture_obvious", gap
        return ("evasion_choice" if gap >= EV_GAP else "benign"), gap
    if v < 0:
        return "sacrifice", gap
    if capture:
        return "capture_obvious", gap
    if m.promotion:
        return ("promo_queen" if m.promotion == chess.QUEEN else "sacrifice"), gap
    if b.gives_check(m):
        return ("check_choice" if gap < CHECK_MAX else "check_forced"), gap
    return "quiet", gap


CRITICAL = {"quiet", "sacrifice", "check_choice", "evasion_choice"}


def excluded_keys():
    """Every bench, regret set and instrument: nothing a later read relies on may enter a training slice."""
    out = set()
    ks = os.path.join(THIS, "ks_sets")
    srcs = [os.path.join(ks, n) for n in ("fitC_mg_sf18.csv", "fitC_eg_sf18.csv", "kp_stress_sf18.csv", "pe_rows_sf18.csv", "dual_val_sf18.csv",
                                          "bench_variant.csv", "bench_odds.csv", "bench_kp.csv", "lowdepth_tuneset.csv", "regret_set.csv",
                                          "playdist_ceiling.csv", "diverse_corpus_wide.csv")]
    srcs += sorted(glob.glob(os.path.join(ks, "game_regret_set*.csv")))
    for p in srcs:
        if os.path.exists(p):
            for r in csv.DictReader(open(p, newline="")):
                f = (r.get("fen") or "").strip()
                if f:
                    out.add(key_of(f))
    n_inst = len(out)
    for p in glob.glob(os.path.join(THIS, "suites", "*.epd")):
        for line in open(p, errors="replace"):
            f = line.split()
            if len(f) >= 4:
                out.add(" ".join(f[:4]))
    print("  excluded: instruments %d keys · with the bench suites %d" % (n_inst, len(out)), flush=True)
    for r in csv.DictReader(open(os.path.join(D, KV.get("CORPUS", "corpus_v3.csv")), newline="")):
        out.add(r["key"])
    print("  excluded: with the corpus %d" % len(out), flush=True)
    return out


def mine():
    SRC = KV.get("SRC", "/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Coach/chess-coach/data/lichess/lichess_db_puzzle.csv")
    CAP = int(KV.get("CAP", 15000))
    LO, HI = int(KV.get("MINRATING", 1800)), int(KV.get("MAXRATING", 3000))
    THEMES = set(KV.get("THEMES", "quietMove,defensiveMove,sacrifice,zugzwang").split(","))
    OUT = os.path.join(D, KV.get("OUT", "CR_fens.csv"))
    ex = excluded_keys()
    seen, rows = set(), []
    # puzzles: the lowest id hashes per stage, so the sample is reproducible and spread over the whole file
    best = {s: [] for s in ("mg", "mgl", "egl", "eg")}
    scanned = passed = 0
    with open(SRC, newline="", encoding="utf-8", errors="replace") as f:
        rd = csv.reader(f)
        next(rd, None)
        for row in rd:
            scanned += 1
            if len(row) < 8:
                continue
            ts = set(row[7].split())
            if not (ts & THEMES) or any("mate" in t.lower() for t in ts):
                continue
            try:
                rt = int(row[3])
            except ValueError:
                continue
            if rt < LO or rt > HI:
                continue
            mv = row[2].split()
            if not mv:
                continue
            passed += 1
            try:
                b = chess.Board(row[1])
                b.push_uci(mv[0])     # the opponent's setup move: the position the solver faces
            except Exception:
                continue
            best[stage_of(b)].append((zlib.crc32(row[0].encode()), row[0], b.fen(), rt, " ".join(sorted(ts & THEMES))))
    print("  puzzles scanned %d · theme, no mate, rating %d-%d: %d · by stage %s" % (scanned, LO, HI, passed, {s: len(v) for s, v in best.items()}), flush=True)
    for s, v in best.items():
        v.sort()
        n = 0
        for h, pid, fen, rt, th in v:
            if n >= CAP:
                break
            k = key_of(fen)
            if k in ex or k in seen:
                continue
            seen.add(k)
            n += 1
            rows.append((fen, s, "stress_hold" if h % 100 < 20 else "train", "puzzle/" + pid, "puzzle", rt, th))
    np_ = len(rows)
    # our games: where the eval fell. Both ends of each recorded collapse.
    games = os.path.join(THIS, "..", "selfplay", "games")
    n_files = 0
    for d in sorted(os.listdir(games)):
        p = os.path.join(games, d, "collapses.csv")
        if not os.path.exists(p):
            continue
        n_files += 1
        try:
            for r in csv.DictReader(open(p, newline="", errors="replace")):
                for col in ("decision_fen", "drop_fen"):
                    fen = (r.get(col) or "").strip()
                    if not fen:
                        continue
                    k = key_of(fen)
                    if k in ex or k in seen:
                        continue
                    try:
                        b = chess.Board(fen)
                    except ValueError:
                        continue
                    seen.add(k)
                    grp = "%s/%s" % (d, (r.get("game") or "").strip())
                    rows.append((fen, stage_of(b), "val" if zlib.crc32(grp.encode()) % 100 < 15 else "train", grp, "collapse", "", col))
        except csv.Error:
            continue
    print("  collapses: %d files → %d positions · puzzles kept %d" % (n_files, len(rows) - np_, np_), flush=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fen", "stage", "split", "group", "source", "rating", "themes"])
        w.writerows(rows)
    print("MINE DONE → %s (%d rows)" % (OUT, len(rows)))


def run_classify():
    files = sorted(glob.glob(os.path.join(D, KV["GLOB"])))
    meta = {}
    if KV.get("META"):
        for r in csv.DictReader(open(os.path.join(D, KV["META"]), newline="")):
            meta[key_of(r["fen"])] = r
    ours = {}
    if KV.get("D10"):
        for p in glob.glob(os.path.join(D, KV["D10"])):
            for r in csv.DictReader(open(p, newline="")):
                if r.get("move"):
                    ours[key_of(r["fen"])] = r["move"]
    if KV.get("CORPUS_MOVES"):
        for r in csv.DictReader(open(os.path.join(D, KV["CORPUS_MOVES"]), newline="")):
            if r.get("ours_move"):
                ours[r["key"]] = r["ours_move"]
    tab, rows_in = {}, 0
    fh = None
    if KV.get("OUT"):
        fh = open(os.path.join(D, KV["OUT"]), "w", newline="")
        w = csv.writer(fh)
        w.writerow(["fen", "key", "source", "stage", "split", "group", "cls", "gap", "best_uci", "best_cp", "moves"])
    for p in files:
        for r in csv.DictReader(open(p, newline="")):
            if not r.get("moves"):
                continue
            rows_in += 1
            mv = [(x.split(":")[0], float(x.split(":")[1])) for x in r["moves"].split(";") if ":" in x]
            c, g = classify(r["fen"], mv)
            k = key_of(r["fen"])
            m = meta.get(k, {})
            src = m.get("source", "corpus")
            t = tab.setdefault((src, c), [0, 0, 0])
            t[0] += 1
            if k in ours:
                t[1] += 1
                t[2] += ours[k] != mv[0][0]
            if fh and c in CRITICAL:
                st = m.get("stage") or stage_of(chess.Board(r["fen"]))
                w.writerow([r["fen"], k, src, st, m.get("split", ""), m.get("group", ""), c, "%.2f" % g, mv[0][0], r.get("best_cp") or mv[0][1], r["moves"]])
    if fh:
        fh.close()
    print("positions classified: %d" % rows_in)
    print("  %-10s %-17s %8s %7s %12s" % ("source", "class", "rows", "share", "d10 misses"))
    for src in sorted({k[0] for k in tab}):
        tot = sum(v[0] for k, v in tab.items() if k[0] == src)
        for (s_, c), v in sorted(tab.items(), key=lambda kv: -kv[1][0]):
            if s_ == src:
                mark = "*" if c in CRITICAL else " "
                print("  %-10s %-16s%s %8d %6.1f%% %12s" % (src, c, mark, v[0], 100.0 * v[0] / tot, "%.1f%% of %d" % (100.0 * v[2] / v[1], v[1]) if v[1] else "-"))
        crit = sum(v[0] for k, v in tab.items() if k[0] == src and k[1] in CRITICAL)
        print("  %-10s critical (*): %d = %.1f%%" % (src, crit, 100.0 * crit / tot), flush=True)


if __name__ == "__main__":
    if KV.get("MODE") == "test":
        cases = [("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1", "e4d5", 1), ("4k3/4p3/3p4/8/8/8/3Q4/4K3 w - - 0 1", "d2d6", -8),
                 ("4k3/8/3p4/2p5/8/8/3R4/4K3 w - - 0 1", "d2d6", 1), ("4k3/8/2p5/8/8/8/3N4/4K3 w - - 0 1", "d2b3", 0),
                 ("4k3/8/2p5/8/4N3/8/8/4K3 w - - 0 1", "e4d6", 0), ("4k3/2p5/8/8/4N3/8/8/4K3 w - - 0 1", "e4d6", -3)]
        for fen, u, want in cases:
            got = see(chess.Board(fen), chess.Move.from_uci(u))
            print("see %-40s %s want %+d got %+d %s" % (fen, u, want, got, "ok" if got == want else "WRONG"))
    elif KV.get("MODE") == "mine":
        mine()
    else:
        run_classify()
