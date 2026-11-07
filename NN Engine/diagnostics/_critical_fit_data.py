# -*- coding: utf-8 -*-
"""Stage 2 of the retune: turn the critical rows of _critical_mine.py into fit data, and read how often a search finds the best move on
critical rows that were held out of the fit.

MODE=assemble  corpus_v3 + the critical rows of CR_classes.csv → corpus_v4.csv. corpus_v3's rows come first, unchanged, so its feature export
               still lines up; the new rows are slices CRP (puzzles) and CRG (our own games), trained as stress rows, and their held-out rows
               have split stress_hold. Also writes
                 CR_rows.csv        the new rows alone, for _px_export.py
                 CR_w_all.csv       key,w for _retune_fit.py ROW_W: own-play critical rows at OWN_W (default 3); puzzle sacrifices scaled so they
                                    weigh no more than the puzzle quiet moves
                 CR_w_lc25 / lc50   the same with 75% / 50% of the new training rows dropped, by a hash of the group
                 CR_scope.csv       key,cls for _retune_fit.py SCOPE_KEYS: every critical row, own-play and new
                 CR_hold.csv        the critical rows no fit trains on: held-out puzzles and collapses, and own-play val rows, with gen 6.7's d10 move
MODE=moves     ARMS=name:glob,name:glob (d10 passes of CR_hold.csv with MOVE=1): how often each arm misses SF18's best move and the win% that costs,
               by source and class, paired against REF (default gen 6.7, whose moves are in the HOLD file) on the same rows, standard errors
               clustered by group. A move outside SF18's top K is charged at the K-th move's value, so regret is a lower bound.
"""
import csv, glob, hashlib, math, os, sys
import chess

KV = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
D = KV.get("D", "/mnt/e/chess_data/retune")
MATE_CP = 50000           # as _corpus_assemble.py
K = 0.00368208
PV = {chess.PAWN: 1000, chess.KNIGHT: 3250, chess.BISHOP: 3450, chess.ROOK: 5000, chess.QUEEN: 10000, chess.KING: 100000}
NPM_LO, NPM_HI = 10000, 40000
csv.field_size_limit(10 ** 7)


def key_of(fen):
    return " ".join(fen.split()[:4])


def wp(cp):
    return 100.0 / (1.0 + math.exp(-K * max(-1500.0, min(1500.0, cp))))


# The next four follow _corpus_assemble.py, which runs on import and cannot be imported.
def phase256(board):
    npm = sum(PV[t] * len(board.pieces(t, c)) for t in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN) for c in (chess.WHITE, chess.BLACK))
    if npm <= NPM_LO:
        return 0
    if npm >= NPM_HI:
        return 256
    return (256 * (npm - NPM_LO)) // (NPM_HI - NPM_LO)


def stage_of(p):
    return "mg" if p == 256 else ("mgl" if p >= 128 else ("egl" if p > 0 else "eg"))


def unquiet_reason(board, best_uci):
    if board.is_check():
        return "in_check"
    try:
        m = chess.Move.from_uci(best_uci)
        if m in board.legal_moves:
            if board.is_capture(m):
                return "best_capture"
            if m.promotion:
                return "best_promo"
            if board.gives_check(m):
                return "best_check"
    except Exception:
        pass
    for m in board.legal_moves:
        if board.is_capture(m) and not board.is_en_passant(m):
            if not board.attackers(not board.turn, m.to_square) or PV[board.piece_type_at(m.to_square)] > PV[board.piece_type_at(m.from_square)]:
                return "win_capture"
    return ""


def band_of(cp):
    a = abs(cp)
    if a >= MATE_CP:
        return "mate", 0.1
    if a < 50:
        return "level", 1.0
    if a < 150:
        return "0.5-1.5", 1.0
    if a < 400:
        return "1.5-4", 1.0
    if a < 1500:
        return "4-15", 0.5
    return ">=15", 0.2


def assemble():
    d10, d10mv = {}, {}
    for p in sorted(glob.glob(os.path.join(D, KV.get("D10", "CR_ours1105_d10_s*of3.csv")))):
        for r in csv.DictReader(open(p, newline="")):
            if r.get("ours_cp_white") not in (None, ""):
                d10[key_of(r["fen"])] = r["ours_cp_white"]
                d10mv[key_of(r["fen"])] = r.get("move", "")
    print("d10 rows on the critical set: %d" % len(d10), flush=True)
    src_path = os.path.join(D, KV.get("CORPUS", "corpus_v3.csv"))
    out_path = os.path.join(D, KV.get("OUT", "corpus_v4.csv"))
    pre = KV.get("PREFIX", "CR")
    in_v3 = {}
    with open(src_path, newline="") as fi, open(out_path, "w", newline="") as fo:
        hdr = fi.readline()
        fo.write(hdr)
        fields = next(csv.reader([hdr]))
        ik, isp, imv, igr = fields.index("key"), fields.index("split"), fields.index("ours_move"), fields.index("group")
        for line in fi:
            fo.write(line)
            r = next(csv.reader([line]))
            in_v3[r[ik]] = (r[isp], r[imv], r[igr])
        new_rows, seen, n_nod10 = [], set(), 0
        for r in csv.DictReader(open(os.path.join(D, KV.get("CLASSES", "CR_classes.csv")), newline="")):
            k = r["key"]
            assert k not in in_v3, "a critical row is already in the corpus: " + k
            if k in seen:
                continue
            seen.add(k)
            b = chess.Board(r["fen"])
            cp = int(float(r["best_cp"]))
            mv = [x.rsplit(":", 1) for x in r["moves"].split(";") if ":" in x]
            gap = abs(int(float(mv[0][1])) - int(float(mv[1][1]))) if len(mv) >= 2 else ""
            p = phase256(b)
            unq = unquiet_reason(b, r["best_uci"])
            band, wb = band_of(cp)
            n_nod10 += k not in d10
            row = dict(zip(fields, [r["fen"], k, "CRP" if r["source"] == "puzzle" else "CRG", r["group"], "train" if r["split"] == "train" else "stress_hold",
                                    p, stage_of(p), len(b.piece_map()), cp, r["best_uci"], int(abs(cp) >= MATE_CP), int(unq == ""), unq, band, wb,
                                    d10.get(k, ""), d10mv.get(k, ""), gap]))
            row["_cls"], row["_moves"] = r["cls"], r["moves"]
            new_rows.append(row)
        w = csv.writer(fo)
        for row in new_rows:
            w.writerow([row[c] for c in fields])
    with open(os.path.join(D, pre + "_rows.csv"), "w", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(fields)
        for row in new_rows:
            w.writerow([row[c] for c in fields])
    tab = {}
    for row in new_rows:
        t = tab.setdefault((row["slice"], row["split"], row["_cls"]), [0, 0, 0])
        t[0] += 1
        t[1] += row["quiet"] == 1
        t[2] += row["ours_d10"] != ""
    print("new rows %d (no d10 score: %d) → %s, %s_rows.csv" % (len(new_rows), n_nod10, out_path, pre))
    print("  %-5s %-12s %-15s %7s %7s %7s" % ("slice", "split", "class", "rows", "quiet", "d10"))
    for key_, v in sorted(tab.items()):
        print("  %-5s %-12s %-15s %7d %7d %7d" % (key_ + tuple(v)))

    # weights: own-play critical rows up, puzzle sacrifices no heavier than puzzle quiet moves
    own_w = float(KV.get("OWN_W", 3))
    own = [r for r in csv.DictReader(open(os.path.join(D, KV.get("OWN_CLASSES", "CRC_classes.csv")), newline=""))]
    tr = [r for r in new_rows if r["slice"] == "CRP" and r["split"] == "train"]
    n_q = sum(r["_cls"] == "quiet" for r in tr)
    n_s = sum(r["_cls"] == "sacrifice" for r in tr)
    sac = min(1.0, n_q / n_s) if n_s else 1.0
    print("own-play critical rows %d at weight %g; puzzle training rows: quiet %d, sacrifices %d at weight factor %.3f" % (len(own), own_w, n_q, n_s, sac))
    base = {r["key"]: own_w for r in own}
    for r in new_rows:
        if r["slice"] == "CRP" and r["_cls"] == "sacrifice":
            base[r["key"]] = sac
    for tag, frac in (("all", 1.0), ("lc25", 0.25), ("lc50", 0.5)):
        with open(os.path.join(D, "%s_w_%s.csv" % (pre, tag)), "w", newline="") as fo:
            w = csv.writer(fo)
            w.writerow(["key", "w"])
            wt = dict(base)
            kept = 0
            for r in new_rows:
                if r["split"] == "train":
                    # md5, not crc32: the puzzles were chosen by lowest crc32, and crc32 is linear, so a second crc32 split is clustered
                    if int(hashlib.md5(r["group"].encode()).hexdigest()[:8], 16) / 2.0 ** 32 >= frac:
                        wt[r["key"]] = 0.0
                    else:
                        kept += 1
            w.writerows(sorted(wt.items()))
        print("  %s_w_%s.csv: %d new training rows kept" % (pre, tag, kept))
    with open(os.path.join(D, pre + "_scope.csv"), "w", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["key", "cls"])
        w.writerows((r["key"], r["cls"]) for r in own)
        w.writerows((r["key"], r["_cls"]) for r in new_rows)

    # rows no fit trains on, with gen 6.7's d10 move
    n = {}
    with open(os.path.join(D, pre + "_hold.csv"), "w", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["fen", "key", "source", "cls", "group", "best_uci", "moves", "g67_move"])
        for r in new_rows:
            if r["split"] != "train":
                s = "puzzle" if r["slice"] == "CRP" else "collapse"
                w.writerow([r["fen"], r["key"], s, r["_cls"], r["group"], r["sf_best"], r["_moves"], r["ours_move"]])
                n[s] = n.get(s, 0) + 1
        for r in own:
            sp, mv, grp = in_v3.get(r["key"], ("", "", ""))
            if sp == "val":
                w.writerow([r["fen"], r["key"], "own", r["cls"], grp, r["best_uci"], r["moves"], mv])
                n["own"] = n.get("own", 0) + 1
    print("held-out critical rows → %s_hold.csv: %s" % (pre, n), flush=True)


def read_moves():
    hold = list(csv.DictReader(open(os.path.join(D, KV.get("HOLD", "CR_hold.csv")), newline="")))
    arms = {"gen 6.7": {r["key"]: r["g67_move"] for r in hold if r["g67_move"]}}
    for a in (x for x in KV.get("ARMS", "").split(",") if x):
        name, pat = a.split(":", 1)
        got = {}
        for p in glob.glob(os.path.join(D, pat)):
            for r in csv.DictReader(open(p, newline="")):
                if r.get("move"):
                    got[key_of(r["fen"])] = r["move"]
        arms[name] = got

    def stats(rows, mv):
        """Per row (group, missed, regret) where this arm has a move."""
        out = {}
        for r in rows:
            m = mv.get(r["key"])
            if not m:
                continue
            s = 1.0 if r["fen"].split()[1] == "w" else -1.0
            ml = [x.rsplit(":", 1) for x in r["moves"].split(";") if ":" in x]
            val = {u: s * float(c) for u, c in ml}
            best = wp(s * float(ml[0][1]))
            out[r["key"]] = (r["group"], m != r["best_uci"], best - wp(val.get(m, s * float(ml[-1][1]))))
        return out

    def paired(a, b):
        """Mean of a - b over common rows, with a standard error clustered by group."""
        keys = [k for k in a if k in b]
        if not keys:
            return 0.0, 0.0, 0
        groups = {}
        for k in keys:
            groups.setdefault(a[k][0], []).append(a[k][1] - b[k][1])
        mean = sum(sum(v) for v in groups.values()) / len(keys)
        var = sum((sum(v) - len(v) * mean) ** 2 for v in groups.values())
        return mean, math.sqrt(var) / len(keys), len(keys)

    scopes = [("all", lambda r: True)] + [("source " + s, (lambda s_: lambda r: r["source"] == s_)(s)) for s in sorted({r["source"] for r in hold})] + \
             [("class " + c, (lambda c_: lambda r: r["cls"] == c_)(c)) for c in sorted({r["cls"] for r in hold})]
    rn = KV.get("REF", "gen 6.7")
    print("held-out critical rows: %d. Paired against %s on the rows both have; minus = fewer misses, less regret." % (len(hold), rn))
    print("  %-22s %-14s %7s %7s %9s %14s %7s %14s" % ("scope", "arm", "rows", "miss%", "regret", "miss% vs ref", "z", "regret vs ref"))
    for name, f in scopes:
        rows = [r for r in hold if f(r)]
        ref = stats(rows, arms[rn])
        for a, mv in arms.items():
            st = stats(rows, mv)
            if not st:
                continue
            miss = 100.0 * sum(v[1] for v in st.values()) / len(st)
            reg = sum(v[2] for v in st.values()) / len(st)
            if a == rn:
                print("  %-22s %-14s %7d %6.2f%% %9.3f" % (name, a, len(st), miss, reg))
                continue
            dm, sem, _ = paired({k: (v[0], float(v[1])) for k, v in st.items()}, {k: (v[0], float(v[1])) for k, v in ref.items()})
            dr, ser, _ = paired({k: (v[0], v[2]) for k, v in st.items()}, {k: (v[0], v[2]) for k, v in ref.items()})
            print("  %-22s %-14s %7d %6.2f%% %9.3f %+7.2f ±%5.2f %+7.2f %+7.3f ±%5.3f" % (name, a, len(st), miss, reg, 100 * dm, 100 * sem, dm / sem if sem else 0, dr, ser))


if __name__ == "__main__":
    if KV.get("MODE") == "assemble":
        assemble()
    elif KV.get("MODE") == "moves":
        read_moves()
    else:
        print(__doc__)
