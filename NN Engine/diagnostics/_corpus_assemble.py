# -*- coding: utf-8 -*-
"""Assemble the retune corpus (step 0, 2026-10-10) from the SF18 d14 single-best label files on E:\\chess_data\\retune.

Two modes, run in order:

  pyrun diagnostics/_corpus_assemble.py MODE=map       [D=/mnt/e/chess_data/retune]
      Walk the stored self-play games once and record, for every own-play FEN in slices A / A2 / V1, the GAME it came
      from (the sampler kept no game id, and its EG_QUOTA reorder splits one game's rows across two blocks, so row
      order cannot recover it). A game = one jsonl file, cut further wherever the piece count rises or the move number
      falls (flat files may hold several games). First occurrence in sorted file order wins, as in the sampler.
      Output: $D/corpus_game_map.csv (key, group). Stops early once every wanted key is mapped.

  pyrun diagnostics/_corpus_assemble.py MODE=assemble  [D=…] [VAL_PCT=15] [STRESS_HOLD_PCT=20]
      One row per POSITION (key = first four FEN fields; clocks ignored), slice priority A > A2 > V1 > KP > VAR
      (A first: it carries our d10 depth row). Rows whose key appears in any held-out / bench / regret set are
      marked split=excluded. Columns:
        fen, key, slice, group, split (train|val|stress_hold|excluded), phase256, stage (mg|mgl|egl|eg = the eval's own
        phase256 stages), pieces, sf_cp (White-POV cp, SF18 d14), sf_best, mate, quiet, unq (reason), band, w_band,
        ours_d10 (White-POV cp, slice A only)
      Split: own-play by GAME hash (VAL_PCT held out); KP / VAR (one independent position per row) by row hash —
      STRESS_HOLD_PCT never trained, the rest is a train slice whose weight the fit decides (0 / 0.25 / 0.5).
      QUIET (static component only): mover not in check, SF best move not a capture / promotion / check, and no
      winning capture for the mover (capture of an undefended or higher-valued piece — the _triangulate_sf11 test).
      Band weights (owner: emphasise ≈ −4…+4 pawns; decided rows are near-saturated in win% anyway) are a DEFAULT
      column only — the fitter may override them.
      Output: $D/corpus_v1.csv + a summary on stdout.
"""
import os, sys, csv, json, glob, hashlib
from collections import Counter, defaultdict

for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v

import chess

THIS = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(THIS)
GAMES = os.path.join(ENGINE, "selfplay", "games")
D = os.environ.get("D", "/mnt/e/chess_data/retune")
MODE = os.environ.get("MODE", "assemble")
MAP_OUT = os.path.join(D, "corpus_game_map.csv")

# Label files per slice, in priority order (an earlier slice wins a duplicate position).
SLICES = [
    ("A",   [os.path.join(D, "A_sf18.csv")]),
    ("A2",  sorted(glob.glob(os.path.join(D, "A2_fens_sf18_s*.csv")))),
    ("V1",  sorted(glob.glob(os.path.join(D, "V1_fens_sf18_s*.csv")))),
    # GAUNT (10-13, round 3): our games vs STRONGER opponents (SF18 @1000 gauntlets + Mediocre) — the positions a stronger
    # engine steers us into; game id from the sampler's `game` column (GAUNT_fens.csv), split by game like own-play
    ("GAUNT", sorted(glob.glob(os.path.join(D, "GAUNT_fens_sf18_s*.csv")))),
    ("KP",  sorted(glob.glob(os.path.join(D, "KP_fens_sf18_s*.csv")))),
    ("VAR", sorted(glob.glob(os.path.join(D, "VAR_sf18_s*.csv")))),
]
OWNPLAY = ("A", "A2", "V1", "GAUNT")
# SLICES_EXTRA=EGX,CUR,VARG,ODDS,STRIP (2026-11-05, corpus_v3): further slices, appended in that order after the six above,
# opt-in so corpus_v1 / v2 reproduce exactly. MODE=map never sees them (their game ids come from <slice>_fens.csv).
#   EGX   endgame-type enriched own-play (10-15)          split by game, val like own-play (nested games share A's groups)
#   CUR   own-play + Mediocre games since 10-19 (gens 4-6.7) split by game, val like own-play
#   VARG  games from the 960-style starts                 split by game, a never-trained stress hold
#   ODDS  games from the odds starts                      split by game, a never-trained stress hold
#   STRIP strip-down stress (gen_kp_fens nopawn/heavy/mixed) split by row, a never-trained stress hold (like KP)
EXTRA_SLICES = {n: sorted(glob.glob(os.path.join(D, n + "_fens_sf18_s*.csv"))) for n in ("EGX", "CUR", "VARG", "ODDS", "STRIP")}
GAME_OWN, GAME_STRESS = ("EGX", "CUR"), ("VARG", "ODDS")
MATE_CP = 50000                  # _build_regret_set scores mates as ±(100000 − distance)
PV = {chess.PAWN: 1000, chess.KNIGHT: 3250, chess.BISHOP: 3450, chess.ROOK: 5000, chess.QUEEN: 10000, chess.KING: 100000}
NPM_LO, NPM_HI = 10000, 40000     # EVAL_V2_EG_LIMIT / EVAL_V2_MG_LIMIT defaults (eval_v2.cpp phase256)


def key_of(fen):
    return " ".join(fen.split()[:4])


def read_rows(paths):
    for p in paths:
        with open(p, newline="") as fh:
            for r in csv.DictReader(fh):
                if r.get("fen"):
                    yield r


# ─────────────────────────────────────────── MODE=map ───────────────────────────────────────────
def run_map():
    # Each slice was sampled from ONE layout: V1 from the flat '*/*.jsonl' files, A / A2 from the nested
    # '*/*/game.jsonl' games. Walk each layout only for its own slices' keys, flat first (A/A2 were all found in the
    # first ~155k of 395k nested files on 10-10; scanning the rest for V1 keys wasted the run). A part ends when its
    # keys are all mapped or after STALL files with no new key. The map is appended every 2,000 files and resumable:
    # $MAP_OUT.progress holds "<part> <files done>".
    stall = int(os.environ.get("STALL", "20000"))
    want = {"flat": set(), "nested": set()}
    for name, paths in SLICES:
        if name in OWNPLAY:
            part = "flat" if name == "V1" else "nested"
            for r in read_rows(paths):
                want[part].add(key_of(r["fen"]))
    mapped = {}
    if os.path.exists(MAP_OUT):
        for r in csv.DictReader(open(MAP_OUT, newline="")):
            mapped[r["key"]] = r["group"]
    prog_path = MAP_OUT + ".progress"
    prog = {}
    if os.path.exists(prog_path):
        for ln in open(prog_path):
            p = ln.split()
            if len(p) == 2:
                prog[p[0]] = int(p[1])
    if not os.path.exists(MAP_OUT):
        with open(MAP_OUT, "w", newline="") as fh:
            csv.writer(fh).writerow(["key", "group"])
    print("  wanted keys: flat %d · nested %d · already mapped %d · progress %s"
          % (len(want["flat"]), len(want["nested"]), len(mapped), prog), flush=True)

    for part, pattern in (("flat", os.path.join(GAMES, "*", "*.jsonl")), ("nested", os.path.join(GAMES, "*", "*", "game.jsonl"))):
        wanted = want[part]
        files = [f for f in sorted(glob.glob(pattern)) if "_archive" not in f and "_bak" not in f]
        start = prog.get(part, 0)
        left = sum(1 for k in wanted if k not in mapped)
        print("  ── %s: %d files, resuming at %d, %d keys left" % (part, len(files), start, left), flush=True)
        pending, last_new = [], start
        for i in range(start, len(files)):
            if left == 0 or i - last_new >= stall:
                break
            fp = files[i]
            rel = os.path.relpath(fp, GAMES)
            try:
                lines = open(fp).read().splitlines()
            except Exception:
                lines = []
            g, prev_pc, prev_mv = 0, 99, 0
            for ln in lines:
                try:
                    fen = (json.loads(ln).get("fen") or "").strip()
                except Exception:
                    continue
                if not fen:
                    continue
                parts = fen.split()
                pc = sum(ch.isalpha() for ch in parts[0])
                mv = int(parts[5]) if len(parts) > 5 and parts[5].isdigit() else 0
                if pc > prev_pc or mv < prev_mv:
                    g += 1
                prev_pc, prev_mv = pc, mv
                k = " ".join(parts[:4])
                if k in wanted and k not in mapped:
                    mapped[k] = "%s#%d" % (rel, g)
                    pending.append((k, mapped[k]))
                    left -= 1
                    last_new = i
            if (i + 1) % 2000 == 0 or left == 0:
                with open(MAP_OUT, "a", newline="") as fh:
                    csv.writer(fh).writerows(pending)
                pending = []
                prog[part] = i + 1
                with open(prog_path, "w") as fh:
                    fh.write("".join("%s %d\n" % kv for kv in prog.items()))
            if (i + 1) % 10000 == 0:
                print("  [%s %6d/%6d] %d keys left" % (part, i + 1, len(files), left), flush=True)
        with open(MAP_OUT, "a", newline="") as fh:
            csv.writer(fh).writerows(pending)
        print("  %s part done: %d keys left unmapped" % (part, left), flush=True)

    groups = Counter(mapped.values())
    print("  MAP DONE: %d/%d keys → %d groups; rows/group median %d, max %d → %s"
          % (len(mapped), len(want["flat"] | want["nested"]), len(groups),
             sorted(groups.values())[len(groups) // 2] if groups else 0, max(groups.values()) if groups else 0, MAP_OUT), flush=True)


# ───────────────────────────────────────── MODE=assemble ─────────────────────────────────────────
def held_out_keys():
    ks = os.path.join(THIS, "ks_sets")
    srcs = [os.path.join(ks, n) for n in ("fitC_mg_sf18.csv", "fitC_eg_sf18.csv", "kp_stress_sf18.csv", "pe_rows_sf18.csv",
                                          "dual_val_sf18.csv", "bench_variant.csv", "bench_odds.csv", "bench_kp.csv",
                                          "lowdepth_tuneset.csv", "regret_set.csv")]
    srcs += sorted(glob.glob(os.path.join(ks, "game_regret_set*.csv")))
    out, per = set(), {}
    for p in srcs:
        if not os.path.exists(p):
            continue
        n0 = len(out)
        try:
            for r in csv.DictReader(open(p, newline="")):
                f = (r.get("fen") or "").strip()
                if f:
                    out.add(key_of(f))
        except Exception:
            continue
        per[os.path.basename(p)] = len(out) - n0
    return out, per


def phase256(board):
    npm = sum(PV[t] * len(board.pieces(t, c)) for t in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
              for c in (chess.WHITE, chess.BLACK))
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
            if not board.attackers(not board.turn, m.to_square) or \
                    PV[board.piece_type_at(m.to_square)] > PV[board.piece_type_at(m.from_square)]:
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


def hpct(s):
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16) % 100


def run_assemble():
    val_pct = int(os.environ.get("VAL_PCT", "15"))
    hold_pct = int(os.environ.get("STRESS_HOLD_PCT", "20"))
    gmap = {}
    if os.path.exists(MAP_OUT):
        for r in csv.DictReader(open(MAP_OUT, newline="")):
            gmap[r["key"]] = r["group"]
    else:
        print("  ☠️ no %s — run MODE=map first" % MAP_OUT); sys.exit(1)
    # slices sampled WITH their game id (10-13: GAUNT) carry it in <slice>_fens.csv's `game` column
    gpath = os.path.join(D, "GAUNT_fens.csv")
    if os.path.exists(gpath):
        for r in csv.DictReader(open(gpath, newline="")):
            if r.get("game"):
                gmap.setdefault(key_of(r["fen"]), "gaunt:" + r["game"])
    extra = [n for n in os.environ.get("SLICES_EXTRA", "").split(",") if n]
    for n in extra:
        SLICES.append((n, EXTRA_SLICES[n]))
        gp = os.path.join(D, n + "_fens.csv")
        if n in GAME_OWN + GAME_STRESS and os.path.exists(gp):
            for r in csv.DictReader(open(gp, newline="")):
                g = r.get("game")
                if not g:
                    continue
                # a nested own-play game is one file; the game map names it <path>#0, so EGX joins A's group for it
                grp = g + "#0" if (n == "EGX" and g.endswith("/game.jsonl")) else n.lower() + ":" + g
                gmap.setdefault(key_of(r["fen"]), grp)
    hold, hold_per = held_out_keys()
    print("  game map: %d keys · held-out/bench/regret keys: %d %s" % (len(gmap), len(hold), hold_per), flush=True)
    # D10_TAG=ours1105 (11-05): every slice's d10 pass of that ship (<slice>_<tag>_d10*.csv on D), with our d10 move;
    # default = the 10-04 pass on slice A only, as corpus_v1 / v2 were built.
    tag = os.environ.get("D10_TAG", "")
    d10, d10mv = {}, {}
    d10_glob = ("*_%s_d10*.csv" % tag) if tag else "A_ours1004_d10_s*of4.csv"
    for p in sorted(glob.glob(os.path.join(D, d10_glob))):
        for r in csv.DictReader(open(p, newline="")):
            if r.get("fen") and r.get("ours_cp_white") not in (None, ""):
                d10[key_of(r["fen"])] = r["ours_cp_white"]
                d10mv[key_of(r["fen"])] = r.get("move", "")
    print("  d10 depth rows (%s): %d" % (d10_glob, len(d10)), flush=True)
    # CRIT_GLOB=<glob> (11-05): SF18 multi-PV files (`moves` = uci:cp;… best-first, White-POV cp) → gap12 = |best − 2nd| cp
    gap12 = {}
    if os.environ.get("CRIT_GLOB"):
        for p in sorted(glob.glob(os.path.join(D, os.environ["CRIT_GLOB"]))):
            for r in csv.DictReader(open(p, newline="")):
                mv = [x.rsplit(":", 1) for x in (r.get("moves") or "").split(";") if ":" in x]
                if len(mv) >= 2:
                    try:
                        gap12[key_of(r["fen"])] = abs(int(float(mv[0][1])) - int(float(mv[1][1])))
                    except ValueError:
                        pass
        print("  multi-PV gap rows (%s): %d" % (os.environ["CRIT_GLOB"], len(gap12)), flush=True)

    out_path = os.path.join(D, os.environ.get("OUT", "corpus_v1.csv"))   # OUT=corpus_v2.csv (10-13) keeps v1 reproducible
    fields = ["fen", "key", "slice", "group", "split", "phase256", "stage", "pieces", "sf_cp", "sf_best", "mate",
              "quiet", "unq", "band", "w_band", "ours_d10"]
    if tag:
        fields.append("ours_move")
    if os.environ.get("CRIT_GLOB"):
        fields.append("gap12")
    d10_miss = Counter()
    seen = set()
    st = defaultdict(Counter)
    a_nod10 = Counter()
    nomap = Counter()
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(fields)
        for name, paths in SLICES:
            n_in = 0
            for r in read_rows(paths):
                n_in += 1
                fen = r["fen"].strip(); k = key_of(fen)
                if k in seen:
                    st[name]["dup"] += 1
                    continue
                seen.add(k)
                try:
                    cp = int(float(r["best_cp"]))
                    board = chess.Board(fen)
                except Exception:
                    st[name]["bad"] += 1
                    continue
                mate = int(abs(cp) >= MATE_CP)
                if name in OWNPLAY or name in GAME_OWN:
                    grp = gmap.get(k)
                    if grp is None:
                        nomap[name] += 1
                        grp = "nomap:" + k
                    split = "val" if hpct(grp) < val_pct else "train"
                elif name in GAME_STRESS:
                    grp = gmap.get(k)
                    if grp is None:
                        nomap[name] += 1
                        grp = name + ":" + k
                    split = "stress_hold" if hpct(grp) < hold_pct else "train"
                else:
                    grp = name + ":" + k
                    split = "stress_hold" if hpct(grp) < hold_pct else "train"
                if k in hold:
                    split = "excluded"
                p256 = phase256(board)
                unq = unquiet_reason(board, r.get("best_uci", ""))
                band, wb = band_of(cp)
                od = ""
                if name == "A" or tag:
                    od = d10.get(k, "")
                    if od == "" and name == "A":
                        a_nod10["mate" if mate else "non-mate"] += 1
                    if od == "":
                        d10_miss[name] += 1
                row = [fen, k, name, grp, split, p256, stage_of(p256), len(board.piece_map()), cp, r.get("best_uci", ""),
                       mate, int(unq == ""), unq, band, wb, od]
                if tag:
                    row.append(d10mv.get(k, ""))
                if os.environ.get("CRIT_GLOB"):
                    row.append(gap12.get(k, ""))
                w.writerow(row)
                s = st[name]
                s["kept"] += 1; s["split:" + split] += 1; s["stage:" + stage_of(p256)] += 1
                s["quiet" if unq == "" else "unq:" + unq] += 1; s["band:" + band] += 1
            st[name]["in"] = n_in
            print("  %-4s in %7d  kept %7d  dup %6d" % (name, n_in, st[name]["kept"], st[name]["dup"]), flush=True)

    print("\n  ── SUMMARY → %s" % out_path)
    for name, _ in SLICES:
        s = st[name]; kept = max(1, s["kept"])
        def pc(key):
            return "%5.1f%%" % (100.0 * s[key] / kept)
        print("  %-4s kept %7d | split train %s val %s hold %s excl %d | stage mg %s mgl %s egl %s eg %s | quiet %s"
              % (name, s["kept"], pc("split:train"), pc("split:val"), pc("split:stress_hold"), s["split:excluded"],
                 pc("stage:mg"), pc("stage:mgl"), pc("stage:egl"), pc("stage:eg"), pc("quiet")))
        print("       unq: in_check %s best_capture %s best_promo %s best_check %s win_capture %s | bands: level %s 0.5-1.5 %s 1.5-4 %s 4-15 %s >=15 %s mate %s"
              % (pc("unq:in_check"), pc("unq:best_capture"), pc("unq:best_promo"), pc("unq:best_check"), pc("unq:win_capture"),
                 pc("band:level"), pc("band:0.5-1.5"), pc("band:1.5-4"), pc("band:4-15"), pc("band:>=15"), pc("band:mate")))
    print("  own-play rows without a game group: %s" % dict(nomap))
    print("  slice A rows without a d10 depth row: %s (expect ≈ 3k, all mate)" % dict(a_nod10))
    if tag:
        print("  rows without a %s d10 row, per slice: %s" % (tag, dict(d10_miss)))
    print("  ASSEMBLE DONE", flush=True)


if MODE == "map":
    run_map()
else:
    run_assemble()
