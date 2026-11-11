# -*- coding: utf-8 -*-
"""LINE TRACE: how does our search treat the moves of Stockfish's line? (2026-11-11; POT due diligence test C, search v2
plan §7d; the owner's game positions of the triage, `ks_sets/pot_owner_cases.txt`).

  MODE=sf     pyrun diagnostics/_line_trace.py                   → SF18 d18 PV (8 plies) per case → /mnt/e/chess_data/linetrace_sf.json
  MODE=trace  V2_PRESET=shipped MAX_DEPTH=10 pyrun diagnostics/_line_trace.py TAG=d10 [CHESSAI_DIR=<test module dir>]
              → for each case, the boards after SF's moves 1..8 go to the C++ line tracer (TRACE_TARGETS, search_engine.cpp
                TraceRec); our normal search at fixed depth; every visit to one of those boards prints how the move INTO it
                was treated (ordering index i, remaining depth rd, kind: first / scout / reduced / LMP / FUT / HIST / SEE /
                SEEC / tt / rep, reduction red, re-search rs) → /mnt/e/chess_data/linetrace_<TAG>.json
  MODE=report TAG=d10 pyrun diagnostics/_line_trace.py           → per case and k, then the registered read

Move classes (python-chess, at the board before the move): tactical = capture, check or promotion; quiet = the rest;
lever = a quiet pawn push that attacks an enemy pawn, or a pawn capture of a pawn (a pawn-structure transformation).
Verdict per case (as `_case_triage.py MODE=report`): ok / EVAL / PRUNING / DEPTH from triage_sf.json, triage_normal.json and
triage_noprune.json (11-10, gen 7.1, d10).

REGISTERED (11-11, before any trace ran; search v2 plan §7d test C):
  Unit = one visit to a board on SF's line at k >= 2 (the move into it is made below the root), quiet SF moves only,
  in the FINAL iteration (it = the case's deepest); all iterations pooled is reported as a secondary read.
  "Buried" = pruned (LMP / FUT / HIST / SEE) or reduced by >= 2 plies.
  KEEP (supports S-2, reduce plan moves less / exempt them from late-move pruning): in the missed cases (DEPTH + PRUNING),
    buried >= 50% of quiet visits, AND >= 1.5x the buried share of quiet visits in the ok + EVAL cases, AND true in at
    least half of the missed cases (per case, so one case cannot carry it).
  STOP (the burial story is not what happens on these lines): buried < 25% of quiet visits in the missed cases.
  Prediction: buried 35-60% in the missed cases, the ok / EVAL cases lower but not by 1.5x; reach falls off fast with k
  (most visits at k = 2-3). Also reported: median ordering index of quiet vs tactical SF moves, by remaining depth.
"""
import os, sys, json, math, tempfile, collections
KV = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
THIS = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(THIS)
if os.environ.get("CHESSAI_DIR") or KV.get("CHESSAI_DIR"):    # a test build of the engine module, kept apart from the live one
    sys.path.insert(0, os.environ.get("CHESSAI_DIR") or KV["CHESSAI_DIR"])
sys.path.insert(0, ENGINE); sys.path.insert(0, THIS)
os.environ.setdefault("PRESET", "LONG_FORMAT"); os.environ.setdefault("USE_OPENING_BOOK", "0")
import chess, chess.engine
MODE = KV.get("MODE", "report")
TAG = KV.get("TAG", "d10")
D = "/mnt/e/chess_data"
SF18 = ("/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/stockfish_18_linux/stockfish-ubuntu-x86-64-avx2")
K = 0.00368208
wp = lambda cp: 100 / (1 + math.exp(-K * max(-1500, min(1500, cp))))
PRUNED = {"LMP", "FUT", "HIST", "SEE", "SEEC"}
ALL_IT = KV.get("ALL_IT", "0") == "1"      # report: 1 = pool every iteration (secondary read), 0 = the final iteration only


def cases():
    out = []
    for line in open(os.path.join(THIS, "ks_sets/pot_owner_cases.txt")):
        if line.startswith("#") or "/" not in line:
            continue
        fen, _, note = line.partition("#")
        out.append((fen.strip(), note.split()[0] if note.split() else "?"))
    return out


def move_class(b, m):
    if b.is_capture(m) or b.gives_check(m) or m.promotion:
        pc = b.piece_at(m.from_square)
        lever = bool(pc and pc.piece_type == chess.PAWN and b.is_capture(m) and b.piece_at(m.to_square)
                     and b.piece_at(m.to_square).piece_type == chess.PAWN)
        return "tactical", lever
    pc = b.piece_at(m.from_square)
    lever = False
    if pc and pc.piece_type == chess.PAWN:
        att = chess.BB_PAWN_ATTACKS[pc.color][m.to_square]
        lever = bool(att & b.pieces_mask(chess.PAWN, not pc.color))
    return "quiet", lever


def mode_sf():
    sf = chess.engine.SimpleEngine.popen_uci(os.environ.get("STOCKFISH_PATH", SF18))
    sf.configure({"Threads": 1, "Hash": 64})
    res = {}
    for fen, tag in cases():
        info = sf.analyse(chess.Board(fen), chess.engine.Limit(depth=18))
        res[fen] = {"tag": tag, "pv": [m.uci() for m in info.get("pv", [])[:8]]}
        print(tag, res[fen]["pv"], flush=True)
    sf.quit()
    json.dump(res, open(os.path.join(D, "linetrace_sf.json"), "w"), indent=1)


def boards_line(fen, pv):
    b = chess.Board(fen)
    rows = []
    for k, u in enumerate(pv, 1):
        m = chess.Move.from_uci(u)
        cls, lever = move_class(b, m)
        b.push(m)
        bb = [b.pawns, b.knights, b.bishops, b.rooks, b.queens, b.kings, b.occupied_co[chess.WHITE], b.occupied_co[chess.BLACK]]
        rows.append((k, u, cls, lever, bb, int(b.turn)))
    return rows


def mode_trace():
    from tactical_test import run_one
    lines = json.load(open(os.path.join(D, "linetrace_sf.json")))
    out = {}
    tmpd = tempfile.mkdtemp(prefix="linetrace_")
    for fen, tag in cases():
        pv = lines.get(fen, {}).get("pv", [])
        if len(pv) < 2:
            continue
        rows = boards_line(fen, pv)
        tp = os.path.join(tmpd, "targets.txt")
        with open(tp, "w") as f:
            for k, u, cls, lever, bb, turn in rows:
                f.write("%d %s %d\n" % (k, " ".join(str(x) for x in bb), turn))
        os.environ["TRACE_TARGETS"] = tp
        ep = os.path.join(tmpd, "err_%s.txt" % tag)
        sys.stderr.flush()
        saved = os.dup(2)
        fd = os.open(ep, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
        os.dup2(fd, 2)
        try:
            o = run_one(fen, [])
        finally:
            os.dup2(saved, 2); os.close(fd); os.close(saved)
        recs = []
        for ln in open(ep, errors="replace"):
            if ln.startswith("[TRACE]"):
                recs.append(dict(kv.split("=", 1) for kv in ln.split()[1:]))
        out[fen] = {"tag": tag, "ours": o.get("uci"), "sf": pv[0], "line": [(k, u, cls, lever) for k, u, cls, lever, _, _ in rows],
                    "recs": recs}
        print("%-5s ours %-6s sf %-6s  trace records %d" % (tag, o.get("uci"), pv[0], len(recs)), flush=True)
    os.environ.pop("TRACE_TARGETS", None)
    json.dump(out, open(os.path.join(D, "linetrace_%s.json" % TAG), "w"))


def verdicts():
    sf = json.load(open(os.path.join(D, "triage_sf.json")))
    nm = json.load(open(os.path.join(D, "triage_normal.json")))
    npn = json.load(open(os.path.join(D, "triage_noprune.json")))
    v = {}
    for fen, s in sf.items():
        n, p = nm.get(fen, {}), npn.get(fen, {})
        gap = 0.0 if s["sf_end"] is None else abs(wp(s["ours_end"] * 1.0) - wp(s["sf_end"]))
        v[fen] = ("ok" if n.get("move") == s["best"] else "EVAL" if gap >= 10 else "PRUNING" if p.get("move") == s["best"] else "DEPTH")
    return v


def mode_report():
    data = json.load(open(os.path.join(D, "linetrace_%s.json" % TAG)))
    ver = verdicts()
    group = {"missed": collections.Counter(), "found": collections.Counter()}
    per_case_keep = []
    idx = collections.defaultdict(list)
    print("%-5s %-8s %2s %-6s %-8s %6s %6s %6s %6s %6s %6s" % ("case", "verdict", "k", "move", "class", "visits", "pruned", "red>=2",
                                                             "full", "tt", "med_i"))
    for fen, d in data.items():
        v = ver.get(fen, "?")
        g = "missed" if v in ("DEPTH", "PRUNING") else "found"
        cls_of = {k: (cls, lever) for k, u, cls, lever in d["line"]}
        mv_of = {k: u for k, u, _, _ in d["line"]}
        by_k = collections.defaultdict(list)
        last = max((int(r["it"]) for r in d["recs"]), default=0)
        for r in d["recs"]:
            if r.get("sh") == "1":
                continue                                    # a prune-shadow re-search, not the real tree
            if ALL_IT or int(r["it"]) == last:
                by_k[int(r["k"])].append(r)
        cq = collections.Counter()
        for k in sorted(cls_of):
            rs = by_k.get(k, [])
            cls, lever = cls_of[k]
            pr = sum(r["kind"] in PRUNED for r in rs)
            rd2 = sum(r["kind"] == "reduced" and int(r["red"]) >= 2 for r in rs)
            full = sum(r["kind"] in ("first", "scout") or (r["kind"] == "reduced" and int(r["red"]) < 2) for r in rs)
            tt = sum(r["kind"] == "tt" for r in rs)
            ii = sorted(int(r["i"]) for r in rs if r["kind"] not in ("tt", "rep"))
            med = ii[len(ii) // 2] if ii else None
            print("%-5s %-8s %2d %-6s %-8s %6d %6d %6d %6d %6d %6s" % (d["tag"], v, k, mv_of[k], cls + ("+lever" if lever else ""),
                                                                     len(rs), pr, rd2, full, tt, med))
            for r in rs:
                if r["kind"] not in ("tt", "rep"):
                    idx[(cls, min(int(r["rd"]), 6))].append(int(r["i"]))
            if k >= 2 and cls == "quiet":
                for r in rs:
                    if r["kind"] in ("tt", "rep"):
                        continue
                    b = r["kind"] in PRUNED or (r["kind"] == "reduced" and int(r["red"]) >= 2)
                    group[g]["visits"] += 1; group[g]["buried"] += b
                    cq["visits"] += 1; cq["buried"] += b
        if g == "missed" and cq["visits"]:
            per_case_keep.append(cq["buried"] / cq["visits"] >= 0.5)
    print("\nordering index of SF's moves where they were searched or pruned (median, n), by class and remaining depth:")
    for cls in ("quiet", "tactical"):
        print("  %-8s %s" % (cls, "  ".join("rd%s %s (n %d)" % (rd if rd < 6 else "6+", sorted(v)[len(v) // 2], len(v))
                                         for (c, rd), v in sorted(idx.items()) if c == cls)))
    sh = {g: (c["buried"] / c["visits"] if c["visits"] else float("nan")) for g, c in group.items()}
    print("\nquiet SF moves at k >= 2, visits buried (pruned, or reduced by >= 2):")
    for g in ("missed", "found"):
        print("  %-6s %d / %d = %.1f%%" % (g, group[g]["buried"], group[g]["visits"], 100 * sh[g]))
    keep = sh["missed"] >= 0.5 and sh["missed"] >= 1.5 * sh["found"] and per_case_keep and sum(per_case_keep) * 2 >= len(per_case_keep)
    print("  missed cases with >= 50%% buried: %d of %d" % (sum(per_case_keep), len(per_case_keep)))
    print("REGISTERED (11-11, test C): %s" % ("KEEP: plan moves on SF's line are buried where we miss" if keep else
                                             "STOP: the burial story is not what happens on these lines" if sh["missed"] < 0.25 else "MISS"))


{"sf": mode_sf, "trace": mode_trace, "report": mode_report}[MODE]()
