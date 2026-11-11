# -*- coding: utf-8 -*-
"""POT TEST D: do POT markers separate WRONG prunes and reductions? (2026-11-11; search v2 plan §7d, POT-REREAD-ANALYSIS §5
test D). The prune-shadow logger (ENABLE_PRUNE_SHADOW + ENABLE_SHADOW_EVENTS) re-searches a sample of LMP / futility skips and
LMR reductions honestly and labels each one wrong (ent=1: the honest search would have improved the node's bound) or not.
Its [SHADOWEV] record carries the node's board and the move (fr / to and bp…bb, tu), so the POT markers are computed here,
offline, with test A's own features (no C++ port of the structure valuation).

  MODE=pass  V2_PRESET=shipped MAX_DEPTH=10 ENABLE_PRUNE_SHADOW=1 ENABLE_SHADOW_EVENTS=1 SHADOW_N=64 \\
             pyrun diagnostics/_pot_shadow.py IN=<csv with a fen column> OUT=<events file> [SHARD=k/n] [N=0] [CHESSAI_DIR=…]
             → runs our fixed-depth search per position (run_one, tables cleared), stderr's [SHADOWEV] lines → OUT (append)
  MODE=auc   pyrun diagnostics/_pot_shadow.py EVENTS=<file>[,<file>…]   → per mechanism: AUC of each POT marker against the
             wrong label, pooled and within strata of the existing leaders; then the registered verdict

POT markers (at the node, before the move; python-chess):
  U          test A's β · feats_sum(board) (the structural-magnitude score of tests A / B; both polarities ranked)
  lever      the move is a quiet pawn push whose new square attacks an enemy pawn, or a pawn capturing a pawn
  heavy_brk  the move takes a rook or queen onto a BREAK FILE (a file holding a pawn in tension or one push from a lever)
  brk_king   the move's break file (lever or heavy_brk) is within one file of either king
Strata (the existing leaders): ordering index i (0-3, 4-7, 8-15, 16+) × remaining depth rd (1, 2, 3, 4+) × history sign.

REGISTERED (11-11, before any pass ran):
  KEEP a marker for a mechanism: pooled AUC >= 0.60 (or <= 0.40) on >= 200 wrong events, AND the stratum-weighted AUC
    (strata with >= 20 wrong and >= 20 right) on the same side >= 0.57 (<= 0.43).
  STOP (today's read): no POT marker beyond 0.55 / 0.45 pooled on any mechanism ⇒ the move-class half closes for today's
    search (the analysis asks for a second read at S5 on search v2).
  Prediction: U pooled 0.52-0.58 on LMR (a trust signal more than a move signal), lever / heavy_brk too rare to reach 200
    wrong events on LMP and futility, ≤ 0.57 where they do; overall MISS, STOP not met (U's magnitude half is real, test A).

RESULT (queue #202, 11-11 14:36; 1,500 positions, d10, 355,326 events): MISS. LMP 2,728 wrong of 300,638 and LMR 378 of
19,607: every marker within 0.02 of 0.5 pooled, within 0.01 inside the strata (U 0.498 / 0.485). The only excursion is U on
futility, 0.433 on 26 wrong events (1.2 SE). Plan-type moves are wrongly pruned slightly LESS often than other late quiets
(LMP: lever 0.65%, heavy_brk 0.84%, brk_king 0.74% vs 0.91%). The prediction for LMR was wrong. Reading and placement: search
v2 plan §7d.
"""
import os, sys, csv, json, collections, math
KV = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
MODE = KV.get("MODE", "auc")
THIS = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(THIS)
sys.path.insert(0, ENGINE); sys.path.insert(0, THIS)
# a test build of the engine module, kept apart from the live one the UI loads: it must come FIRST on the path and be
# imported here, before tactical_test puts the engine folder (and its live module) in front again
CHESSAI_DIR = os.environ.get("CHESSAI_DIR") or KV.get("CHESSAI_DIR")
if CHESSAI_DIR:
    sys.path.insert(0, CHESSAI_DIR)
    import ChessAI as _chessai_module
    assert os.path.dirname(os.path.abspath(_chessai_module.__file__)) == os.path.abspath(CHESSAI_DIR), _chessai_module.__file__
    print("engine module: %s" % _chessai_module.__file__, flush=True)
os.environ.setdefault("PRESET", "LONG_FORMAT"); os.environ.setdefault("USE_OPENING_BOOK", "0")
import chess
BETA_CACHE = "/mnt/e/chess_data/pot_test_a_beta.json"


def mode_pass():
    from tactical_test import run_one
    rows = [r["fen"] for r in csv.DictReader(open(KV["IN"], newline="")) if r.get("fen")]
    k, n = (int(x) for x in KV.get("SHARD", "0/1").split("/"))
    rows = [f for i, f in enumerate(rows) if i % n == k]
    if int(KV.get("N", 0)):
        rows = rows[:int(KV["N"])]
    out = KV["OUT"]
    saved_out, saved_err = os.dup(1), os.dup(2)
    null = os.open(os.devnull, os.O_WRONLY)
    err = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_APPEND)
    os.dup2(null, 1); os.dup2(err, 2)
    try:
        for j, fen in enumerate(rows):
            run_one(fen, [])
            if j % 100 == 99:
                os.write(saved_out, ("  %d / %d positions\n" % (j + 1, len(rows))).encode())
    finally:
        os.dup2(saved_out, 1); os.dup2(saved_err, 2)
    print("done: %d positions → %s" % (len(rows), out))


def board_of(d):
    b = chess.Board(None)
    W, B = d["bw"], d["bb"]
    for pt, key in ((chess.PAWN, "bp"), (chess.KNIGHT, "bn"), (chess.BISHOP, "bbi"), (chess.ROOK, "br"), (chess.QUEEN, "bq"),
                    (chess.KING, "bk")):
        m = d[key]
        for sq in chess.SquareSet(m):
            b.set_piece_at(sq, chess.Piece(pt, bool((W >> sq) & 1)))
    b.turn = bool(d["tu"])
    return b


def break_files(b):
    files = set()
    for c in (chess.WHITE, chess.BLACK):
        own, opp = b.pieces(chess.PAWN, c), b.pieces(chess.PAWN, not c)
        fwd = 8 if c == chess.WHITE else -8
        for sq in own:
            if chess.BB_PAWN_ATTACKS[c][sq] & int(opp):
                files.add(chess.square_file(sq))                      # in tension now
            to = sq + fwd
            if 0 <= to < 64 and not b.piece_at(to) and chess.BB_PAWN_ATTACKS[c][to] & int(opp):
                files.add(chess.square_file(sq))                      # one push from a lever
    return files


def markers(b, fr, to, U):
    pc = b.piece_at(fr)
    tgt = b.piece_at(to)
    lever = 0
    if pc and pc.piece_type == chess.PAWN:
        if tgt and tgt.piece_type == chess.PAWN:
            lever = 1
        elif not tgt and chess.BB_PAWN_ATTACKS[pc.color][to] & b.pieces_mask(chess.PAWN, not pc.color):
            lever = 1
    bf = break_files(b)
    heavy = int(bool(pc and pc.piece_type in (chess.ROOK, chess.QUEEN) and chess.square_file(to) in bf))
    kfiles = {chess.square_file(k) for k in (b.king(chess.WHITE), b.king(chess.BLACK)) if k is not None}
    brk_file = chess.square_file(to) if heavy else (chess.square_file(fr) if lever else None)
    brk_king = int(brk_file is not None and any(abs(brk_file - kf) <= 1 for kf in kfiles))
    return {"U": U, "-U": -U, "lever": lever, "heavy_brk": heavy, "brk_king": brk_king}


def auc(scores, labels):
    pairs = sorted(zip(scores, labels))
    npos = sum(labels); nneg = len(labels) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    rsum, i = 0.0, 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1
        rsum += avg * sum(l for _, l in pairs[i:j + 1])
        i = j + 1
    return (rsum - npos * (npos + 1) / 2.0) / (npos * nneg)


def mode_auc():
    import re
    FIELD = re.compile(r"(\w+)=(-?\d+)")
    if os.path.exists(BETA_CACHE):
        beta = json.load(open(BETA_CACHE))
    else:
        saved = sys.argv; sys.argv = [sys.argv[0]]
        import _pot_mg_fit
        sys.argv = saved
        beta = [float(x) for x in _pot_mg_fit.test_a_beta()]
        json.dump(beta, open(BETA_CACHE, "w"))
    import numpy as np
    from _pot_mg_fit import feats_sum
    ev = collections.defaultdict(list)
    ucache = {}
    for path in KV["EVENTS"].split(","):
        for line in open(path, errors="replace"):
            if "[SHADOWEV]" not in line or " bp=" not in line:
                continue
            kind = re.search(r"kind=(\w+)", line).group(1)
            d = {k: int(v) for k, v in FIELD.findall(line)}
            key = (d["bp"], d["bn"], d["bbi"], d["br"], d["bq"], d["bk"], d["bw"], d["tu"])
            b = board_of(d)
            if key not in ucache:
                ucache[key] = float(np.r_[1.0, feats_sum(b)] @ np.array(beta))
            m = markers(b, d["fr"], d["to"], ucache[key])
            m.update(ent=d["ent"], i=d["i"], rd=d["rd"], hist=d["hist"])
            ev[kind].append(m)
    print("events: %s · distinct nodes %d" % (", ".join("%s %d" % (k, len(v)) for k, v in sorted(ev.items())), len(ucache)))
    MK = ["U", "-U", "lever", "heavy_brk", "brk_king"]
    keep_any, beyond_55 = [], False
    for kind, rs in sorted(ev.items()):
        lab = [r["ent"] for r in rs]
        npos = sum(lab)
        print("\n=== %s: n %d · wrong %d (%.2f%%)" % (kind, len(rs), npos, 100.0 * npos / max(1, len(rs))))
        strata = collections.defaultdict(list)
        for r in rs:
            ib = 0 if r["i"] < 4 else 1 if r["i"] < 8 else 2 if r["i"] < 16 else 3
            strata[(ib, min(r["rd"], 4), (r["hist"] > 0) - (r["hist"] < 0))].append(r)
        for mk in MK:
            sc = [r[mk] for r in rs]
            fires = sum(1 for r in rs if r[mk]) if mk in ("lever", "heavy_brk", "brk_king") else len(rs)
            wrong_f = sum(1 for r in rs if r[mk] and r["ent"]) if mk in ("lever", "heavy_brk", "brk_king") else npos
            a = auc(sc, lab)
            num = den = 0.0
            for srs in strata.values():
                sl = [r["ent"] for r in srs]
                if sum(sl) >= 20 and len(sl) - sum(sl) >= 20:
                    sa = auc([r[mk] for r in srs], sl)
                    if not math.isnan(sa):
                        num += sa * len(srs); den += len(srs)
                sa_w = num / den if den else float("nan")
            sa_w = num / den if den else float("nan")
            hi = (a >= 0.60 and sa_w >= 0.57) or (a <= 0.40 and sa_w <= 0.43)
            ok = hi and npos >= 200
            if not math.isnan(a) and abs(a - 0.5) > 0.05:
                beyond_55 = True
            if ok:
                keep_any.append((kind, mk))
            print("   %-10s pooled AUC %.4f · within strata %.4f · fires %d (wrong among them %d)%s"
                  % (mk, a, sa_w, fires, wrong_f, "  <-- KEEP" if ok else ""))
    if keep_any:
        print("\nREGISTERED (11-11, test D): KEEP — %s" % ", ".join("%s on %s" % (m, k) for k, m in keep_any))
    elif not beyond_55:
        print("\nREGISTERED (11-11, test D): STOP — no POT marker beyond 0.55 / 0.45 on any mechanism (today's search)")
    else:
        print("\nREGISTERED (11-11, test D): MISS — a marker moves past 0.55 / 0.45 somewhere, none reaches the keep bar")


{"pass": mode_pass, "auc": mode_auc}[MODE]()
