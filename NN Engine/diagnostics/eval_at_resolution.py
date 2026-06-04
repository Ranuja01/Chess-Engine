# -*- coding: utf-8 -*-
"""
Eval-at-resolution — settle whether the deep-misevals are an EVAL-function bug or
a SEARCH-pruning bug.

The engine reports the *winning* side as losing on a cluster of tactics
(WAC.213 root ~ -9 pawns vs Stockfish ~ 0; worse with depth). That -9000 is the
SEARCH-backed value, not the static eval, so it cannot indict the eval function
on its own. This tool reads the RAW STATIC eval (placement_and_piece_eval, exposed
via ChessAI.ev) at the positions the engine *should* reach — the resolution of the
winning line, found ONCE by Stockfish's deep PV and FROZEN — and lays it beside
Stockfish there. The verdict:

  - our static tracks SF's sign/magnitude along the won line  -> eval is FINE,
    the SEARCH (pruning/depth) is the bug -> dial back LMR/futility next.
  - our static stays wrong (winner reads as losing) deep in the won line
    -> EVAL bug -> term-hunt king-safety / attacking layer.
  - our -9000 matches our static at a real lost-branch leaf -> SEARCH (faithfully
    scoring a genuinely-lost position it was wrongly forced into).

Three modes (run in WSL, from NN Engine/, at the SAME interpreter that built ChessAI):

    python eval_at_resolution.py --sanity
        Cheap, no Stockfish. Validates ev() before we trust it: start position ~0,
        material imbalances have the right sign+magnitude, and a loose cross-check
        against the search on quiet positions.

    python eval_at_resolution.py --generate
        Calls Stockfish. Builds the FROZEN dataset eval_resolution_set.json ONCE
        (base position -> won-line + lost-branch FEN trajectories + SF eval at each
        ply). Re-run only to ADD base positions; do NOT regenerate per build (it
        would re-pay SF's deep search and make runs non-reproducible).

    python eval_at_resolution.py --eval [tag]
        No Stockfish. Re-scores OUR static ev() over the frozen set and prints the
        trajectory beside SF's, with a per-line verdict. This is the experiment;
        re-run it on every build whose eval/search you want to compare.

Conventions:
  - ev() returns the engine's ABSOLUTE static score (positive favours Black; the
    search flips it once via Config::side_to_play). We normalize every number to
    WHITE-POV centipawns: white_cp = -ev / 10 (engine units ~ centipawns x 10), so
    a trajectory down an alternating-move PV reads as one clean line.
  - Stockfish scores are taken White-POV too (info["score"].white()).
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # silence TensorFlow startup chatter

import re
import sys
import csv
import json
import platform
import tempfile
import contextlib

import chess

# diagnostics layout: tool lives in diagnostics/, ChessAI .so one level up, the
# frozen dataset + per-run CSVs in results/.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.path.dirname(THIS_DIR)            # NN Engine/  (has ChessAI*.so)
SUITES_DIR = os.path.join(THIS_DIR, 'suites')
RESULTS_DIR = os.path.join(THIS_DIR, 'results')
sys.path.insert(0, ENGINE_DIR)

# Stockfish is only needed for --generate; import lazily there so --sanity/--eval
# work without python-chess's engine subprocess.

# Engine units are ~ centipawns x 10 (pawn = 1000). Normalize to centipawns.
ENGINE_UNITS_PER_CP = 10.0

FROZEN_SET = os.path.join(RESULTS_DIR, "eval_resolution_set.json")

# How deep Stockfish searches at each node when freezing the lines, and how many
# plies of each line to keep. Only used by --generate; the result is frozen.
SF_GEN_DEPTH = 22
LINE_MAX_PLIES = 16

# Base positions to derive resolution trajectories for. Start focused on the
# deep-miseval cluster + the trivial control (same set line_probe.py uses); the
# third field is the KNOWN BEST move (for reference only — the lost branch uses
# OUR engine's actual choice, derived at generate time). Expand later by pointing
# --generate at a larger source.
BASE_PROBES = [
    ("WAC.213", "3r1r1k/1b4pp/ppn1p3/4Pp1R/Pn5P/3P4/4QP2/1qB1NKR1 w - - 0 1", "h5h7"),
    ("WAC.204", "r1b1qrk1/1p3ppp/p1p5/3Nb3/5N2/P7/1P4PQ/K1R1R3 w - - 0 1", "e1e5"),
    ("WAC.283", "3q1rk1/4bp1p/1n2P2Q/3p1p2/6r1/Pp2R2N/1B4PP/7K w - - 0 1", "h3g5"),
    ("WAC.018", "R7/P4k2/8/8/8/8/r7/6K1 w - - 0 1", "a8h8"),
]


# --- Load the keras models once (the C++ search doesn't use them, but the
#     ChessAI constructor signature requires them — mirrors main.py / tactical_test). ---
def _load_engine():
    # Models unused by the C++ engine — return None and skip TensorFlow (CHESS_ENABLE_TF to restore).
    from ChessAI import ChessAI
    return ChessAI, None, None


@contextlib.contextmanager
def _captured_stdout():
    """Capture fd 1 (Python prints AND the C++ engine's std::cout) into a buffer."""
    sys.stdout.flush()
    saved_fd = os.dup(1)
    tmp = tempfile.TemporaryFile(mode='w+')
    try:
        os.dup2(tmp.fileno(), 1)
        yield tmp
    finally:
        sys.stdout.flush()
        os.dup2(saved_fd, 1)
        os.close(saved_fd)


_EVAL_RE = re.compile(r'Evaluation:\s*(-?\d+)')


def white_cp_from_ev(ev_abs):
    """ev() absolute (Black-positive) -> White-POV centipawns."""
    return -ev_abs / ENGINE_UNITS_PER_CP


# ---------------------------------------------------------------------------
# --sanity : validate ev() before trusting it
# ---------------------------------------------------------------------------
def run_sanity():
    ChessAI, blackModel, whiteModel = _load_engine()

    # A single warm instance suffices: ev() reads everything from the board arg and
    # the globally-initialized attack tables; it does not depend on self state.
    seed = chess.Board()
    ai = ChessAI(blackModel, whiteModel, seed, seed.turn)

    print("=== ev() sanity (raw ABSOLUTE units: positive favours Black; pawn=1000) ===\n")

    # (label, fen, predicate on raw ev, human expectation)
    Q = 10000
    checks = [
        ("start position (symmetric)",
         chess.STARTING_FEN,
         lambda e: abs(e) < 1000,
         "~0 (|ev| < 1000)"),
        ("White up a queen",
         "4k3/8/8/8/8/8/8/3QK3 w - - 0 1",
         lambda e: e < -Q // 2,
         "strongly NEGATIVE (< -5000)"),
        ("Black up a queen",
         "3qk3/8/8/8/8/8/8/4K3 w - - 0 1",
         lambda e: e > Q // 2,
         "strongly POSITIVE (> +5000)"),
        ("White up a rook",
         "4k3/8/8/8/8/8/8/R3K3 w - - 0 1",
         lambda e: e < -1500,
         "NEGATIVE (< -1500)"),
        ("Black up a rook",
         "r3k3/8/8/8/8/8/8/4K3 w - - 0 1",
         lambda e: e > 1500,
         "POSITIVE (> +1500)"),
    ]

    all_ok = True
    print(f"{'check':<32} {'ev(abs)':>9} {'white_cp':>9}  {'expect':<24} res")
    for label, fen, pred, expect in checks:
        ev_abs = ai.ev(chess.Board(fen))
        ok = pred(ev_abs)
        all_ok &= ok
        print(f"{label:<32} {ev_abs:>9} {white_cp_from_ev(ev_abs):>9.0f}  "
              f"{expect:<24} {'OK' if ok else 'FAIL'}")

    # Loose cross-check vs the search on quiet positions. The search reports a
    # side-to-move-POV value (Config::side_to_play-flipped); on a quiet position it
    # should be in the same ballpark as the static eval normalized to STM-POV. This
    # is informational (search adds q-search + small adjustments), NOT a hard gate.
    print("\n--- informational: ev() vs search on quiet positions (STM-POV units) ---")
    print(f"{'fen (quiet)':<44} {'ev_stm':>8} {'search':>8}  note")
    quiet = [
        "8/5k2/8/8/3K4/8/8/8 w - - 0 1",          # bare kings, ~0
        "8/8/4k3/8/8/4K3/4P3/8 w - - 0 1",        # K+P vs K, White edge
        "8/4p3/4k3/8/8/4K3/8/8 w - - 0 1",        # K+P vs K, Black edge
    ]
    for fen in quiet:
        b = chess.Board(fen)
        ev_abs = ai.ev(b)
        ev_stm = ev_abs if not b.turn else -ev_abs  # absolute(Black+) -> STM-POV
        with _captured_stdout() as buf:
            ai2 = ChessAI(blackModel, whiteModel, b, b.turn)
            ai2.alphaBetaWrapper()
            buf.seek(0)
            out = buf.read()
        m = _EVAL_RE.findall(out)
        search = m[-1] if m else "n/a"
        print(f"{fen:<44} {ev_stm:>8} {str(search):>8}  "
              f"{'(book?)' if 'Book Move' in out else ''}")

    print()
    print("SANITY: " + ("PASS — ev() sign+magnitude look correct."
                        if all_ok else
                        "FAIL — ev() is miscalibrated; fix the wrapper before --generate/--eval."))
    return 0 if all_ok else 1


# ---------------------------------------------------------------------------
# --generate : build the FROZEN resolution set (calls Stockfish + our engine once)
# ---------------------------------------------------------------------------
def _find_stockfish():
    import shutil
    for var in ("STOCKFISH_PATH", "STOCKFISH"):
        p = os.environ.get(var)
        if p and os.path.exists(p):
            return p
    w = shutil.which("stockfish")
    if w:
        return w
    for c in (
        "/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/stockfish/stockfish-windows-x86-64-avx2.exe",
        "/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/stockfish/stockfish.exe",
        "../stockfish/stockfish-windows-x86-64-avx2.exe",
    ):
        if os.path.exists(c):
            return c
    return None


def _sf_line(engine, start_board, depth, max_plies):
    """Walk Stockfish's own best line from start_board, re-analysing at each node.
    Returns [{ply, fen, sf_cp_white}], one entry per visited position."""
    import chess.engine
    nodes = []
    b = start_board.copy()
    for ply in range(max_plies):
        info = engine.analyse(b, chess.engine.Limit(depth=depth))
        score = info.get("score")
        cp = score.white().score(mate_score=100000) if score else None
        nodes.append({"ply": ply, "fen": b.fen(), "sf_cp_white": cp})
        if b.is_game_over():
            break
        pv = info.get("pv")
        if not pv:
            break
        b.push(pv[0])
    return nodes


def _our_move(ChessAI, blackModel, whiteModel, fen):
    """Run OUR engine on a position and return its chosen move UCI (or None / 'BOOK')."""
    b = chess.Board(fen)
    with _captured_stdout() as buf:
        ai = ChessAI(blackModel, whiteModel, b, b.turn)
        mv = ai.alphaBetaWrapper()
        buf.seek(0)
        out = buf.read()
    if "Book Move" in out:
        return "BOOK"
    return mv.uci() if mv is not None else None


def run_generate():
    import chess.engine
    sf = _find_stockfish()
    if not sf:
        print("Stockfish not found. Set STOCKFISH_PATH=/path/to/stockfish (or put it on PATH).")
        return 1

    ChessAI, blackModel, whiteModel = _load_engine()

    print(f"Generating frozen resolution set -> {FROZEN_SET}")
    print(f"Stockfish: {sf}   depth={SF_GEN_DEPTH}, max {LINE_MAX_PLIES} plies/line\n")

    engine = chess.engine.SimpleEngine.popen_uci(sf)
    try:
        engine.configure({"Threads": 1})  # determinism
    except Exception:
        pass

    dataset = {}
    try:
        for label, fen, best in BASE_PROBES:
            base = chess.Board(fen)

            # WON line: Stockfish's best play from the base.
            won = _sf_line(engine, base, SF_GEN_DEPTH, LINE_MAX_PLIES)

            # LOST branch: OUR engine's actual move, then Stockfish plays it out.
            our = _our_move(ChessAI, blackModel, whiteModel, fen)
            lost = []
            if our and our != "BOOK":
                try:
                    lb = base.copy()
                    lb.push(chess.Move.from_uci(our))
                    lost = _sf_line(engine, lb, SF_GEN_DEPTH, LINE_MAX_PLIES)
                except Exception as e:
                    print(f"   !! {label}: could not build lost branch from our move {our}: {e}")

            dataset[label] = {
                "base_fen": fen,
                "best_move": best,        # reference only
                "our_move": our,          # what the lost branch is built on
                "won_line": won,
                "lost_line": lost,
            }

            won_end = won[-1]["sf_cp_white"] if won else None
            lost_end = lost[-1]["sf_cp_white"] if lost else None
            print(f"{label:<9} our={str(our):<6} "
                  f"won_line {len(won):>2} plies (SF end {str(won_end):>7} cp w)  |  "
                  f"lost_line {len(lost):>2} plies (SF end {str(lost_end):>7} cp w)")
    finally:
        engine.quit()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(FROZEN_SET, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"\nFrozen {len(dataset)} base positions -> {FROZEN_SET}")
    print("Eyeball: along each won_line, SF's cp should swing strongly toward the "
          "winner. This file is now FROZEN — re-run --generate only to add bases.")
    return 0


# ---------------------------------------------------------------------------
# --eval : re-score OUR static eval over the frozen set (the experiment)
# ---------------------------------------------------------------------------
# |white_cp| beyond this is a forced-mate score, not a positional evaluation.
MATE_CP = 30000


def _verdict(line_nodes):
    """Judge whether our static eval AGREES with SF at the RESOLUTION.

    Crucial: a static eval can never represent a forced mate. If SF's line ends in
    mate, comparing our static eval to a +99990 mate score is meaningless — the
    win is TACTICAL, so the only meaningful question is whether the line is a
    sacrifice (static rightly shows our side materially worse) that SEARCH/depth
    must carry. We therefore compare at the last node where SF is still a
    positional number (material settled, mate not yet scored)."""
    nodes = [n for n in line_nodes if n["sf_cp_white"] is not None]
    if not nodes:
        return "no SF data"
    sf_end = nodes[-1]["sf_cp_white"]
    settled = [n for n in nodes if abs(n["sf_cp_white"]) < MATE_CP]
    ref = settled[-1] if settled else nodes[0]
    sf_ref, our_ref = ref["sf_cp_white"], ref["our_cp_white"]

    if abs(sf_end) >= MATE_CP:
        return (f"MATING line (SF->mate); static can't represent mate. "
                f"Last positional node: SF={sf_ref:+.0f} ours={our_ref:+.0f} cp "
                f"=> win is TACTICAL -> SEARCH/depth, not eval.")

    winner = "balanced" if abs(sf_ref) < 75 else ("White" if sf_ref > 0 else "Black")
    same_sign = (sf_ref >= 0) == (our_ref >= 0)
    agree = (abs(our_ref) < 300) if winner == "balanced" else (same_sign and abs(our_ref) > 150)
    return (f"SF={sf_ref:+.0f} ours={our_ref:+.0f} cp  "
            f"-> {'AGREE (eval sees it)' if agree else 'DISAGREE (eval blind here)'}")


def run_eval(tag):
    if not os.path.exists(FROZEN_SET):
        print(f"{FROZEN_SET} not found — run `python eval_at_resolution.py --generate` first.")
        return 1
    with open(FROZEN_SET) as f:
        dataset = json.load(f)

    ChessAI, blackModel, whiteModel = _load_engine()
    seed = chess.Board()
    ai = ChessAI(blackModel, whiteModel, seed, seed.turn)  # one warm instance for all FENs

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, f"eval_at_resolution_{tag}.csv")
    rows = []

    print(f"Re-scoring OUR static eval over {FROZEN_SET}  (tag={tag})")
    print("All numbers WHITE-POV centipawns. our_cp = -ev/10.\n")

    for label, rec in dataset.items():
        print(f"=== {label}   base={rec['base_fen']}")
        print(f"    best={rec.get('best_move')}  our_move={rec.get('our_move')}")
        for line_name in ("won_line", "lost_line"):
            nodes = rec.get(line_name) or []
            if not nodes:
                print(f"    [{line_name}] (none)")
                continue
            print(f"    [{line_name}]  {'ply':>3} {'sf_cp_w':>8} {'our_cp_w':>9} {'delta':>8}")
            scored = []
            for n in nodes:
                ours = white_cp_from_ev(ai.ev(chess.Board(n["fen"])))
                sf = n["sf_cp_white"]
                delta = (ours - sf) if sf is not None else None
                scored.append({"sf_cp_white": sf, "our_cp_white": ours})
                print(f"    {'':>9} {n['ply']:>3} {('' if sf is None else f'{sf:>8.0f}')} "
                      f"{ours:>9.0f} {('' if delta is None else f'{delta:>8.0f}')}")
                rows.append({"base": label, "line": line_name, "ply": n["ply"],
                             "sf_cp_white": sf, "our_cp_white": round(ours, 1),
                             "delta": (None if delta is None else round(delta, 1)),
                             "fen": n["fen"]})
            print(f"    [{line_name}] verdict: {_verdict(scored)}")
        print()

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["base", "line", "ply", "sf_cp_white",
                                          "our_cp_white", "delta", "fen"])
        w.writeheader()
        w.writerows(rows)

    print(f"-> {csv_path}")
    print("\nReading it: on the WON line, if our_cp tracks SF toward the winner by the "
          "resolution, the eval is FINE and the bug is SEARCH/pruning. If our_cp stays "
          "on the wrong side deep into the won line, it's an EVAL bug. On the LOST "
          "branch, our_cp at the leaf ~ our reported -9000 means the search faithfully "
          "scored a genuinely-lost line it was wrongly forced into.")
    return 0


def main():
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    mode = args[0]
    if mode == "--sanity":
        return run_sanity()
    if mode == "--generate":
        return run_generate()
    if mode == "--eval":
        tag = args[1] if len(args) > 1 else "run"
        return run_eval(tag)
    print(f"Unknown mode {mode!r}. Use --sanity | --generate | --eval [tag].")
    return 2


if __name__ == "__main__":
    sys.exit(main())
