# -*- coding: utf-8 -*-
"""Self-play driver / referee.

Spawns two `engine_server.py` subprocesses (one per side, fully cache-isolated by being separate
OS processes), relays moves between them over the PUSH/GO/QUIT protocol, enforces the rules with
python-chess, and reports the result. Configs A and B are env-knob sets passed per process, so each
side can run different settings (`--config-a "ASPIRATION_DELTA=500"` vs `--config-b
"ASPIRATION_DELTA=0"`).

This is the Step-2 minimal version: ONE game, no Stockfish arbiter, no rich logging — just prove two
servers play a legal game to a correct termination. Steps 3–5 add the SF arbiter, per-move JSONL
logs, opening seeding, the N-game tournament, and adjudication.
"""

import os
import sys
import subprocess
import argparse
import json

ENGINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ENGINE_DIR)
import chess
import chess.pgn

from arbiter import Arbiter, find_stockfish

_SERVER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "engine_server.py")


def _parse_config(s):
    """'KNOB=v KNOB2=v2' -> {'KNOB': 'v', ...} to overlay on the child env."""
    env = {}
    for tok in (s or "").split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            env[k] = v
    return env


class EngineProc:
    """A persistent engine_server subprocess for one side, with a line-protocol wrapper."""

    def __init__(self, color, config, start_fen, label, stderr_path):
        self.color = color
        self.label = label
        env = os.environ.copy()
        env.update(_parse_config(config))
        self._stderr = open(stderr_path, "w")
        self.proc = subprocess.Popen(
            [sys.executable, "-u", _SERVER, "--color", color, "--start-fen", start_fen],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr,
            cwd=ENGINE_DIR, text=True, bufsize=1, env=env,
        )
        self._stderr_path = stderr_path
        self.backtrace_path = None  # set when running under gdb (--gdb), else None

    def _readline(self):
        line = self.proc.stdout.readline()
        if not line:  # EOF = the server died
            raise RuntimeError(
                f"engine '{self.label}' ({self.color}) exited unexpectedly; see {self._stderr_path}")
        return line.strip()

    def _send(self, cmd):
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def wait_ready(self):
        while True:
            if self._readline() == "READY":
                return

    def go(self):
        """Ask for a move. Returns (kind, uci|None, calc_dict). kind in {'MOVE','RESIGN'}."""
        self._send("GO")
        line = self._readline()
        if line.startswith("MOVE "):
            rest = line[5:]
            uci, _, js = rest.partition(" ")
            return "MOVE", uci, (json.loads(js) if js else {})
        if line.startswith("RESIGN"):
            js = line[6:].strip()
            return "RESIGN", None, (json.loads(js) if js else {})
        raise RuntimeError(f"engine '{self.label}' bad GO reply: {line!r}")

    def push(self, uci):
        self._send("PUSH " + uci)
        line = self._readline()
        if not line.startswith("OK"):
            raise RuntimeError(f"engine '{self.label}' rejected PUSH {uci}: {line!r}")

    def quit(self):
        try:
            self._send("QUIT")
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()
        finally:
            self._stderr.close()


def _write(fh, record):
    """Single-writer, append-one-JSON-line, flushed + fsync'd so the file can be tailed live."""
    fh.write(json.dumps(record) + "\n")
    fh.flush()
    os.fsync(fh.fileno())


def _move_comment(rec):
    """Compact per-move PGN comment: the engine's eval (pawn=1000 scale) and, when present, the
    Stockfish cp (standard centipawns, pawn=100). E.g. 'A +4.72/d13  SF +1.10/d18 bm e6e7'."""
    if rec.get("opening"):
        return ""   # seeded opening moves carry no eval — keep the PGN clean
    parts = []
    label = rec.get("label", "")
    ev = rec.get("eval_white_pov")
    if isinstance(ev, int):
        d = rec.get("depth")
        parts.append(f"{label} {ev / 1000.0:+.2f}" + (f"/d{d}" if d is not None else ""))
    elif rec.get("booked"):
        parts.append(f"{label} book")
    sf = rec.get("sf_cp")
    if isinstance(sf, int):
        s = f"SF {sf / 100.0:+.2f}"
        if rec.get("sf_depth") is not None:
            s += f"/d{rec['sf_depth']}"
        if rec.get("sf_best"):
            s += f" bm {rec['sf_best']}"
        parts.append(s)
    return "  ".join(parts)


def write_pgn(logdir, meta, move_records, result, reason, start_fen, filename="game.pgn"):
    """Build games/<tag>/<filename> from the accumulated move records, with the engine eval (and SF cp
    when present) in each move's comment. Returns the PGN text. Robust to a partial/crashed game."""
    game = chess.pgn.Game()
    game.headers["Event"] = meta.get("event", "selfplay")
    game.headers["White"] = meta.get("white", "A")
    game.headers["Black"] = meta.get("black", "B")
    game.headers["Result"] = result
    game.headers["WhiteConfig"] = meta.get("config_white") or "-"
    game.headers["BlackConfig"] = meta.get("config_black") or "-"
    if reason:
        game.headers["Termination"] = reason
    if start_fen and start_fen != chess.STARTING_FEN:
        game.setup(chess.Board(start_fen))
    node = game
    for rec in move_records:
        try:
            move = chess.Move.from_uci(rec["uci"])
        except (ValueError, KeyError, TypeError):
            continue
        node = node.add_main_variation(move)
        comment = _move_comment(rec)
        if comment:
            node.comment = comment
    pgn_str = str(game)
    with open(os.path.join(logdir, filename), "w") as f:
        f.write(pgn_str + "\n")
    return pgn_str


def _write_crash_bundle(logdir, label, color, config, start_fen, moves, crash_fen, backtrace_path):
    """On any engine crash, dump a self-contained bundle with everything needed to replay it — so a
    rare/non-reproducible crash is still fully recorded the one time it fires."""
    rel = "selfplay/replay.py"
    sf = "" if start_fen == chess.STARTING_FEN else f" --start-fen '{start_fen}'"
    env = (config + " ").lstrip()
    replay_cmd = f"{env}python {rel} --color {color} --moves '{' '.join(moves)}'{sf}"
    bundle = {
        "crashed_label": label, "crashed_color": color, "config": config,
        "start_fen": start_fen, "moves": list(moves), "crash_fen": crash_fen,
        "backtrace": backtrace_path, "replay_cmd": replay_cmd,
        "note": "Replay may not reproduce a timing-specific crash under a time preset; for a "
                "deterministic attempt prefix PRESET=LONG_FORMAT MAX_DEPTH=<N>. Run under gdb for a "
                "backtrace (add -g to setupAI.py).",
    }
    path = os.path.join(logdir, "crash.json")
    with open(path, "w") as f:
        json.dump(bundle, f, indent=2)
    return path


class Adjudicator:
    """Heuristic-gated, single-Stockfish-confirm game adjudication.

    Cheap python-chess signals (no-progress halfmove clock, low piece count + flat eval, repetition)
    GATE the check; only when a gate fires is ONE arbiter.evaluate() call spent to confirm. On
    confirmation the game is flagged to end after the NEXT played move (the position gets one more
    move), which cuts the long won-game grind to the -15p resign and the dead-drawn shuffle tails. A
    per-window cooldown bounds Stockfish to at most one call per `window` plies. All thresholds are
    White-POV; engine evals arrive as mover-normalised White-POV milli-pawns."""

    def __init__(self, arbiter, draw_cp=40, win_p=5.0, noprog_plies=40, low_pieces=12, window=6,
                 do_win=True, do_draw=True):
        self.arbiter = arbiter
        self.do_win = do_win             # adjudicate clearly-won positions (off => let wins grind to mate/resign)
        self.do_draw = do_draw           # adjudicate dead-drawn positions
        self.draw_cp = draw_cp            # SF |cp| below this confirms a draw
        self.win_p = win_p               # engine win-margin gate, in pawns (White-POV)
        self.win_confirm_cp = 300        # SF must agree the leader is up >= this (cp) to confirm decisive
        self.flat_p = 1.0                # |engine eval| below this (pawns) counts as "flat" for the draw gate
        self.noprog_plies = noprog_plies
        self.low_pieces = low_pieces
        self.window = window
        self.evals = []                  # recent White-POV pawn evals (one per played move; None if absent)
        self.pending = None              # (result, reason) once a gate is confirmed
        self.pending_at = None           # len(moves) when pending was set
        self.last_probe = -10 ** 9       # last ply an SF confirmation was spent (cooldown anchor)

    def after_move(self, board, ewp, n_moves, fh):
        """Call once per played (non-opening) move with the resulting board and the mover's White-POV
        milli-pawn eval. Returns (result, reason) to terminate the game, or None to continue."""
        if self.pending is not None:                      # confirmed earlier -> end one move later
            return self.pending if n_moves > self.pending_at else None
        self.evals.append(ewp / 1000.0 if isinstance(ewp, int) else None)
        if self.arbiter is None or n_moves - self.last_probe < self.window:
            return None
        recent = [e for e in self.evals[-self.window:] if e is not None]
        full = len(recent) >= self.window
        side = gate = None
        if self.do_win and full and all(e >= self.win_p for e in recent):
            side = "white"
        elif self.do_win and full and all(e <= -self.win_p for e in recent):
            side = "black"
        elif self.do_draw and board.halfmove_clock >= self.noprog_plies:
            gate = "no-progress %d plies" % board.halfmove_clock
        elif self.do_draw and chess.popcount(board.occupied) <= self.low_pieces and full and all(abs(e) <= self.flat_p for e in recent):
            gate = "low material (%d) + flat eval" % chess.popcount(board.occupied)
        elif self.do_draw and board.is_repetition(2):
            gate = "repetition"
        if side is None and gate is None:
            return None
        # A gate fired: spend exactly one Stockfish call to confirm before adjudicating.
        self.last_probe = n_moves
        cp, _, _ = self.arbiter.evaluate(board)
        if cp is None:
            return None
        if side is not None:
            if (side == "white" and cp >= self.win_confirm_cp) or (side == "black" and cp <= -self.win_confirm_cp):
                self._set_pending("1-0" if side == "white" else "0-1",
                                  "adjudicated win (%s, SF %+dcp)" % (side, cp), n_moves, "win", cp, fh)
        elif abs(cp) < self.draw_cp:
            self._set_pending("1/2-1/2", "adjudicated draw (%s, SF %+dcp)" % (gate, cp), n_moves, "draw", cp, fh)
        return None

    def _set_pending(self, result, reason, n_moves, kind, cp, fh):
        self.pending = (result, reason)
        self.pending_at = n_moves
        if fh:
            _write(fh, {"type": "adjudication", "ply": n_moves, "kind": kind, "reason": reason, "sf_cp": cp})


def play_game(config_a, config_b, label_a, label_b, start_fen, max_plies, logdir,
              jsonl_path=None, verbose=True, arbiter=None, opening_moves=None, adjudicator=None):
    """Play one game: A is White, B is Black. Returns a dict with result + the move list (and the PGN).
    If jsonl_path is given, writes one record per move (the driver is the sole writer) live. If an
    arbiter is given, each position is scored by Stockfish (White-POV cp) between moves. If
    opening_moves is given, those UCI moves are PUSHed to both engines + the master board first (no
    search) and the engines are forced to USE_OPENING_BOOK=0 so they THINK from the seeded position.
    If an adjudicator is given, it may end the game early once a heuristic gate is Stockfish-confirmed
    (the `arbiter` observation path and the `adjudicator` confirmation path are independent)."""
    os.makedirs(logdir, exist_ok=True)
    if opening_moves:
        # Seeding implies the engines must not march their own book from the seed.
        config_a = (config_a + " USE_OPENING_BOOK=0").strip()
        config_b = (config_b + " USE_OPENING_BOOK=0").strip()
    white = EngineProc("white", config_a, start_fen, label_a, os.path.join(logdir, "white.stderr"))
    black = EngineProc("black", config_b, start_fen, label_b, os.path.join(logdir, "black.stderr"))
    board = chess.Board(start_fen)
    moves = []
    move_records = []   # one dict per played move, reused for the JSONL line and the PGN comment
    pgn_str = None
    result, reason = "*", "in-progress"
    fh = open(jsonl_path, "w") if jsonl_path else None
    try:
        if fh:
            _write(fh, {"type": "meta", "white": label_a, "black": label_b,
                        "config_white": config_a, "config_black": config_b, "start_fen": start_fen,
                        "opening": " ".join(opening_moves) if opening_moves else ""})
        white.wait_ready()
        black.wait_ready()
        # Seed the opening (if any): PUSH each move to both servers + the master board, no search.
        for uci in (opening_moves or []):
            mv = chess.Move.from_uci(uci)
            if mv not in board.legal_moves:
                raise ValueError(f"illegal opening move {uci} at {board.fen()}")
            move_no, color = board.fullmove_number, ("white" if board.turn else "black")
            board.push(mv)
            white.push(uci)
            black.push(uci)
            moves.append(uci)
            rec = {"type": "move", "ply": len(moves), "move_no": move_no, "color": color,
                   "label": "opening", "uci": uci, "fen": board.fen(), "opening": True}
            move_records.append(rec)
            if fh:
                _write(fh, rec)
            if verbose:
                print(f"  {len(moves):>3}. {color[0].upper()} {uci:<6} [opening]", flush=True)
        while True:
            if board.is_game_over(claim_draw=True):
                result, reason = board.result(claim_draw=True), board_outcome_reason(board)
                break
            if board.ply() >= max_plies:
                result, reason = "1/2-1/2", f"max-plies ({max_plies})"
                break
            mover = white if board.turn else black
            other = black if board.turn else white
            move_no = board.fullmove_number
            try:
                kind, uci, calc = mover.go()
            except RuntimeError as e:
                # The engine process died (segfault, etc.) — forfeit this game, keep the run alive,
                # and dump a full crash bundle (moves/FEN/config/replay cmd) for forensics.
                result = "0-1" if mover.color == "white" else "1-0"
                cfg = config_a if mover.color == "white" else config_b
                bpath = _write_crash_bundle(logdir, mover.label, mover.color, cfg, start_fen,
                                            moves, board.fen(), mover.backtrace_path)
                reason = f"{mover.label} ({mover.color}) crashed at {board.fen()}  [bundle: {bpath}]"
                if fh:
                    _write(fh, {"type": "crash", "color": mover.color, "label": mover.label,
                                "fen": board.fen(), "error": str(e), "bundle": bpath})
                break
            if kind == "RESIGN":
                result = "0-1" if board.turn else "1-0"
                reason = f"{mover.label} resigned"
                if fh:
                    _write(fh, {"type": "move", "ply": len(moves) + 1, "move_no": move_no,
                                "color": mover.color, "label": mover.label, "uci": None,
                                "resign": True, "fen": board.fen(), **(calc or {})})
                break
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                result, reason = "*", f"ILLEGAL move {uci} by {mover.label} at {board.fen()}"
                break
            board.push(move)
            try:
                other.push(uci)
            except RuntimeError as e:
                result = "0-1" if other.color == "white" else "1-0"
                cfg = config_a if other.color == "white" else config_b
                bpath = _write_crash_bundle(logdir, other.label, other.color, cfg, start_fen,
                                            moves + [uci], board.fen(), other.backtrace_path)
                reason = f"{other.label} ({other.color}) crashed on PUSH {uci}  [bundle: {bpath}]"
                if fh:
                    _write(fh, {"type": "crash", "color": other.color, "label": other.label,
                                "fen": board.fen(), "error": str(e), "bundle": bpath})
                break
            moves.append(uci)
            # Each engine reports eval in its own side_to_play frame (White-positive vs Black-positive);
            # normalise to a single White-POV column for readability + SF compare.
            ev = (calc or {}).get("eval")
            ewp = (ev if mover.color == "white" else -ev) if isinstance(ev, int) else None
            rec = {"type": "move", "ply": len(moves), "move_no": move_no,
                   "color": mover.color, "label": mover.label, "uci": uci,
                   "fen": board.fen(), "eval_white_pov": ewp, **(calc or {})}
            # Stockfish scores the resulting position between moves (both engines idle) — never
            # competes for CPU with a search. Skip book plies (known theory, not worth arbitrating).
            # Failures are recorded but never abort the game.
            if arbiter is not None and not (calc or {}).get("booked"):
                try:
                    sf_cp, sf_best, sf_depth = arbiter.evaluate(board)
                    rec["sf_cp"], rec["sf_best"], rec["sf_depth"] = sf_cp, sf_best, sf_depth
                except Exception as sf_err:
                    rec["sf_error"] = str(sf_err)
            move_records.append(rec)
            if fh:
                _write(fh, rec)
            if verbose:
                if (calc or {}).get("booked"):
                    tag = "book"
                else:
                    tag = f"d{calc.get('depth')} ev {calc.get('eval')}"
                    sf_cp = rec.get("sf_cp")
                    if isinstance(sf_cp, int):
                        # Show SF in the same frame + unit as the engine's mover-POV milli-pawn `ev`
                        # (SF is stored White-POV centipawns): flip to mover-POV, ×10 to milli-pawns.
                        tag += f"  SF {(sf_cp if mover.color == 'white' else -sf_cp) * 10}"
                print(f"  {len(moves):>3}. {mover.color[0].upper()} {uci:<6} [{tag}]", flush=True)
            if adjudicator is not None:
                adj = adjudicator.after_move(board, ewp, len(moves), fh)
                if adj is not None:
                    result, reason = adj
                    break
    finally:
        white.quit()
        black.quit()
        if fh:
            _write(fh, {"type": "result", "result": result, "reason": reason,
                        "plies": len(moves)})
            fh.close()
        try:
            pgn_meta = {"event": os.path.basename(logdir), "white": label_a, "black": label_b,
                        "config_white": config_a, "config_black": config_b}
            pgn_str = write_pgn(logdir, pgn_meta, move_records, result, reason, start_fen)
        except Exception as pgn_err:
            print(f"[selfplay] PGN export failed: {pgn_err}", flush=True)
    return {"result": result, "reason": reason, "plies": len(moves), "moves": moves,
            "fen": board.fen(), "pgn": pgn_str, "pgn_path": os.path.join(logdir, "game.pgn")}


def board_outcome_reason(board):
    if board.is_checkmate():
        return "checkmate"
    if board.is_stalemate():
        return "stalemate"
    if board.is_insufficient_material():
        return "insufficient material"
    if board.is_seventyfive_moves():
        return "75-move rule"
    if board.is_fivefold_repetition():
        return "fivefold repetition"
    if board.can_claim_fifty_moves():
        return "50-move claim"
    if board.can_claim_threefold_repetition():
        return "threefold claim"
    return "game over"


def main():
    ap = argparse.ArgumentParser(description="Self-play driver: one game, optional Stockfish arbiter + PGN.")
    ap.add_argument("--config-a", default="", help="env knobs for side A (White), e.g. 'ASPIRATION_DELTA=500'")
    ap.add_argument("--config-b", default="", help="env knobs for side B (Black)")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--start-fen", default=chess.STARTING_FEN)
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--tag", default="smoke")
    ap.add_argument("--sf-path", default=None, help="Stockfish binary (default: auto-detect)")
    ap.add_argument("--sf-movetime", type=float, default=0.3,
                    help="Stockfish per-position time budget in seconds (time-gated mode; default 0.3)")
    ap.add_argument("--sf-depth", type=int, default=None,
                    help="Stockfish fixed depth (overrides --sf-movetime if set)")
    ap.add_argument("--no-sf", action="store_true", help="disable the Stockfish arbiter")
    ap.add_argument("--opening", default="", help="space-separated UCI moves to seed before play "
                    "(forces USE_OPENING_BOOK=0 so the engines think from the seed)")
    args = ap.parse_args()

    logdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "games", args.tag)
    os.makedirs(logdir, exist_ok=True)
    jsonl_path = os.path.join(logdir, "game.jsonl")

    # The arbiter is on by default when a Stockfish binary is found; a missing binary just disables it
    # (the game still plays, the SF columns are simply absent).
    arbiter = None
    if not args.no_sf:
        sf = args.sf_path or find_stockfish()
        if sf:
            try:
                arbiter = Arbiter(sf, movetime=args.sf_movetime, depth=args.sf_depth)
                mode = f"depth {args.sf_depth}" if args.sf_depth else f"{args.sf_movetime}s/move"
                print(f"[selfplay] Stockfish arbiter ON ({sf}, {mode})", flush=True)
            except Exception as e:
                print(f"[selfplay] WARNING: could not start Stockfish ({e}); arbiter OFF", flush=True)
        else:
            print("[selfplay] WARNING: no Stockfish binary found (set STOCKFISH_PATH or --sf-path); "
                  "arbiter OFF", flush=True)

    print(f"[selfplay] {args.label_a} (White) vs {args.label_b} (Black)", flush=True)
    print(f"[selfplay] live game log -> {jsonl_path}  (tail -f it)", flush=True)
    try:
        g = play_game(args.config_a, args.config_b, args.label_a, args.label_b,
                      args.start_fen, args.max_plies, logdir, jsonl_path=jsonl_path, arbiter=arbiter,
                      opening_moves=(args.opening.split() or None))
    finally:
        if arbiter is not None:
            arbiter.close()
    print(f"\n[selfplay] result {g['result']} ({g['reason']}) in {g['plies']} plies", flush=True)
    print(f"[selfplay] PGN -> {g['pgn_path']}\n", flush=True)
    if g.get("pgn"):
        print(g["pgn"], flush=True)


if __name__ == "__main__":
    main()
