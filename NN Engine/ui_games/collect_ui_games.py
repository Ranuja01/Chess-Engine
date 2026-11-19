"""
Append the games in a UI session log (or a pasted PGN file) to ui_games/ui_games.pgn, with the build that played them.

	python ui_games/collect_ui_games.py ui_games/logs/<session>.log
	python ui_games/collect_ui_games.py pasted.pgn --build "gen 7.6 (chess_engine_uci_g76pgo)" --knobs "ENABLE_QSTANDPAT_SEED=1" --note "..."

A session log written by selfplay/play_ui.sh starts with "# build:", "# knobs:" and "# started:" lines; those fill the headers,
and --build / --knobs override them. Every PGN block in the file (the UI prints one when a game ends, and the partial game when
the window closes early) is read; a game whose moves are already in the archive is skipped, so re-running on the same log is safe.

@author: Ranuja Pinnaduwage
"""
import argparse, datetime as dt, io, os, re
import chess, chess.pgn

HERE = os.path.dirname(os.path.abspath(__file__))
ARCHIVE = os.path.join(HERE, "ui_games.pgn")


def key_of(game):
	"""Identity of a game: its start position and its first 60 moves."""
	return (game.headers.get("FEN", ""), " ".join(m.uci() for m in list(game.mainline_moves())[:60]))


def read_games(text):
	"""Every PGN block in the text, each parsed from its own [Event tag."""
	starts = [m.start() for m in re.finditer(r"^\[Event ", text, re.M)]
	for i, s in enumerate(starts):
		game = chess.pgn.read_game(io.StringIO(text[s:starts[i + 1] if i + 1 < len(starts) else len(text)]))
		if game is not None and len(list(game.mainline_moves())) > 0:
			yield game


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("source")
	ap.add_argument("--build", default=None)
	ap.add_argument("--knobs", default=None)
	ap.add_argument("--note", default=None)
	a = ap.parse_args()

	with open(a.source, encoding="utf-8", errors="replace") as fh:
		text = fh.read()
	meta = dict(re.findall(r"^# (\w+): (.*)$", text, re.M))
	build = a.build or meta.get("build", "not recorded")
	knobs = a.knobs if a.knobs is not None else meta.get("knobs", "")
	started = meta.get("started", dt.datetime.now().strftime("%Y-%m-%d %H:%M"))

	known, rounds = set(), 0
	if os.path.exists(ARCHIVE):
		with open(ARCHIVE, encoding="utf-8") as fh:
			for g in read_games(fh.read()):
				known.add(key_of(g))
				rounds += 1

	added = []
	for g in read_games(text):
		if key_of(g) in known:
			continue
		known.add(key_of(g))
		rounds += 1
		h = g.headers
		if h.get("Event", "?") == "?":
			h["Event"] = "AlphaUX UI game"
		h["Site"] = "ChessUI (local)"
		h["Date"] = started[:10].replace("-", ".")
		h["Round"] = str(rounds)
		h["AlphaUXBuild"] = build + (" + " + knobs if knobs else "") + " - recorded at play time"
		h["Played"] = started
		h["Source"] = os.path.basename(a.source)
		if a.note:
			h["OwnerNote"] = a.note.replace('"', "'")
		end = g.end().board()
		if h.get("Result", "*") == "*" and end.is_checkmate():
			h["Result"] = "0-1" if end.turn == chess.WHITE else "1-0"
		added.append(g.accept(chess.pgn.StringExporter(headers=True, variations=False, comments=True)))

	if added:
		with open(ARCHIVE, "a", encoding="utf-8") as fh:
			fh.write("\n" + "\n\n".join(added) + "\n")
	print(f"{len(added)} new game(s) added to ui_games/ui_games.pgn ({rounds} in the archive)")


if __name__ == "__main__":
	main()
