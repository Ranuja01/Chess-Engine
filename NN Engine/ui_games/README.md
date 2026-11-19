# AlphaUX UI games

Games the owner played against AlphaUX through the pygame UI (`ChessUI/chess_ui_v2.py`), including games relayed by hand
against chess.com / lichess / Chessiverse bots. They are kept so a game can be found again as "played by generation X".

- **`ui_games.pgn`** — every game, oldest first. Headers: `AlphaUXBuild` (the build that played it), `Played` / `Pasted`
  (local time), `OwnerNote` (what the owner said with it), `Session` (the Claude Code transcript it came from).
- **`logs/`** — the full terminal output of each session launched with `selfplay/play_ui.sh` (per-move eval, depth, nodes).
- **`collect_ui_games.py`** — appends the games in a session log, or in a pasted PGN, to `ui_games.pgn` (skips games already there).

## Recording a game

From `NN Engine/` in WSL: `bash selfplay/play_ui.sh` (the installed build) or `bash selfplay/play_ui.sh KNOB=VALUE ...`.
The build id, binary checksum and knobs are written at the top of the log and into every game's `AlphaUXBuild` header
("recorded at play time"). A game played without the launcher can be added from its printed PGN:
`python ui_games/collect_ui_games.py game.pgn --build "gen 7.6" --knobs "..." --note "..."`.

## How the first 46 were recovered (2026-11-18)

The UI only prints a game's PGN to its terminal, so nothing was saved until 11-18. Games 1-46 were recovered from the PGNs the
owner pasted into Claude Code sessions (10-01 to 11-17). For those, `AlphaUXBuild` is **inferred** from the production install
times in `dev_notes/current/OPTIMIZATION_LOG.md` at the moment of the paste, not recorded at play time:
- before 10-19 the UI ran AlphaUX in-process, and the preset used (v1, or v2 with `V2_PRESET=shipped`) was not recorded;
- 10-26 to 10-30 reads "gen 6.2 or 6.3" (gen 6.3's install time is not in the log);
- the UI's PGN has no player names or result, so most read `White` / `Black` and `*`; a mate is marked in the index. Who
  played which side and the time odds are in the owner's note where given;
- two pastes had an illegal move near the end (`ParseNote`); those games stop before it.

## Index

| # | pasted (local) | build at that time (inferred) | White - Black | start | moves | result | owner's words |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 10-01 20:11 | pre-10-11 (in-process UI; build not recorded) | White - Black | from FEN | 35 | 0-1 (mate) | this time here's a lightning game against SF18 0.005 with knight odds, which we sometimes  |
| 2 | 10-01 20:16 | pre-10-11 (in-process UI; build not recorded) | White - Black | from FEN | 25 | * |  |
| 3 | 10-01 20:20 | pre-10-11 (in-process UI; build not recorded) | White - Black | from FEN | 38 | * | here's a position that is winning for black from a point where V1 won against an online IM |
| 4 | 10-01 21:18 | pre-10-11 (in-process UI; build not recorded) via Chess.com | ranmal100 - world-champion-carlsen | start | 70 | 0-1 | final game using the STANDARD preset against the Magnus Carlsen bot in chess.com. Disappoi |
| 5 | 10-04 21:44 | pre-10-11 (in-process UI; build not recorded) | White - Black | start | 85 | * | first game lost as white: " |
| 6 | 10-04 21:58 | pre-10-11 (in-process UI; build not recorded) | White - Black | from FEN | 54 | * | bit of a dissapointing loss from a winning position, but then it was purely drawn, but the |
| 7 | 10-15 19:28 | v2 gen 3 (in-process UI; preset not recorded) | White - Black | from FEN | 57 | * | first game in: " |
| 8 | 10-15 20:09 | v2 gen 3 (in-process UI; preset not recorded) via lichess.org | Ranmal101 - lichess AI level 6 | start | 46 | 1-0 | we were white here. Here's another as white: " |
| 9 | 10-15 20:44 | v2 gen 3 (in-process UI; preset not recorded) via chessiverse.com | RoyalBishop6821 - Paul Morphy | start | 49 | 0-1 | ok, here's one as white against the Chessiverse Morphy bot (tuned to be online standard ac |
| 10 | 10-18 15:50 | v2 gen 3 (in-process UI; preset not recorded) | White - Black | from FEN | 34 | 1-0 (mate) | ok, knight odds game against SF18 with 0.005s vs our lightning: " |
| 11 | 10-18 16:10 | v2 gen 3 (in-process UI; preset not recorded) | White - Black | start | 65 | * | here's a loss against SF18 0.005s with no odds: " |
| 12 | 10-18 16:41 | v2 gen 3 (in-process UI; preset not recorded) | White - Black | start | 57 | * | heres a loss at BLITZ time: " |
| 13 | 10-18 17:53 | v2 gen 3 (in-process UI; preset not recorded) via Chess.com | ranmal100 - Hikaru-bot | start | 44 | 0-1 | Gane played against the hikaru bot on chess.com. One thing ill say is that this was played |
| 14 | 10-19 19:29 | gen 4 | White - Black | start | 52 | 1-0 (mate) | also if this really was us, for the first time ever, we just beat SF18 at 0.005 seconds wi |
| 15 | 10-19 19:33 | gen 4 | White - Black | start | 78 | 0-1 (mate) | sure. Here's a game we lost right after: " |
| 16 | 10-24 21:58 | gen 6 | White - Black | start | 56 | 1-0 (mate) | first game against 0.005s time SF18 with 2.5s movetime for us: " |
| 17 | 10-24 22:51 | gen 6 via Chess.com | ranmal100 - world-champion-carlsen | start | 78 | 1-0 | i am proud to persent our first win ever against this bot: " |
| 18 | 10-27 16:26 | gen 6.2 or 6.3 | White - Black | from FEN | 28 | 0-1 (mate) |  |
| 19 | 10-27 16:35 | gen 6.2 or 6.3 | White - Black | from FEN | 83 | * | we lost an odds game: " |
| 20 | 10-30 20:22 | gen 6.4 | White - Black | from FEN | 40 | * | here's the first. I believe it started as a winning position and was 3x the time given but |
| 21 | 10-30 20:40 | gen 6.4 | White - Black | from FEN | 51 | * | another loss from a winning position, but i think this one was just poor calculation based |
| 22 | 10-30 21:58 | gen 6.4 | White - Black | from FEN | 34 | 1-0 (mate) | i did one more game here with rook odds but equal time which we won: " |
| 23 | 11-03 21:13 | gen 6.5 | White - Black | from FEN | 32 | 1-0 (mate) | quick warmup win with odds. T his is using the classic tiiming of the engine. 0.005s speed |
| 24 | 11-03 21:15 | gen 6.5 | White - Black | start | 54 | * | This time a loss: " |
| 25 | 11-03 21:43 | gen 6.5 via chessiverse.com | RoyalBishop6821 - Paul Morphy | start | 35 | 1-0 | ok, this was a win, but honestly the bot played some bad moves in the beginning i think: " |
| 26 | 11-03 22:11 | gen 6.5 via Chess.com | Francis-BOT - ranmal100 | start | 68 | 0-1 | the interesting part is that during the bookphase, it was able to get some increment time  |
| 27 | 11-03 22:24 | gen 6.5 | White - Black | from FEN | 17 | 1-0 (mate) | 1. clean win and quickly: " |
| 28 | 11-03 22:29 | gen 6.5 | White - Black | from FEN | 58 | * | 2. yeah it had no idea what to do: " |
| 29 | 11-03 22:35 | gen 6.5 | White - Black | start | 52 | 1-0 (mate) | 3. it seems the engine won long before any time trouble came to be an issue: " |
| 30 | 11-03 22:56 | gen 6.5 | White - Black | from FEN | 49 | 1-0 (mate) | here it is in full: " |
| 31 | 11-03 23:08 | gen 6.5 | White - Black | from FEN | 18 | 1-0 (mate) | custom endgame position with 30 seconds + 0: " |
| 32 | 11-03 23:08 | gen 6.5 | White - Black | from FEN | 24 | 1-0 (mate) | custom endgame position with 30 seconds + 0: " |
| 33 | 11-03 23:08 | gen 6.5 | White - Black | from FEN | 48 | * | custom endgame position with 30 seconds + 0: " |
| 34 | 11-07 17:14 | gen 7 | White - Black | start | 51 | 1-0 (mate) | first game agaisnt SF at 0.0005s: " |
| 35 | 11-07 17:16 | gen 7 | White - Black | start | 46 | 1-0 (mate) | ok, it seems we've won against this time too: " |
| 36 | 11-07 17:37 | gen 7 | White - Black | start | 124 | 1-0 (mate) | here's an interesting game agaisnt SF at 0.05s against our BLITZ preset: " |
| 37 | 11-07 18:24 | gen 7 | White - Black | start | 39 | 0-1 (mate) | ok, we won the game but this one was pretty weird. I've included the move evals from our e |
| 38 | 11-10 16:09 | gen 7.1 | White - Black | from FEN | 23 | 0-1 (mate) | for POT, this was an idea of my design. It seemed to fail in eval, but we did consider loo |
| 39 | 11-12 15:19 | gen 7.2 | White - Black | from FEN | 36 | 0-1 (mate) | first odds win at 60+0.6 each against SF18: " |
| 40 | 11-12 15:29 | gen 7.2 | White - Black | start | 95 | * | A loss on equal time 60 + 0.6 against SF1.1: " |
| 41 | 11-16 20:05 | gen 7.5 | Stockfish 1.1 - AlphaUX gen 7.5 | start | 42 | * | first somewhat disappointing game. It was 600 + 0.6, where we actually had a fair advantag |
| 42 | 11-16 20:25 | gen 7.5 | White - Black | from FEN | 38 | 1-0 (mate) |  |
| 43 | 11-16 20:28 | gen 7.5 | White - Black | from FEN | 18 | 1-0 (mate) | faster win on the first: " |
| 44 | 11-16 20:28 | gen 7.5 | White - Black | from FEN | 15 | 1-0 (mate) | faster win on the first: " |
| 45 | 11-16 20:58 | gen 7.5 via Chess.com | world-champion-carlsen - AlphaUX gen 7.5 | start | 34 | 0-1 | a nice win: " |
| 46 | 11-17 19:45 | gen 7.6 | AlphaUX (v2 gen 7.6) - Stockfish 1.1 | start | 51 | * | a very interesting opening was taken. We played as white this time, and since the opening  |
