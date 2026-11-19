# AlphaUX UI games

Every game the owner has played against AlphaUX through the pygame UI (`ChessUI/chess_ui_v2.py`), including games relayed by
hand against chess.com / lichess / Chessiverse bots, labelled with the build and settings each was played under. The PGN
imports straight into a lichess study (one chapter per game; the Event tag is the chapter label, and a comment before the
first move repeats the settings in case lichess drops the custom tags).

- **`ui_games.pgn`**: every game, in local play order. Real names with settings in White / Black. Custom tags:
  - `AlphaUXBuild`, `AlphaUXTime`, `OpponentSettings`, `Odds` and `OurSide` record the settings.
  - `Confidence` says, per field, whether it was stated, carried from earlier in the session, inferred or unknown.
  - `OwnerNote`, `Evidence` and `Pasted` / `Session` say where the record came from.
  - Site games also keep their original names, UTC time, Elo, link and termination.
- **`logs/`**: the session record of each game played through `selfplay/play_ui.sh` (or a pasted terminal output).
- **`collect_ui_games.py`**: appends the games in a session log, or in a pasted PGN, to `ui_games.pgn`; duplicates are skipped.

## Recording a game

From `NN Engine/` in WSL, run `bash selfplay/play_ui.sh` (the installed build) or `bash selfplay/play_ui.sh KNOB=VALUE ...`.
- The launcher writes the build id, binary checksum and knobs at the top of the log, and into every game's `AlphaUXBuild` header
  ("recorded at play time").
- To add a game played without the launcher, use its printed PGN:
  `python ui_games/collect_ui_games.py game.pgn --build "gen 7.6" --knobs "..." --note "..."`.
- Every game the owner gives Claude is saved into this archive in the same turn.

## How games 1-54 were recovered (2026-11-18)

The UI only prints a game's PGN to its terminal, so nothing was saved before 11-18.
- **Source of the moves:** games 1-54 come from the PGNs the owner pasted into Claude Code sessions, 10-01 to 11-17. That
  includes 8 games pasted while a reply was in progress.
- **Source of the settings:** what the owner and Claude said around each paste, as recorded in the transcripts.
- **The build** usually comes from the ship log's install times, or from Claude's launch instructions at the time. It is rarely
  in the owner's own words, so most builds read "inferred".
- **Unresolved:**
  - 4 games have no recorded AlphaUX time control.
  - 5 have no recorded opponent.
  - Games 14-17 and 47-48 could be one of two builds; which one ran was not recorded.
  - Several results come from the final position when the owner did not state them.

  The `Confidence` tag says which, per game.
- **Game 55 onward** are recorded at play time.

## Index

All 55 games in `ui_games.new.pgn`, in local play order (the time is when the owner pasted the game, minutes after it ended; the last game was recorded at play time). Settings were recovered from the Claude Code transcripts on 11-18; the Confidence column says for each field whether it was stated for that game, carried from a run of games, inferred, or unknown.

| # | local date/time | AlphaUX build | AlphaUX time | opponent + settings | our side | odds/start | moves | result | confidence | owner's words |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2026-10-01 20:08 EDT | v2 V2_PRESET=shipped as of 10-01 (Fit A PSTs + KS + winnability; fingerprint 251 / 49,211,859), in-process Cython UI | not stated (that evening used LIGHTNING and STANDARD) | Stockfish 18 at 0.005 s per move (fixed) | Black | none (standard start) | 81 | loss (1-0) | side inferred, opponent stated, time unknown, build inferred, result stated | here's the first game. We've never been able to beat SF18 at 0.005 seconds of time per ... |
| 2 | 2026-10-01 20:11 EDT | v2 V2_PRESET=shipped as of 10-01 (Fit A PSTs + KS + winnability; fingerprint 251 / 49,211,859), in-process Cython UI | LIGHTNING preset (~1 s/move) | Stockfish 18 at 0.005 s per move (fixed) | Black | knight odds to AlphaUX: White (Stockfish) starts without the b1 knight | 35 | win (0-1) | side inferred, opponent stated, time stated, build inferred, result stated | this time here's a lightning game against SF18 0.005 with knight odds, which we sometim... |
| 3 | 2026-10-01 20:16 EDT | v2 V2_PRESET=shipped as of 10-01 (Fit A PSTs + KS + winnability; fingerprint 251 / 49,211,859), in-process Cython UI | LIGHTNING preset (carried) | Stockfish 18 at 0.005 s per move (fixed) (not restated; carried from the game before) | Black | variant start, no material odds: Black's knights replaced by bishops (rbbqkbbr) | 25 | loss (1-0) | side stated, opponent carried, time carried, build inferred, result stated | this time no odds, but instead of knights we only have bishops, which constitutes a wha... |
| 4 | 2026-10-01 20:20 EDT | v2 V2_PRESET=shipped as of 10-01 (Fit A PSTs + KS + winnability; fingerprint 251 / 49,211,859), in-process Cython UI | not stated | Stockfish ('SF'); setting not restated, likely SF18 at 0.005 s as in the games before | Black | custom FEN: move 20 of V1's win against the Chessiverse Bartholomew bot, winning for Black | 39 | loss (1-0) | side stated, opponent inferred, time unknown, build inferred, result stated | here's a position that is winning for black from a point where V1 won against an online... |
| 5 | 2026-10-01 21:18 EDT | v2 V2_PRESET=shipped as of 10-01 (Fit A PSTs + KS + winnability; fingerprint 251 / 49,211,859), in-process Cython UI | STANDARD preset | chess.com 'world-champion-carlsen' bot (2882 on chess.com's bot scale), untimed | White | none (standard start) | 70 | loss (0-1) | side stated, opponent stated, time stated, build inferred, result stated | final game using the STANDARD preset against the Magnus Carlsen bot in chess.com. Disap... |
| 6 | 2026-10-04 21:44 EDT | v2 V2_PRESET=shipped as of 10-04 (connected-pawn ship; fingerprint 254 / 50,622,239), in-process Cython UI | LIGHTNING preset | Stockfish 18 at 0.005 s per move (fixed) | White | none (standard start) | 85 | loss (0-1) | side stated, opponent stated, time stated, build stated, result stated | first game lost as white ... This is against SF18 at 0.005s. Still haven't been able to... |
| 7 | 2026-10-04 21:48 EDT | v2 V2_PRESET=shipped as of 10-04 (connected-pawn ship; fingerprint 254 / 50,622,239), in-process Cython UI | LIGHTNING preset (carried) | Stockfish 18 at 0.005 s per move (fixed) (carried from the game before) | White | knight odds to AlphaUX: Black (Stockfish) starts without the b8 knight | 28 | win (1-0) | side inferred, opponent carried, time carried, build carried, result stated | here's a win with knight odds |
| 8 | 2026-10-04 21:50 EDT | v2 V2_PRESET=shipped as of 10-04 (connected-pawn ship; fingerprint 254 / 50,622,239), in-process Cython UI | LIGHTNING preset (carried) | Stockfish 18 at 0.005 s per move (fixed) (carried) | White | variant start: White has bishops for knights, Black knights for bishops (rnnqknnr / RBBQKBBR) | 31 | win (1-0) | side stated, opponent carried, time carried, build carried, result stated | interesting game of all knights vs all bishops where we win as white |
| 9 | 2026-10-04 21:58 EDT | v2 V2_PRESET=shipped as of 10-04 (connected-pawn ship; fingerprint 254 / 50,622,239), in-process Cython UI | LIGHTNING preset (carried) | Stockfish 18 at 0.005 s per move (fixed) (carried) | White | custom FEN (move 33, White to move, a middlegame White was winning) | 54 | loss (0-1) | side inferred, opponent carried, time carried, build carried, result stated | bit of a dissapointing loss from a winning position, but then it was purely drawn, but ... |
| 10 | 2026-10-15 19:28 EDT | v2 gen 3 (V2_PRESET=shipped, fingerprint 259 / 47,869,841), in-process Cython UI | LIGHTNING preset | Stockfish 18 at 0.005 s per move (fixed) | White | custom FEN (the 10-04 position, move 33) | 57 | loss (0-1) | side stated, opponent inferred, time stated, build inferred, result inferred | first game in ... we were white here |
| 11 | 2026-10-15 20:09 EDT | v2 gen 3 (V2_PRESET=shipped, fingerprint 259 / 47,869,841), in-process Cython UI | LIGHTNING preset | lichess AI level 6 (of 8), no time limit | White | none (standard start) | 46 | win (1-0) | side stated, opponent stated, time stated, build inferred, result stated | Here's another as white ... a game played against the lichess level 6 computer (out of ... |
| 12 | 2026-10-15 20:44 EDT | v2 gen 3 (V2_PRESET=shipped, fingerprint 259 / 47,869,841), in-process Cython UI | LIGHTNING preset | Chessiverse Paul Morphy bot ('3231' on Chessiverse's scale); its time not stated | White | none (standard start) | 49 | loss (0-1) | side stated, opponent stated, time stated, build inferred, result stated | here's one as white against the Chessiverse Morphy bot (tuned to be online standard acc... |
| 13 | 2026-10-15 20:58 EDT | v2 gen 3 (V2_PRESET=shipped, fingerprint 259 / 47,869,841), in-process Cython UI | LIGHTNING preset | Stockfish 18 at 0.005 s per move (fixed) | White | knight odds to AlphaUX: Black (Stockfish) starts without the b8 knight | 27 | win (1-0) | side inferred, opponent stated, time stated, build inferred, result inferred | here's a game agaisnt SF18 @ 0.005s (which is often stronger than us like in that first... |
| 14 | 2026-10-18 15:50 EDT | gen 3 (V2_PRESET=shipped, 259 / 47,869,841) or gen 3 + C64 (V2_PRESET=candidate, 258 / 47,689,413) - not recorded | LIGHTNING preset | Stockfish 18 at 0.005 s per move (fixed) | White | knight odds to AlphaUX: Black (Stockfish) starts without the b8 knight | 34 | win (1-0) | side inferred, opponent stated, time stated, build unknown, result stated | ok, knight odds game against SF18 with 0.005s vs our lightning ... Good win but i will ... |
| 15 | 2026-10-18 16:10 EDT | gen 3 (V2_PRESET=shipped, 259 / 47,869,841) or gen 3 + C64 (V2_PRESET=candidate, 258 / 47,689,413) - not recorded | LIGHTNING preset (carried) | Stockfish 18 at 0.005 s per move (fixed) | White | none (standard start) | 65 | loss (0-1) | side inferred, opponent stated, time carried, build unknown, result stated | here's a loss against SF18 0.005s with no odds ... we get a clearly winning position, b... |
| 16 | 2026-10-18 16:41 EDT | gen 3 (V2_PRESET=shipped, 259 / 47,869,841) or gen 3 + C64 (V2_PRESET=candidate, 258 / 47,689,413) - not recorded | BLITZ preset | not stated; likely Stockfish 18 at 0.005 s as in the sitting's other games | White | none (standard start) | 57 | loss (0-1) | side inferred, opponent inferred, time stated, build unknown, result stated | heres a loss at BLITZ time |
| 17 | 2026-10-18 17:53 EDT | gen 3 (V2_PRESET=shipped, 259 / 47,869,841) or gen 3 + C64 (V2_PRESET=candidate, 258 / 47,689,413) - not recorded | STANDARD preset | chess.com Hikaru-bot (2820 on chess.com's bot scale), untimed | White | none (standard start) | 44 | loss (0-1) | side stated, opponent stated, time stated, build unknown, result stated | Gane played against the hikaru bot on chess.com ... this was played with STANDARD, wher... |
| 18 | 2026-10-19 19:29 EDT | v2 gen 4, native UCI binary through NNEngineUCI (its first evening in the UI) | LIGHTNING preset (~1 s/move) | Stockfish 18 at 0.005 s per move (fixed) | White | none (standard start) | 52 | win (1-0) | side inferred, opponent stated, time stated, build inferred, result stated | also if this really was us, for the first time ever, we just beat SF18 at 0.005 seconds... |
| 19 | 2026-10-19 19:33 EDT | v2 gen 4, native UCI binary through NNEngineUCI (its first evening in the UI) | LIGHTNING preset (carried) | Stockfish 18 at 0.005 s per move (fixed) (carried) | White | none (standard start) | 78 | loss (0-1) | side inferred, opponent carried, time carried, build inferred, result stated | Here's a game we lost right after ... One thing i noticed is that the resigning is gone |
| 20 | 2026-10-24 21:58 EDT | v2 gen 6 + PGO, production UCI binary build_uci/chess_engine_uci | fixed 2.5 s per move (NNEngineUCI(movetime=2.5)) | Stockfish 18 at 0.005 s per move (fixed) | White | none (standard start) | 56 | win (1-0) | side inferred, opponent stated, time stated, build inferred, result inferred | first game against 0.005s time SF18 with 2.5s movetime for us |
| 21 | 2026-10-24 22:51 EDT | v2 gen 6 + PGO, production UCI binary build_uci/chess_engine_uci | STANDARD preset, no movetime | chess.com 'world-champion-carlsen' bot (2882 on chess.com's bot scale), untimed | White | none (standard start) | 78 | win (1-0) | side stated, opponent stated, time stated, build inferred, result stated | im now gonna use standard preset (no movetime set) against the chess.com magnus bot ...... |
| 22 | 2026-10-27 16:26 EDT | v2 gen 6.2, production UCI binary (gen 6.3 shipped the evening of 10-28) | 60 s + 0.6 s clock (NNEngineUCI(clock=(60, 0.6))) | Stockfish 18, full strength, on its own 60+0.6 clock | Black | custom FEN: a won bishop ending for Black (owner: 'from winning positions') | 28 | win (0-1) | side inferred, opponent inferred, time stated, build inferred, result stated | perhaps ill do this from winning positions so our engine has a chance ... first win ...... |
| 23 | 2026-10-27 16:35 EDT | v2 gen 6.2, production UCI binary (gen 6.3 shipped the evening of 10-28) | 60 s + 0.6 s clock | Stockfish 18, full strength, 60+0.6 clock | White | knight odds to AlphaUX: Black (Stockfish) starts without the b8 knight | 83 | loss (0-1) | side inferred, opponent carried, time carried, build inferred, result stated | we lost an odds game |
| 24 | 2026-10-27 16:43 EDT | v2 gen 6.2, production UCI binary (gen 6.3 shipped the evening of 10-28) | '5 + 0.6' (read as 5 min + 0.6 s; the UI file later held clock=(300, 0.6)) | Stockfish 18, '1 + 0.6' (read as 1 min + 0.6 s) | White | knight odds to AlphaUX: Black (Stockfish) starts without the b8 knight, plus 5x time odds | 33 | win (1-0) | side inferred, opponent inferred, time inferred, build inferred, result stated | this time igave 5 + 0.6 vs 1 + 0.6 and we won |
| 25 | 2026-10-30 20:22 EDT | v2 gen 6.4, production UCI binary | 3x the opponent's time (base not stated; the UI file had held 300+0.6 = 5x) | Stockfish (version not stated), 60+0.6 per the assistant's read of chess_ui_v2.py | White | custom FEN: pawn ending, winning for White | 40 | loss (0-1) | side inferred, opponent inferred, time stated, build inferred, result stated | here's the first. I believe it started as a winning position and was 3x the time given ... |
| 26 | 2026-10-30 20:40 EDT | v2 gen 6.4, production UCI binary | not stated | Stockfish (version and clock not stated; the UI file had it on 60+0.6) | White | custom FEN: White to move with a winning attack | 51 | loss (0-1) | side inferred, opponent inferred, time unknown, build inferred, result stated | another loss from a winning position, but i think this one was just poor calculation based |
| 27 | 2026-10-30 20:50 EDT | v2 gen 6.4, production UCI binary | '5 times more time' than the game before (base not stated) | Stockfish (version and clock not stated; the UI file had it on 60+0.6) | White | custom FEN: same position as the game before | 52 | loss (0-1) | side inferred, opponent inferred, time stated, build inferred, result stated | here is the same position but this time 5 times more time given ... Closer but still a ... |
| 28 | 2026-10-30 21:34 EDT | v2 gen 6.4, production UCI binary | not stated | Stockfish (version and clock not stated) | Black | custom FEN: the Bartholomew position of 10-01 (Black to move, winning for Black) | 56 | loss (1-0) | side inferred, opponent inferred, time unknown, build inferred, result stated | here's another loss from a winning position |
| 29 | 2026-10-30 21:58 EDT | v2 gen 6.4, production UCI binary | equal to the opponent's (value not stated) | Stockfish (version not stated), same clock as ours | White | rook odds to AlphaUX: Black (Stockfish) starts without the a8 rook | 34 | win (1-0) | side inferred, opponent inferred, time stated, build inferred, result stated | i did one more game here with rook odds but equal time which we won |
| 30 | 2026-11-03 21:13 EST | v2 gen 6.5 + speed batch 8 (production chess_engine_uci_g65b8pgo, installed that evening; same moves as gen 6.5) | LIGHTNING preset ('the classic timing') | Stockfish 18 at 0.005 s per move (fixed) | White | rook odds to AlphaUX: Black (Stockfish) starts without the a8 rook | 32 | win (1-0) | side inferred, opponent stated, time stated, build inferred, result stated | quick warmup win with odds. This is using the classic tiiming of the engine. 0.005s spe... |
| 31 | 2026-11-03 21:15 EST | v2 gen 6.5 + speed batch 8 (production chess_engine_uci_g65b8pgo, installed that evening; same moves as gen 6.5) | LIGHTNING preset | Stockfish 18 at 0.005 s per move (fixed) | White | none (standard start) | 54 | loss (0-1) | side inferred, opponent stated, time stated, build inferred, result stated | same time controls but no odds this time. Keep in mind, with lightning, we've only won ... |
| 32 | 2026-11-03 21:43 EST | v2 gen 6.5 + speed batch 8 (production chess_engine_uci_g65b8pgo, installed that evening; same moves as gen 6.5) | BLITZ preset | Chessiverse Paul Morphy bot ('3231' on Chessiverse's scale); its time not stated | White | none (standard start) | 35 | win (1-0) | side stated, opponent stated, time carried, build inferred, result stated | Im gonna try a BLITZ preset game against an online bot now ... ok, this was a win, but ... |
| 33 | 2026-11-03 22:11 EST | v2 gen 6.5 + speed batch 8 (production chess_engine_uci_g65b8pgo, installed that evening; same moves as gen 6.5) | 60 s + 0.6 s clock (the bot untimed) | chess.com Francis-BOT (2300), untimed | Black | none (standard start) | 68 | win (0-1) | side stated, opponent stated, time stated, build inferred, result stated | a game against Francis, rated 2300 on chess.com. There's no timer for Francis, but i'm ... |
| 34 | 2026-11-03 22:24 EST | v2 gen 6.5 + speed batch 8 (production chess_engine_uci_g65b8pgo, installed that evening; same moves as gen 6.5) | 60 s + 0.6 s clock | Stockfish ('against SF'); its time not stated (no Stockfish clock lines, so fixed per move) | White | custom FEN: K+R v K with the 50-move count at 60 | 17 | win (1-0) | side inferred, opponent stated, time stated, build inferred, result stated | 1. clean win and quickly ... that was in 60 + 0.6 by the way, and so will the rest be |
| 35 | 2026-11-03 22:29 EST | v2 gen 6.5 + speed batch 8 (production chess_engine_uci_g65b8pgo, installed that evening; same moves as gen 6.5) | 60 s + 0.6 s clock (carried) | Stockfish (carried); its time not stated | White | custom FEN: K+B+N v K | 58 | draw (1/2-1/2) | side inferred, opponent carried, time carried, build inferred, result stated | 2. yeah it had no idea what to do |
| 36 | 2026-11-03 22:35 EST | v2 gen 6.5 + speed batch 8 (production chess_engine_uci_g65b8pgo, installed that evening; same moves as gen 6.5) | 180 s, no increment (NNEngineUCI(clock=(180))) | Stockfish at 0.0005 s per move, no clock (version not stated) | White | none (standard start) | 52 | win (1-0) | side stated, opponent stated, time stated, build inferred, result stated | im doing white_chess_engine = NNEngineUCI(clock=(180)) vs just stockfish (no clock) at ... |
| 37 | 2026-11-03 22:56 EST | v2 gen 6.5 + speed batch 8 (production chess_engine_uci_g65b8pgo, installed that evening; same moves as gen 6.5) | 60 s, no increment | Stockfish (not stated for this game); its time not stated (no Stockfish clock lines) | White | custom FEN: a won bishop ending for White | 49 | win (1-0) | side stated, opponent inferred, time stated, build inferred, result inferred | im pretty sure as white, our engine would win ... For now illl try the UI with 60 + 0 a... |
| 38 | 2026-11-03 23:08 EST | v2 gen 6.5 + speed batch 8 (production chess_engine_uci_g65b8pgo, installed that evening; same moves as gen 6.5) | 30 s, no increment | not stated (presumably the UI's Stockfish) | White | custom FEN: two rooks v queen | 19 | win (1-0) | side inferred, opponent unknown, time stated, build inferred, result inferred | custom endgame position with 30 seconds + 0 |
| 39 | 2026-11-03 23:08 EST | v2 gen 6.5 + speed batch 8 (production chess_engine_uci_g65b8pgo, installed that evening; same moves as gen 6.5) | 20 s, no increment | not stated (presumably the UI's Stockfish) | White | custom FEN: two rooks v queen (same position) | 25 | win (1-0) | side inferred, opponent unknown, time stated, build inferred, result inferred | same position with 20 seconds + 0 |
| 40 | 2026-11-03 23:08 EST | v2 gen 6.5 + speed batch 8 (production chess_engine_uci_g65b8pgo, installed that evening; same moves as gen 6.5) | 10 s, no increment | not stated (presumably the UI's Stockfish) | White | custom FEN: two rooks v queen (same position) | 48 | unfinished (*) | side inferred, opponent unknown, time stated, build inferred, result unknown | and finally 10 + 0 |
| 41 | 2026-11-07 17:14 EST | v2 gen 7, production UCI binary chess_engine_uci_g7pgo | LIGHTNING preset | Stockfish at 0.0005 s per move (version not stated) | White | none (standard start) | 51 | win (1-0) | side inferred, opponent stated, time stated, build inferred, result stated | first game agaisnt SF at 0.0005s ... We have been able to beat this version for a while... |
| 42 | 2026-11-07 17:16 EST | v2 gen 7, production UCI binary chess_engine_uci_g7pgo | LIGHTNING preset (carried) | Stockfish at 0.005 s per move (version not stated) | White | none (standard start) | 46 | win (1-0) | side inferred, opponent carried, time carried, build inferred, result stated | ok, it seems we've won against this time too |
| 43 | 2026-11-07 17:37 EST | v2 gen 7, production UCI binary chess_engine_uci_g7pgo | BLITZ preset (up to 3.5 s/move) | Stockfish at 0.05 s per move (version not stated) | White | none (standard start) | 124 | win (1-0) | side inferred, opponent stated, time stated, build inferred, result stated | an interesting game agaisnt SF at 0.05s against our BLITZ preset ... SF apparantly didn... |
| 44 | 2026-11-07 18:24 EST | v2 gen 7, production UCI binary chess_engine_uci_g7pgo | STANDARD preset (45 s cap per move) | chess.com 'world-champion-carlsen' bot (2882 on chess.com's bot scale), untimed | Black | none (standard start) | 39 | win (0-1) | side stated, opponent stated, time inferred, build inferred, result stated | ill give it a try again as black ... we won the game but this one was pretty weird ... ... |
| 45 | 2026-11-10 16:09 EST | v2 gen 7.1 (lazy eval), production | 60 s + 0.6 s clock, both sides | not stated (the saved PGN says 'Opponent'), 60+0.6 | Black | custom FEN: the won bishop ending for Black of 10-27 | 23 | win (0-1) | side stated, opponent unknown, time stated, build stated, result inferred | ill run some games to see how lazy eval operates ... Here's a test game at 60 +0.6 each... |
| 46 | 2026-11-10 16:21 EST | v2 gen 7.1 (lazy eval), production | 60 s + 0.6 s clock, both sides | not stated, 60+0.6 ('same time controls') | Black | knight odds to AlphaUX: White starts without the b1 knight | 37 | win (0-1) | side inferred, opponent unknown, time stated, build carried, result inferred | here's another game ... This time knight odds with the same time controls |
| 47 | 2026-11-12 15:19 EST | gen 7.2, or gen 7.2 + SEE_PER_MOVE=3 (gen 7.3 behaviour) - not recorded | 60 s + 0.6 s clock | Stockfish 18, 60+0.6 clock | Black | knight odds to AlphaUX: White (Stockfish 18) starts without the b1 knight | 36 | win (0-1) | side inferred, opponent stated, time stated, build unknown, result stated | first odds win at 60+0.6 each against SF18 |
| 48 | 2026-11-12 15:29 EST | gen 7.2, or gen 7.2 + SEE_PER_MOVE=3 (gen 7.3 behaviour) - not recorded | 60 s + 0.6 s clock | Stockfish 1.1, 60+0.6 clock | Black | none (standard start) | 95 | loss (1-0) | side inferred, opponent stated, time stated, build unknown, result stated | A loss on equal time 60 + 0.6 against SF1.1 ... We ended up in a winning position but i... |
| 49 | 2026-11-16 20:05 EST | v2 gen 7.5, production UCI binary chess_engine_uci_g75pgo, plain 'python ChessUI/chess_ui_v2.py' | 600 s + 0.6 s clock | Stockfish 1.1, 600+0.6 clock | Black | none (standard start) | 42 | loss (1-0) | side inferred, opponent stated, time stated, build carried, result stated | first somewhat disappointing game. It was 600 + 0.6, where we actually had a fair advan... |
| 50 | 2026-11-16 20:25 EST | v2 gen 7.5, production UCI binary chess_engine_uci_g75pgo, plain 'python ChessUI/chess_ui_v2.py' | 10 s + 0.1 s clock | Stockfish 1.1, 10+0.1 clock | White | custom FEN: K+B+N v K | 38 | win (1-0) | side inferred, opponent inferred, time inferred, build stated, result stated | with that short amount of time it was able to mate |
| 51 | 2026-11-16 20:28 EST | v2 gen 7.5, production UCI binary chess_engine_uci_g75pgo, plain 'python ChessUI/chess_ui_v2.py' | 10 s + 0.1 s clock (carried) | Stockfish 1.1, 10+0.1 clock (carried) | White | custom FEN: K+B+N v K, defender in the wrong corner | 18 | win (1-0) | side carried, opponent carried, time carried, build carried, result stated | faster win on the first |
| 52 | 2026-11-16 20:28 EST | v2 gen 7.5, production UCI binary chess_engine_uci_g75pgo, plain 'python ChessUI/chess_ui_v2.py' | 10 s + 0.1 s clock (carried) | Stockfish 1.1, 10+0.1 clock (carried) | White | custom FEN: K+B+B v K | 15 | win (1-0) | side carried, opponent carried, time carried, build carried, result inferred | and second |
| 53 | 2026-11-16 20:58 EST | v2 gen 7.5, production UCI binary chess_engine_uci_g75pgo, PRESET=STANDARD with NNEngineUCI() and no clock | STANDARD preset, no clock | chess.com 'world-champion-carlsen' bot (2882 on chess.com's bot scale), untimed | Black | none (standard start) | 34 | win (0-1) | side stated, opponent stated, time stated, build inferred, result stated | ill do one game against the chess.com magnus bot ... id like to see if we can beat it r... |
| 54 | 2026-11-17 19:45 EST | v2 gen 7.6, production UCI binary chess_engine_uci_g76pgo (shipped 11-17 18:23); no knobs stated | 600 s + 0.6 s clock | Stockfish 1.1 on its own clock, 600 s base (increment inferred 0.6) | White | none (standard start; book to ply 23, French Winawer Poisoned Pawn) | 51 | loss (0-1) | side stated, opponent stated, time inferred, build inferred, result stated | a very interesting opening was taken. We played as white this time ... at ply 57, white... |
| 55 | 2026-11-18 23:15 EST | gen 7.6 + ENABLE_QSTANDPAT_SEED=1 (recorded at play time) | 600+0.6 | Stockfish 1.1 (600+0.6) | White | standard start | 61 | loss (0-1, AlphaUX resigned at -23.40) | stated (recorded at play time) | We were winning as white but couldn't make a breakthrough and eventually got into a losing |

## Old round to new round

| new | from |
|---|---|
| 1 | not in the old archive: bare-movetext paste of 2026-10-01 20:08 EDT (session 97a258a9) |
| 2 | old 1 |
| 3 | old 2 |
| 4 | old 3 |
| 5 | old 4 |
| 6 | old 5 |
| 7 | not in the old archive: queued-message paste of 2026-10-04 21:48 EDT (session 10e4d60e) |
| 8 | not in the old archive: queued-message paste of 2026-10-04 21:50 EDT (session 10e4d60e) |
| 9 | old 6 |
| 10 | old 7 |
| 11 | old 8 |
| 12 | old 9 |
| 13 | not in the old archive: queued-message paste of 2026-10-15 20:58 EDT (session ed0b1a06) |
| 14 | old 10 |
| 15 | old 11 |
| 16 | old 12 |
| 17 | old 13 |
| 18 | old 14 |
| 19 | old 15 |
| 20 | old 16 |
| 21 | old 17 |
| 22 | old 18 |
| 23 | old 19 |
| 24 | not in the old archive: queued-message paste of 2026-10-27 16:43 EDT (session 2a8f69b3) |
| 25 | old 20 |
| 26 | old 21 |
| 27 | not in the old archive: queued-message paste of 2026-10-30 20:50 EDT (session 2a8f69b3) |
| 28 | not in the old archive: queued-message paste of 2026-10-30 21:34 EDT (session 2a8f69b3) |
| 29 | old 22 |
| 30 | old 23 |
| 31 | old 24 |
| 32 | old 25 |
| 33 | old 26 |
| 34 | old 27 |
| 35 | old 28 |
| 36 | old 29 |
| 37 | old 30 |
| 38 | old 31 |
| 39 | old 32 |
| 40 | old 33 |
| 41 | old 34 |
| 42 | old 35 |
| 43 | old 36 |
| 44 | old 37, replaced by its chess.com PGN |
| 45 | old 38 |
| 46 | not in the old archive: queued-message paste of 2026-11-10 16:21 EST (session 2a8f69b3) |
| 47 | old 39 |
| 48 | old 40 |
| 49 | old 41 |
| 50 | old 42 |
| 51 | old 43 |
| 52 | old 44 |
| 53 | old 45 |
| 54 | old 46 |
| 55 | old 47 (recorded 11-18, verbatim) |
