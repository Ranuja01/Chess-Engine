# UCI BINARY — design for owner review (2026-10-19; search transition step 1)

Goal: a native, UCI-compliant engine binary built from the SAME engine sources as the Cython module, so that (a) any UCI GUI —
the owner's UI, cutechess / fastchess, a lichess bot — can run it like Stockfish, (b) engine-vs-engine tests run without Python
in the loop (cheaper, and real clocks become trustworthy), (c) timed v1-vs-v2 and time-management experiments become fair.
The Cython path stays exactly as it is (every diagnostic depends on it); a `bench` command proves both front ends play the same.

## 1. What the audit found (file:line in the engine)
- **Time today = fixed per move, no game clock.** Presets (`search_engine.h:45-193`): `TIME_LIMIT` = hard cap from search start
  (LIGHTNING 1.0 s · BLITZ 3.5 · STANDARD 45 · LONG_FORMAT 600); `MOVE_TIMES[d]` = soft gate — depth d+1 starts only if the
  previous iteration took ≤ MOVE_TIMES[d] (`search_engine.cpp:3530`). Default preset STANDARD (`h:220`).
- **Cut-off is already in the recursion**: `time_up` (global `std::atomic<bool>`, `cpp:149`) checked per node; the wall clock every
  200,000 nodes (`h:16`); NODE_LIMIT exact. Only completed iterations are adopted (`cpp:3619-3663`). ⇒ the owner's "hard-woven"
  cut-off is exactly the HARD STOP a modern time manager needs; only where the deadline comes from must change.
- **Entry points** (`h:3480-3494`): `initialize_engine` (pushes the root, sets `side_to_play`, parses env knobs ONCE per process),
  `set_current_state` (pushes a position), `get_engine_move` (searches; then applies its own move to the history, `cpp:3846`),
  `clearSearchTables` (does not clear the eval cache `evalCacheNew`).
- **No FEN / UCI-move parsing in C++.** `update_state` (`cpp_bitboard.h:1444`) applies a move to bitboards (castling, ep, promotion,
  rights; throws on a non-pseudo-legal move); clocks + turn live in `make_move` (`cpp:3237`, `inline`). Repetition counts are keyed
  by a Zobrist hash of placement + side to move (castling / ep excluded).
- **Single-threaded search**; no Python inside the C++ path. `Cython_Chess.inititalize()` only initialises its OWN .so copy — not
  needed by the binary.
- ☠️ **stdout is used for debug prints** (~275 `std::cout` sites: cache stats, "SEARCHING DEPTH", PV, "TIME LIMIT EXCEEDED") ⇒ would
  corrupt the UCI stream. ☠️ **Eval cache and TT store side_to_play-relative values** ⇒ must be cleared when the engine's colour
  changes in one process (GUIs alternate colours between games).

## 2. Files (owner-approved set)
| file | contents |
|---|---|
| `uci_main.cpp` | entry point: stdin loop on the main thread; search on ONE worker thread; `uci / isready / ucinewgame / position / go / stop / quit / setoption`; `info` + `bestmove`; `bench`; stdout hygiene |
| `position.h/.cpp` | FEN parse/emit; UCI move strings ↔ engine moves; applying game moves (update_state + clocks + hash + history); `perft` for correctness |
| `book.h/.cpp` | native polyglot reader (same `.bin` files; standard polyglot keys); weighted choice; options below |
| `timeman.h/.cpp` | **later** (TM upgrade). For now `uci_main` maps `go` parameters onto the existing preset mechanism (§4) |
Build: `overnight_runner.sh build_uci` → `build_uci/chess_engine_uci` (same flags as `setupAI.py`: -O2, LTO, BMI2/PEXT), sources =
the three new files + `cpp_bitboard.cpp eval_v2.cpp search_engine.cpp` (+ `nnue.cpp` if still linked). Linux/WSL first; Windows
(MinGW) later only if needed (the Oracle code uses fork/pipe; MSVC unsupported).

## 3. Engine-side changes (minimal; each byte-identical for the Cython path, verified by fingerprint)
1. **`uci_stop`** — a second atomic flag. `stop` sets `uci_stop` and `time_up`; `get_engine_move` re-reads `uci_stop` right after
   it resets `time_up` (`cpp:3450`) ⇒ a `stop` arriving at the very start is not lost. Cython path: flag always false.
2. **Iteration callback** — an optional function pointer called after each completed iteration (depth, score, nodes, time, PV).
   Null in the Cython path. The binary prints `info depth … score cp|mate … nodes … nps … time … pv …` from it.
3. **Cache clear incl. `evalCacheNew`** — a new `clearAllCaches()` (does not change `clearSearchTables`, which diagnostics use).
   Called on `ucinewgame` and whenever the side the engine searches for differs from the last `go`.
4. **stdout** — the binary swaps `std::cout`'s buffer to stderr at start-up and writes UCI through its own handle on the real
   stdout. No engine print is touched; GUIs ignore stderr.

## 4. Behaviour
- **position**: `startpos | fen <FEN>` + `moves …` ⇒ rebuild the whole history from scratch every time (correct repetition counts;
  undoes the move `get_engine_move` pushed). Malformed / illegal input ⇒ `info string` error, position unchanged (never crash).
- **go**: `depth N` → MAX_DEPTH · `nodes N` → NODE_LIMIT · `movetime T` → a ConfigData with TIME_LIMIT = T, MOVE_TIMES = T ·
  `infinite` → no limits until `stop` · **`wtime/btime/winc/binc/movestogo`** → **minimal clock rule** (needed for ANY clock play):
  budget = remaining / (movestogo or 30) + 0.8 × increment, hard cap = min(4 × budget, remaining / 4); MOVE_TIMES = the active
  preset's shape scaled to the budget. The real adaptive time manager (soft/hard limits, best-move stability, score drops) is the
  later `timeman` item — judged against this rule at EQUAL average time per game (owner, 10-19).
- **Preset option** (`Preset = LIGHTNING | BLITZ | STANDARD | LONG_FORMAT | Clock`): with a named preset, `go` without limits plays
  EXACTLY today's fixed-per-move behaviour (parity with the UI / Python harness). `Clock` (default when wtime/btime are given)
  uses the rule above.
- **Eval / knobs**: the binary defaults `V2_PRESET=shipped` (gen 4) when unset; every env knob still works. Knobs latch at the
  first engine init ⇒ the binary initialises lazily at the first `isready` / `position` / `go`; a knob-changing `setoption` after
  that answers `info string … needs restart`. Options: `EvalArm` (v1/v2), `V2Preset`, `Preset`, `OwnBook`, `BookFile`,
  `BookMaxPly` (30, as today), `Resign` (default OFF — GUIs adjudicate; the engine's resign returns no move today).
- **Book**: used only for games from `startpos` and only within BookMaxPly (today's rule: book OFF for FEN starts); promotion
  pieces handled (the 10-16 fix). `Hash` / `Threads`: fixed for now (TT is statically sized; search is single-threaded).
- **Scores**: engine millipawns, side-to-move POV ⇒ `cp = score / 10`; mate scores (≥ 9,000,000) decoded to `mate N`.

## 5. Verification (each milestone gated; nothing ships on reasoning)
1. **perft** from `position.cpp` vs python-chess on the standard perft suite (startpos, Kiwipete, the 6 CPW positions) — proves FEN
   parsing, move application and the engine's move generator together.
2. **bench = the WAC fingerprint**: 300 WAC positions at depth 10, tables cleared per position (as `tactical_test.run_one`) ⇒ must
   print **258 solved / 47,689,413 nodes** (shipped) and 250 / 35,310,778 (`EvalArm=0`) — byte-identical to the Cython path.
3. **UCI conformance**: cutechess / fastchess smoke match (no illegal moves, no time losses, no hangs) + `stop` mid-search returns a
   legal bestmove.
4. **Parity in play**: fixed-depth games binary vs Cython engine from the same openings ⇒ identical moves.
5. **First timed baselines**: v1 vs v2 at 1 s / move and at LIGHTNING-like clocks (the owner's request), through the binary.

## 6. Milestones
M1 position + FEN + moves + perft · M2 UCI loop + go depth/nodes/movetime/infinite + stop + info/bestmove · M3 bench parity ·
M4 clock rule + book + options · M5 UI hook + cutechess/fastchess + first timed v1 vs v2 baseline. Each milestone = its own commit.

## 7a. OWNER DECISIONS (2026-10-19)
1. UI: add a small `UciEngine` class to `ChessUI/chess_engine.py` (like its `Stockfish` class); keep `AlphaUX` (the Cython path,
   incl. v1) as the legacy route.
2. Clock rule placeholder: fine. ★ Owner observation: LIGHTNING "1 s" can take ~1.5 s — CAUSE: the wall clock is checked only every
   200,000 nodes (≈ 0.44 s at ~450k NPS) ⇒ add a finer check interval (a few thousand nodes) when standardising time — it changes
   timing only, never fixed-depth / node-limited results. ☠️ Old-vs-new timed comparisons must equalise the ACTUAL average time per
   move (old LIGHTNING ≈ 1.0-1.5 s), so the binary logs real time per move.
3. Book: ON for official / UI games only; OFF in all engine-vs-engine tests (opening sets supply the starts: UHO 1,000, UHO-ext
   14,784, variant 2,376, endgame venues 600 / 464). Future: a custom book (memory `custom-opening-book-future`).
4. Runner: **fastchess as a SECOND runner** (real clocks, low overhead, pentanomial SPRT, any UCI opponent) — `tournament.py` stays
   for node-limited, analysis-heavy tests (per-move logs feed ~a dozen tools). Glue: a fastchess PGN → `summary.csv` converter so
   the pair / venue tools read both.
5. Quirks (repetition hash without castling / ep, resign return, redundant Cython_Chess init): kept for parity now, fixed in the
   correctness step. + found in M1: ☠️ `make_move` (search_engine.cpp:3237) tests the pawn bit AFTER `update_state` moved the
   pawn ⇒ inside search only CAPTURES reset the halfmove clock (pawn moves do not) ⇒ the in-search 50-move bookkeeping is wrong.
   The binary's game history uses the FIDE rule (= python-chess, what the Cython path hands the engine at the root).

**M1 DONE (2026-10-19, 4ed2490):** perft of the SEARCH's generator (`generateLegalMovesReordered`) == python-chess on 361
positions at depth 3 (300 held-out game positions + 60 random-play positions with castling / ep) and the standard CPW positions.

**M2 DONE (2026-10-19):** UCI loop (uci / isready / ucinewgame / setoption / position / go depth|nodes|movetime|wtime..|infinite /
stop / quit), search on a worker thread, `info depth … score cp|mate … nodes … nps … pv` (PV replayed from the root and cut at the
first illegal move — the engine's pv_table holds stale tail entries), bestmove always legal. Engine hooks inert in the Cython path
(fingerprints 258 / 47,689,413 and 250 / 35,310,778 unchanged). python-chess as GUI: all limits, stop during infinite (1.00 s),
a 12-ply game, mate-in-1 = `Mate(+1)`. ⚠️ movetime 500 took 0.66-0.73 s = the 200k-node clock check (fix in M4).

**M3 DONE (2026-10-19): `bench` == the WAC fingerprint EXACTLY** — shipped 258 / 47,689,413, `EvalArm 0` 250 / 35,310,778 (same
per-position setup as tactical_test.run_one; suite `diagnostics/suites/wac_uci.txt` = wac.epd with UCI best moves, raw ep field).
⇒ the native binary searches node-for-node like the Cython path. Run: `printf 'bench
quit
' | build_uci/chess_engine_uci`.

**M4 DONE (2026-10-19):** (1) `Config::TIME_CHECK_NODES` (Cython path keeps 200000; binary 2048) ⇒ `movetime 500` returns at
0.497-0.500 s (was 0.66-0.85). The remaining overshoot in a naive test was clearAllCaches() on a COLOUR CHANGE inside `go`
(0.13-0.38 s) — in real matches the clear runs on `ucinewgame`, outside the clock. (2) native polyglot book (`book.cpp`,
keys generated from python-chess into `book_keys.h`): keys == python-chess, book choice == python-chess on 69/69 in-book
positions; startpos games only, < BookMaxPly. Options OwnBook / BookFile / BookMaxPly / Resign / Preset / EvalArm / V2Preset.
(3) ☠️ BUG FOUND + FIXED: a stop (or an expired clock) during alpha_beta's FIRST pass crashed it (moves_list[0] on an empty
list, search_engine.cpp ~3956) — the M2 hook made a pre-start stop hit it. Engine guard (empty list ⇒ return, only on the path
that crashed) + the front end applies a stop only after the first completed iteration. Fingerprints unchanged.

**M5 (2026-10-19, in progress):** UI class `NNEngineUCI` added to `ChessUI/chess_engine.py` (NOT committed — ChessUI is the
friend's submodule and holds the owner's own uncommitted edits; swap `AlphaUX(board, True)` → `NNEngineUCI()` in chess_ui_v2.py
to play it; LIGHTNING move returned at 1.04 s). fastchess v1.8.2-alpha at `E:\chess_data\engines\fastchess\`; UHO 1,000 →
`E:\chess_data\fastchess\openings_uho.epd`; option `Debug` (default off) silences engine prints. Smoke: v2 vs v1, 4 + 4 games
at 10+0.1, all normal terminations. Next: the timed v1-vs-v2 baselines (overnight) and a PGN → summary.csv converter.

## 7. Questions for the owner (answered in §7a)
1. **UI hook**: `ChessUI/` is your friend's project (a git submodule). Add a small `UciEngine` class there (like its `Stockfish`
   class), or keep a wrapper on our side?
2. **Clock rule**: OK as a placeholder until the time-manager upgrade?
3. **Defaults**: book ON for startpos games, Resign OFF, `V2_PRESET=shipped` — agreed?
4. **Test harness**: fastchess (modern, fast) or cutechess-cli for engine-vs-engine matches — any preference?
5. Known engine behaviours kept for PARITY now and listed for the correctness step instead of fixed here: the repetition hash
   ignores castling / ep; `get_engine_move`'s resign return; the redundant `Cython_Chess` init.
