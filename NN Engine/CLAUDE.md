# Chess Engine (C++) — Claude Code Instructions

## Scope / Orientation

This repo (`Chess-Engine`) is a large, loosely-organized hobby project, and **most of it is inactive**. All active development lives in this `NN Engine/` directory.

Despite the name, the current engine is **not** the neural-network engine the directory was originally built for. Recent work is a **(mostly) standalone C++ chess engine**, driven through a Cython entry point and compiled/run under WSL. The older NN engine, the rest of the repo, and most other files in this directory are legacy.

**Default assumption:** work concerns the C++ engine files listed in the file map below. Treat only those as the active engine. Anything outside that set — flag it, don't silently touch it. If the user wants to expand scope (e.g. revive the NN side, or add a new subsystem), map it out here first before working on it.

---

## Build & Run (WSL workflow)

Everything compiles and runs under **WSL** (Anaconda `base` env). From this `NN Engine/` directory:

```bash
# Navigate (from a fresh WSL shell)
cd "/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine/"

# Compile the Cython + C++ extension (produces ChessAI*.so)
python setupAI.py build_ext --inplace

# Run the playable UI (pygame)
python ChessUI/chess_ui_v2.py

# Run an isolated single-position test
python main.py
```

- To test a specific position with `main.py`, edit the `chess.Board("...")` FEN near the top of [main.py](main.py#L31).
- `setupAI.py` compiles with `-Ofast -march=native -flto -fopenmp -mpopcnt -mbmi2`, C++20, `-fno-rtti`. It is tuned for the host CPU (`-march=native`), so the built `.so` is machine-specific.
- `main.py` still loads two keras models at startup only because the `ChessAI` constructor signature requires them — the engine logic itself is C++.

### Unattended / autonomous runs — the ONLY prompt-free invocation form

For overnight/unattended work, calls MUST auto-approve or they hang waiting on a prompt. The
`.claude/settings.local.json` allowlist is a **prefix match**: `wsl.exe -e bash -lc "bash '<abs overnight_runner.sh>'` followed by `*` (any suffix). So:

- ✅ **Auto-approved** — the command *begins* with, verbatim:
  `wsl.exe -e bash -lc "bash '/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine/selfplay/overnight_runner.sh' <sub> <args…>"`
  The trailing `*` also covers `KEY=VALUE` knobs after the sub **and** a trailing `… 2>&1 | grep … >> '<literal path>'`.
- ❌ **Prompts (hangs unattended)** — anything that does NOT start with `bash '<runner>'` right after `-lc "`: a leading `R='…';`/`cd`/`export` wrapper, `wsl.exe bash -c` (missing `-e`/`-lc`), or raw commands (`pgrep`, `pkill`, `env … python`, `find`, `wsl.exe --shutdown`).

**Rules for unattended sequences:** (1) launch **each** run as its own auto-approved `bash '<runner>' …`
call (background for long ones) — do NOT chain them in one `R='…'` wrapper (that prompts); (2) write
paths/knobs **literally** — shell vars (`$R`) expand empty in this wrapper; (3) read results **with the
Read tool** on the Windows-path results file (no shell → no prompt), not via `grep`; (4) never `pgrep`/
`pkill` unattended. To adapt between runs, go step-by-step: launch → Read the result file → decide → launch next.

**Waiting on long runs — do NOT use `ScheduleWakeup` to poll.** It is unreliable here (the timed wakeup
often never fires). Instead launch the run as a **background Bash task** and wait for the harness's
automatic completion notification — a background task pings you when it exits, so no self-scheduled poll is
needed. For a *mid-run* directional read (e.g. an SPRT that runs to a game cap), just `Read` the task's
output file when you're next active; the interim lines (running Elo/LLR) are all there. Never sit on a
`ScheduleWakeup` timer expecting it to wake you.

---

## C++ Engine File Map

| File | Role | Key contents |
| --- | --- | --- |
| `ChessAI.pyx` | Cython entry point / Python↔C++ bridge | `ChessAI` class; `alphaBetaWrapper()` is called from the pygame UI to get a move; opening-book lookup; `extern` declarations into the three C++ headers; converts results back to `chess.Move` (UCI) |
| `setupAI.py` | Build script | Compiles `cpp_bitboard.cpp`, `threadpool.cpp`, `search_engine.cpp`, `ChessAI.pyx` into the `ChessAI` extension; OpenMP + C++20 + native-arch flags |
| `search_engine.cpp` / `search_engine.h` | Search entry point & core algorithm | `get_engine_move()` (C++ entry from Cython); iterative deepening; `alpha_beta()`, `minimizer()` / `maximizer()`, `qSearch()`; PVS, null-move pruning, LMR, futility, razoring; structs `BoardState`, `MoveData`, `SearchData`, `TTEntry`; search constants (`MAX_QDEPTH`, `MIN_MATERIAL_FOR_NULL_MOVE`, futility margins, time-check interval) |
| `move_gen.h` | Legal move generation + ordering | `generateLegalMoves()`, `generatePseudoLegalMoves()`, `generateEvasions()`, castling / en-passant / pawn move gen; `generateLegalMovesReordered()` — scores captures by MVV-LVA + SEE and quiets by history / killer / counter-move / move-frequency heuristics; precomputed attack masks |
| `cache_management.h` | Hashing, caches, heuristics | Zobrist hashing (`generateZobristHash`, incremental update); transposition table (`searchEvalCache` + `TTEntry`, depth-preferred replacement); eval / quiescence / move-gen caches; killer / counter-move / history / move-frequency tables and their decay functions |
| `cpp_bitboard.cpp` / `cpp_bitboard.h` | Evaluation function (+ bitboard ops & move-gen impl) | `placement_and_piece_eval()` — main static eval; per-piece midgame/endgame evaluators (pawns, knights, bishops, rooks, queens, kings); placement / attacking layers; passed pawns, king safety, bishop color complexity, rook open files; `advanced_endgame_eval()`, `get_latent_threat_score()`; piece values and most engine constants |
| `ThreadPool.cpp` / `ThreadPool.h` | Worker thread pool | `enqueue()` / `wait()` work-stealing pool. Compiled in (`setupAI.py` lists it as `threadpool.cpp`, which resolves on the case-insensitive `/mnt/c` mount), but **not currently wired into the active search** — treat as latent / future SMP. |

---

## Data Flow

```
pygame UI (ChessUI/chess_ui_v2.py)
  → ChessAI.alphaBetaWrapper()           [ChessAI.pyx — Cython bridge]
  → get_engine_move()                    [search_engine.cpp — C++ entry]
  → iterative deepening
  → alpha_beta()  →  minimizer()/maximizer()   (+ transposition table, LMR,
                                                  null-move/futility/razoring,
                                                  qSearch at the horizon)
  → placement_and_piece_eval()           [cpp_bitboard.cpp — static eval]
  → best MoveData  →  UCI  →  chess.Move back to Python
```

Move ordering at each node comes from `generateLegalMovesReordered()` (`move_gen.h`), backed by the heuristic tables in `cache_management.h`.

---

## Search Techniques Already Present

Know what exists before adding anything — **reuse, don't reinvent**:

- Alpha-beta (fail-soft) with Principal Variation Search (null-window scout + re-search)
- Iterative deepening with time control
- Transposition table (depth-preferred replacement)
- Null-move pruning, Late Move Reduction (LMR), futility pruning, razoring
- Quiescence search (captures/checks at the horizon)
- Killer-move, history, and counter-move heuristics + move-frequency PV bonus
- SEE-based capture ordering (MVV-LVA fallback)
- Zobrist hashing for position identity

---

## Development Rules & Code Style

### 🚨 COLOUR-SYMMETRY IS A SHIP GATE, NOT A DEBUGGING TOOL

**Any new or modified EVAL term must pass the mirror test before it ships.** `eval(board.mirror())` must
equal `-eval(board)` — mirror flips ranks, swaps colours AND swaps side-to-move, so a *correctly*
implemented side-to-move term still passes. Run it mid-development or at the end of a run, but run it:

```
pyrun diagnostics/_eval_symmetry.py N=800 [TERMS=1] [<your knobs>]
```

**Why this is a hard gate.** A 2026-08-08 sweep found **seven** colour defects and cut violations from
74.5% to 1.4%. Two of them were introduced *by recent, game-validated ships*: the signed-shift rounding
bug rode in with `MOD_KS_REALIZ` (+36.7 Elo bundle), and the rook `DBLCOUNT` contradiction rode in with
capped threats (+45 Elo bundle). **Winning Elo is no protection against carrying a colour bug in**, and
every constant fitted afterwards silently absorbs the breakage.

The recurring shapes, so they can be recognised while writing rather than months later:
- **Wrong constant per colour** — one branch pays 10, its twin pays 15.
- **Swapped wrap guards** — `<<9` wraps onto file A, `<<7` onto file H; getting them backwards both
  admits the wrap AND deletes a legitimate neighbour.
- **Non-mirrored rank windows** — White `> 4` (ranks 5-7) must mirror to Black `< 3`, not `< 5`.
- **Two alternative fixes both shipped** — they are alternatives, not a bundle; shipping both recreates
  the defect inverted. Grep the sibling knob's default before flipping either.
- **`>>` on a SIGNED value** — an arithmetic shift rounds toward −∞, so `v>>8` and `(-v)>>8` are not
  negatives of each other. Use `/ 256`, which truncates toward zero. ⚠️ A one-unit rounding error is not
  automatically small: `KING_SAFETY_MAG=3000` amplified one unit into exactly 30 mp on 68 positions.
- **Unstable sort with no tie-break** — `std::sort` leaves ties in insertion order, and insertion order
  is usually square order, which reverses under a mirror. Tie-break on something colour-relative.

⚠️ **Colour is the invariant axis; PHASE is not.** A white/black difference *inside one function* is
presumptively a bug. A midgame/endgame difference is a legitimate design choice — the two evaluators
exist so the phases can price things differently. Sweep white-branch vs black-branch, never midgame vs
endgame.

⚠️ **Both benches are colour-skewed** (`sts300` 177w/123b, `wac` 190w/110b), so neither can judge a
colour change alone — the skewed STS once ranked two candidate fixes *backwards*. Use the mirrored twins
and score `orig + mirror`:
`sts_suite sts300_mirror.epd <tag>` · `wac_suite wac_mirror.epd <tag>`.

★ When symmetry alone cannot choose (both repair directions are symmetric, e.g. 10-vs-15), it is a
TUNING question — decide it on balanced STS, and choose on the balanced TOTAL, never the colour gap.

- **Build & run via the WSL workflow above.** Don't invent new build steps or compile flags. On Windows, use PowerShell syntax for any host-side commands.
- **Reuse, don't redefine.** Call the existing eval / move-gen / cache / hashing helpers rather than reimplementing detection or scoring logic. Define a piece of logic once.
- **Match the existing C++ style** (lifted from the codebase, not invented):
  - Functions: `camelCase` (e.g. `generateLegalMoves`, `placement_and_piece_eval` — note some eval functions use `snake_case`; follow the neighbouring code).
  - Local/global variables and struct members: `snake_case`.
  - Constants / macros: `UPPER_SNAKE_CASE`.
  - Type aliases / structs: `PascalCase` (`BoardState`, `MoveData`, `TTEntry`).
  - **Tabs** for indentation.
  - `uint64_t` for bitboards; prefer explicit types over `auto`; `constexpr` for compile-time constants.
  - `/* ... */` block header comments documenting parameters/returns on non-trivial functions; keep the `@author: Ranuja Pinnaduwage` file banner on new C++ files.
- **Comment guidelines:** explain *what* the code does and *why*, never *that it was added*. No `// Phase 7 fix`, `// ISSUE 15 FIX`, `// NEW:`, or AI-dialogue comments. Good: `// Skip pinned defenders when scoring captures`.
- **No magic numbers** for evaluation or search thresholds. Use (or add to) the named constants in `cpp_bitboard.h` / `search_engine.h` — piece values, futility margins, `MAX_QDEPTH`, `MIN_MATERIAL_FOR_NULL_MOVE`, time-check interval, etc. — rather than inlining literals.
- **Performance-critical code.** Eval and move generation run millions of times per search. Avoid heap allocations and any logging inside hot loops; respect and use the existing caches instead of recomputing.
- 🚨 **NEVER USE A SHELL TO READ — it prompts, and a prompt BLOCKS THE QUEUE.** Read / Grep / Glob never
  prompt. Any shell invocation that is not the exact allowlisted runner prefix does, and while the owner
  is asleep that stalls every job behind it until a human wakes up. One careless read can cost a whole
  overnight block.

  | need | ✅ use | ❌ never |
  |---|---|---|
  | read a file or task output | **Read** | `cat`, `tail`, `pyrun -c "print(open(...))"` |
  | search contents | **Grep** | `grep`, `rg`, `Select-String` |
  | find files | **Glob** | `find`, `ls -R`, `Get-ChildItem -Recurse` |
  | file size / existence | **Read it, or don't check** | `Get-Item .Length`, `stat`, `wc` |

  ☠️ **PowerShell prompts, always** — there is no read-only PowerShell command worth running here.
  ☠️ **Multi-line `pyrun -c` prompts** — the allowlist needs a SINGLE-LINE command; embedded newlines break
  the prefix match. Put logic in a `.py` file and run `pyrun diagnostics/<file>.py`.
  ☠️ **Never pipe a long run through `| tail`** — output buffers until exit, so a crash is indistinguishable
  from a slow run (this cost 47 minutes waiting on an already-dead job).
  ★ **Before any Bash call, ask: "is this reading something?" If yes, it is the wrong tool.**
  Detail + incident log: memory `never-shell-for-reading-it-prompts`.

## 🧰 Diagnostics toolkit — READ BEFORE WRITING ANY NEW PROBE

`diagnostics/` already holds ~200 scripts and nearly every question we ask has a tool for it. **Check
[`dev_notes/DIAGNOSTICS-TOOLKIT.md`](dev_notes/DIAGNOSTICS-TOOLKIT.md) first** — it is the index of what
exists, what each script answers, and the conventions. Rebuilding a probe wastes time and usually produces a
weaker version (the rebuilt one lacks the *control set* that made the original trustworthy).

- **Extend the canonical tool, do not fork it.** `probe_fens.py` is THE per-FEN probe (ours + SF11 +
  SF15.1 classical + SF15.1 NNUE + SF18 static + SF18 search, `--table` for one row per FEN).
- **Rank eval errors by win% (Lichess k=0.00368208), not centipawns** — the same logistic the fit scripts
  use. Two pawns of error at +8 barely matters; two pawns at 0.0 flips the game.
- **Carry the whole reference ladder.** The SF versions are a progression: whichever generation is closest
  to truth for a situation is the source to read for that concept.

## 📐 Subsystem model docs — READ BEFORE CHANGING THAT SUBSYSTEM

Each records the measured system, the principles behind it, the evidence, and **what would falsify each
claim**. Read the relevant one before touching that area of the eval, and **update it if you change the
system** — a stale model doc is worse than none. These are canonical: they hold knowledge that does not
expire, and superseded claims are struck through and kept, never deleted.

- **Pawns → [`dev_notes/PAWN_MODEL.md`](dev_notes/PAWN_MODEL.md)** — rank/file tables, chain & wall,
  isolated/backward, passer detection and realizability, the per-pawn clamp. Includes a **refutation
  record** of ideas already measured and killed; check it before re-proposing one.
- *(King safety, and other subsystems, to follow the same pattern.)*

## Eval diagnosis — the three-way triangulation (ours / SF11-static / SF18-search)

The standing method for hunting eval bugs: for a suspect position compare **ours** (`ai.ev_breakdown(board)`,
a clean partition — fields sum to `total`), **SF11-static** (classical HCE `eval` with a labeled per-term
table — the hand-fixable ceiling; compare our term vs SF11's same term), and **SF18-search** (the truth, but
includes tactics we can't encode statically). Act on positions where **SF11-static agrees with SF18 but ours is
wrong** (statically fixable → read SF11's classical source); skip where SF11 also misses (search's job); leave
alone where we already beat SF11. Tools live in `diagnostics/`: `probe_fens.py` (explicit FEN list),
`sf11_collapse_gap.py` (corpus per-term over-read), `ks_failure_hunt.py`. SF11 binary + full recipe:
memory `sf11-sf18-triangulation-method`.
