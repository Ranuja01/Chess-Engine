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
- **Use the dedicated tools** (Grep / Read / Glob), not shell `grep` / `cat` / `find`, when exploring this codebase.
