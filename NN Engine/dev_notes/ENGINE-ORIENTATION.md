# Engine orientation — file map, data flow, what already exists

Reference material split out of [`../CLAUDE.md`](../CLAUDE.md) so it is read **when needed** rather than
loaded into every session. CLAUDE.md keeps the rules you must know *before* acting; this holds the facts
you can look up on demand.

⚠️ **Read this before adding anything to the engine** — the recurring failure is reinventing a helper that
already exists, or "adding" a technique the search already has.

---

## C++ engine file map

| File | Role | Key contents |
| --- | --- | --- |
| `ChessAI.pyx` | Cython entry point / Python↔C++ bridge | `ChessAI` class; `alphaBetaWrapper()` is called from the pygame UI to get a move; opening-book lookup; `extern` declarations into the three C++ headers; converts results back to `chess.Move` (UCI) |
| `setupAI.py` | Build script | Compiles `cpp_bitboard.cpp`, `threadpool.cpp`, `search_engine.cpp`, `ChessAI.pyx` into the `ChessAI` extension; OpenMP + C++20 + native-arch flags |
| `search_engine.cpp` / `search_engine.h` | Search entry point & core algorithm | `get_engine_move()` (C++ entry from Cython); iterative deepening; `alpha_beta()`, `minimizer()` / `maximizer()`, `qSearch()`; PVS, null-move pruning, LMR, futility, razoring; structs `BoardState`, `MoveData`, `SearchData`, `TTEntry`; search constants (`MAX_QDEPTH`, `MIN_MATERIAL_FOR_NULL_MOVE`, futility margins, time-check interval). **`namespace Config` holds every knob.** |
| `move_gen.h` | Legal move generation + ordering | `generateLegalMoves()`, `generatePseudoLegalMoves()`, `generateEvasions()`, castling / en-passant / pawn move gen; `generateLegalMovesReordered()` — scores captures by MVV-LVA + SEE and quiets by history / killer / counter-move / move-frequency heuristics; precomputed attack masks |
| `cache_management.h` | Hashing, caches, heuristics | Zobrist hashing (`generateZobristHash`, incremental update); transposition table (`searchEvalCache` + `TTEntry`, depth-preferred replacement); eval / quiescence / move-gen caches; killer / counter-move / history / move-frequency tables and their decay functions |
| `cpp_bitboard.cpp` / `cpp_bitboard.h` | Evaluation function (+ bitboard ops & move-gen impl) | `placement_and_piece_eval()` — main static eval; per-piece midgame/endgame evaluators (pawns, knights, bishops, rooks, queens, kings); placement / attacking layers; passed pawns, king safety, bishop colour complexity, rook open files; `advanced_endgame_eval()`, `get_latent_threat_score()`; piece values and most engine constants |
| `ThreadPool.cpp` / `ThreadPool.h` | Worker thread pool | `enqueue()` / `wait()` work-stealing pool. Compiled in (`setupAI.py` lists it as `threadpool.cpp`, which resolves on the case-insensitive `/mnt/c` mount), but **not currently wired into the active search** — treat as latent / future SMP. |

⚠️ The engine is **non-negamax**: separate `minimizer()` / `maximizer()` / `pre_minimizer()` / `qSearch()`,
an **absolute Black-positive** eval flipped once at the root, and **millipawn** units.

## Data flow

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

Move ordering at each node comes from `generateLegalMovesReordered()` (`move_gen.h`), backed by the
heuristic tables in `cache_management.h`.

## Search techniques already present — reuse, don't reinvent

- Alpha-beta (fail-soft) with Principal Variation Search (null-window scout + re-search)
- Iterative deepening with time control; a **root pre-search** that warms the TT and ordering
- Transposition table (depth-preferred replacement)
- Null-move pruning, Late Move Reduction (LMR), futility pruning, razoring, IIR (gated)
- Quiescence search (captures/checks at the horizon)
- Killer-move, history, and counter-move heuristics + move-frequency PV bonus; history gravity
- SEE-based capture ordering (MVV-LVA fallback)
- Zobrist hashing for position identity

🐛 **`LMP` keys REMAINING depth; `LMR` keys ITERATION depth.** They are not the same axis.

---

## Colour symmetry — the recurring defect shapes

The gate itself is in CLAUDE.md and is mandatory. These are the shapes, so they can be recognised while
writing rather than found months later. A 2026-08-08 sweep found **eleven** defects and cut violations from
**74.5% to 1.4%**; two rode in with *game-validated ships*, so winning Elo is no protection.

- **Wrong constant per colour** — one branch pays 10, its twin pays 15.
- **Swapped wrap guards** — `<<9` wraps onto file A, `<<7` onto file H; getting them backwards both admits
  the wrap AND deletes a legitimate neighbour.
- **Non-mirrored rank windows** — White `> 4` (ranks 5-7) must mirror to Black `< 3`, not `< 5`.
- **Two alternative fixes both shipped** — they are alternatives, not a bundle; shipping both recreates the
  defect inverted. Grep the sibling knob's default before flipping either.
- **`>>` on a SIGNED value** — an arithmetic shift rounds toward −∞, so `v>>8` and `(-v)>>8` are not
  negatives of each other. Use `/ 256`, which truncates toward zero. ⚠️ A one-unit rounding error is not
  automatically small: `KING_SAFETY_MAG=3000` amplified one unit into exactly 30 mp on 68 positions.
- **Unstable sort with no tie-break** — `std::sort` leaves ties in insertion order, which is usually square
  order, which reverses under a mirror. Tie-break on something colour-relative.
- **Asymmetry hides in constant TABLES too**, not only in branches.

⚠️ **Colour is the invariant axis; PHASE is not.** A white/black difference *inside one function* is
presumptively a bug. A midgame/endgame difference is a legitimate design choice — the two evaluators exist
so the phases can price things differently. Sweep white-branch vs black-branch, never midgame vs endgame.

⚠️ **Both benches are colour-skewed** (`sts300` 177w/123b, `wac` 190w/110b), so neither can judge a colour
change alone — the skewed STS once ranked two candidate fixes *backwards*. Use the mirrored twins and score
`orig + mirror`: `sts_suite sts300_mirror.epd <tag>` · `wac_suite wac_mirror.epd <tag>`.

★ When symmetry alone cannot choose (both repair directions are symmetric, e.g. 10-vs-15), it is a TUNING
question — decide it on balanced STS, and choose on the balanced TOTAL, never the colour gap.

---

## Eval diagnosis — the three-way triangulation (ours / SF11-static / SF18-search)

The standing method for hunting eval bugs. For a suspect position compare:

- **ours** — `ai.ev_breakdown(board)`, a clean partition (fields sum to `total`).
  ⚠️ `pieces` already contains `material`; `pt_*` and `material` are **SUB-VIEWS**, not additive extras, and
  `pt_*` is **material-inclusive** — never place it beside SF11's placement-only rows.
- **SF11-static** — classical HCE `eval` with a labeled per-term table: the hand-fixable ceiling.
- **SF18-search** — the truth, but includes tactics we cannot encode statically.

Act where **SF11-static agrees with SF18 but ours is wrong** (statically fixable → read SF11's classical
source). Skip where SF11 also misses (search's job). Leave alone where we already beat SF11.

Tools: `probe_fens.py` (explicit FEN list, `--table` for win% error), `sf11_collapse_gap.py` (corpus
per-term over-read), `ks_failure_hunt.py`. SF11 binary + full recipe: memory
`sf11-sf18-triangulation-method`. ⚠️ SF binaries and source live in the **sibling** `Programming/Chess
Engine/` directory, and SF11 **has** a linux build — never infer a path from one tool's mapping.

★ **Ask questions as a difference between two CANDIDATES, not against a null** — a candidate-vs-candidate
gap is null-independent, which is why the SF11-vs-SF15c comparison was readable when nothing else was.
