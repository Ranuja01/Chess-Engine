# Chess Engine (C++) — Claude Code Instructions

**This file holds only what must be known BEFORE acting.** Everything else is a pointer, so it is read
on demand rather than loaded every session.

| you need | read |
| --- | --- |
| file map · data flow · what the search already has · symmetry defect shapes · triangulation | [`dev_notes/ENGINE-ORIENTATION.md`](dev_notes/ENGINE-ORIENTATION.md) |
| which probe exists (~200 of them) | [`dev_notes/DIAGNOSTICS-TOOLKIT.md`](dev_notes/DIAGNOSTICS-TOOLKIT.md) |
| **what a probe can RESOLVE and how it LIES** | [`dev_notes/INSTRUMENT-MAP.md`](dev_notes/INSTRUMENT-MAP.md) |
| has this been tried, and was it RESOLVED or just UNREADABLE | memory `KNOWLEDGE-MAP.md` (all 274) — or the `record-check` agent |
| where the program stands today | memory `eval-v2-rebuild-state` → [`dev_notes/SESSION-HANDOFF-2026-09-25.md`](dev_notes/SESSION-HANDOFF-2026-09-25.md) |
| the ship record | [`dev_notes/OPTIMIZATION_LOG.md`](dev_notes/OPTIMIZATION_LOG.md) |

## Scope

This repo is a large, loosely-organized hobby project and **most of it is inactive**. All active work lives
in `NN Engine/`. Despite the name the current engine is **not** the neural-network engine the directory was
built for — it is a (mostly) standalone **C++ engine** driven through a Cython entry point, compiled under
WSL. The older NN engine and the rest of the repo are legacy. **Default assumption:** work concerns the C++
files in the orientation doc's file map; anything outside that set — flag it, don't silently touch it.

🎯 **Roadmap:** strongest single-threaded **hand-written** eval first, then train our own NN from it — now
validated numerically, since SF11 (purely hand-written) reaches ~90% of a modern classical eval's gain.
⚠️ The owner does not want an SF clone: **strength while staying distinctive.** Ours with no SF analogue —
the attacking-layer heat map, `ovd_imbalance`, `approximate_capture_gains`, latent bishop/rook activity, the
multiplicative passer R, `piece_value_boost`, `EG_EXIST_*`, material-as-realizability-conditioner.

---

## Build & run (WSL)

Everything compiles and runs under **WSL** (Anaconda `base` env), from this directory:

```bash
cd "/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine/"
python setupAI.py build_ext --inplace     # compile Cython + C++ → ChessAI*.so
python ChessUI/chess_ui_v2.py             # playable UI (pygame)
python main.py                            # isolated single-position test
```

- To test a position with `main.py`, edit the `chess.Board("...")` FEN near [main.py:31](main.py#L31).
- `setupAI.py` uses `-Ofast -march=native -flto -fopenmp -mpopcnt -mbmi2`, C++20, `-fno-rtti`. The `.so` is
  **machine-specific**. ☠️ **Do not hand-optimize against `-Ofast -flto`** — and note that adding code at
  all can move NPS by tens of percent through layout alone.
- `main.py` loads two keras models at startup only because the `ChessAI` constructor requires them.

🚨 **BASELINE — reverify every build:** `250 / 35,310,778 / EBF 3.784 / STS 1796` · quiet **249,014**
(marginal EBF 1.914) · depth@1s **12** · STS d8 **1676**. NPS 450,201 is the **LIGHTNING** mean
(LONG_FORMAT ~386k). Canonical byte-identity command: `MAX_DEPTH=10 USE_OPENING_BOOK=0 PRESET=LONG_FORMAT`
— without `LONG_FORMAT` hard positions time-abort and node counts become machine-load dependent.

### The ONLY prompt-free invocation form

Unattended calls MUST auto-approve or they hang on a prompt. The `.claude/settings.local.json` allowlist is
a **prefix match**: `wsl.exe -e bash -lc "bash '<abs overnight_runner.sh>'` followed by `*`.

- ✅ **Auto-approved** — begins verbatim with:
  `wsl.exe -e bash -lc "bash '/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine/selfplay/overnight_runner.sh' <sub> <args…>"`
  The trailing `*` also covers `KEY=VALUE` knobs after the sub **and** a trailing
  `… 2>&1 | grep … >> '<literal path>'`.
- ❌ **Prompts (hangs)** — anything not starting with `bash '<runner>'` right after `-lc "`: a leading
  `R='…';`/`cd`/`export` wrapper, `wsl.exe bash -c` (missing `-e`/`-lc`), or raw commands (`pgrep`,
  `pkill`, `env … python`, `find`, `wsl.exe --shutdown`).

**Rules:** (1) launch **each** run as its own auto-approved call — never chain them in one `R='…'` wrapper;
(2) write paths and knobs **literally** — shell vars expand empty here; (3) read results with the **Read
tool** on the Windows path; (4) never `pgrep`/`pkill` unattended. Adapt step-by-step: launch → Read →
decide → launch next.

☠️ **`sts`/`wac` take `<tag>` FIRST, knobs after.** Reversing it is the tag artifact, and the harness
discards the guard's output — it fails **silently**.
☠️ **conc ≤4 · never `nohup` (it orphans the job and a relaunch double-books cores → OOM) · never rebuild
while a job runs · never `| tail` a long run** (output buffers until exit, so a crash is indistinguishable
from a slow run — this once cost 47 minutes waiting on a dead job).
⚠️ **RAM, not cores, is the binding constraint** — cap ~2 engine-loading runs.
⚠️ The owner games ~9pm–midnight and weekend afternoons: **fixed-depth work is safe then, timed work is
not**, and machine suspend corrupts fixed-time games.

**Waiting on long runs — do NOT use `ScheduleWakeup` to poll**; it is unreliable here. Launch as a
**background task** and wait for the harness's completion notification. For a mid-run read, `Read` the
task's output file.

---

## 🚨 NEVER USE A SHELL TO READ — it prompts, and a prompt BLOCKS THE QUEUE

Read / Grep / Glob never prompt. Any shell invocation that is not the exact allowlisted runner prefix does,
and while the owner is asleep that stalls every job behind it. One careless read can cost an overnight block.

| need | ✅ use | ❌ never |
|---|---|---|
| read a file or task output | **Read** | `cat`, `tail`, `pyrun -c "print(open(...))"` |
| search contents | **Grep** | `grep`, `rg`, `Select-String` |
| find files | **Glob** | `find`, `ls -R`, `Get-ChildItem -Recurse` |
| file size / existence | **Read it, or don't check** | `Get-Item .Length`, `stat`, `wc` |

☠️ **PowerShell prompts, always.** ☠️ **Multi-line `pyrun -c` prompts** — the allowlist needs a SINGLE-LINE
command; put logic in a `.py` file and run `pyrun diagnostics/<file>.py`.
★ **Before any Bash call, ask: "is this reading something?" If yes, it is the wrong tool.**

---

## 🚨 Three pre-flight checks — run the check, don't rely on care

All three recurred in one 2026-08-08 session *after* two were already written down. Memory:
`three-recurring-self-check-failures`.

**1. Before concluding anything from a code fragment, prove it EXECUTES at defaults.** Check the enclosing
branch, the gate, and the phase. Removing a queen "proved the queen was involved" when it actually pushed
`phase_score` past the endgame threshold and switched *evaluators*; a `forward_mask` fix landed behind
`ENABLE_CHEAP_BISHOP_COMPLEX` and changed **0 of 1200 positions**.
✅ Then verify the knob MOVES the engine — **changed-rate, not byte-identity**. A null from a knob you have
not proven live is worthless. ☠️ Byte-identical-to-control is the signature of a **silent fallback**.

**2. To explain WHY a number moved, ABLATE — do not read.** Reading is the best tool for *finding* defects
(5 of 9 symmetry defects came from reading) and unreliable for *explaining measurements*: four such stories
were falsified in one session.
★ **When a violation magnitude is CONSTANT, divide it by the candidate knobs.** 30 mp = rounding ×
`KING_SAFETY_MAG`; 50 = `ROOK_ENEMY_PAWN_PEN`; 85 = `EG_SUPPORT − EG_LATENT`; 24 = 2×`CHEAP_BISHOP_KING`.
Four for four.

**3. Every diagnostic print goes behind the existing flag guard FIRST, `getenv` second.** Use
`if (g_capture_eval_breakdown && std::getenv("X"))` — the flag is false in the search path and
short-circuits the `getenv` away. An unguarded `getenv` in the capture loop cost **7.5% peak NPS** and
inflated run-to-run spread from 0.8% to 6.7%. ☠️ **Byte-identity cannot see this** — node counts were
identical throughout.

## 🚨 Colour symmetry is a SHIP GATE, not a debugging tool

**Any new or modified EVAL term must pass the mirror test before it ships.** `eval(board.mirror())` must
equal `-eval(board)` — mirror flips ranks, swaps colours AND side-to-move, so a correctly implemented
side-to-move term still passes.

```
pyrun diagnostics/_eval_symmetry.py N=800 [TERMS=1] [<your knobs>]
```

Zero Stockfish, seconds, exact. A 2026-08-08 sweep found **eleven** defects and cut violations 74.5% → 1.4%;
**two rode in with game-validated ships**, so winning Elo is no protection, and every constant fitted
afterwards silently absorbs the breakage. Defect shapes and the mirrored-bench rule:
[`dev_notes/ENGINE-ORIENTATION.md`](dev_notes/ENGINE-ORIENTATION.md).

## 🚨 Diagnostic-harness contamination

Any diagnostic searching MANY positions in ONE process (`run_one` and everything on it — `wac`/`sts`/
`movematch`/the regret rulers) shares the engine's file-scope C++ learning tables ACROSS positions. A fresh
`ChessAI` per FEN does **not** reset them; `get_engine_move` clears only per-ply scratch, so
`historyHeuristics`/`counterMoveHeuristics`/`moveFrequency` bleed ordering from unrelated prior FENs and
silently change the chosen move. Worst at low material. This FAKED an entire "endgame KS hurt".

- `run_one` now calls `ai.clear_search_tables()` per position. Default ON; **`DIAG_NO_CLEAR=1`** opts out
  (pure-timing NPS benches, or to reproduce old numbers).
- **DIAGNOSTIC-ONLY** — the game path never clears, so the shipped engine is byte-unchanged.
- Cost of the bug (contaminated→clean): **STS 1670→1771**. ⇒ **any search-based POSITIONAL/REGRET number
  from before 2026-08-14 is confounded.** Games/SPRT, static-eval diagnostics and byte-id are unaffected —
  but "byte-id cancels" holds only for a LITERAL-identical eval, never for a real-change DELTA.

---

## Before you measure anything — full protocol in `INSTRUMENT-MAP.md` §G

**`record-check`** (tried already? and was it RESOLVED or merely UNREADABLE — the record is ~85 eval
attempts, mostly unresolved nulls) → **`knob-audit`** (live at defaults? ECHOED ≠ WIRED) → **look up the
instrument's resolution and count the resolvable effect BEFORE calling a null** → **measure the null**
(rate-matched neutral arm, per corpus AND per stratum) → **register the prediction** → run.
★ **Cross-set or it didn't happen** — one corpus at +1pp is noise. ★ **Play games after any search
change.** ★ Knobs latch at init ⇒ **one process per setting**.

## 📐 Subsystem model docs — read before changing that subsystem

Each records the measured system, the evidence, and **what would falsify each claim**. Update the doc if you
change the system — a stale model doc is worse than none, because it reads as authoritative. Superseded
claims are struck through and kept, never deleted.

- **Pawns → [`dev_notes/PAWN_MODEL.md`](dev_notes/PAWN_MODEL.md)** — rank/file tables, chain & wall,
  isolated/backward, passer detection and realizability, the per-pawn clamp, plus a **refutation record**
  (§8a). Check it before re-proposing a killed idea.
- **King safety → [`dev_notes/KING_SAFETY_MODEL.md`](dev_notes/KING_SAFETY_MODEL.md)** (§4a refutation
  record) and memory `ks-twelve-attempt-history-and-the-channel-law`. ☠️ Additive KS changes are **0-for-11**;
  only subtractive ones have won.
- **Before ANY eval RETUNE →
  [`collinearity-why-the-eval-cannot-be-tuned.md`](dev_notes/collinearity-why-the-eval-cannot-be-tuned.md)**
  (code map: `eval-architecture-degeneracy-map.md`). The eval is DEGENERATE (~30 terms, ~2 signals) — that,
  not the corpus, is why every retune flattens (the −85.6-Elo fit). ★ The fix is STRUCTURAL re-shaping,
  **never ridge** (it resolves collinearity *by* shrinking) and **never removal** (that sheds load-bearing
  terms). ⚠️ **Bounded 09-09:** it does *not* hold for the three heat channels (max r=0.42).
- **To TUNE/VALIDATE eval → the LOW-DEPTH REGRET method, not static corpus fit** (which is anti-correlated
  with Elo). Score our fixed-**d7** search's move by `winpct(SF18_best) − winpct(SF18_of_our_move)` on
  `diagnostics/ks_sets/game_regret_set.csv` (~15k FENs, SF18 multi-PV @d14 cached). ⚠️ Read **win%**, not
  the mean, against a **measured** null band, and replicate on `_v2`. ⚠️ It is a d7 EVAL screen — **blind to
  anything gated at depth ≥6**. Full protocol: `INSTRUMENT-MAP.md` §B.

## Code style

- **Reuse, don't redefine.** Call the existing eval / move-gen / cache / hashing helpers. Define logic once.
- Functions `camelCase` (some eval functions are `snake_case` — follow the neighbouring code); locals,
  globals and struct members `snake_case`; constants `UPPER_SNAKE_CASE`; types `PascalCase`. **Tabs** for
  indentation. `uint64_t` for bitboards; explicit types over `auto`; `constexpr` for compile-time constants.
  `/* ... */` header comments on non-trivial functions; keep the `@author: Ranuja Pinnaduwage` banner.
- **Comments explain *what* and *why*, never *that it was added*.** No `// Phase 7 fix`, `// NEW:`, or
  AI-dialogue comments. Good: `// Skip pinned defenders when scoring captures`.
- **No magic numbers** for eval or search thresholds — use (or add to) the named constants in
  `cpp_bitboard.h` / `search_engine.h`. **In persistent docs cite STABLE symbols**, not line numbers.
- **Hot code.** Eval and move generation run millions of times per search: no heap allocations or logging in
  hot loops; use the existing caches rather than recomputing.
- **New experiment knobs default OFF and byte-identical**, with the rationale and the measured result in the
  comment beside them.
