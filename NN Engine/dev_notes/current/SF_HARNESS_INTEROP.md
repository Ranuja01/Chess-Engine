# Running the Stockfish-dependent harness under WSL without interop deaths

**Symptom:** the SF arbiter / annotation process dies mid-run ("Exec format error",
broken pipe, hangs), especially in chats where the assistant drives the runs.

**Root cause:** `find_stockfish()` (in `selfplay/arbiter.py` and `diagnostics/stockfish_oracle.py`)
resolves the **Windows Stockfish `.exe`** — `$STOCKFISH_PATH` and every hardcoded fallback end in
`.exe`. Launching a `.exe` from WSL runs it through **Windows interop (`binfmt_misc`)**. That bridge is
fragile when SF is spawned **repeatedly, in short-lived shells, left idle between calls, or
concurrently** — it goes stale and the process dies. It is NOT fragile when SF is spawned **once and
kept alive** in a long-running process.

## The pattern that works (ran a full overnight tournament + annotation cleanly)

Spawn Stockfish **exactly once, inside a single long-lived WSL process**, and let nothing else
repeatedly spawn it.

**Roles:**
- A human (or one persistent interactive WSL shell) **OWNS the engine/SF runs**: `setupAI.py` build,
  `tournament.py` / `sprt.py`, `annotate.py`, the WAC/STS suites. Launch them in a WSL terminal that
  stays open for the whole run. `tournament.py` keeps **one shared `Arbiter` (one SF process) alive for
  the entire run** — that single long-lived process is the happy path for interop.
- The assistant stays on the **READ side**: it reads `games/<tag>/` outputs (`summary.csv`,
  `game_*/game.jsonl[.annotated]`, `tournament.json` / `sprt.json`) and runs the **pure-Python analyzers
  that need no engine/SF** (`summary.py`, `tournament_diag.py`, `deep_diag.py`, `term_diag.py`). It does
  **not** invoke SF/engine through its own tool — the assistant's shell is short-lived (and may be
  Git-Bash-on-Windows or a fresh WSL shell per call), which is exactly the fragile spawn pattern.

**Do / Don't:**
- DO run long SF/engine jobs in ONE persistent WSL shell; let that single process own SF for its lifetime.
- DO use the **WSL Anaconda CPython** that built `ChessAI` (a Windows `python3` may resolve to **PyPy**,
  which breaks the SF pipe comms).
- DON'T spawn SF per-command or in a loop of short-lived shells; don't leave the interop bridge idle and
  then reuse it.
- DON'T have the assistant drive the SF runs itself.

## Durable fix (removes interop entirely; required for `sprt.py` concurrency)

A native Linux Stockfish makes `popen_uci` / `Popen` spawn ordinary Linux processes — no `binfmt`, no
interop, robust even for concurrent or short-lived spawns:

```bash
sudo apt-get install stockfish              # native ELF -> /usr/games/stockfish (has the NNUE `eval` cmd)
export STOCKFISH_PATH=/usr/games/stockfish  # env beats `which`, so set it explicitly in WSL
```
(or drop an official `stockfish-ubuntu-x86-64-avx2` in the repo, `chmod +x`, point `STOCKFISH_PATH` at it).

## Confirm which binary is in play (from `NN Engine/`, in WSL)

```bash
python3 -c "import sys; sys.path.insert(0,'diagnostics'); from stockfish_oracle import find_stockfish; print(find_stockfish())"
```
A `…\.exe` path ⇒ interop (fragile). `/usr/games/stockfish` ⇒ native (robust).
