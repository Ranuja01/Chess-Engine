# Mediocre matchup — READY (java-in-WSL installed) — 2026-07-15

## STATUS: BLOCKED on python-chess ↔ Mediocre UCI protocol quirk (2026-07-15 overnight, 2nd finding)
Mediocre RUNS standalone (mediocre_test.py handshake perfect: id, options, plays go depth/time). BUT driving it
through vs_sf.py + python-chess `SimpleEngine` FAILS: a flood of `AssertionError` in Protocol._line_received on
Mediocre's `id name`/`id author`/`uciok` lines (they arrive when no command is ACTIVE) — i.e. Mediocre emits
these lines at times python-chess doesn't expect (likely re-sends id on ucinewgame/isready — non-standard for a
2007 engine). Consequence: Mediocre's MOVE handling is corrupted → wrong/stale opponent moves get pushed to our
engine → our engine plays from desynced positions and "loses" ~90% (score 8-10% for us over conc2 AND conc1 —
NOT a real result; a ~2700 engine does not lose 90% to ~2100 Mediocre; it's a desync artifact). Persists at
concurrency=1, so it's NOT a threading race — it's the protocol interaction itself.
FIX (attended): don't use python-chess SimpleEngine for Mediocre. Write a tolerant raw-pipe UCI driver (send
uci/isready/ucinewgame/position/go, read bestmove, ignore stray id/uciok lines) OR ask Mediocre's author about
its UCI emission quirks. Then plug that as the opponent move-source into a small standalone match loop reusing
EngineProc for our side. The vs_sf.py Threads-on-opponent bug was fixed (skip Threads if not in opponent.options)
— that part is good and byte-safe for SF. Book restored (performance.bin back in place).

## STATUS: WIRED + VERIFIED (2026-07-15 overnight)
User installed default-jre (OpenJDK 11) in WSL. Confirmed: `java` at /usr/bin/java; Mediocre handshakes as
"Mediocre 0.5 by Jonatan Pettersson" (options Hash/EvalHash/PawnHash/Ponder/OwnBook), plays `go depth 8` = Nc3
23.8k nodes. Native-fs launcher created + exec-verified: **/home/ranuja/mediocre_uci.sh** (DrvFs /mnt/c couldn't
hold the exec bit; WSL-home works). popen_uci(WRAP) drives it fine.
RUN COMMAND (dispatcher-form, our 250k nodes vs Mediocre 1s/move handicap-in-its-favor, SF18 arbiter, book off
via --our USE_OPENING_BOOK + Mediocre's book stays on unless configured — minor):
```
bash '<runner>' pyrun selfplay/vs_sf.py --our-label ours \
  --our-config "PRESET=LONG_FORMAT MAX_DEPTH=64 NODE_LIMIT=250000 USE_OPENING_BOOK=0" \
  --sf-path /home/ranuja/mediocre_uci.sh \
  --sf-arb-path "/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/stockfish_18_linux/stockfish-ubuntu-x86-64-avx2" \
  --sf-elo 0 --sf-movetime 1.0 --games 30 --concurrency 2 --seed 0 \
  --openings selfplay/openings_uho.txt --adjudicate-draw --quiet --tag mediocre
```
(--sf-elo 0 is REQUIRED — Mediocre has no UCI_Elo option; a nonzero cap would try to set it and error.)
Slot into a gap between the priority strength gauntlets to avoid keras-OOM (≤3 concurrent our-engine workers).

## (original parking note below, superseded)
# Mediocre matchup — PARKED (needs java-in-WSL) — 2026-07-15

For-fun exhibition: our engine vs **Mediocre v0.5** (Jonatan Pettersson; author is a current professional
colleague of the user). Classic Java HCE UCI engine (~2000-2200 Elo era): PVS, null-move, LMR, futility, SEE,
killers, TT, tapered eval, polyglot book. Located `C:\Users\Kumodth\OneDrive\Desktop\Programming\Chess Engine\
Mediocre\` (`mediocre_v0.5.jar`, `Mediocre.bat` = `java -Xmx1024M -jar mediocre_v0.5.jar`, `performance.bin`
polyglot book, no source).

## Why parked (2026-07-15 overnight)
`shutil.which('java')` in WSL = **None** → java not installed in the WSL env our harness runs under. Running it
otherwise (install JRE via sudo, or a Windows-side harness against our WSL .so) needs prompting commands → can't
be done unattended. Not a priority (user framed it as fun-if-time). Deferred to an attended session.

## Ready-to-run recipe (when attended)
1. **Install a JRE in WSL** (one-time, user runs): `sudo apt-get update && sudo apt-get install -y default-jre`
   then verify `java -version`.
2. **Wrapper** so a single "engine path" launches the jar — create `selfplay/mediocre_uci.sh`:
   ```sh
   #!/bin/bash
   exec java -Xmx1024M -jar "/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Mediocre/mediocre_v0.5.jar"
   ```
   `chmod +x`. (popen_uci keeps it a persistent process per game = fair to it.)
3. **Match via vs_sf.py** — it separates OPPONENT (`--sf-path`) from ARBITER (`--sf-arb-path`), so point the
   opponent at the wrapper and keep SF18 as arbiter:
   `--sf-path selfplay/mediocre_uci.sh --sf-arb-path <SF18> --sf-movetime <t>` with UHO openings.
   TRAP: old engines often DON'T support a `nodes` limit → use **equal MOVETIME or a depth cap**, not
   `--sf-nodes` (our fixed-node gauntlet won't apply). Consider disabling Mediocre's book (`setoption name
   OwnBook value false`) for fairness — vs_sf.py may need a small tweak to pass opponent UCI options, else accept
   its book (both sides get UHO opening moves fed anyway).
4. Optionally add a runner sub `mediocre)` wrapping the above for a clean dispatcher-form invocation.

## Expectations
We're ~2700 vs Mediocre ~2000-2200 → we should win handily; this is an exhibition/for-fun, NOT a strength
gate. Fun framing: fair time control, its book on or off, best-of-N.
