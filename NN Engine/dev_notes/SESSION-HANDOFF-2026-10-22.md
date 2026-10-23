# Session handoff — 2026-10-20 → 10-22 (search transition: correctness, void-match correction, speed batch)

## STATE
- **Engine = v2 gen 5 + speed batch S1-S4.** Fingerprint **258 / 47,689,586 / 3.926** (`V2_PRESET=shipped`; UCI `bench` identical);
  v1 legacy control unchanged **250 / 35,310,778 / 3.784**. `V2_SHIP_GEN=4` → 258 / 47,689,413.
- Gen 5 = gen 4 + B3 `FIX_DRAW_EXIT` + C1 `FIX_QS_CHECK` (v2 block only). Real harm check +2.6 ± 29.4 (400 g).
- Speed (result-identical, both arms): S1 qsearch quiet skip · S2 `QGEN_NOISY_ONLY` · S3 huge-page tables · S4 `REP_STACK`. Bench
  ≈ +26% vs gen 5 morning (≈ 600-635k UCI bench nps); real play ≈ +30% (the bench spends ~19% clearing tables per position).
- **Running at handoff:** queue #93 — speed batch vs this morning's binary, st=1, 400 g (`E:\chess_data\q93_speed_batch.log`,
  `fastchess\speed_batch.log`). Registered +10 … +30 Elo. Elo here is pure speed (identical search).
- Commits through `830c247` + the docs commit of this handoff; pushed at the owner's request (10-22).

## ☠️ HONESTY — errors this phase (all caught; read before relying on any 10-19…10-22 game number)
1. **Every fastchess `Knob_` match was gen-vs-gen (void).** fastchess drops undeclared options; my "knob is live" check typed
   `setoption` by hand — it proved the ENGINE, not the CHANNEL. I then built a story ("A2's first screen inflated, fresh seeds pulled
   it back") on two nulls. Fixed with the declared `Knobs` option + a channel pre-flight in every knob queue.
2. **Scope overreach:** launched four whole-search audits after the owner asked WHETHER cross-engine research should be done in a
   corrhist discussion (a reply with only a link is not a yes). Owner accepted the result; memory updated: confirm scope first.
3. **Bundle design:** A4 (a general change) bundled with narrow fixes. **Test design:** the B2 position had ...Rh1+ winning the queen
   (I registered "Black losing"). **Wrong form parked:** NULL_NPM_GUARD mode 1 (replaces the legacy rule) instead of mode 2 (adds SF's).
4. **A fabricated commit hash** in a report ("a1f5c8e…") — the real one was `7ef89b3`. Never quote a hash without reading it.
5. **Premature lever:** proposed "pin-aware movegen" from a profile row before reading `is_safe` — it is already pin-aware; the row
   was mostly rdtsc-scope overhead. **Misattributed:** #75's 29 promotion voids were the known 10-16 book bug, not a new one.
6. Ops: a PowerShell `>` to a `/mnt/e` path (nothing ran); source edits during a running production build (the build stayed valid).
Pattern again: every correction came from a control — a stdin-logging wrapper, a counter, a bench identity, a game-identity run.

## OWNER CALLS (this phase)
- Transition = correctness bugs, result-identical speed, the time manager, eval-coupled items; everything else → SEARCH V2.
- Search v2 = likely a NEW file (negamax parity port, then consensus-form features); the old search becomes the frozen control.
  Prefer SHARED components now (movegen, eval, caches, time manager, corrhist tables) — they carry over.
- A2/A1/LMR rework, null reform, improving, IIR, singular → search v2. Gen 5 = B3 + C1 on correctness; B1, B2, A4 stay off.
- Corrhist: rebuild in the consensus form (audit) — the 08-24 closure covered one implementation, not corrhist.
- Table footprint (step 6) includes HISTORY tables (piece-keyed). Speed changes are batched toward ~+30%, then games.
- Release-readiness sweep when v2 is frozen: plain code comments + a Windows binary near Linux speed (memory `release-readiness-sweep`).
- SIMD: not for the HCE; design it into the NNUE inference later.
- dev_notes cleanup: inventory → owner approval → `git mv` to `archive/` with link fixes → `INDEX.md`.

## NEXT (in order)
1. Read #93. If positive, the speed batch is confirmed in games.
2. More result-identical speed: KS cost in eval (~10% of real play), the 16-pass movegen collapse (~5-8%), TT density (fold
   `g_ttMoveTable` into `TTEntry`), the main-search move-gen cache (memory; step 6).
3. Corrhist stage 0-1 (`research-2026-10-22/CORRHIST-REFERENCE-AUDIT.md` §7): instruments, then the consensus minimum.
4. Time manager: ~26-31% of move time is spent in a discarded iteration (TMCNT); depth-1 ID start; equal ACTUAL time judging.
5. Audit correctness items, each starting with a counter: mate-loss gate on move-loop pruning, VERIFY downgrade, endgame LMR bypass
   (`reduced_search_depth` phase ≥ 117), killer-splice swap, mate-distance pruning, TT never storing mates/draws. Logged defect:
   `isNearGameEnd` read uninitialised in `generateLegalMovesReordered`.
6. Odds and ends: NULL_NPM_GUARD mode 2, v2 tree blow-ups, triage cases, optional PE_PASSER_EXT venue re-run (its games were void;
   move test +0.012 ± 0.011). Then threats eval+search, POT-as-search, margins re-sweep LAST, bench checkpoint #3 → search v2.
7. dev_notes cleanup (inventory for the owner).

## WHERE THINGS ARE
- Plan + every number: `SEARCH-CORRECTNESS-PLAN-2026-10-19.md` (transition scope, speed tally, void list, queues #88-#93).
- Record: `OPTIMIZATION_LOG.md` 2026-10-21/22 · `BASELINE_PERF.md` top section.
- Search audits (filed for search v2): `research-2026-10-22/` — corrhist, pruning, reductions/extensions, move ordering, qsearch/TT/root/time.
- Tools: `_fix_firecount.py` (fire counts, move test, game identity, crash watch, time accounting), `_loss_autopsy.py`,
  `_perf_symbolise.py`, `QGEN_VERIFY`, profiler scopes MG_CACHE/QGEN/REPETITION (DIAGNOSTICS-TOOLKIT).
- Data: `E:\chess_data\` (queue logs q88-q93, autopsy, fire counts, profiles `profile_v2_g5*.log`, `perf_*.log`, fastchess PGNs).
