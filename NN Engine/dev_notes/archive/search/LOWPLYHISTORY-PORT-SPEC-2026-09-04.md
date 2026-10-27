# `lowPlyHistory` — port spec + the six pre-port questions (2026-09-04)

**Why now**: IIR reads small-positive-but-unresolvable (~+8 ±25). Per
[[the-sf11-gap-is-two-thirds-node-efficiency]] the instrument can only resolve a bundle worth ≳+15-20, and
IIR is **0-for-2 at bundling** — both failures were with other NODE-SAVERS (same population). We need a
partner from a **different class**. This is the best-profiled ordering-class candidate.

## WHAT IT IS (SF17, read from source — not recalled)
A **ply-indexed butterfly history covering only the top 4 plies**, refilled every `go`.
| site | code |
|---|---|
| size | `history.h:39` `constexpr int LOW_PLY_HISTORY_SIZE = 4;` → `Stats<int16_t, 7183, 4, 64*64>` = **32 KB** |
| **WRITE** | `search.cpp:1911` in `update_quiet_histories`: `if (ss->ply < 4) lowPlyHistory[ss->ply][move.from_to()] << bonus * 829 / 1024;` (~81% of the main-history bonus) |
| **READ** | `movepick.cpp:184`: `if (ply < 4) m.value += 8 * (*lowPlyHistory)[ply][m.from_to()] / (1 + 2 * ply);` |
| refill | `search.cpp:332` `lowPlyHistory.fill(92)` — **once per `go`, BEFORE the ID loop** (so it survives across iterations WITHIN a search, and resets between moves) |
| new game | `search.cpp:579` `fill(105)` in `Worker::clear()` |
Read weight by ply: **ply0 8× · ply1 2.67× · ply2 1.6× · ply3 1.14×** — a steep decay, so it is
overwhelmingly a **ply-0/1** device.

## ⇒ WHAT IT ACTUALLY DOES
It is **cross-iteration ordering memory scoped to the top of the tree and to the current search**. That is
precisely the job our pre-search's WARMING channel does (measured worth: **+42 STS**), and precisely the
gap at cascade **tier 3**, whose current key is move-gen heuristic order — recorded in our own comment as
measured WORSE (−12 WAC / −44 STS).

## THE SIX PRE-PORT QUESTIONS ([[six-occupancy-failures-and-the-transfer-model]])
1. **OCCUPANCY** — what already produces this decision here? `historyHeuristics[turn][from][to]`. ⚠️ It is a
   butterfly history too, but **global and long-lived** (decayed, never reset per search). SF's is
   **ply-local and per-search**. Near the root SF weights recent, ply-specific evidence **8×** over its own
   general history. ⇒ the SLOT IS GENUINELY EMPTY; the signal is different, not a re-keying.
2. **TRIGGER OVERLAP** — `moveFrequency`'s PV bonus is the nearest live mechanism and also acts near the
   root. ★ **RUN THE 2×2 WITH `moveFrequency` FIRST**, and with IIR.
3. **MARKER ≠ GUARD** — no guard here; this is an ordering term, so the honest pre-measure is the
   accuracy map: does a ply-local history score better than the global one against the wrong-order label at
   plies 0-3? Measure BEFORE wiring.
4. **CONSUMER AUDIT** — our ordering consumers are `generateLegalMovesReordered` and the cascade sort.
   ⚠️ Adding a term changes BOTH; the cascade is default-off so start with the main orderer only.
5. **PREREQUISITES** — needs a per-`go` reset hook. `clearSearchTables` is DIAGNOSTIC-ONLY (games never
   call it) ⇒ **a per-search fill must be added to the GAME path**, or the table silently becomes global and
   we have merely built a second `historyHeuristics`. ☠️ This is the single most likely way to get a null.
6. **COMMENSURABILITY** — bonus units are ours; scale `829/1024` and the `8/(1+2·ply)` read weight are
   SF-tuned against SF's bonus magnitudes. **Do not port the constants blind** — our
   `HISTORY_BONUS_SCALE` sweep was already a null, so the shape matters more than the numbers.

## ⚠️ THE COUNTER-EVIDENCE (must be read before building)
[[history-sparsity-is-coverage-not-architecture]] — the uninformative statScore bucket shrinks
84.7%@d8 → 65.2%@d12, i.e. it is fixed by DEPTH, and **denser re-keying did NOT shrink it**. Conclusion
recorded there: *history levers are PREMATURE, not dead.*
⇒ This is a real precedent for a null. The counter-argument is that `lowPlyHistory` is **not** a denser
re-key of the same signal — it is a different signal (ply-local recency, per-search scope). That distinction
is the whole bet, and it is exactly the kind of story I have been wrong about twice this week.
★ **So measure the signal BEFORE building the table** (question 3), rather than building then measuring.

## ▶️ ORDER OF WORK
1. Accuracy-map probe: ply-local vs global history AUC against the wrong-order label at plies 0-3. **If it
   does not separate, STOP** — that is the cheap null and it costs no build.
2. Only then build, behind `ENABLE_LOW_PLY_HISTORY`, default off, byte-identity verified.
3. 2×2 against `moveFrequency`, then 2×2 against IIR.
4. Bundle only if both corners hold.
