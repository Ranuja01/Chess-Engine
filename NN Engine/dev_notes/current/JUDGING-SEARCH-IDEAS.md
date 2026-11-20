# JUDGING SEARCH IDEAS — how to tell whether an idea is good when the engine was tuned around the one it replaces

Written 2026-11-13 at the owner's request, during the root work (`ROOT_SEARCH_MODEL.md`). The owner's concern, in their words:
"the results may not immediately reflect the actual order of how good something is given the dozens of other parameters
that surround it ... it's possible that the very best could come out looking like one of the worst. And if we get rid of
that idea early on, it's possible then that we never come back to it."

This document is the protocol that guards against that. It applies to any search change that replaces or competes with an
existing system: the root, the move source, pruning, history, the time manager. Instrument details (what each tool resolves
and how it misleads) are in [`INSTRUMENT-MAP.md`](INSTRUMENT-MAP.md).

## 1. Why games mislead in this zone

| effect | what it does | record |
|---|---|---|
| **The incumbent was tuned around itself** | Margins, thresholds and schedules were fitted with the incumbent present. A challenger tested on them carries their misfit, so an untuned challenger starts behind. | memory `a-correctness-fix-into-absorbed-tuning-is-not-free` (retune LAST) |
| **Load-bearing parts** | A part's consumers lose their supplier when it is removed, so the test measures "the part minus its supplier", not the idea. | v1 capture gains; the September pre-search removals (`ROOT_SEARCH_MODEL.md` §5) |
| **Antagonism** | Parts that win alone can lose together, and the reverse. Bundles hide which part did what. | memory `search-changes-are-antagonistic-not-additive`, `bundling-is-refuted-components-cancel-26-percent` |
| **Early reads inflate** | An interim Elo, or one at the bound a sequential test stops on, is biased away from zero. | SEE bit 4: +15.9 at 550 g → −0.7 at 2,010; A4: −32 at 420 g → −15 at 1,280 (memory `sprt-point-estimates-inflate-at-the-bound-they-stop-on`) |
| **Fixed depth is not equal time** | Accuracy per depth is not strength, and nodes are not time: a change can save nodes and lose nodes per second. | memory `ebf-metric-is-not-comparable`, `byte-identity-does-not-imply-speed-identity` |
| **Small savings are invisible** | Node savings below roughly 35% (about half a ply) have been Elo-neutral in our record. | memory `node-savings-below-35-percent-are-elo-neutral-dont-scale-from-a-bundle`, `half-a-ply-is-elo-neutral-the-speed-bar` |
| **Time-control dependence** | A result at 10+0.1 may not hold at longer controls. Ideas that buy accuracy tend to gain with depth, and ideas that buy speed tend to lose some. | — |
| **The harness has its own null** | Two identical engines do not score exactly 50%. | memory `games-harness-null-is-plus4.6-elo-and-true-candidate-spread-is-16` |
| **Instruments can disagree fairly** | Self-play, SF18 and Mediocre can each be right about different things. | memory `two-fair-instruments-can-genuinely-disagree` |

## 2. The protocol, in order

1. **Mechanism first.** Before any test, add counters for what the idea supplies and what it consumes, and prove it is live
   (a changed node count or a changed counter, never a byte-identical result). Bound the mechanism, not a proxy (memory
   `bound-the-mechanism-not-a-proxy`).
2. **Map the dependencies.** List every consumer of the part being replaced (the subsystem's model document, e.g.
   `ROOT_SEARCH_MODEL.md` §2). Measure each consumer's weight with counters before deciding what the replacement must supply.
3. **Complete before comparing.** A replacement gets a new supplier for every consumer, or a measurement showing the
   consumer no longer needs one. Never switch a part off alone and call the result a verdict on the idea.
4. **Screen at fixed depth on four venues**: WAC (tactical sanity only, never a node judge for root work), STS300
   (accuracy), the quiet corpus and the game set (nodes). Screens rank and prune the field; they are never verdicts.
5. **Register before any game**: the rule (thresholds in σ, pentanomial), the prediction, the game count, and what each
   outcome leads to. Report PGNs as separate arguments (memory `pool-pgns-as-separate-args-never-cat`); check pairs = games / 2.
6. **Games at equal time.** Read headroom signals beside the Elo:
   - accuracy per depth (what the idea has left to spend);
   - nodes and nodes per second (where the time goes);
   - for finalists, the same comparison at a second time control or depth (scaling);
   - what the idea enables next (a step that only becomes possible with it).
7. **Name the verdict precisely.** Only the last label retires an idea, and even then its code stays behind its switch:

   | label | meaning |
   |---|---|
   | SHIP | passes the ship rule (self-play ≥ +2σ, SF18 at equal nodes > −1σ, Mediocre not negative) |
   | LEVEL | within ±2σ; kept as an option, with its headroom signals recorded |
   | BEHIND, INCOMPLETE | a consumer has no supplier; the trigger is "when the supplier exists" |
   | BEHIND, UNTUNED | complete but not refitted; the trigger is "after its tuning pass" |
   | LOST, COMPLETE AND TUNED | complete, retuned, and still worse at two time controls; retired as a default, kept in the code |

8. **Register every non-winner** in its subsystem's register (for the root, `ROOT_SEARCH_MODEL.md` §5), with a re-test
   trigger that is an EVENT ("when a cheap ply-1 node exists"), not a date. Reread every trigger at the start of each stage
   (memory `a-park-trigger-nobody-rereads-is-a-silent-rejection`).
9. **Compare finalists only after each is tuned** (replicated SPSA, memory `spsa-tuning-needs-replication-not-convergence-stats`),
   at two time controls. An untuned challenger is never judged against a tuned incumbent as if they were equals.
10. **Never delete the incumbent.** A winner becomes the default; the old path stays behind its switch for hybrids and
    re-tests (memory `keep-v1-search-for-hybrid-ideas`). Removal is a separate decision, the owner's.

## 3. Reading a result

- An interim Elo is a progress report. Only the registered game count gives the verdict.
- "Behind" asks which supplier or which tuning is missing. Use the counters (step 1) to find it before any conclusion about the
  idea itself.
- "Level" with better accuracy per depth means there is headroom, not that the idea is neutral.
- A win on one venue and a loss on another is information about the mechanism, not noise to average away. Read it per venue
  and per stratum (memory `criticality-split-reads-signal-not-average`).
- Before calling anything a null, count the effect the test could resolve (memory `count-the-resolvable-effect-before-calling-a-null`).

## 4. Hygiene fixes ship on "not harmful" (owner, 2026-11-20)

Owner: "if there is a justifiable reason to say something is cleaner, then I suppose we should go with it. It may if truly
cleaner, support future development better."

- **What counts as hygiene:** today's behaviour breaks an invariant we can state, and the fix has one right form with no
  constant to choose. The invariants: a chess rule (repetition, 50-move, en passant); the definition of a returned bound or a
  TT flag (a fail-soft upper bound must not be below the true value; EXACT only for an in-window result); the symmetry of
  the two search halves; no state leaking across searches or games. Three audits finding the same thing is supporting
  evidence, not the test.
- **What does not:** a form or a constant choice, even when it fixes a real defect, if more than one form is defensible (how
  far toward beta to return, what a lazy cut should return, how aggressive a pruning rule is). Those keep the +2σ self-play
  bar and are parked when neutral, with a trigger.
- **The bar:** self-play z > −1 (the point estimate above −1σ; at 2,000 games σ ≈ 6 Elo), then the outside judges pooled
  z > −1 (SF18 + SF1.1), then the owner's yes.
- **How:** each hygiene item still gets its own self-play read (so a harmful one is caught alone). The items that clear z > −1
  are then confirmed together in one bundle against production, self-play and outside judges, before shipping. That checks
  their interaction (memory `search-changes-are-antagonistic-not-additive`) and costs one set of judges instead of one per item.
- **Why it helps later work:** every constant tuned afterwards is fitted to the corrected search instead of absorbing the
  defect (memory `a-correctness-fix-into-absorbed-tuning-is-not-free`: retune last, on the clean base).
