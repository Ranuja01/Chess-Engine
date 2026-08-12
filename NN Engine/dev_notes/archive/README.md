# dev_notes/archive — superseded notes

Nothing here is wrong; it is **superseded or one-off**. Findings that outlived their document were
promoted into the canonical docs (`OPTIMIZATION_LOG.md`, `PAWN_MODEL.md`, `DIAGNOSTICS-TOOLKIT.md`, the
subsystem maps) or into memory. Read these only for provenance — "why did we conclude X" — never as
current state.

⚠️ **Anything here predates the 2026-08-08 symmetry sweep**, which removed ten colour/file defects and
changed the eval materially. Numbers in these files were measured on a different engine and are not
comparable to the current baseline register.

## What was moved (2026-08-09, 38 files)
- **`SESSION-HANDOFF-2026-07-04` … `-2026-08-06`** — per-session state. The three most recent
  (`-08-07`, `-08-08`, `-08-09`) stay in `dev_notes/` because they are still the live entry points.
- **`fable-question-*`, `fable-followup-*`, `fable-audit-*`** — one-off external-model consultations.
  Their conclusions, where they survived contact with measurement, are in the canonical docs. Several did
  not survive; treat any recommendation here as unvalidated unless you can find the measurement.
- **`ROADMAP-2026-07-*`** — superseded plans.

## Still in `dev_notes/` on purpose
Canonical and cited: `OPTIMIZATION_LOG.md` · `DIAGNOSTICS-TOOLKIT.md` · `PAWN_MODEL.md` ·
`BASELINE_PERF.md` · `collapse-reduction-ledger.md` · the subsystem maps (passed-pawn, king-safety,
search-architecture, search-ordering-pruning) · `speed-and-qsearch-findings-2026-07-24.md` ·
`search-reopening-angles-2026-07-24.md` · `search-architecture-fable-review-2026-07-28.md` ·
the SF reference docs.

▶️ A second pass could archive more of the dated July one-offs (campaigns, briefs, dissections), but
several are still cited from memory — check the citation before moving anything else.
