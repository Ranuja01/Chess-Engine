# dev_notes/archive: superseded and one-off notes

Nothing here is current state. These are superseded plans, one-off investigations, campaign logs and old session handoffs,
kept for provenance: "why did we conclude X, and on what evidence". Findings that outlived their document were promoted
into `current/` (the ship record `OPTIMIZATION_LOG.md`, the instrument map, the subsystem documents) or into memory.
Each file's creation and last-update dates are in the index of [`../README.md`](../README.md).

| folder | contents |
|---|---|
| `handoffs/` | Session handoffs, 2026-07-04 to the one before the latest (the latest stays in `active/`), plus the June `HANDOFF.md`. |
| `eval/` | Eval work, v1 and v2: the v2 rung and slice designs, gap audits, Texel fits, POT, triangulation, the v1 term campaigns. |
| `king-safety/` | The king-safety campaigns (July-August): detectors, the signed accumulator, phase gating, sweeps. |
| `pawns/` | Pawn and passed-pawn work: maps, ground-truth findings, the passer redesigns. |
| `search/` | Search campaigns: ordering, pruning, pre-search, EBF, sweeps, corrhist feasibility, the June crash playbook. |
| `speed/` | NPS work before the October speed batches. |
| `collapse/` | The July "collapse" investigations (sudden eval collapses in games). |
| `games/` | Analyses of individual games and matches. |
| `planning/` | Audits, roadmaps, strategy resets, backlogs, and the one-off external-model consultations (`fable-*`). |

**Caveats that still apply:**

- Anything dated before the **2026-08-08 symmetry sweep** was measured on an engine with ten colour/file defects that
  the sweep removed. Its numbers are not comparable to the current baseline register.
- Any search-based positional or regret number from before **2026-08-14** is confounded by diagnostic-harness history
  contamination (see `current/INSTRUMENT-MAP.md`). Games and static-eval numbers are unaffected.
- The `fable-*` consultations are external-model advice. Where a recommendation survived measurement it is in the
  current documents; treat any other recommendation as unvalidated.

History: a first pass on 2026-08-09 moved 38 files here; on 2026-10-27 the whole of `dev_notes` was sorted into
`current/`, `active/`, `reference/` and these topic folders.
