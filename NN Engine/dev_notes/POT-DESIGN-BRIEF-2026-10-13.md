# POT design brief — what the retuned eval still gets wrong (2026-10-13)

**For the owner's POT design discussion.** Groundwork only: a residual map of ship gen 3 and fit-only probes of candidate POT
inputs (`diagnostics/_pot_probe.py`). Nothing here touches the engine. Context: `research-2026-10-08/POT-DESIGN-PREPLAN.md`
(+ your framing notes at its end), the 10-12 bench (`dev_notes` / memory `eval-v2-rebuild-state`), and the agreed order: POT is
designed on the RESIDUAL of the retuned eval, fitted on top with everything else frozen, then a light joint refit.

## 1. Where gen 3 is still wrong (held-out own-play games, endgame-leaning + endgame)

Win% MSE, and SIDE-AHEAD BIAS in brackets (win% points; negative = the side that is ahead is under-rated = too drawish).

| endgame type | static eval | our d10 search |
|---|---|---|
| pawn ending, pawns equal | 586 (**−23.8**) | 273 (**−14.6**) |
| pawn ending, a pawn up | 321 (**−15.2**) | 146 (**−6.5**) |
| minor only, equal / up | 129 (−12.9) / 163 (−6.4) | 39 (−2.0) / 106 (−2.9) |
| rook only, equal / up | 79 (−9.1) / 134 (−6.3) | 39 (−3.9) / 69 (−2.5) |
| rook + minor, equal / up | 93 (−7.7) / 93 (−4.7) | 38 (−2.5) / 37 (−0.9) |
| mixed / imbalanced, equal / up | 102 (−4.2) / 86 (−3.3) | 60 (+0.5) / 43 (+1.4) |
| queen ending, up | 162 (−4.5) | 121 (+1.3) |

- **Statically, every endgame type is under-confident.** Our search removes most of it — **except pawn endings**, where even the d10
  search stays badly under-confident (−14.6 / −6.5).
- Cross-check with the 10-12 bench (vs SF11 on the bench corpora): the same three types lead the gap — pawn, queen, rook+minor
  endings — and our effective endgame pawn is under-valued against pieces (pawn/knight 0.41 vs SF18 search 0.50).

## 2. Candidate POT inputs, each fitted ALONE on top of the frozen gen-3 eval

Endgame-weighted ((256 − phase)/256). Val change vs the ship, static (quiet) and depth (our d10), all positions and EG stages:

| candidate (form) | static all | static EG | depth all | depth EG |
|---|---|---|---|---|
| DIRECTION · passers (count, most advanced rank, protected, outside) | +0.0% | +0.1% | +0.3% | +0.4% |
| DIRECTION · majority wings | 0.0% | 0.0% | 0.0% | 0.0% |
| DIRECTION · king centralisation · king-to-passer race | 0.0% | 0.0% | 0.0% | 0.0% |
| **MAGNITUDE · complexity (SF12+ `winnable()` inputs) as sign(eval) × input** | **−3.4%** | **−5.9%** | **−1.05%** | **−1.5%** |
| direction + magnitude together | −3.7% | −6.3% | −0.8% | −1.1% |

Complexity weights (mp, endgame leg; positive = MORE confident): **pure pawn ending +580** · pawns on both flanks +132 ·
pawn count +20/pawn · king outflanking +20 · king infiltration −52 · passer count −32. Effect on the depth bias: pawn endings
equal −14.6 → −11.0, a pawn up −6.5 → −4.4; rook+minor −2.5 → −1.7; others ~unchanged.

**Reading:**
1. **POT's MAGNITUDE half (complexity: how decisive the existing edge is) carries the signal; the simple DIRECTION half
   (potential) adds nothing** — today's PX / passer / KFL terms already own passers, majorities and king activity.
   This matches your framing that complexity and potential are separate quantities, and says which one is missing now.
2. The magnitude gain is mostly STATIC (search already resolves much of the under-confidence) — but static discrimination
   matters for pruning / ordering (your 10-07 principle), so a static gain is not worthless.

## 3. Decisions for you

1. **Cause split for pawn endings.** The biggest single weight (+580, pure pawn ending) may be compensating a SEARCH gap: SF11
   wins pawn endings by search (no null move / no shallow pruning with only pawns, passed-pawn push extension), which the search
   transition is scheduled to add. Options: (a) build POT magnitude WITHOUT the pure-pawn-ending input now and let the search
   transition fix pawn endings, then re-measure; (b) include it now and expect to shrink it after the transition; (c) do the
   two SF11 pawn-ending search rules FIRST (small, verified), then design POT on that residual.
   *My recommendation: (c) or (a)* — the record's rule is "split endgame gaps by cause; search-caused → the transition", and (c)
   is small.
2. **Form.** SF12+ applies complexity as `sign(eg) × max(complexity, −|eg|)` on the endgame leg (it can only move toward a
   draw by up to the whole edge). The probe used an unbounded linear form. Proposal: adopt SF's bounded form (it cannot flip a
   sign), with our own inputs and weights fitted nested on top.
3. **Owner inventions to keep distinctive.** Your POT concept also has a middlegame half (potential while a subsystem is still
   unresolved, pushing either way) — parked since 10-01 as null at depth. The probe says nothing new about it; keep parked, or
   re-test once complexity is in?
4. **Winnability vs POT boundary.** Winnability (the multiplicative scale) stays standalone; complexity would be additive and
   live in POT. Round 3 (running) prices the four winnability caps — its result tells us how much of the endgame bias the
   scale side can absorb before POT's additive side starts.

## 4. Next steps once you've decided
Build the chosen POT inputs as engine detectors (byte-identical at 0, symmetry-gated), fit them nested on the round-3 ship with
everything else frozen, light joint refit, then the usual gate (self-play → SF18 → Mediocre) and the bench checkpoint (#2 of 3).
