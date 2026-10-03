# Passer system — design for owner review (2026-10-03)

Owner (10-03): path support/defence never got the "map to the giants" treatment (it came from v1/SF11); with a full
reference mapping and a design discussion about what we want and why, complete pawns + passers and Texel-tune them to
perfection — "hopefully improvements like we did with KS, as this is a pivotal endgame concept".
Source: Opus engine-contrast over 11 engines / 7 lineages (SF11, SF15.1, Ethereal v12.75, Weiss master, Berserk 4.7.0,
Laser, Xiphos, Stash v35, Igel 2.4, Defenchess, Koivisto v4). Fetched line numbers UNVERIFIED. Pawn-structure contrast and
record check: C3 doc §19-19a. Status: DESIGN — nothing built beyond what exists.

## 1. What we score today (V2_PRESET=shipped)
- Passer + candidate (50%) rank table — **endgame leg only** (mg ×0).
- King distance to the stop square, both kings, rank weight w = 5r−13, **eg only**; no second-push term.
- **Nothing else live:** the SF11 path ladder is built but OFF; no defended/connected passer term; no rook-behind term;
  no passer-file term; no square rule; passer count enters no scale (POT winnability's passer input ships at 0).
- Queue #14 (running tonight) re-prices only the EXISTING columns (rank tables both legs, 4 king coefficients).

## 2. What the giants do (concept × lineage; ours)
| concept | engines / lineages | design flavours | ours |
|---|---|---|---|
| king distance (escort / shepherding) | 10 / 6 | stop vs pawn square · linear w vs rank tables · eg-only (SF, Xiphos) vs **both legs** (6 of 10) · SF adds a 2nd-push term | stop, linear w, eg only |
| pawn-defended / connected passer | 8 / 5 | generic Connected (SF, Ethereal — ⚠️ our docs wrongly say Ethereal denies it) vs dedicated `PassedDefended[r]` (Weiss, r7 S(158,96)) | **none** |
| stop square defended by own pieces | 5 / 4 | SF +5 rung · Laser DEFENDED/FULLY_DEFENDED · Berserk inside safe-advance | none |
| blocked stop square | 7 / 4 | **withhold** (SF, Berserk, Laser) · **table-select** (Ethereal, Koivisto) · **subtract** (Weiss PassedBlocked) · own+enemy blocker (SF, Ethereal, Weiss) vs enemy only (Berserk, Laser) | none |
| stop-square safety / free advance | 7 / 3 | SF one rank-weighted scalar ladder · Ethereal `PassedPawn[canAdv][safeAdv][r]` · Weiss `FreeAdv[r]` (**mg NEGATIVE, eg positive**) · Berserk multi-condition flag (both legs +) | none (ladder off) |
| whole path safe | 4 / 3 | SF rungs · Ethereal SafePromotionPath S(−49,57) · Laser FREE_PROMOTION | none |
| own R/Q behind (Tarrasch) | 5 / 3 | SF inside the ladder (x-ray) · Weiss PassedRookBack S(21,46) · Berserk/Laser line-of-sight | none |
| enemy R/Q behind | 4 / 3 | SF unsafe span · Berserk S(29,−135) · Laser blocks the path | none |
| passer file (edge > centre) | 4 / 3 | SF S(11,8)·edge dist · Laser · Berserk | none |
| candidates | 5 / 3 | SF halving rule · Ethereal/Berserk own `[r]` tables · SF15.1 helper filter | flat 50% |
| square rule / unstoppable | 2 / 1 | Weiss S(−26,422), Berserk S(0,440), defender has no pieces | KPK exact only |
| passer count in scaling | 2 / 1 | SF complexity +9·n; SF15.1 OCB sf 18+4·n | built at 0 |
| passers in KS / threats | 0 / 0 | — (no reference wires passers into king danger or threats) | — |
Nobody prices the blockader's PIECE TYPE (only Berserk's outside-passer-vs-knight). Every reference form is ADDITIVE —
the v1 graveyard ("no 5th passer VALUATION mechanism") was about MULTIPLICATIVE valuation and does not cover these.

## 3. Proposed feature set (Texel fit on the DEPTH target, rank cells r4-r7, BOTH legs unless noted)
1. Passer + candidate rank tables (have; mg leg freed — queue #14 tests this).
2. **Stop state**, mutually exclusive: S1 blocked [r] · S2 free AND not enemy-attacked [r] · (S3 free-but-attacked =
   baseline). Optional S1 split own vs enemy blocker.
3. **Whole path free** of enemy pieces/attacks [r] (nested inside S2).
4. **Stop defended by any own piece** [r].
5. **Pawn-defended passer** [r] (+ optional phalanx passer [r]) — owns "connected passers"; the generic connected term
   stays parked (one owner).
6. **King escort**: both kings' distance to the stop, per-rank cells, eg AND mg legs; + SF's second-push own-king term.
7. **Own R/Q behind** and **enemy R/Q behind** (line of sight), eg-weighted.
8. **Passer file** (distance from the edge), one coefficient, both legs.
9. **Square rule** (defender has no non-pawn material; eg only), gated off where KPK-exact already decides.
10. Passer count as an INPUT to the winnability scale factor (a scale, not a sum) — separate from per-pawn value.
Ownership: none of 2-9 is read by KS, threats or mobility in any reference; mobility counts piece moves, not pushes. The
fit prices them jointly with the pawn-structure columns and the refit isolated/backward exclusion (§19a), then the
gates go PER PART (stop-state block · support block (4,5,7) · king escort block · the rest).

## 4. Questions for the owner
1. **Blocked passers:** withhold the bonus (SF), separate fitted cells (Ethereal), or an explicit penalty (Weiss)? The
   fit can decide if we give it the cells (my lean: Ethereal-style cells — the fit picks the sign).
2. **Free-advance middlegame sign:** Ethereal/Weiss fit it NEGATIVE in mg (a free passer matters in the endgame, not
   the middlegame), SF positive. Leave both legs free and let the depth fit decide? (It also fits the §14 lead: we
   over-rate the leader's mg edge when passers are on the board.)
3. **King escort in the middlegame** (6 of 10 engines) — include the mg leg? (your shepherding point)
4. **Pieces escorting** — rook/queen behind (Tarrasch, 3 lineages) and "stop defended by own pieces" (4 lineages) cover
   piece shepherding; nobody prices the blockader's type. Enough, or do you want blockade QUALITY as an invented term
   (would go in at 0 under the single-lineage rule)?
5. **Square rule** (single lineage) — include at 0 for the fit to price, or skip?

## 5. Plan
1. Owner review of §3-4.
2. C++: extend `v2_features` with the new passer cells (same pattern as C3; mirror-safe), knobs at 0 ⇒ byte-identical.
3. Feature pass on the labelled rows; fit jointly with the pawn-structure columns on the depth target (current ship).
4. Closure + symmetry → per-part gates on the calibrated judge (SF18 @800) + self-play.
5. Then the isolated/backward exclusion refit folds in (§19a).

## 6. Owner review, round 1 (2026-10-03)
- **Order:** passers FIRST as their own pass, pawn structure SEPARATELY after (owner: "not convolute them just because
  they are both pawns"). The passer fit holds structure at whatever ships from queue #14; the structure pass (incl. the
  isolated/backward exclusion) holds the passer system fixed.
- **Free-advance mg sign:** leave every cell per-rank with both legs free — the owner's case (closed mg, an opened flank,
  a far passer forcing sacs) must be expressible as a strongly POSITIVE r6-r7 mg cell; nothing forces a sign.
- **King escort in the middlegame — "inherently available, like central":** mg legs for BOTH kings (escort and the
  commoner DEFENDING king in its own backyard), rank-weighted so they only matter for far passers; KS remains the
  counterweight against kings marching out (both fitted, the fit sets the balance).
- **Blockade quality:** split the blocked-passer cells by blocker type (minor vs rook/queen), entered at 0 (single-lineage
  rule); reasoning why references skip it: a pawn block is not a passer; a minor blockader is usually an outpost (already
  rewarded); a heavy blockader being chased is a threat that search resolves — mostly priced elsewhere, cheap to test.
- **Square rule:** IN, defender without non-pawn material, overridden by the exact KPK bitbase where that applies.
