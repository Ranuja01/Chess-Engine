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

## 7. Can a STATIC eval price passers? (owner's question; `diagnostics/_passer_static_study.py`, 2026-10-03)
Mean |win% gap to SF18 d14 search| on SF18-labelled positions with an advanced passer (relative rank ≥ 5):
| set | n | SF11 static | OURS static | OURS d10 search |
|---|---|---|---|---|
| middlegame, all | 2,123 | 8.43 | 8.78 | **5.80** |
| middlegame r5 / r6 / r7 | 1,162 / 767 / 194 | 8.27 / 8.28 / 10.00 | 8.39 / 9.04 / 10.05 | 5.90 / 5.72 / 5.46 |
| **endgame, all** | 8,839 | **6.86** | 7.45 | (no d10 pass) |
| endgame r5 / r6 / r7 | 3,838 / 3,435 / 1,566 | 6.15 / 7.02 / 8.24 | 6.82 / 7.77 / 8.30 | — |
⇒ **Middlegame: SF11's static eval is NO better than ours** on passers — the rare cases are priced by SEARCH (both
statics ≈ 8.5 vs our search 5.8). **Endgame: SF11 is modestly better** (6.86 vs 7.45, ≈ 0.6-0.75pp at ranks 5-6; equal at
rank 7) ⇒ the classical static headroom over us is real but small, concentrated in eg ranks 5-6 — exactly where the
stop-state / support / escort cells live. The depth-target fit can exceed SF11-level (KS did), so this bounds the
"copy SF11" gain, not the fit's.

## 8. BUILD STATE (2026-10-03 night) — code written, NOT built (the gauntlet queues run from the working tree)
- C++ (commit b09a2fc): `px_counts` — 51 cells per side (layout in its header comment), `PX_V2` + `PX_V2_FILE` loader
  (c3_load_table contract), scored like the C3 blocks (needs the shared attack maps), published as `v2_pxpass`
  (EB_V2_PXPASS, v2-only, masked from EB_ALL); exported in `v2_features` at 184-234 (V2F_PER_SIDE 184 → 235; ChessAI.pyx
  buffers updated). Syntax-clean. Default PX_V2 = 0 ⇒ byte-identical expected — the build guard must prove it.
- Pipeline: `_px_export.py` (features + live theta for the ~34k LABELLED rows only) → `_px_depth_fit.py` (passer rank
  columns 77-96 + PX cells, both legs, on the DEPTH target, mg + eg sets, structure fixed; per-block arms RANK / STOP /
  SUPPORT / ESCORT / MISC / ALL, each also on top of RANK) → closure (`_texel_feature_pass.py` with PX_V2=1, block
  `v2_pxpass`) + symmetry → per-block gates (calibrated SF18 @800 + self-play).
- Needs first: a depth pass on the ENDGAME labelled set (`fitC_eg_sf18.csv`) with the current ship — passers matter most
  there and no eg d10 pass exists.

## 9. FIRST PX FIT (queue #15, 2026-10-04 02:26) — prep only, gates await the owner
Build guards: v1 250 / 35,310,778 · shipped v2 255 / 47,218,480 (PX off ⇒ unchanged). Endgame depth pass of the current
ship complete (19,280 rows; the log's "rows: 0" is my wrong relative path in the count line). Export: 34,621 labelled
FENs. Fit on the DEPTH target, mg + eg (32,707 rows, val 4,998; a PX cell fires on 55.6%), λ 1e-2, val vs base:
| arm | val | | arm | val |
|---|---|---|---|---|
| RANK (77-96 only) | −0.22% | | RANK+STOP | −0.35% |
| STOP | −0.11% | | RANK+SUPPORT | −0.40% |
| SUPPORT | −0.16% | | RANK+ESCORT | −0.26% |
| ESCORT | −0.16% | | RANK+MISC | −0.24% |
| MISC | −0.02% | | **ALL** | **−0.66%** |
Closure: `v2_pxpass` max 1.0 mp (9,662 live / 20k), `v2_passers` max 3.2 mp · symmetry colour 0/4000, file 0/3170.
⇒ Real but MODEST on this target (compare KS joint −4.07%, pawn joint −1.26% on mg only). Consistent with §7: in the
middlegame passers are priced by search; the static headroom (eg ranks 5-6) is small. Blocks are roughly additive
(STOP + SUPPORT + ESCORT ≈ ALL − RANK). Files: `E:/chess_data/texel/px_depth_{c1,px}.txt` (ALL arm).

## 10. v1-era PASSER CORPORA — have we plugged the weaknesses? (owner, 10-04; `diagnostics/_passer_corpus_check.py`)
`ks_sets/passer_corpus.csv` (288, SF18 labels, tiers) + `suites/passers.csv` (405, SF labels, blockade categories); mean
|win% gap to the SF label|, STATIC evals:
| corpus / tier | v1 | v2 shipped | v2 + PX fit | SF11 static |
|---|---|---|---|---|
| passer_corpus all (288) | 12.51 | 12.01 | 11.86 | **9.94** |
| blowup_guard (140) | 15.57 | **11.19** | 11.05 | 10.01 |
| control (79) | **5.20** | 7.68 | 7.63 | 5.57 |
| **under_fire (69)** | 14.65 | **18.63** | 18.35 | 14.79 |
| passers_suite all (405) | 12.23 | 11.02 | 10.97 | **9.38** |
| suite clear / major / minor / other | 12.66 / 12.29 / 11.69 / 11.59 | 10.81 / 12.22 / 11.01 / 10.21 | ≈ v2 | 9.73 / 9.73 / 9.44 / 8.45 |
⇒ v1's passer BLOW-UPS (over-reading) are fixed (15.6 → 11.2 ≈ SF11), but **UNDER-FIRE passers got WORSE** (v2 18.6 vs v1
14.7 / SF11 14.8) — v2 now under-reads (or misjudges) contested/attacked passers; controls also regressed (7.7 vs 5.2).
The PX fit moves these 0.1-0.3pp only. SF11 static still ≈ 2pp better overall ⇒ a concrete passer weakness remains: the
under-fire class. (Depth-10 search comparison running.)
**10a. DEPTH-10 SEARCH on the same corpora (2026-10-04):**
| corpus / tier | v1 d10 | **v2 shipped d10** | v2 + PX d10 |
|---|---|---|---|
| passer_corpus all | 9.25 | **6.68** | 6.59 |
| blowup_guard | 11.47 | **7.08** | 7.01 |
| control | 6.38 | 5.16 | 4.84 |
| under_fire | 8.17 | **7.60** | 7.70 |
| passers_suite all | 9.10 | **5.86** | 5.87 |
⇒ **v1's passer weaknesses are largely PLUGGED in v2 at search depth** (6.7 vs 9.3; 5.9 vs 9.1), INCLUDING under_fire
(7.6 vs 8.2 — the static 18.6 was search-resolvable, not a persistent eval hole). **The PX fit adds nothing measurable at
depth** (6.59 vs 6.68; 5.87 vs 5.86). VERDICT (owner's question "is there measurable gain from these additions?"): NO —
PX stays built at 0, NOT gated; revisit only inside the giant-corpus final retune (where it is free to be priced jointly).

## 11. STEP 0 — is SF11's static edge WEIGHTING or MISSING KNOWLEDGE? (owner, 10-04; `diagnostics/_joint_depth_preview.py`)
Diagnostic only (static closeness is not the goal; §10a shows our d10 search already beats both static evals here).
(a) JOINT DEPTH FIT, a preview of the final retune: PST (file-mirror-tied, mg/eg) + all 235 v2 columns (mg/eg) + Kaufman
cells and N/B/R/Q value corrections, 894 params, fitted together on the depth target (SF18 d14 vs our d10 search of the
10-03 ship; 32,707 rows, val 4,998), then applied to the SHIPPED static on the §10 corpora. (ceiling) The same features
re-weighted on the corpora's OWN labels from the static base, 5-fold CV — the most reweighting alone could buy here.
Guard: shipped static reproduced §10 exactly (12.01 / 18.63 / 11.02).
| λ | depth val | passer_corpus all | under_fire | control | passers_suite all |
|---|---|---|---|---|---|
| shipped | — | 12.01 | 18.63 | 7.68 | 11.02 |
| joint 1e-1 | −4.65% | 11.73 | 18.37 | — | 10.94 |
| joint 1e-2 | −8.56% | 11.39 | 17.89 | — | 10.79 |
| **joint 1e-3** | **−9.74%** | **11.08** | **17.45** | 6.47 | **10.65** |
| ceiling 1e-3 / 1e-2 / 3e-2 / 1e-1 / 1 / 10 | — | 12.54 / 11.20 / 10.82 / **10.67** / 11.14 / 11.46 | **14.43** / 14.47 / 14.71 / 14.93 / 16.26 / 17.43 | 12.39 / 9.77 / 8.50 / 7.74 / — / 7.68 | 14.99 / 12.04 / 11.36 / 11.05 / 11.00 / 10.98 |
| SF11 static | — | 9.94 | 14.79 | 5.57 | 9.38 |
Predictions (registered first): joint passer_corpus 11.3-11.9 (MISS — 11.08, better) · under_fire 17.5-18.5 (borderline
miss, 17.45) · ceiling ≈ 10.5-11 and short of SF11 (HIT, 10.67) · under_fire ceiling reaching SF11 — NOT predicted.
⇒ **MIXED, and it splits by corpus:**
- **passer_corpus: about half WEIGHTING.** The general depth fit closes 45% of the gap to SF11 (12.01 → 11.08); even
  in-domain reweighting stops at 10.67 ⇒ the remaining ≈ 0.7pp is not expressible by our features.
- **under_fire: EXPRESSIBLE, but in CONFLICT with the general fit.** In-domain weights reach SF11 (14.4-14.9), but only
  by wrecking the controls (control 7.7 → 12.4 at λ 1e-3); the depth target moves it just 31%. ⇒ a missing INTERACTION /
  gate (the weights that fix contested passers are wrong elsewhere), not a missing raw feature — and §10a shows d10
  search resolves this class (7.60), so it is search's job today.
- **passers_suite: MISSING KNOWLEDGE (or a label mismatch).** NO reweighting improves it (best ceiling 10.98 vs 11.02;
  SF11 9.38); the depth fit buys 0.37. ⚠️ The suite's `sf_cp` labels are a fixed-MOVETIME SF search (`gen_passer_corpus.py`,
  `Arbiter(find_stockfish(), movetime=…)`), not SF18 d14 — relabel before charging the 1.6pp to our eval. Largest category gap: major (12.2 vs 9.7).
Side finding (do NOT act on it alone): the JOINT depth fit gains −9.7% val vs ≈ −1% for every per-part fit (pawns, PX) —
joint pricing is where the final retune's room is. ☠️ No scale nuisance and Kaufman moves of ~243 mp rms at λ 1e-3 ⇒ part
may be piece-value stretching; corpus fit is anti-correlated with Elo until gated (memory
`corpus-fit-is-anti-correlated-with-elo`, Fit K "parts cancel"). It is a preview, not a candidate.
