# Texel C3 — new king-related detectors, designed from the references, priced by the fit

@author: Ranuja Pinnaduwage (maintained with Claude)

Status: DESIGN for owner review (2026-09-27). Built from three read-only cross-engine studies (SF1.1, SF11, SF15.1,
Ethereal, Weiss) on Opus agents. Context: C1 (re-fitting existing tables) was Elo-null; new COVERAGE is the lever
(`TEXEL-FIT-C-DESIGN-2026-09-27.md` §6). The owner's rule: study the giants, keep what is universal, and make the
rest our own.

## 0. Record check: all three are NEVER TRIED in v2
| detector | v2 | v1 | why not built |
|---|---|---|---|
| king shelter + pawn storm | never | `KS_SHIELD` / `KS_STORM` units, never measured alone; flat 185/75 shelter's sweep was a silent no-op ⇒ UNREADABLE | rung 2d scheduled, rung 2 shipped as 2a/2b only |
| pawnless flank + king-pawn distance | never | never | "NEVER TRIED, NEVER CONSIDERED" (REG:118) |
| KingProtector | never | never | deferred on an overlap premise the register itself later called half-false (REG:156-159) |
| king-zone defenders | never | `KS_DEFENDER` flat count: v1-channel, no games verdict ⇒ unreadable · `KS_DEFAWARE_MODE=1` shipped inside a +20.8 bundle, never separated | "DEFER — tiny" (RUNG1-KS-DESIGN) |

## 1. What is universal (kept) vs divergent (ours to choose)
| concept | universal across lineages | divergent |
|---|---|---|
| shelter | own pawns on king file ±1, at or ahead of the king's rank, graded by rank; mostly mg; direct score (5/5) | edge clamp, file weights (SF edge-distance vs Ethereal absolute file ☠️ fails our mirror gate), attacked-pawn exclusion, eg leg, feedback into danger, castling MAX |
| storm | 4/5 have it (~2 lineages); blocked vs unblocked split (SF + Ethereal); "enemy pawn stuck on the edge beside the king is good for us" | per-rank vs distance indexing |
| flank / pawn distance | endgame-dominated; penalty rises with distance and saturates by ~5; a cliff at ~3 files | own vs both colours; Chebyshev vs file metric; the no-pawn case |
| KingProtector | minors only; direct score | SF linear from distance 1 (own king) vs Ethereal nothing below 4 (nearer king) |
| defenders | implicit via weak squares / safe checks (v2 already has this) | SF counts attacked squares, subtracts from danger; Ethereal counts occupying pieces, as a direct score |

## 2. OWNERSHIP (the decision the owner asked for, proposed)
- **Fitted king mg PST owns the SQUARE; shelter/storm (KS-B) owns the PAWNS in front of it.** The PST has probably
  absorbed some average shelter ⇒ fit shelter JOINTLY with the king mg PST and pawn mg PST ranks 2-3 left free (C1
  showed collinear blocks trade value).
- **KS-A (attack units) keeps attackers.** Defenders enter KS-A later as a SUBTRACTIVE channel (C2, below), never
  additive.
- **OvD (the owner's concept) owns the UNCOMMITTED king; KS-B owns the COMMITTED king.** KS-B is silent while the king
  stands on its home square with a castling right. OvD gets "a storm makes castling there bad" and the
  opposite-castling race. The split is predicate-disjoint, so there is no shared signal. It also avoids the hazard of
  a shelter computed at e1 taxing central pawn moves.

## 3. OUR DESIGNS (linear features for the Texel pipeline; every one is colour- and file-mirror symmetric)
**C3-a Shelter + storm** (direct score; separate from `ks_units`), per committed king:
- Window: 3 files, centre clamped to b..g. File class F = min(f, 7−f).
- Shelter state (nearest own pawn at or ahead of the king's rank; d = relative rank gap): none · beside · 1 · 2 · 3 ·
  ≥4 · lever-attacked. "none" is pinned to 0 per file class.
- Storm state (the enemy's most advanced pawn in the span): unblocked d 1 / 2 / 3 / 4 / ≥5 · blocked d 1-2 / 3 / ≥4 ·
  none (pinned 0).
- **56 cells per leg × 2 legs = 112 params.** Fit from 0, plus a second run from an SF11 prior. If eg ≈ 0, drop the eg
  leg.
- Cost: ~40 ops, about the size of `build_pawn_entry`.

**C3-b Pawnless flank + king-to-pawn distance** (direct score):
- Chebyshev distance king→nearest OWN pawn and king→nearest ENEMY pawn, one-hot {2, 3, 4, ≥5} with 1 as reference:
  8 params, eg only. Splitting own/enemy is ours: it separates "guarding my pawns" from "attacking theirs".
- Flank (SF's KingFlank table, both colours): no pawns (mg + eg) · only enemy pawns (eg): 3 params.
- **11 params.** Covers king races with NO passer (v2 prices king-to-pawn distance only via passers).

**C3-c KingProtector** (direct score):
- Knight / bishop Chebyshev distance to OUR king, bins 1..6+, both legs, smoothed: **24 params.** The fit chooses
  between SF's linear shape and Ethereal's "nothing below 4".
- Optional: split each bin by "within 2 of the enemy king" (48).

**C2-d Defenders** (KS channel, SUBTRACTIVE; belongs to the KS fit, not C3):
- Per king: D1 zone squares defended · D2 doubly defended · D3 knight defends the king ring · D4 flank-camp defence.
- Enter as u −= Σ δ·D before the onset, with δ ≥ 0 and 0 by default (byte-identical). Plus our variant "defaware-v2":
  weight each attacker by the share of its footprint we hold.

**Existing zero-weight detectors** (threats, rook files, connected, reach, long diagonal, latent, tempo, bishop pair)
join the same fit as extra blocks at no design cost. Kaufman stays out (collinear with material).

## 4. Screening order (the record says build the move test before the term)
1. **Move tests first**, per concept: find positions where our move disagrees with SF18 on king shelter / king
   placement / minor retreat, and keep only failures that PERSIST as depth rises (d8 and d14). If fewer than ~2%, the
   concept has no eval-shaped headroom: stop before C++.
2. Build the detectors (weight 0 = byte-identical) and add them to `v2_features`; per-detector oracle + differential
   fire check.
3. Joint fit: C3 blocks + PST (+ the zero-weight existing detectors) on the fresh `fitC_std_d6` data, with Fit B
   variant data weighted (exposed kings make shelter fire).
4. Held-out gain must clearly exceed C1's −0.5%. Then closure, symmetry, SPRT, replication, the SF18 gauntlet, the
   variant gate.
5. Then **C2: KS fit** (shape + scale + defenders + a shelter→danger coupling tested as ONE scalar `u −= k·shelter`).

## 5. Risks
- Collinearity with the PSTs: the C1 pattern (held-out gain, Elo null).
- The Texel "already winning" confound: minors near the enemy king; king near enemy pawns.
- Endgame payoff rarely shows in Elo (`eval-payoff-is-opening-midgame-not-endgame`). C3-b is the most exposed to this.
- Castling-gate leaks distorting the opening: check the root-move distribution for e4/d4 before games.
- Storm cells with unstable bootstrap signs ⇒ ship shelter alone and hand storm to OvD.

## 6. KS RECALL STUDY (2026-09-27, `diagnostics/_ks_recall_study.py`)
Ground truth: SF11's classical king-safety row on `cond_corpus_v2.csv` (37,222 positions; |sf11_kingsafety| ≥ 1.5
pawns marks the endangered king) plus game results. Ours: `ks_counts` units under V2_PRESET=shipped; we "fire" past
KS_V2_ONSET = 450.

| | endangered kings | recall | endangered side's score: caught / missed | quiet kings |
|---|---|---|---|---|
| all positions | 7,098 | **7.4%** | 0.146 / **0.235** | 0.500 |
| **near-equal only** (corpus status) | 1,296 | **2.8%** | 0.542 (n small) / **0.451** | 0.500 |

**Readings (registered here before any design uses them):**
1. **Most of SF11's "king danger" is correlated with already losing.** The score gap shrinks from −26pp to about −5pp
   once material and eval are level. The pure king-danger effect is REAL (~3.5σ) but MODEST. ⇒ Temper "KS tuning
   could be huge": the upside lives in these near-equal ~5pp cases, and loosening KS has to be judged there.
2. **Recall is low mainly because of our THRESHOLD, not only missing detectors.** Missed kings DO have attackers
   (n_att 1.84 vs quiet 0.78; zone_net 1.72 vs 0.43), but units stay at ~91, far under the 450 onset. ⇒ The onset /
   curve belongs in the C2 fit, judged on near-equal positions.
3. **The best separators of missed vs quiet kings:**

   | condition | lift |
   |---|---|
   | zone squares attacked-more-than-defended ≥ 2, with a queen | ×9.6 |
   | 2+ enemy pieces attacking the zone | ×7.0 |
   | shelter ≤ 1 with any attacker | ×4.0 |
   | semi-open files ≥ 2 | ×3.1 |
   | open file | ×2.2 |
   | storm | ×1.8 |

   ⇒ The attack/defence BALANCE per zone square (the owner's point) and shelter gaps are the signals. Storm is weak.
4. **Uncommitted kings are UNDER-represented among the misses** (lift 0.34). The dangerous misses are castled kings
   under attack, which is KS-B / C2 territory rather than OvD's. OvD's case has to be made on a different axis
   (long-term pressure before attackers arrive), not KS recall.
5. Caveat: SF11's row includes shelter/storm and pawnless-flank terms that fire WITHOUT attackers; some "misses" are
   SF pricing structure, not tactical danger.

### 6a. ☠️ CORRECTION (same day): the recall study applied the onset TWICE
`ks_counts` "units" is already `max(0, u − KS_V2_ONSET)` (eval_v2.cpp:836), and the script then required
`units > 450`. So "caught" meant raw u > 900 (≈1.4 pawns of our danger). The script is fixed: `FIRE=0` = fires at
all. Re-run:

| | first run (wrong) | **corrected (fires at all)** |
|---|---|---|
| recall, all | ~~7.4%~~ | **54.4%** |
| recall, near-equal | ~~2.8%~~ | **46.5%** |
| near-equal scores, caught / missed / quiet | ~~0.542 / 0.451~~ | **0.453 / 0.454** / 0.500 |

- ~~Reading 2 "recall is low because of our threshold"~~: **RETRACTED.** KS fires on ~half of the SF-endangered kings.
- ~~Reading 3's "zone_net ≥ 2 & queen, lift 9.6"~~: an artifact of the wrong threshold; correctly measured, **1.76**.
- **NEW, load-bearing:** on near-equal positions, whether v2 KS fires does NOT separate outcome (0.453 vs 0.454). Both
  are ~5pp under quiet kings. The danger is real, but v2 KS does not discriminate it.
- **The misses are STRUCTURAL**, not tactical:
  - enemy queen present in only 47% (vs quiet 64%);
  - separated by shelter ≤ 1 (×2.2), semi-open files ≥ 2 (×3.3), storm (×2.1).
  - SF11's row prices pawn cover without attackers.
- ⇒ The recall gap is **shelter/storm (C3-a)** territory. The attack/defence balance redesign is still worth doing,
  but for PRECISION / DISCRIMINATION (fire where games actually go badly), not recall.
- Reading 1 (the already-losing confound) and reading 4 (uncommitted kings under-represented, lift 0.36) stand.

**Balance design research returned (Opus, 2026-09-27):**
- No reference compares attacker vs defender COUNTS per square. Defence enters as binary weak/safe sets (SF, Ethereal)
  or not at all (Weiss). A numeric balance is OUR concept (v1 `KS_DEFAWARE_MODE` = weight × contested/footprint,
  shipped inside a +20.8 bundle, UNREADABLE alone; v1 `KS_OVERLOAD` Σmax(0, att−def) was NO-GO additive on v1's
  channels, which does not transfer).
- Gaps in our `ks_units` vs SF11:
  - n_att ignores x-ray, although the maps are x-rayed;
  - adjacency counts distinct squares, not attack instances;
  - defender maps are not pin-restricted;
  - no unsafe-check / blocker / flank channels;
  - the defence share of u is ~1/3 of SF's (a weak square is worth 57 vs SF 185, against a similar attacker product).
- Arms:
  - **A** SF-faithful fixes (x-ray n_att, adjacency instances, unsafe checks, pinned defenders + blockers).
  - **B** our per-square balance from bit-sliced attack-count planes: B1 Σmax(0, nA − nD), B2 #squares with nA ≥ 2 & nA > nD, B3 = B2 × queen.
  - **C** subtractive hybrid: defaware-v2 replaces the attacker term + B2.
  - All channels start at 0 (byte-identical); the onset is refit JOINTLY.
- Falsifiers: the corrected recall study must show (i) better near-equal DISCRIMINATION (the caught side scores clearly
  below the missed side), (ii) quiet-king fires within +1pp split by phase (attempt #24 failed on opening over-fire),
  (iii) channel r with w_att·n_att < 0.8. Then regret and games.
