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

### 6b. CHANNEL SCREEN on near-equal kings (2026-09-27 evening, `_ks_recall_study.py` section 2)
32,250 near-equal kings (1,296 SF-endangered, 13,016 quiet). Section 1 reproduces 54.4 / 46.5 / 0.453 vs 0.454.
Split = endangered median; hi/lo SE ≈ 0.028; r over all near-equal kings (1/√n ≈ 0.006, ~0.008 with kings paired).

| channel | s_hi | s_lo | r | r\|u | quiet fire early/mid/late (endangered) | r_wn |
|---|---|---|---|---|---|---|
| units (shipped KS) | 0.453 | 0.454 | −0.014 | — | 1/1/1 (51/55/30) | 0.50 |
| n_att | 0.471 | 0.427 | −0.025 | −0.022 | 11/15/12 | 0.88 |
| w_att | 0.450 | 0.464 | −0.024 | −0.020 | 18/19/10 | 0.90 |
| weak | 0.450 | 0.459 | −0.017 | −0.014 | 1/8/35 | 0.29 |
| adj | 0.440 | 0.473 | −0.019 | −0.016 | 0/5/26 | 0.43 |
| checks | 0.455 | 0.450 | −0.025 | −0.023 | 4/7/20 | 0.09 |
| n_att_x | 0.469 | 0.428 | −0.025 | −0.022 | 12/16/12 | 0.87 |
| adj_inst | 0.452 | 0.455 | −0.022 | −0.018 | 1/4/15 | 0.53 |
| unsafe | 0.456 | 0.452 | −0.012 | −0.011 | 19/31/34 | 0.28 |
| blockers | 0.493 | 0.445 | −0.005 | −0.004 | 8/8/7 | 0.21 |
| **flank_att** | **0.437** | **0.477** | **−0.044** | **−0.042** | 26/17/8 (74/67/37) | 0.46 |
| **flank_def** | **0.475** | **0.432** | +0.009 | +0.008 | 98/72/19 | −0.06 |
| knight_def | 0.467 | 0.446 | −0.019 | −0.020 | 85/52/25 | −0.07 |
| contest_excess | 0.455 | 0.452 | −0.021 | −0.018 | 1/9/35 | 0.36 |
| contest_sq | 0.464 | 0.447 | −0.026 | −0.023 | **2/5/10** (35/43/42) | 0.35 |
| w_att_contest | 0.442 | 0.466 | −0.019 | −0.015 | 2/7/17 | 0.52 |
| gate | 0.458 | 0.430 | −0.020 | −0.018 | 54/32/13 | 0.57 |

**Reading:**
- ☠️ **This instrument cannot rank the channels.** Even the shipped KS total reads r = −0.014 against the result. The
  whole SF-danger effect on near-equal positions is ~0.047, so a perfect split would be worth only ~2σ here. Every
  channel sits at |r| 0.01–0.04, and their differences are 1–2σ.
- The only standout is the **flank pair**: `flank_att` has r|u −0.042 (~5σ) and a −0.040 split, and `flank_def` has
  a +0.043 split in the defending direction. That is SF's kingFlankAttacks/Defense pair, and it measures region
  control rather than ring contact.
- **Precision:** `contest_sq` is the cleanest firer (quiet 2/5/10% vs endangered 35–43%). `weak`, `adj`,
  `contest_excess` and `unsafe` over-fire on LATE quiet kings (26–35%), which is the phase risk to control for in
  the fit. `n_att`, `n_att_x` and `w_att` are collinear with w·n (≥ 0.87), as expected; `adj_inst` is not (0.53).
- **Registered predictions, scored 1 of 4:** contest separation (wrong, reversed split), `knight_def` positive
  (wrong, −0.019; it fires mostly in the opening), unsafe/blockers null (right), `adj_inst` collinear (wrong).
- **Decision:** arm C screened on this instrument would be equally unreadable, so it is NOT run here. Every channel
  goes into the **joint KS fit** (millions of rows, where r ≈ 0.02 is resolvable), then the move test and games. The
  flank pair is the prior favourite; this is a hypothesis, not a verdict.

## 7. DECISIONS AND OPEN DISCUSSIONS (as of 2026-09-27, context handoff)
**Decided with the owner:**
- **Order:**
  1. KS attack/defence balance, "the highest ROI, it is what lets KS be computed";
  2. then the OvD design;
  3. then the other detectors (shelter/storm, KingProtector, pawnless flank);
  4. then a joint KS tune of everything together;
  5. then games.
- **Design rule:** unique where we have a better idea, universal ideas kept, single-lineage ideas NOT dropped (added
  at weight 0; the fit decides). Memory `unique-where-better-never-self-nerf`.
- **KS must fire only when danger is real.** Never loosen thresholds for fire rate. Judge on near-equal discrimination
  plus quiet-king fires by phase.
- **Ownership:** the king's SQUARE belongs to the PST (average placement) and DANGER belongs to KS (situational).
  They are separate but fitted jointly. Shelter/storm is a separate direct term; a coupling into KS danger is tested
  later as one scalar.
- **Adjacency:** keep the distinct-squares channel (breadth) AND add ring attack instances (convergence); the fit
  weights them.
- **KingProtector:** minors only (every reference that has it); rooks/queens defend from range, so their defence is
  counted in the defender channels.
- **The RFP / pruning-margin re-sweep waits until the eval block ends.**

**Open (need the owner):**
- **OvD design:** what the concept is, how it becomes ADDITIVE rather than duplicating KS / shelter / threats, and
  where it lives. My earlier "OvD owns the uncommitted king" was RETRACTED as overstepping. The recall study says OvD
  is not a KS-recall fix (uncommitted kings ×0.36 among misses), so its case rests on long-term pressure before
  attackers arrive.
- **Pawn storm:** a later discussion. It could be ours, or separate from shelter.
- Whether the committed/uncommitted split for shelter survives the OvD design.

**Built (all at 0, byte-identical, `0518604`):** `ks_channels` shared by the scorer and the probe; knobs:
- modes `KS_V2_ATT_XRAY`, `KS_V2_PIN_DEF`, `KS_V2_GATE`, `KS_V2_DEFAWARE`;
- weights `KS_V2_ADJ_INST`, `KS_V2_UNSAFE`, `KS_V2_BLOCKERS`, `KS_V2_FLANK_ATT`, `KS_V2_FLANK_ATT2`,
  `KS_V2_FLANK_DEF`, `KS_V2_KNIGHT_DEF`, `KS_V2_CONTEST_EXCESS`, `KS_V2_CONTEST_SQ`, `KS_V2_CONTEST_SQ_Q`.

**Channel screen DONE (§6b):** the per-channel game-outcome screen cannot rank the channels; the flank pair stands
out. All channels go to the joint fit.

**NEXT concrete step:** item 2 of the order, the **OvD design discussion WITH the owner**. Then build the
shelter/storm, KingProtector and pawnless-flank detectors at 0 (move test first). Then the joint KS fit: KS
reproduced exactly in Python from the channels, onset/curve refit jointly, phase over-fire of weak/adj/contest_excess
controlled, fed by `fitC_std_d6` plus the variant data.

## 8. OvD REDESIGN — long-term pressure as TRANSFORMATION POTENTIAL (agreed with the owner 2026-09-27 evening)
**The owner's concept (their words, condensed).** v1's OvD was never aimed at the king per se: the offence/defence
accumulators summed positional pressure as a whole (central control, attacking presence), and the heat map only
weighted the king zone higher; attacks INTO an area (a long diagonal at the king) were read as pressure that could
eventually cause collapse. As KS matured, that became a cruder KS. The idea worth keeping is LONG-TERM PRESSURE AND
CHANGE: which side has the better chances of a favourable positional TRANSFORMATION — e.g. a break that leaves the
opponent two isolanis and us a majority on that side, central tension resolving our way, a majority cementing — and
in the endgame this becomes winnability. "A way to get long-term data in without search." Not only passers, not only
pawns ("if we can look beyond pawns, that might help too"). Delicate: a sensible concept can be implemented harmfully
or duplicatively, so it is built at 0 and priced by the fit. ★ It may be RENAMED (it no longer sums offence and
defence scores); it carries the legacy of the long-term-pressure term.

**Decisions:**
- **Its own subsystem, not a KS feeder.** Two legs through the (mg,eg) pair plumbing, both ADDITIVE (not multipliers):
  - mg leg = transformation potential, SIGNED (can favour either side), saturating. Novel (no reference engine scores
    a flank majority, a break's outcome or central tension as a forward-looking asset) ⇒ added at 0, fit decides.
  - eg leg = winnability, universal shape (SF `initiative`/`winnable`, Ethereal `evaluateComplexity`): ☠️ it may
    grow or shrink an advantage but NEVER flips the sign. Inputs: both flanks, pawn count, outflanking, pawn ending.
- **KING-FREE by construction.** Storm-against-a-king stays with shelter/storm (KS-B), where every reference with a
  storm puts it (SF11 `pawns.cpp:186-215` + castling MAX `:233-237`; SF15.1; Ethereal `evaluateKingsPawns`; Weiss
  has none). OvD reads no king square ⇒ predicate-disjoint from KS and KS-B (Ethereal's complexity is king-free too).
  The committed/uncommitted split therefore stays INSIDE KS-B (castling squares).
- **Tempo returns only as TIMING.** The flat side-to-move bonus stays closed on MECHANISM, not on a null: a constant
  shared by every sibling cannot reorder moves, and its node effect flipped sign when the margins moved
  (`EVAL-V2-CURRENT-CONFIG.md:101`) ⇒ threshold coupling. Inside OvD, tempo = how many moves each side needs to
  execute its transformation, with the side to move breaking the race. Position-dependent ⇒ it CAN reorder siblings.
- **Score the OPTION, not the execution** (from the screen below: premature transformations are ~3× commoner than
  critical ones). OvD prices a transformation each side can FORCE; once executed, the resulting structure is priced by
  the structure terms that already exist. ⚠️ Risk the fit must watch: if the held option is priced above its realised
  structure, the engine never cashes it ("tension forever").

**First-cut features (all at 0; pawn-only ones live in the pawn cache ⇒ ~free per node):**
1. LEVER OUTCOME — for each lever a side can play (pawn×pawn now, or a push into contact), resolve the pawn exchange
   and score the resulting structure with v2's own pawn scorer (isolated / doubled / backward / majority / passer);
   feature = best Δ per side, discounted by moves needed (the timing tempo). The owner's "two isolanis + a majority".
2. MOBILE MAJORITY — flank majority that can still advance (not rammed), per flank.
3. TENSION — unresolved central contact, signed by who benefits from its resolution (reuses feature 1 on d/e).
4. INDUCE (beyond pawns) — a piece capture on a pawn-guarded square that forces a structure-changing recapture
   (Bxc6 bxc6). Largest class in the screen below; needs piece info ⇒ per-node, keep it small.
5. eg: winnability inputs above.
- Overlaps to control: candidate passers are already scored (`PASSER_V2_CAND_PCT=50`) — fit jointly, never count a
  majority's candidate twice · `space_mp` (built, 0, inert) · KS flank channels · Texel "already winning" confound
  (a majority travels with a material edge) ⇒ near-equal positions only. v1 winnability (`ENABLE_WINNABILITY`) and
  v1 majority (`PAWN_MAJORITY_*`) are both UNREADABLE parks (STS-only / contamination-era), not refutations.

### 8a. Move-class screen (2026-09-27, `_position_class.py MOVECLASS=1`, no engine, `game_regret_set.csv` 15k, SF18 d14 multi-PV)
Registered predictions scored **0 of 4 outright** (P3's direction and P1's advance share were right): P1 transform-best ~25% → **10.3%** ❌ (advance ~12% →
11.1% ✅) · P2 critical ~4% → **1.6%** ❌ · P3 traps ≈1.5× critical → **3.3×** (direction ✅) · P4 levers > pawn captures
in the critical set → 50 vs 56, and INDUCE dominates ❌.

| | GAP 5pp | GAP 2pp |
|---|---|---|
| SF18 best = pawn_capture / lever / induce / advance / piece | 3.4 / 3.7 / 3.2 / 11.1 / 78.7 % | same |
| transform_critical (best is a transformation, best non-transform ≥ GAP worse, near-equal, not a recapture) | **239 (1.6%)** | **382 (2.5%)** |
| — by kind: pawn_capture / lever / induce | 56 / 50 / 133 | 96 / 111 / 175 |
| transform_trap (best is NOT a transformation, a listed one is ≥ GAP worse, near-equal) | **792 (5.3%)** | **1,146 (7.6%)** |

- ☠️ The printed "only-moves 96% / 94%" is **definitional, not evidence of tactics**: the critical filter already requires
  every non-transform move to trail by ≥ GAP, so `n_good=1` is nearly automatic. Do NOT read it as "search's job".
- ⇒ A ONE-MOVE gap is the wrong shape for a long-term concept (long-term value shows as many small preferences and in
  outcomes, not as a single sharp choice). The OUTCOME fit (Texel on near-equal positions) is OvD's primary
  instrument; the move test stays as a HARM check (does the term create premature transformations on
  `transform_trap`?) and a headroom read on `transform_critical`.
- **Next (needs engine slots — after `fitC_std_d6` finishes):** our shipped engine at d8 and d14 on
  `classes_move_g2/transform_critical.csv` (382) and `transform_trap.csv`; count failures that PERSIST at d14 (and
  quote nodes). Then build features 1-5 at 0, the oracle + differential fire check + mirror gate, and add to
  `v2_features` for the joint fit.

### 8b. OWNER CALL (2026-09-27): OvD is NEW, the KS items are KNOWN — fit them NESTED, never only jointly
The KS items (channels, shelter/storm, KingProtector, pawnless flank) are universal and known to work; OvD is new. A
failed joint fit must not take the proven block down with it. ⇒ From ONE feature pass, run two fits:
- **Fit K** = KS block + PST (the proven block) — gated vs shipped; this is what ships first.
- **Fit K+O** = Fit K + OvD features — gated vs **Fit K**, not vs shipped, so OvD must pay INCREMENTALLY (held-out,
  then its own SPRT + SF18 read).
- Collinearity check: if adding OvD shifts the KS weights materially, OvD is taking value from KS rather than adding
  its own — counts against OvD even when the joint number looks fine.
If K+O fails, Fit K ships unchanged and OvD returns to design with the result in hand; no redo is needed.

## 9. BUILD LOG
**C3-a shelter + storm — BUILT 2026-09-27 night, at 0 (compile-only; the fitC run held the cores).**
- Engine: `ksb_cells` / `ksb_side` in `eval_v2.cpp`; knobs `KSB_V2` (1 = on with `KSB_V2_FILE`, same line format as
  `C1_V2_FILE`, k = 106..161) and `KSB_V2_CASTLE` (SF castling max, by blended value; the extractor flags it as
  unmodelled). Published as `ev_breakdown["v2_shelter"]` (new v2-only bit `EB_V2_SHELTER`, masked out of `EB_ALL`).
- Features: `v2_features` widened 106 → **162 per side**; cells 106-129 shelter (F × 6), 130-161 storm (F × 8), counted
  at the actual king square UNCONDITIONALLY. `ChessAI.pyx` buffers resized; `_texel_feature_pass.py` has a
  `v2_shelter` block; `_texel_c1_fit.py` no longer assumes 106 columns; the C1 loader now REJECTS k ≥ 106.
- ✅ **Oracle PASS** (`diagnostics/_ksb_oracle.py`, no engine instance): 3/3 hand rows · **0 mismatches** vs an
  independent per-square python-chess implementation on 17,856 positions (game_regret_set + variant + odds) ·
  colour mirror 0 · file mirror 0 · all 56 cells fire (thinnest: b/g blocked ≤2, 193 positions).
- ⏳ **Pending (needs an engine instance, after fitC finishes):** (1) WAC fingerprints byte-identical at defaults
  (v1 `250 / 35,310,778 / 3.784`, v2 shipped `249 / 53,405,821 / 3.973`); (2) DIFFERENTIAL fire check + feature-pass
  CLOSURE with the synthetic non-zero table `E:/chess_data/texel/ksb_test_table.txt` (`KSB_V2=1 KSB_V2_FILE=…`) — the
  default-off closure is VACUOUS; (3) `_eval_symmetry.py` colour gate with the table on; (4) pair-mode bound.

**C3-b pawnless flank + king-pawn distance and C3-c KingProtector — BUILT 2026-09-27 night, at 0.**
- Knobs `KFL_V2` / `KPROT_V2` with `KFL_V2_FILE` / `KPROT_V2_FILE`; one shared loader `c3_load_table` and init
  `v2_c3_init` (replaces `v2_ksb_init`); each block has its own knob, table and `ev_breakdown` key (`v2_kflank`,
  `v2_kprot`) so the nested fits can switch them separately and closure is checked per block.
- C3-b cells 162-171: nearest own pawn d 2/3/4/5+ · nearest enemy pawn d 2/3/4/5+ (d 1 = reference) · flank empty (SF
  KingFlank, both colours) · flank only enemy. C3-c cells 172-183: knight d 1..6+ · bishop d 1..6+ to OUR king
  (uncapped Chebyshev; `ps_kdist` caps at 5).
- `v2_features` now **184 per side**. The oracle is renamed **`diagnostics/_c3_oracle.py`** and covers all three blocks:
  ✅ 8/8 hand rows · 0 mismatches · colour 0 · file 0 on 17,856 positions · every cell fires.
- Test tables for the pending engine checks: `E:/chess_data/texel/{ksb,kfl,kprot}_test_table.txt`.

**Engine checks — ALL PASS (2026-09-28 morning, after fitC_std_d6 finished 30,000/30,000):**
1. WAC d10 fingerprints byte-identical at defaults: v1 **250 / 35,310,778 / 3.784**, v2 shipped **249 / 53,405,821 / 3.973**.
2. Fire + closure with the synthetic tables (`_texel_feature_pass.py LIMIT=20000`, all three on): live rows shelter
   17,769 · kflank 15,040 · kprot 16,069 of 19,983; |engine − Σ count×θ| median ≤ 0.05, **max 1.0 mp** (the per-side
   `>> 8` truncation) for every block ⇒ NOT vacuous.
3. `_eval_symmetry.py N=4000 TERMS=1` with all three on: colour swap **0 / 4000**, file mirror **0 / 3170**.
   (Fixed in passing: `v2_phase256` was not in the gate's SKIP_TERMS and would have read as a signed violation.)

**OWNER CALL (2026-09-28): no move test for the built KS detectors — straight to the fit and games.** Rationale: the move
test's job was to stop C++ being written for concepts with no headroom; here the C++ is built, cheap and verified, and
the fit + games decide. The move test stays in the plan for OvD (no C++ yet).

**Fit K pipeline (2026-09-28):** `_texel_extract` (fitC → 1,844,613 quiet rows; W 40.6 / D 21.6 / L 37.8) ·
`_texel_feature_pass` (184/side; C1 blocks close ≤ 4.2 mp) · `_texel_ks_pass.py` (NEW; KS reproduced in Python
**EXACTLY** on all rows, 275,660 live) · `_texel_engine_pass MODE=zero` · `_texel_k_fit.py` (NEW; nested arms P / PK /
PKC, KS non-linear with analytic gradients, attacker scale pinned — no engine knob). Smoke (200k rows, feasibility
only): P −0.005% · PK −0.32% / −0.22% · PKC −0.35% / −0.24% (val_hash / val_block). ⚠️ Smaller than C1's −0.74%, which
was Elo-null ⇒ games decide, not the held-out number.

### 9a. FIT K RESULTS (2026-09-28, all 1,844,465 rows; `E:/chess_data/texel/fitK1*`)
| arm | val_hash | val_block |
|---|---|---|
| P (PST re-fit) | −0.04% | −0.02% |
| PK over P (KS re-priced) | −0.76% | −0.68% |
| PKC over PK (+ C3 detectors) | −0.22% | −0.11% |
| **Fit K1** final (onset free) | **−0.99%** | **−0.82%** |
| **Fit K1p** final (`PIN=ONSET`, 450) | **−0.95%** | **−0.72%** |
Both nested increments pay on both holdouts. Fit K1 moved the onset 450→242 (sd 38); K1p shows that buys only
0.04-0.10 pp — the gain is the re-priced checks (Q ≈ 250-260, R ≈ 190, B ≈ 145, N ≈ 190 vs 126/122/80/152), ADJ
61→42-50, NO_QUEEN ≈ 390-400, and the new channels (CONTEST_SQ ≈ 31, CONTEST_SQ_Q ≈ 17-22, CONTEST_EXCESS ≈ 11-15,
UNSAFE ≈ 19, ADJ_INST ≈ −12..−15, KNIGHT_DEF 15-25 subtractive, FLANK_ATT 3-11); BLOCKERS and FLANK_DEF → 0.
**Firing (Fit C data, per king):** shipped 4.5-11.5% · K1 18.6-26.3% · K1p 11.0-19.4%. Near-equal score firing vs quiet:
all three ≈ 0.43-0.48 vs ≈ 0.50. ★ Danger MAGNITUDE vs own score on near-equal firing kings: shipped r ≈ 0 (−0.011 /
+0.002 — confirms the recall study), K1 −0.05, K1p −0.055 / −0.048 ⇒ the fit made KS magnitude informative.
Closure: engine with K1 loaded == Python model (KS EXACT incl. balance channels; C3 blocks ≤ 1 mp).
Owner call: the "never loosen" rule binds hand-tuning; a fit-moved threshold is flagged, a pinned control is kept, and
GAMES decide. `sprt_fitK1` (K1 vs shipped, NODE_LIMIT=50000, seed 40, 0/+10) launched 2026-09-28.
**SPRT `sprt_fitK1` — H1 ACCEPTED (2026-09-28):** Fit K1 vs shipped, NODE_LIMIT=50000, UHO, seed 40, 0/+10:
**+420 −323 =169 of 912 (55.3%), elo ≈ +37.1 ± 26.5, LLR +3.004.** ⚠️ SPRT point estimate — magnitude comes from the
replication + SF18 gauntlet. `sprt_fitK1p` (onset pinned, seed 41, same terms) launched next; K1p closure EXACT.
**SPRT `sprt_fitK1p` — H1 ACCEPTED (2026-09-28):** onset pinned at 450, seed 41, same terms: **+410 −315 =190 of 915
(55.2%), elo ≈ +36.2 ± 26.4, LLR +3.013** — statistically identical to K1 (+37.1). ⇒ lowering the onset bought nothing
in games either; **K1p is the candidate** (keeps the owner's threshold rule at equal strength). Next: `fitK1p_rep`,
fixed 2,000 games, seed 42 (Fit A's replication recipe), then the SF18 gauntlet and variant/odds starts.

## 10. FIT K2 — KS STRUCTURE, gated vs K1p (owner 2026-09-28: "maximise efficiency AND coverage; give everything a fair chance")
K1p re-priced every CONTINUOUS KS number (unit weights, checks, onset-pinned curve, the ten balance channels). Still
UNPRICED — each gets its own arm, read against K1p on held-out, then games:
1. **Attacker weight per type** (N 31 · B 31 · R 47 · Q 78, compile-time until now) → knobs `KS_V2_W_N/B/R/Q` (defaults =
   byte-identical); the probe now exports zone attackers by type so `w_att = Σ W_t · att_t` is exact.
2. **Coordination** `KS_V2_COORD` (fixed 256 = product form) → free, jointly with 1.
3. **Mode arms** (a fit cannot choose a mode; each is its own arm): x-ray attackers (`KS_V2_ATT_XRAY`; exports
   `att_x_*`, `w_att_x`) · defence-aware attacker weight (`KS_V2_DEFAWARE`; exports `share_*`) · the two-attacker gate
   (`KS_V2_GATE`) · pin-aware defence (`KS_V2_PIN_DEF` — changes weak/safe sets, needs its own channel pass).
4. **Fuller defender design** (C2-d: zone squares defended / doubly defended as SUBTRACTIVE channels) — not built.
5. **Shelter→danger coupling** as ONE scalar `u −= k · shelter` — not built; testable as a fit arm from the C3 cells.
6. **Definitions** (zone, weak, safe check) — structural; only if 1-5 leave a gap.
Order: 1+2 (one fit) → 3 (x-ray, defaware from counts; gate cheap) → 5 → 4. Build after `fitK1p_rep` frees the engine.

### 8c. OvD lever-outcome OUTCOME PILOT (2026-09-28, `_ovd_lever_proto.py N=300000`, pure Python, feasibility only)
Feature = best pawn-for-pawn structural Δ a side can FORCE (v2's own pawn-structure values), White − Black; tested as
the correlation with the result residual BEYOND the shipped eval on near-equal rows (79,302).
| feature | fires (near-eq) | r(feature, residual) |
|---|---|---|
| lever_now (immediate exchange) | 3.5% | **+0.043 ± 0.019 (2.3σ)** — signal sits in the extreme bin (≈ +2 pawns of structure ⇒ residual +0.09); middle bins not monotonic |
| lever_push (push into contact) | 6.9% | +0.012 ± 0.014 (0.9σ) — NULL |
Predictions (registered): now +0.02 (dir ✓, size larger), push +0.03 (✗, null). ⇒ Lever outcome ALONE is thin:
suggestive for large forced exchanges, null for push levers, and its 3-7% coverage caps what it can carry. OvD's case
rests on the broader block (mobile majority, tension resolution, eg winnability) fitted jointly, lever_now as one input.
**REPLICATION `fitK1p_rep` — CONFIRMED (2026-09-28):** fixed 2,000 games, seed 42: **+895 −709 =396 (54.6%), Elo +32.4 ±
17.9**; White +457 −347 =196, Black +438 −362 =200. Registered +20..+35 ✓. **Pooled with the SPRT (2,915 games) ≈ +34.**
K2 build (attacker-weight knobs + per-type exports) fingerprints byte-identical (v1 250 / 35,310,778 / 3.784; v2 249 /
53,405,821 / 3.973). SF18 gauntlet (ours NODE_LIMIT=250000 vs SF18 @400 nodes, 500 per seed, seeds 36 + 37, paired
against the saved `gauntlet_fitA[_s37]` = today's shipped config) running; knobs verified live in the per-game stderr
and by divergence from the baseline games.
☠️ **EXTERNAL CHECK — K1p does NOT transfer (2026-09-28).** SF18 gauntlet, ours NODE_LIMIT=250000 vs SF18 @400 nodes,
paired by opening + colour (`_gauntlet_pair` in the session scratchpad; results.csv by game index):
| seed | shipped baseline | K1p | paired diff |
|---|---|---|---|
| 36 | 71.50% | 69.20% | −2.30pp ± 4.56 |
| 37 | 70.10% | 69.80% | −0.30pp ± 4.68 |
| **pooled 1,000** | 70.80% (+154) | 69.50% (+143) | **−1.30pp ± 3.27 ⇒ −10.8 Elo [−36.8, +16.9]** |
Registered +8..+20 ✗. Harness null MEASURED: a FRESH shipped baseline (`gauntlet_shipped_0928`, seed 36) reproduces the
saved `gauntlet_fitA` to 2 of 500 changed results (71.3% vs 71.5%) ⇒ reusing saved baselines is sound here, and K1p vs
the fresh baseline reads the same (−17.5 Elo, seed 36). ⇒ **Self-play +34 (2,915 g) vs external ≈ −11**: unlike Fit A
(+111 → +38), K1p's gain is not established outside self-play. NOT SHIPPED.
Two hypotheses, discriminated next: (1) SELF-PLAY EXPLOITATION (fit on v2-vs-v2 games; the checks re-price, Q 126→260,
is the prime suspect); (2) DEPTH (SPRTs at 50k nodes, gauntlet at 250k). Test: the same gauntlet with OUR engine at 50k
nodes, both arms, seed 36 (`gauntlet50k_shipped` / `gauntlet50k_fitK1p`): K1p ahead at 50k ⇒ depth; level ⇒ exploitation.
**DISCRIMINATION (2026-09-28): it is DEPTH, not exploitation.** Same SF18@400n gauntlet, ours at **50k** nodes, seed 36,
500 paired: shipped 47.5% → K1p 53.4%, **+5.90pp ± 4.75, z 2.43 ⇒ +41 Elo [+8, +75]** — the gain TRANSFERS at the SPRT
budget and vanishes at 250k. Cause (best explanation): Fit C's labels are d6 games, where kings get walked into attacks,
so the fit priced king danger for SHALLOW play (checks ~2×); deeper search finds the defences. Fit A (positional PSTs)
was not exposed. Memory `fit-data-depth-must-match-play-depth`. Options: (a) play-depth training games for the KS block;
(b) shrink K1p's KS change toward shipped and re-test at 250k; (c) keep the depth-robust parts (C3 detectors, PST) and
re-test with shipped KS.

### 9b. OVERNIGHT QUEUE 2026-09-28 → 29 (which parts of K1p survive depth; does it upgrade CRITICAL positions?)
Owner's question: K1p fires rarely but should win the critical moments — is the flat 250k result "critical gains,
other positions not helped" or "critical gains offset by WORSE play elsewhere"? Games alone cannot say (any eval change
diverges ~56% of games; a fresh shipped rerun changes only 2/500 ⇒ the divergence is caused by K1p, not its quality).
**Exploratory game split** (baseline-game sharpness; v1 definition ≥300 cp swing was DEGENERATE, 995/1000; v2 = a ≥150 cp
swing from a still-balanced |eval| ≤ 300 position, chosen before reading its numbers): sharp 911 games **+0.1pp ± 3.5**
(259 better / 263 worse) · quiet 89 games **−14.6pp ± 8.9** (9 / 27) ⇒ hypothesis: not a sharp-game gain at depth, and
worse in QUIET games (over-reaction without real danger) — to be tested at position level by (3).
Queue (each launched when the previous finishes):
1. `gauntlet_K1p_pos` (PST + C3 detectors, SHIPPED KS) and `gauntlet_K1p_half` (K1p with the KS change halved toward
   shipped, onset 450), 250k vs SF18@400n, seed 36, conc 2 each.
2. Same two arms, seed 37 ⇒ 1,000 paired each (pair vs `gauntlet_shipped_0928` / `gauntlet_fitA_s37`).
3. Move regret at play depth: `pyrun diagnostics/_ks_footprint_regret.py SET=ks_sets/game_regret_set.csv DEPTH=11
   JOBS=4 MAXN=4000 BASE_KNOBS='V2_PRESET=shipped' CAND_KNOBS='<K1p>;<K1p_half>;<K1p_pos>;ASPIRATION_DELTA=300'
   CAND_NAME='K1p;K1p_half;K1p_pos;null_asp300'` — read win% of CHANGED moves vs the neutral arm, split BY_CRIT / BY_PHASE.
Knob strings: K1p = §9a `ks_fitK1p.txt` + `PST_V2_FILE`/`KSB_V2`/`KFL_V2`/`KPROT_V2` files; K1p_half KS = WEAK 61 ADJ 56
CHK_R 158 CHK_Q 193 CHK_B 111 CHK_N 171 NO_QUEEN 362 ONSET 450 ADJ_INST −6 UNSAFE 10 FLANK_ATT 6 KNIGHT_DEF 8
CONTEST_EXCESS 7 CONTEST_SQ 15 CONTEST_SQ_Q 10 HALF 623 (rest as K1p).
**Overnight results (2026-09-29):** `K1p_pos` (PST + C3 detectors, SHIPPED KS), 250k vs SF18@400n, 1,000 paired:
seed 36 −0.20pp ± 4.52 · seed 37 −2.50pp ± 4.58 · **pooled −1.35pp ± 3.22 ⇒ −11.2 Elo [−36.8, +16.0]** — identical to full
K1p ⇒ the structural detectors + re-fitted PST do NOT transfer at depth either (the seed-36 "only KS hurts" hint did not
replicate). `K1p_half` seed 36: −2.10pp ± 4.72 (seed 37 running). Move-regret breakdown (§9b item 3) launched with JOBS=2.
`K1p_half` pooled 1,000: seed 36 −2.10pp · seed 37 −0.90pp · **−1.50pp ± 3.33 ⇒ −12.4 Elo [−38.8, +15.6]**. ⇒ All three
arms (full / half-KS / positional-only) ≈ −11..−12 at 250k, none significant. Shared element = K1p's re-fitted PST + C3
detectors ⇒ `gauntlet_K1p_ks` (K1p KS ONLY: shipped PST, detectors off), seed 36, launched to separate the parts.
⚠️ **PROVISIONAL (1 seed): `K1p_ks` (K1p KS knobs ONLY — shipped PST, C3 off), 250k vs SF18, seed 36: 75.9% vs 71.3%,
+4.60pp ± 4.35, z 2.07 ⇒ +41.2 Elo [+2.1, +85.5].** On the same seed every arm WITH the re-fitted PST + C3 read −2.1 /
−2.1 / −0.2. If seed 37 replicates, the KS re-pricing DOES transfer at depth and the positional parts (PST + C3, fitted
jointly on d6 data) are what cancel it — which would REVISE `fit-data-depth-must-match-play-depth` (the depth effect
would sit in the positional block, not KS). Seed 37 (`gauntlet_K1p_ks_s37`) running; do not act before it.
**Move regret @ d11 (`_ks_footprint_regret.py`, 4,000 positions, base shipped) — within noise.** win% of CHANGED moves:
null_asp300 **52.0%** (1,190 changed) · K1p 49.1% (1,266; Δ +0.079) · K1p_half 51.5% (Δ −0.157) · K1p_pos 50.9% (Δ −0.102).
All |Δ| < 0.2 = unresolvable by this tool; K1p mildly below the neutral arm (weakly consistent with the gauntlet). The
CRITICAL split is UNREADABLE: changed moves are ~1,100 benign vs cr3 22-27 and cr4 2-3 per arm (the known `n_crit` limit)
⇒ the owner's "does it upgrade critical positions?" needs a CRITICAL-position corpus, not this set.
★★★ **`K1p_ks` REPLICATES (2026-09-29):** seed 37 +2.30pp ± 4.40; **pooled 1,000: 70.70% → 74.15%, +3.45pp ± 3.09, z 2.18
⇒ +30.0 Elo [+3.0, +59.3] vs SF18 at 250k.** ⇒ The KS re-pricing TRANSFERS at play depth; the PST refit + C3 detectors,
fitted jointly with it on d6 data, are what cancel it. My 09-28 "KS is depth-fragile" was WRONG (memory
`fit-data-depth-must-match-play-depth` corrected). Ship path launched: `sprt_K1p_ks` (KS knobs only vs shipped,
NODE_LIMIT=50000, seed 43, 0/+10). `K1p_c3` (detectors only) seed 36 running to split C3 vs the PST refit.
`K1p_c3` (C3 detectors ONLY: shipped PST + shipped KS), 250k vs SF18, pooled 1,000: seed 36 −1.50pp · seed 37 −0.50pp ·
**−1.00pp ± 3.18 ⇒ −8.3 Elo [−33.8, +18.7]** — flat. ⇒ The d6-fitted C3 cell values do not pay at depth (not shown
harmful); they stay at weight 0 in the engine (built, verified) until a play-depth fit or an OvD/K2 round prices them.
**SPRT `sprt_K1p_ks` — H1 ACCEPTED (2026-09-29):** KS knobs only vs shipped, NODE_LIMIT=50000, seed 43, 0/+10:
**+737 −633 =398 of 1,768 (52.9%), elo ≈ +20.5 ± 19.0** (LLR crossed the bound; in-flight drain left it at +2.930).
⇒ KS-only: **+20 self-play @50k, +30 [+3, +59] vs SF18 @250k** — holds at depth, unlike the bundle. Replication
`K1p_ks_rep` (2,000 fixed, seed 44) launched; then variants/odds; ship = owner.
**Game-type split of K1p_ks vs SF18 (250k, 1,000 paired), classes from the BASELINE game.** ☠️ Classes tied to the
baseline's own outcome (length, sharp/quiet) carry REGRESSION TO THE MEAN — any perturbation lifts the baseline's bad games
and lowers its good ones — so each class is read against a CONTROL arm with ~zero net effect (`K1p_c3`, same games):
net (K1p_ks − control): sharp (911) **+4.8pp** · quiet (89) +1.1 · same-side castling (528) +2.6 · king uncastled at ply 30
(429) +6.4 · opposite-side (43) +8 (unreadable n) · long/medium/short +2.4/+5.5/+4.7. ⇒ Broad gain, carried by sharp
games, largest where king danger arises; no class where KS hurts relative to the control (per-class ±~5pp).
☠️ CORRECTION: the 09-28 "K1p loses QUIET games −14.6pp" was this artefact — the null-like control shows −10.7pp there.
