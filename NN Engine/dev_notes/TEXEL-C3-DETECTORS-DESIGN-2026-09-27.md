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
