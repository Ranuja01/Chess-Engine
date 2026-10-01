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
**REPLICATION `K1p_ks_rep` — CONFIRMED (2026-09-29):** 2,000 fixed games, seed 44: **+848 −711 =441 (53.4%), Elo +23.8 ±
17.9**; White +433 −353 =214, Black +415 −358 =227. **Pooled with the SPRT (3,768 games) ≈ +22 self-play; external +30
[+3, +59] vs SF18 @250k.** ⇒ KS-only is the ship candidate (owner's call; variants/odds pending).
Collapse revisit launched: v1's 276 classified collapse positions (`ks_sets/collapse_dataset_classified.csv`, labelled
SF18 d14 multi-PV 16 → `collapse_regret_set.csv`, class in `phase_bucket`), base v1 vs v2 shipped / v2+K1p_ks / null, d11.
**COLLAPSE REVISIT (2026-09-29):** v1's 276 classified collapse positions, base = v1, d11, changed-move win% (SF18 d14
multi-PV 16): null (v1 ASPIRATION_DELTA=300) **43.5%** (+1.87) · v2 shipped **62.4%** (−1.61) · v2+K1p_ks **57.0%**
(−1.69). ks_attack: null 40% (+2.70) · v2 63% (−2.57) · v2+KS 58% (**−3.29**); positional: null 45% (+0.94) · v2 62%
(−0.90) · v2+KS 56% (−0.56). ⇒ The collapses are REAL v1 defects (a perturbation of v1 gets worse there, not better);
v2 fixes a large share of both classes; the re-priced KS gives the largest mean regret cut on the KS-attack collapses
(n ≈ 45 changed per class ⇒ KS-vs-shipped is a lean, not a verdict).
**ODDS vs SF18 (K1p_ks, 250k vs SF18@400n, `openings_odds.txt`, seed 42) — NEUTRAL (2026-09-29):** paired with
`oddsSF_fitA` (= shipped) on 475 games (WSL restarted at 474/480; results rebuilt from the log): **−0.95pp ± 3.47**;
defending −2.10pp ± 6.22, converting +0.21pp ± 3.07. ⇒ no harm; Fit A's odds gain was PST material handling, which KS
does not touch. ⚠️ WSL restarted mid-run (uptime 0 at 16:21) with 2+2 concurrency — cause unknown; variant gate resumed.
**VARIANT GATE (K1p_ks, 2,000 games, seed 40):** +833 −759 =408 (51.8%), **+12.9 [−0.7, +26.5]**; pairs [100 136 481
156 127], net +74 half-points. Queen families positive (NQ_only +77 [+3, +158], RRvQ +63, array_std +63,
QminorsvRRR +54, QvRB +49); queenless lean negative (noQ −52, KRNvRB −38, KNP −21, minors −17) ⇒ K2 target: KS in
queenless positions (gate / NO_QUEEN channel).
✅✅ **SHIPPED 2026-09-29 (owner sign-off): Fit K1p's KS knobs only** in `V2_PRESET=shipped` (search_engine.cpp preset,
runner `V2=`, EVAL-V2-CURRENT-CONFIG §1). Fingerprints: v1 unchanged 250 / 35,310,778 / 3.784; **v2 shipped NOW 252 /
49,094,807 / 4.012**; preset == explicit tested knobs (byte-identical WAC). C3 detectors stay built at weight 0.

### 10a. FIT K2 RESULT (2026-09-29) — structure arms NULL
Start = shipped KS (K1p knobs), PST = Fit A, C3 off; `fitC_ks2.npz` (per-type exports; `w_att == Σ W·att_t` checked 0
mismatches; KS pass EXACT under the new ship). Held-out vs start (val_hash / val_block):
base −0.152 / −0.092 (control; not 0 because the shipped KS was fitted jointly with K1p's PST+C3) · att −0.222 / −0.080
(W 31/31/42/54, COORD 188) · xray −0.220 / −0.060 · defaware −0.260 / **+0.010** · gate −0.155 / −0.072.
⇒ Against the base control no structure arm improves BOTH holdouts (each gains ~0.07-0.11 on val_hash and loses on
val_block). The fit consistently wants lighter R/Q attacker weights and COORD ≈ 185, but it does not generalise. The
gate does not address the variant gate's queenless lean. Predictions 1/5 (gate ≈ 0). ⇒ KS block DONE for now; base
re-fit (−0.1%) is below the C1 Elo-null level, not gamed. Next = OvD.

### 11. BASELINE SNAPSHOT before OvD (2026-09-29; re-run the same rows after any OvD ship)
Accuracy = win%-MSE of the static eval vs SF18 d14 search, 3,000 rows, val split (`_accuracy_arm_grid.py`; references from
the 09-26 `_reference_ceiling.py` table on the same rows). STS = sts300, d10 and at 249,014 nodes. WAC d10.
| evaluator | own-play acc | diverse acc | STS d10 | STS @249k n | WAC |
|---|---|---|---|---|---|
| SF18 static | 61.93 | 68.85 | | | |
| SF15.1 NNUE | 72.76 | 61.61 | | | |
| SF11 classical | 151.41 | 95.26 | | | |
| **v2 shipped now (Fit A + Fit K KS)** | **168.08** | **124.18** | **1891** | **1734** | **252** |
| v2 before the KS ship (Fit A) | 170.17 ✓ | 126.93 ✓ | 1888 | 1760 ✓ | 249 |
| SF15.1 classical | 192.35 | 139.20 | | | |
| v1 | 188.85 ✓ | 238.77 ✓ | 1796 ✓ | 1752 ✓ | 250 |
✓ = reproduces the stored value exactly (deterministic harness). The KS ship: accuracy −1.2% / −2.2%; STS +3 (d10) /
−26 (@249k), inside the ±150 STS floor; STS is known blind to KS.
**OvD pilot, full middlegame set** (300k rows, near-equal residual r): lever_now +0.043 (2.3σ, 3.5% fire) · tension_centre
+0.032 (1.5σ, 2.7%) · lever_push +0.012 (0.9σ) · **mobile_majority +0.005 (0.9σ) on 51%: NULL** (the passer/candidate terms
likely already own it). ⇒ the mg block carries little beyond the eval; next = the eg WINNABILITY pilot.
**OvD eg WINNABILITY pilot (2026-09-29, `_ovd_winnability_proto.py`): STRONG.** 358,727 rows, phase256 < 96, stronger side's
edge 1-4 pawns; r(input, stronger side's residual beyond the eval): SF11 complexity composite **+0.153** · pawns +0.153 ·
both flanks +0.152 · pawn ending +0.072 · outflanking −0.058 · passed −0.011. Complexity quintiles: lowest fifth scores
**72.6% (residual −7.9pp)**, the rest 80-82% (+1..+3pp). ⇒ v2 OVERRATES edges in low-complexity endgames (few pawns / one
flank) and slightly underrates the rest — exactly SF's winnability correction, which v2 lacks. ⚠️ rows are not independent
(30k games): the σ is inflated, but the effect is far above the mg features. ⇒ OvD's evidence sits in its eg leg.

### 12. OvD eg WINNABILITY — BUILT + FITTED (2026-09-29)
Built at 0 (`WIN_V2` + `WIN_V2_{PASSED,PAWNS,OUTFLANK,INFILT,FLANKS,PAWN_END,UNWIN,BASE}`; eval_v2.cpp `win_inputs` /
`win_adjust`, applied once to the finished total, sign-preserving, eg-weighted (256−phase)/256; `ev_breakdown["v2_winnab"]`;
probe `ChessAI.win_inputs`). Fingerprints byte-identical at 0. `_win_oracle.py` PASS (hand 2/2, 0 mismatches / 17,856,
colour mirror 0). `_texel_win_pass.py` closure EXACT with test weights (16,554 live / 20k) and with the fit (18,005).
`_eval_symmetry` with the term on: colour 0/4000, file 0/3170.
**Fit W** (`_texel_win_fit.py`, nested on the shipped eval, 1.84M rows): ☠️ first run's L2 (λ 1e-8 ×1e4) was ~10× the
effect and pinned the weights near 0 (the UNFITTED SF prior beat it) — fixed by a λ grid. λ 1e-11 ≈ 1e-13, both starts
converge to the same point: **val_hash −0.30% · val_block −0.32% · ENDGAME rows −1.27% / −1.38%** (SF prior unfitted:
−0.06 / −0.10 / −0.25 / −0.52). Weights (mp): PASSED 14 · PAWNS 27 · OUTFLANK 28 · INFILT −200 · FLANKS 795 · PAWN_END
1734 · UNWIN 573 · BASE −877 ⇒ one-flank / few-pawn endgame edges shrink up to ~0.9 pawn; pure pawn endings boosted.
⚠️ Large adjustments (up to ~1.7 pawns) ⇒ straight to the depth gate: SF18 gauntlet @250k (seeds 36 + 37), then SPRT.

### 13. NAME: **POT (Potential) — OvD reworked** (owner, 2026-09-29)
The owner's long-term-pressure term, OvD ("offensive vs defensive", v1 — the change the owner credits with taking the
engine from struggling vs 1600 bots to competing with 2000s on chess.com), is carried forward as **POT (Potential)**:
*transformation potential* in the middlegame (who can force a favourable structural change) and *winning potential* in
the endgame (winnability). The name no longer sums offence and defence scores, but the concept — long-term pressure and
change, "long-term data without search" — is OvD's, and every POT note should say so. Code: the winnability knobs stay
`WIN_V2_*` until the depth gate decides the ship; a shipped POT family moves to a `POT_V2_*` prefix.
**POT winnability depth gate (2026-09-29): Fit W does NOT transfer as fitted.** SF18 @250k, paired vs the shipped
config (`gauntlet_K1p_ks[_s37]`): seed 36 −2.40pp ± 4.36 · seed 37 −0.70pp ± 4.50 · **pooled −1.55pp ± 3.13 ⇒ −13.8 Elo
[−40.2, +14.6]**. Held-out endgame −1.4% did not carry to games. Next: half-strength arm (`gauntlet_fitW_half`, all
weights ×0.5) — magnitude (up to ~1.7-pawn swings) is the prime suspect; if also flat, POT winnability is parked
priced-not-shipped (like the C3 detectors).
`gauntlet_fitW_half` (all weights ×0.5), seed 36: −3.20pp ± 4.57 — no better than full strength ⇒ magnitude is not it.
☠️ **GATING BIAS FOUND (2026-09-29): don't pair new candidates against the SHIP-SELECTION seeds.** The shipped config
(K1p_ks) was chosen partly on its seed-36/37 gauntlet results (seed 36 75.9% vs its 74.15% mean). A candidate paired
against a baseline selected-high on those seeds regresses and reads NEGATIVE by construction — the sharp/quiet artefact
one level up. Every arm on seed 36 reads negative (fitW −2.4, half −3.2). ⇒ From now on each new gate uses FRESH seeds
(38, 39, …) with a fresh shipped baseline on the same seeds. POT winnability re-gated on seeds 38 + 39 (4×500 games).
☠️ **POT winnability Fit W — FRESH-SEED VERDICT (2026-09-30): NEGATIVE, PARKED.** SF18 @250k, paired vs a FRESH shipped
baseline on unused seeds: seed 38 −4.20pp ± 4.47 · seed 39 −1.50pp ± 4.60 · **pooled −2.85pp ± 3.20, z −1.74 ⇒ −24.9 Elo
[−51.2, +3.2]**. All five readings over four seeds negative. The misjudgement it targets is real (pilot §11: low-complexity
edges over-rated ~8pp) but THIS fit (d6 game-outcome labels, swings up to ~1.7 pawns) hurts at depth. Next attempt: fit to
SF18 search labels (depth-independent target) with the adjustment capped (~½ pawn), gated on fresh seeds (40, 41).

### 14. POT NEXT ROUND — depth-independent labels (owner, 2026-09-30 night)
Owner's framing: POT = the whole arc — opening/middlegame TRANSFORMATIONS that carry a position into a strong
middlegame and a winnable endgame, with winnability continuing the shift at the end; the variant corpora add unusual
structures / exposed kings, so piece and KING placement matter for how they affect TRANSFORMING the position (not KS).
Idea to test: **projected winnability** — score the expected convertibility of where today's structure is heading
(flanks, majorities, minor-vs-structure fit, king placement for the coming pawn battle), weighted by distance to the
endgame, so the side ahead steers to convertible structures and the side behind to holdable ones.
Why the last round failed and this one is different: the mg pilot and Fit W were fitted to d6 GAME OUTCOMES (priced for
shallow play). This round uses SF18 SEARCH labels (depth-independent) and capped magnitudes, gated on fresh seeds.
Samples (made 09-30): `ks_sets/fitC_eg_sample.csv` (20,000 endgame rows, phase < 96, `row` joins `fitC_win.npz`) and
`ks_sets/fitC_mg_sample.csv` (10,000 standard midgame rows 96-224 + 5,000 variant-game rows).
Overnight queue: C3 re-gate seeds 38 → 39 (`gauntlet_c3v2_s3x`, vs `gauntlet_ship2_s3x`); then label with
`_build_regret_set.py IN=ks_sets/fitC_eg_sample.csv K=4 SF_DEPTH=14 OUT=ks_sets/fitC_eg_sf18.csv`, then the mg sample
→ `fitC_mg_sf18.csv`.
☠️ **Owner condition (09-30): no forced overlap — "we may end up with a mess like v1 again".** Each POT feature: a stated
OWNER distinct from existing terms · a COLLINEARITY check vs the existing features on the labelled data BEFORE any C++ ·
INCREMENTAL value (held-out, then games) on top of the shipped eval AND the other POT features · few features, each its
own signal. Projected winnability overlaps winnability + pawn structure by construction ⇒ admitted only if it carries
"where the structure is heading", not "what it is now".
**C3 detectors — FRESH-SEED re-gate (2026-09-30):** K1p cell values on top of the shipped config, SF18 @250k vs a fresh
baseline: seed 38 −4.10pp ± 4.59 · seed 39 −1.00pp ± 4.66 · **pooled −2.55pp ± 3.27 ⇒ −22.4 Elo [−49.4, +6.6]**. Agrees
with the earlier −8 ⇒ with d6-outcome values the detectors do not pay at depth; they stay built at weight 0, to be
re-priced on SF18 labels with POT. (Note: winnability and detectors both read ≈ −4.1 on seed 38 — one seed, not a pattern.)
SF18 labelling of `fitC_eg_sample.csv` started (then the mg sample).
**POT winnability Fit W-SF (2026-09-30, `_texel_win_sf_fit.py`): fitted to SF18 d14 SEARCH labels** on the 20k endgame
sample (18,488 joined rows, val by game hash 15%), adjustment capped ±500 mp: MSE of win% vs SF18 search **train −10.0%,
val −8.4%**. Weights (mp): PASSED 1 · PAWNS 449 · OUTFLANK 69 · INFILT 136 · FLANKS 33 · PAWN_END 1346 · UNWIN 62 · BASE
−1155 ⇒ ≤2-pawn endgame edges shrink, ≥3-pawn edges grow (within ±½ pawn); pure pawn endings boosted. New knob
`WIN_V2_CAP` (0 = uncapped = byte-identical; shipped fingerprint unchanged 252 / 49,094,807 / 4.012); closure EXACT with
the fit + cap (18,117 live / 20k). ⚠️ Accuracy vs SF is not Elo — gate on FRESH seeds 40 + 41 with fresh baselines.
**POT middlegame screen vs SF18 SEARCH labels (2026-09-30, `_pot_mg_screen.py`, 9,829 standard mg rows):** residual =
win%(SF18) − win%(shipped). lever_now r −0.054 (−1.1σ, fires 4.5%) · tension_centre −0.055 (−1.0σ) · mobile_majority
+0.014 (+1.1σ, 61%) · projected winnability inputs (leader-relative): passed **−0.043 (−4.3σ)**, pawns +0.9σ, outflanking
+0.4σ, infiltration −1.3σ, both_flanks +0.2σ. Overlap with the 106 existing features ≤ 0.35 for all.
⇒ No transformation feature explains our mg disagreement with SF — POT's mg half does NOT pass the fair test with these
features; nothing built. The one signal (we over-rate the leader's mg edge when passers are on the board) is OWNED by the
passer terms (no-overlap rule) ⇒ logged as a passer-term lead, not a POT feature. Variant mg rows (4,944) labelled but not
yet engine-passed.
☠️ **Fit W-SF — FRESH-SEED VERDICT (2026-09-30 night): NEGATIVE.** vs fresh shipped baselines: seed 40 −2.90pp ± 4.13 ·
seed 41 −2.10pp ± 4.44 · **pooled −2.50pp ± 3.03, z −1.62 ⇒ −22.9 Elo [−48.9, +5.1]**. Two very different fits (d6
outcomes; SF18 labels + ½-pawn cap) both ≈ −23..−25 ⇒ suspect the FORM, not the fit.
★ **Diagnosed: the additive form is DISCONTINUOUS at a level score.** adj = sign(T)·C with C > 0 turns +1 mp into +500
and −1 into −500. On the Fit C endgame rows: Fit W-SF GROWS the edge on 84.7% and amplifies 98.0% of near-level rows
(|T| < 200 mp) beyond their own size (median +500 mp = the cap); Fit W 57.3% / 39.3% (median +131). Near-level rows are
only 2.5% of the (quiet, finished-game) training data, so the fit barely paid for it — but search lives in balanced lines.
⇒ Proposal (owner discussion): winnability as a multiplicative endgame SCALE FACTOR, E_eg = T·f(C), f ∈ ~[0.5, 1.2] —
continuous at 0, cannot manufacture an edge from noise, still shrinks unwinnable edges. It is the universal form (4/4
reference engines have an eg scale factor) and closes the recorded v2 gap ("the endgame can only say draw or full").

## 15. STATE + DECISIONS AT THE 2026-09-30 CONTEXT HANDOFF
**Shipped:** Fit K KS knobs only (09-29) — v2 fingerprint **252 / 49,094,807 / 4.012**; v1 unchanged 250 / 35,310,778 / 3.784.
**Built at weight 0 (verified; not shipped):** C3 detectors (`KSB_V2`, `KFL_V2`, `KPROT_V2`); K2 attacker-weight knobs
(`KS_V2_W_N/B/R/Q`); POT winnability (`WIN_V2_*`, `WIN_V2_CAP`).
**Owner decisions (09-30 morning):**
1. **POT mg half: the ALGORITHM, not the concept, is unproven** — "it's never been seen before, so consider what we have,
   what's good, what's not". Next: a design re-think WITH the owner (no forced overlap — §14 condition), not more
   variants of the tested features (lever / tension / majority / projected-winnability inputs all null vs SF18).
2. **POT eg winnability: rebuild the FORM from the references** — SF11 `initiative`, SF12+/15 `winnable` (+ its scale
   factor), Ethereal `evaluateComplexity`, Weiss `ScaleFactor`. The recorded failure is the additive discontinuity at 0
   (§12 end) ⇒ a continuous, reference-shaped form (likely a multiplicative eg scale factor), fitted on the SF18-labelled
   eg sample, gated on fresh seeds (next unused: 42+).
3. **Passer re-tune** (the mg lead: we over-rate the leader's edge when passers are on the board) — same Texel method as
   PST / KS — AFTER POT.
**Data on disk:** `E:/chess_data/texel/` fitC_stage1 (1.84M), fitC_features.npz (184/side), fitC_ks.npz (old KS) /
fitC_ks2.npz (per-type, new KS), fitC_pass/zero, fitC_win.npz, fit outputs (`*_fitK1p*`, `win_fitW*.txt`);
`diagnostics/ks_sets/fitC_eg_sf18.csv` (19,779 SF18-labelled eg) + `fitC_mg_sf18.csv` (14,842 mg incl. 4,944 variant;
variant rows lack an engine pass) + samples; `collapse_regret_set.csv`.
**Gate protocol now:** closure + symmetry + fingerprints → SF18 gauntlet @250k, 1,000 paired on FRESH seeds with a fresh
baseline (`_gauntlet_pair.py`) → per-part ablation if it's a bundle → SPRT @50k + 2,000-game replication → variants/odds.

## 16. POT WINNABILITY — REFERENCE FORM: ENDGAME SCALE FACTOR (2026-09-30)
**Reference extraction** (Opus engine-contrast agent; SF lines cited locally, Ethereal/Weiss fetched, line numbers approx.):
all FOUR references scale the endgame MULTIPLICATIVELY (SF11 `evaluate.cpp:743-760` + material.cpp pawnless factor;
SF15.1 `:908-947`; Ethereal `evaluateScaleFactor`; Weiss `ScaleFactor`) — continuous at eg = 0 because eg·sf → 0 from both
sides. ★ Surprise: SF11/SF15/Ethereal's ADDITIVE complexity leg has the SAME step (2C) whenever C > 0 — they survive it
because the base constant keeps C ≤ 0 except in pawn-rich / pure-pawn endings; our W-SF fit made C > 0 from ~3 pawns up.
Weiss has no complexity term at all. Universal inputs (4/4, but one SF lineage): strong-side pawn count · pawns on one
flank · opposite bishops; 3/4: pawn ending, bare-minor/pawnless edge; SF-only: passers, outflanking, infiltration.
**Form:** total' = total·(1 + eg·(f − 64)/64), eg = (256−phase)/256, f = clamp(64 + BASE + SP·strong pawns + ONEFLANK +
OCB (bishops only, opposite colours) + PASSED·strong passers, 0, 64); strong = sign(total). Non-pair approximation of
eg·f (exact on the eg share when mg = eg). |adjustment| ≤ |total| ⇒ it can never create or grow an edge.
**Fit** (`_texel_win_sf_fit.py MODE=scale`, SF18 d14 labels, 18,488 eg rows, val by game hash 15%), val MSE vs shipped:
SF15-shaped prior UNFITTED +5.65% (worse) · HI 64 all 9 features −10.18% · HI 72 −14.75% · HI 80 −15.88% · HI 64
4 features −10.18% · **HI 64, SP + OCB only −9.86% ⇒ SHIP ARM: BASE −37, SP 34, OCB −80** (f: 0 pawns 0.42, 1 pawn 0.95,
≥2 pawns 1.0; OCB-only 0 at ≤1 pawn → 1.0 at 4). ☠️ HI > 64 REJECTED: it grows 89% of endgame evals ×1.1-1.2 = matching
SF18's search-score SCALE (search magnitudes exceed static ones) — mean correction, not winnability.
Predictions registered before fitting: prior helps 1-3% (✗, it hurt) · fit −3..−6% (✗, −10) · HI>64 helps (✓ but
rejected) · SP > 0 (✓).
**Build** (`WSF_V2`, `WSF_V2_{BASE,SP,ONEFLANK,OCB,PASSED}`, `win_scale_adjust`, publishes into `v2_winnab`): fingerprints
byte-identical at 0 (252 / 49,094,807 / 4.012) · closure EXACT (`_texel_win_pass.py WSF_V2=1 …`, 1,919 live / 20k with
ONEFLANK exercised; PASSED also exact, 1,060 live) · symmetry on the eg sample: colour 0/4000, file 0/3916.
**Gate** (queued 2026-09-30, `selfplay/_queue_wsf_gate.sh`): SF18 @250k, 500 games per arm, FRESH seeds 42 + 43, fresh
shipped baselines `gauntlet_ship4_s4x` vs `gauntlet_wsf_s4x`. **Predictions:** pooled in [−10, +15], most likely ≈ +3
(the endgame has little eval-shaped headroom — inventory caveat); NOT clearly negative like W / W-SF (the discontinuity
is gone); seeds differ by < 3pp.

## 17. POT — THE OWNER'S DEFINITION (2026-09-30) ★ supersedes the §8 "pawn-transformation" reading
**Lineage:** v1 OvD (piece pressure on a heat map vs the defender's coverage) was retired because it was mostly KS; KS now
owns that. POT is the TRUE meaning: the POTENTIAL of a position to transform. (§8's lever/tension/majority features and
v1's heat-map pressure were both the wrong mechanism — the §14 screen nulls stand, the concept does not fall with them.)
**Owner's definition (paraphrase, key phrases verbatim):**
- Subsystems (mobility, pawns, KS, passers …) are ABSOLUTES about what we see now; KS already carries short predictions
  (storms, attackers + safe checks). POT is the SUPER-LONG-TERM prediction: can either side (both sides matter; the side
  to move gets the tempo) STRUCTURALLY change the position so that a subsystem will later "shine through" — e.g. a closed
  centre with a king stuck in it that one side can blow open: KS says nothing yet (lines blocked), POT gives a slight boost.
- ★ **Potential vs kinetic:** "if things are already blown up and KS is in full effect, then this midgame potential is
  not needed anymore as the potential is now kinetic." ⇒ **per subsystem, POT speaks only where that subsystem CANNOT yet
  judge the position reliably (unresolved), and goes silent where it can — regardless of whether it fires high or zero.**
  Non-KS transformations (pawn/majority/piece isolation…) count the same way.
- The ideal: see what deep search sees (a push that yields favourable connections 30 plies later, when search stops at
  15) — impossible exactly, but that is the direction. Part of the job is knowing WHEN something is unresolved.
- **Method:** catalogue the TYPES of transformation and their END RESULTS, find what makes them happen, and work BACKWARDS
  to detect the precursors early — a GM's intuition without calculation — then tune on it.
- **Endgame:** potential of this kind fades ⇒ POT hands over to the known endgame metric, WINNABILITY (§16 scale factor).
**Architecture this implies (my formalisation, for owner review):**
    POT = Σ_k  U_k(position) · P_k(side can force transformation k) · E[Δ owner_k after k]      (mg; → winnability in eg)
  k = transformation type; owner_k = the subsystem whose score it will change (KS, pawn structure, passers, mobility,
  winnability); U_k ∈ [0,1] = how UNRESOLVED owner_k is (1 = cannot judge yet, 0 = already kinetic). ⇒ no toe-stepping by
  construction: POT carries only the LATENT part of another subsystem's future score, and fades as that subsystem resolves.
**Plan:** (1) transformation EVENT catalogue from game sequences: detect when an owner subsystem's state jumps (lines open
toward a king, centre opens with an uncastled king, a passer/majority conversion, a weakness created, files opened …),
count types and outcomes; (2) PRECURSORS: which structural features N plies earlier predict a SUCCESSFUL event
(premature attempts are ~3× commoner, §8a — predict success, not attempts); (3) UNRESOLVED-ness per owner: predict
|owner(t+N) − owner(t)| from structure (locked pawns, tension, uncastled king, closed files …); (4) validate against SF18
search residuals (the depth-independent "what search sees and static eval doesn't"), with the no-overlap collinearity
check; (5) only then C++ and the Texel fit, gated on fresh seeds.
☠️ **Owner correction (2026-09-30): U_k must NEVER be read from a subsystem's OUTPUT.** A quiet KS can be KINETIC with
nothing to say (king safe, position settled) — a low score is not a potential zone. U_k = how much owner_k's INPUTS can
still change (locked vs tense pawns, closed vs openable files, castling still pending …), judged from STRUCTURE only;
empirically calibrated as the PREDICTED FUTURE MOVEMENT of owner_k, never its current level.
**Next (owner):** distil human + computer knowledge of transformations first (Soviet "transformation of advantages",
Nimzowitsch, Kmoch's levers, Flores/Soltis structure families, Shereshevsky, AlphaZero concept probing …) ⇒ a taxonomy
(type → end result → precursors → early signs → owning subsystem), then work backwards to detection and relative scoring.

## 18. POT T1 STUDY — central opening vs an uncastled king (2026-09-30, `_pot_t1_study.py GAMES=3000 N=20`)
First full pass of the owner's work-backwards method (§17; knowledge doc `POT-TRANSFORMATION-KNOWLEDGE-2026-09-30.md`).
Structure only: UNRESOLVED gate = every file kf−1..kf+1 (c-f) still holds a defender pawn; EVENT = by t+20 plies the
king is still on d-f and one of those files has lost all defender pawns.
- Rows: KINETIC 41,893 · UNRESOLVED 29,340. Event 31.5% (predicted 10-20% ✗). Defender scores 0.470 with the event vs
  0.505 without — the event itself costs only ~3.5pp in d6 games.
- Precursors → event: levers 29/39/48% (✓ Kmoch) · rams 3 → 2.4% (✓ freezes) · supported pushes FLAT (✗) · castling:
  can castle now 18% vs rights lost 43% (dominant, partly mechanical: a king that castles leaves before it opens).
  Logistic AUC val 0.651 (predicted 0.70 ✗). Overlap with engine KS +0.05 (KS silent on 89% of these rows).
- **SF18 check** (1,426 unresolved central-king rows of the labelled mg sample): corr(P(event) score, SF18−ours residual
  toward the attacker) **+0.10 (3.8σ); balanced +0.15 (3.3σ)** — but lopsided (low-score quintiles −2.1/−2.7pp, the rest ≈0)
  and **per precursor the STRUCTURAL ones carry nothing** (levers +0.009, push −0.007, rams +0.023); the signal is
  development (+0.128), castling (rights −0.084, block +0.090), heavy pieces (−0.096), and **side-to-move +0.232** (a
  static-vs-search TEMPO effect, not POT — the flat tempo bonus is closed on mechanism; worth checking on ALL mg rows).
- ⇒ Reading: the structural precursors predict THAT lines open, but SF18 does not price that as missing from our eval;
  what it prices is development / castling tempo (dynamic, T10-like, possibly owned elsewhere). n is small (1,426) and the
  event model was trained on d6 outcomes — a feasibility read, not a verdict.
**18a. Follow-ups (2026-09-30).** `MODE=stm`: the side-to-move residual is GENERAL, not T1 — +2.39pp toward the mover on
all 9,898 mg rows (central king +2.53, none +2.27, balanced +2.98) ⇒ a static-vs-search gap (the mover picks its best
move in search), parked as a separate lead; NOT POT.
`MODE=race` (owner: the castling race must not overlap KS — at most a KS FEEDER, not a POT rescoring): SF18 residual
toward the attacker, ridge-residualised out-of-fold on 235 control columns (engine KS + all 68 KS channels + C1/C3 incl.
the shelter/castle cells + stm), 1,426 unresolved central-king rows:
| feature | raw r (σ) | BEYOND controls r (σ) | max overlap |
|---|---|---|---|
| castle_tempi (moves until D can castle; 4 = cannot) | +0.102 (3.9) | **+0.102 (3.8)** | 0.34 |
| dev_lead | +0.128 (4.8) | +0.008 (0.3) — owned already | 0.31 |
| heavy_centre (A rooks/queens on the king's files) | −0.096 (−3.6) | **−0.131 (−5.0)** | 0.39 |
| levers | +0.009 (0.3) | +0.013 (0.5) | 0.42 |
| race = dev + tempi | +0.156 (5.9) | +0.090 (3.4) | 0.23 |
⇒ (1) the STRUCTURAL T1 core (levers) carries no SF-priced signal, raw or beyond; (2) **castling delay survives every
control** — but it is a king-SAFETY STATE (the king can't reach safety soon), i.e. KS's concept ⇒ per the owner's rule
it is a **KS FEEDER candidate** (e.g. a danger input for an uncastled central king), not a POT score; (3) development is
already carried by the eval; (4) **we OVER-credit the attacker's heavy pieces on the (still closed) king files** (−5.0σ
beyond controls) — an existing term scoring latent pressure as kinetic; find the owner (PST rook/queen files? KS attacker
count? rook-file terms?) — the inverse of POT's own rule, and worth a look.
T1 verdict so far: nothing for POT proper; one KS feeder lead; one over-scoring lead. Predictions for the race check
were not registered (I did not write them down before running — noted).
**18b. WHO over-credits heavy pieces on closed files? (2026-09-30, `_pot_t1_study.py MODE=heavy`, all 9,898 labelled
std mg positions × both sides, ev_breakdown only.)** Predictions: general not T1-only ✓ · open files ≈ 0 ✓ · the PST
absorbs it ✗.
- heavy (R+Q) on files holding an enemy pawn: −7.6σ raw, stepping +0.89 → −1.32pp from 0 to 3+; on open/half-open files
  0.6σ. No published term absorbs it (pieces/PST, KS, mobility, placement, pawns, passers — partials all ≈ −0.08).
- ☠️ my first material control was DEGENERATE (closed + open = R+Q count, so the partial was 0 by construction). The valid
  placement test — the SHARE of A's heavy pieces on closed files, controls = stm + all terms + both sides' R/Q + rams —
  leaves only −0.019 (≈ −2.7σ): **closed-file placement is a small effect; the signal is MATERIAL.**
- ★ **QUEEN IMBALANCE:** A has a queen, the opponent none (n 1,499): residual toward A **−5.31pp (se 0.17)**; by rook
  difference 0 (Q vs minors/pawns, n 1,145) **−6.13pp** · −1 (n 238) −1.95 · −2 (Q vs 2R, n 35) +1.20; both queens /
  none: 0.00. ⇒ **v2 OVER-values the queen against minor-piece compensation by ~6pp win%.** Not POT — a material-imbalance
  lead (record check running; Kaufman was parked 09-18). ⚠️ static vs SF SEARCH: confirm on the depth residual (our
  d10-12 search) before acting; quiet-filtered rows, but imbalance positions can be transient.
- §18a's "heavy_centre" finding is most likely this (queens sit on the king's files).
**16a. WSF DEPTH GATE — PASSED (2026-09-30).** SF18 @250k, paired vs FRESH shipped baselines on unused seeds:
seed 42 +1.60pp ± 2.52 · seed 43 +2.80pp ± 2.53 · **pooled +2.20pp ± 1.79, z 2.41 ⇒ +20.2 Elo [+3.7, +37.4]** (base
73.45% → 75.65%). Predictions: not clearly negative ✓ · seeds within 3pp ✓ (1.2) · pooled in [−10, +15] ✗ (above).
⇒ the FORM was the problem: the same idea (winnability), in the reference engines' continuous multiplicative form, turns
−23..−25 into +20. Next (protocol): SPRT @50k seed 45 → 2,000-game replication seed 46 (`selfplay/_queue_wsf_sprt.sh`),
then variants/odds, then the owner ships. Knobs would move to a `POT_V2_*` prefix on ship (§13).
**18c. KAUFMAN, TEXEL-FITTED on SF18 labels (2026-09-30, `_texel_kauf_fit.py`; owner: "Kaufman was never Texel-tuned,
so that may be where it shines").** 28,317 SF18-labelled std mg + eg rows, val by game hash 15%; every arm AND the
baseline fit a side-to-move nuisance and (`SCALE=1`) a global-scale nuisance α (fitted +0.095 = SF search scores run
~9.5% larger than our static), neither shipped — so no arm can win by stretching evals (without it the cell arms
stretched ×1.06, the rejected WSF-HI pattern). Val MSE vs baseline (overall · mg · eg · queen-imbalance):
SF tables unfitted +8.03 · +9.57 · +7.24 · +11.18 (worse — agrees with 09-18) ·
**CELLS λ 0.01 −2.60 · −3.17 · −2.31 · −8.52** (stretch 0.986) · **QUEEN cells λ 0.001 −1.01 · −2.09 · −0.46 · −9.42** ·
piece values only −0.26 · −0.37 · −0.20 · −2.66.
Cells: CELLS = pieces gain with pawns on the board (own N×P +32, R×P +21; their R×P +28, N×P +24, B×P +25), pair
×own pawns +28 / ×their pawns −25; QUEEN = queen loses with pawns (own Q×P −63, their −43) and is redundant with rooks
(own Q×R −77; SF −134). Exported to `E:/chess_data/texel/kauf_full.txt` / `kauf_queen.txt`.
Engine: `KAUF_V2_FORM=3` + `KAUF_V2_FILE` (cells in mp, MAG 1000 = as fitted), coded; ⚠️ NOT BUILT YET — the WSF SPRT
runs from the working tree. After it: build → fingerprints byte-identical at 0 → closure (engine kaufman_imbalance =
Python model) → symmetry → SF18 gauntlet @250k on fresh seeds, BOTH arms (per-part rule), vs a fresh baseline.
Also: confirm the queen lead on the depth residual (`_depth_residual_pass.py`) in the same engine window.
**16b. WSF — the instruments DISAGREE (2026-09-30, 20:45).** Self-play SPRT @50k (seed 45) at 3,480 games: +1362 −1381
=737, **elo ≈ −2, LLR −2.52 → heading to H0** (H0 ≤ 0 / H1 ≥ +10), against the external +20.2 [+3.7, +37.4]. The usual
pattern is the reverse (self-play inflates: Fit A +111 vs +38). Not read as harm: self-play ≈ 0 ± ~10 says "no self-play
gain", the external gate says "+20 with a lower bound of +3.7". Tie-breaker queued (`selfplay/_queue_wsf_regate.sh`, after
queue #2): a SECOND SF18 gate on new seeds 50 + 51 with fresh baselines. Ship decision waits for it.
**16c. WSF self-play verdicts (2026-09-30 night).** SPRT @50k seed 45: **H0 accepted** at 3,972 games, +1552 −1575 =845,
elo −2.0 ± 12.7. Replication (2,000 fixed, seed 46): +782 −792 =426, **elo −1.7 ± 17.9**. Pooled self-play ≈ −1.9 over
5,972 games (≈ ±10) ⇒ no self-play gain, no harm. External +20.2 [+3.7, +37.4] stands alone; the seed-50/51 re-gate
decides (queue #3). Hypotheses (owner Q&A): (1) statistics — a true ≈ +5 fits both; (2) DEPTH — the external gate ran
at 250k nodes, the SPRT at 50k, and an eg scale factor bites only where search reaches endgames; if the re-gate holds,
test with a self-play match at 250k. Fallback if ≈ 0: universal term (4/4 refs) kept at its fitted value only via the
final joint retune (owner rule: universal kept), with per-part ablation.
**18d. OVERNIGHT 09-30 → 10-01 RESULTS.**
- Build guard: fingerprints unchanged (v2 252 / 49,094,807 · v1 250 / 35,310,778). Kaufman FORM 3 closure EXACT (full
  12,564 live / queen 7,764 live of 14,842) · symmetry colour 0/4000, file 0/3170, both arms.
- ☠️ **Kaufman Texel cells LOSE at depth** (SF18 @250k, fresh seeds 47 + 48, fresh baselines `gauntlet_ship5_s4x`):
  **full −3.30pp ± 3.16, z −2.04 ⇒ −29.6 Elo [−56.1, −1.3]** · **queen −2.20pp ± 3.05 ⇒ −20.0 [−46.1, +8.0]** — both
  arms negative on both seeds. The 4th "fits the labels, loses at depth" case (C3, Fit W, Fit W-SF additive, Kaufman).
- ★ **Depth residual** (`_depth_residual_pass.py` d10 on 14,696 rows → `_depth_residual_read.py`, 9,827 std mg rows):
  mean |residual| static 8.39 → depth 5.63pp · side-to-move static +2.41 → **depth +0.80** (search fixes most of it — the
  instrument reframe confirmed) · **queen imbalance static −5.35 → depth −5.71pp (se 0.14): PERSISTS** ⇒ real knowledge
  search cannot supply. ⇒ The lead is right; the FORM was wrong: the census cells fire on 52-85% of positions (any queen ×
  any pawn/piece count) while the misjudgement lives only where ONE side lacks the queen. Next shape: a NARROW conditional
  term (queen vs no-queen only, by compensation), fitted on the DEPTH residual, gated at 250k.
- ★★ **v2 vs v1 (equal nodes 50k, 2,000 games, seed 49): +1072 −700 =228, 59.3% ⇒ +65.4 ± 17.9 Elo.** v2 now clearly
  beats v1 even at equal nodes (v2 gets ~40% more at equal time) — on 09-18 they were level.
**18e. Kaufman "collateral" check — INCONCLUSIVE, a regression artefact (2026-10-01).** `_gauntlet_gametype_split.py`
gained a material class (baseline game held a queen imbalance ≥ 10 plies; a transient mid-trade imbalance ≠ class — the
naive version put 786/1000 games in it). Kaufman full: persistent-QI games (n 277, baseline 86.3%) −15.5pp, rest +1.4;
queen cells −11.4 / +1.3. ☠️ CONTROL: the C3 arm (no material content) shows −14.9 / +2.6 on its own pairs; WSF (few changed
games) −3.5 / +4.5 ⇒ the class selects openings where the BASELINE scored ~86% and any arm that changes many games
regresses there. Says nothing about Kaufman. (The outcome-selected-class trap, one level up: a class defined by the
baseline game's CONTENT still correlates with the baseline's success.)
**18f. IMBALANCE DEPTH SCREEN (2026-10-01, `_imbalance_depth_screen.py`, 9,827 std mg rows, residual toward the first-named
side, static → DEPTH pp):** Q vs no-Q rooks 0 (n 1,115) −6.19 → **−6.31** · Q vs no-Q rooks −1 (n 235) −1.97 → −3.65 ·
R vs 2 minors (n 197) −3.61 → −2.74 · minor vs ≥2 pawns (n 283) +5.78 → **+4.65** · bishop pair (n 1,349) +2.42 → +1.79
· exchange (n 798) +2.06 → +0.95 · B vs N (n 1,964) +0.80 → +0.51 — every class PERSISTS at depth. One direction: queen and
rook too DEAR vs minors/pawns; minors and the pair too CHEAP.
**B vs N × openness: NULL** (r +0.012 raw, +0.013 after the bishop/knight mobility counts; levers −0.005) ⇒ the
openness-conditioned minor value is DROPPED (owner's mobility-overlap caution confirmed).
**Hypothesis — OPPONENT-STRENGTH dependence of the gauntlet:** labels = SF18 d14 (strong play); the gauntlet opponent =
SF18 @400 nodes (we score ~75%). Vs a much weaker opponent, a queen in a tactical imbalance is worth MORE than strong-play
theory says ⇒ fitting strong-play labels can lower the weak-opponent gauntlet while making the eval truer. Would also fit
WSF (+20 vs SF@400, flat self-play). Test: Kaufman queen arm in SELF-PLAY (never run) — queued after the WSF re-gate.
**18g. NARROW MATERIAL CLASSES (MCL_V2), fitted on the DEPTH target (2026-10-01, `_material_class_fit.py`).** Each term
fires ONLY in its class (the Kaufman lesson: same SHAPE as the error). Base = OUR d10 search, target = SF18 d14, std +
VARIANT rows (14,518; the depth target needs no static total), × phase/256 (fades into the endgame, which POT winnability
owns), STM nuisance fitted not shipped. Val MSE −1.87% overall; by class QUEEN (2,122 rows) −6.70% · PAIR (1,632) −4.46% ·
MINOR vs ≥2 pawns (350) −4.29% · R vs 2 minors (294) −2.89%. Fitted (cp, A-oriented): q0 −43, per opposing extra rook +8,
per extra minor −19, per extra pawn +2 (⇒ Q vs 3 minors ≈ −1 pawn, Q vs 2R ≈ neutral) · r2m −17 · mp0 +32 (+2 per extra
pawn) · pair +20. Engine `MCL_V2` + `MCL_V2_{Q0,QR,QM,QP,R2M,MP0,MPP,PAIR}` (mp), default 0, syntax-clean; built by queue #5.
Gates queued (`selfplay/_queue_mcl.sh`): closure + symmetry → SF18 @250k seeds 54/55 (all classes · queen class alone)
→ self-play 2,000 @50k. ⚠️ PAIR re-opens a lane closed as "owned by PST + mobility" (flat pair, 09-18) — the depth
residual says the pair is still +1.8pp under-valued; the per-part gate decides.
**16d. WSF SHIP BUILT (2026-10-01).** Fingerprints: v1 250 / 35,310,778 (unchanged) · shipped with POT_V2_WIN=0 252 /
49,094,807 (exactly the old v2 ⇒ the ship changed nothing else) · **NEW shipped v2: 251 / 49,211,859 / 4.014**. Runner
`V2=` line + CURRENT-CONFIG §1 carry the POT_V2_WIN knobs.
**18h. Kaufman SELF-PLAY (queue #4, 2,000 each @50k, seed 52):** queen cells +3.3 ± 17.9 · **full table +11.8 ± 17.9**
(vs −29.6 [−56, −1.3] in the SF18 @250k gauntlet ⇒ ~41-Elo swing, ≈2.5σ). Supports an instrument difference, but
CONFOUNDED: self-play @50k vs gauntlet @250k (depth) AND equal vs weak opponent. Queue #5's 250k self-play (WSF) separates
the two for winnability; a 250k Kaufman self-play would do the same for material.
