# Eval v2 — Rung 2 design: pawn placement, structure and support

@author: Ranuja Pinnaduwage (design captured with Claude)
Status: DESIGN. Nothing built. Written 2026-09-12 after the four mandatory pre-rung scans.

⚠️ Read `EVAL-V2-REBUILD-LOG.md` 09-12 entries first — three measured results below are load-bearing and
were surprises: the pawn clamp hides HARM, the chain bonus is a confirmed central double-count, and the
masking does NOT generalise past pawns.

---

## 0. What the scans found (the reason this is a rewrite, not a port)

| scan | finding |
|---|---|
| 1 — consumers by data dependency | **24 functions** touch pawn state. **Every piece evaluator takes the passer masks as parameters** ⇒ passers are a GLOBAL dependency, not pawn-local, and the detector must run before all of them |
| 2 — knob inventory | ~40 pawn knobs; **~12 sit at 0** (majority x5, isolated, backward, outposts x2, closedness, obstruction blend, passer danger, passer resid). `STRUCT_R_MG_PCT`/`_EG_PCT` are rank-indexed ARRAYS |
| 3 — producers / side effects | `evaluate_pawns_*` writes **four** other subsystems: `attack_bitmasks`, the OvD accumulators, `central_score`/`update_global_central_scores`, and `g_passer_{mid,end}_deferred` — plus the passer masks via `getPPIncrement` |
| 4 — clamps / shared budgets | `total -= min(PAWN_CLAMP_MID, structural + positional)` — two competing quantities, ONE 225mp budget. Masked chain **14x** and struct **23x**, and INVERTED both signs |

⇒ The pawn layer cannot be ported. It is entangled with four subsystems by side effect, its structural
terms are net-harmful under a clamp that exists to hide them, and a third of its knobs are dead.

---

## 1. Architecture — three layers, and the cache boundary IS the detector/scorer boundary

### Layer A — `PawnEntry` : DETECTOR. Pure function of the two pawn bitboards. CACHEABLE on `pawnKey`.

```
struct PawnEntry {
    uint64_t passed[2], candidate[2];      // promotion-relevant
    uint64_t isolated[2], backward[2], doubled[2];
    uint64_t phalanx[2], supported[2], opposed[2], lever[2];
    uint64_t attacks[2], attacks2[2];      // pawn attack + double-attack spans
    uint8_t  openFiles, halfOpen[2];
    uint8_t  obstruction[64];              // ★ OURS: graded passed-ness
    int16_t  score_mg, score_eg;           // Layer B's output, stored in the entry
};
```

★ **Nothing here reads a piece, a king, or the side to move.** That is what makes it cacheable, and it is
also exactly the owner's DETECTOR stage. The cache boundary and the testability boundary coincide — the
single most useful structural fact of this rung.
✅ `pawnKey` already exists (`cache_management.h:161`, computed for corrhist). Only the table is missing.

### Layer B — structure SCORE. Reads only Layer A. Stored INSIDE the entry ⇒ computed once per structure.

### Layer C — `passer_value()` : piece/king dependent ⇒ NOT cacheable, recomputed per node.

☠️☠️ **MUST BE ADDITIVE. My first draft of this layer was MULTIPLICATIVE — the documented root defect.**
`passer-law-multiplicative-vs-additive-and-valuation-graveyard`: ours is `mag x R/256`; **SF and Ethereal
grant the rank bonus UNCONDITIONALLY and treat path-safety as UPSIDE ONLY, never as a factor that can
zero.** One wrong realizability call zeros a whole pawn, and because each failure position is stopped by a
DIFFERENT mechanism (blockade / path-attack / rear-rook / king-escort), **no scalar magnitude knob closes
it** — two-sided error, same wall as KS.
⇒ Layer C form: `value = Rank[r] (unconditional) + bonuses(blockade quality, king proximity, rook-behind,
path safety)`. Every modifier is an ADDITIVE term with its own sign. **No multiplier anywhere.**
☠️ **Standing owner instruction: "do not propose a 5th passer VALUATION mechanism."** Layer C is therefore
NOT a new mechanism — it is the STRUCTURAL replacement of the multiplicative form under a SINGLE owner,
which the graveyard explicitly names as what remains genuinely unbuilt.
☠️ **Do NOT re-enable `PASSER_ENEMY_CREDIT_PCT`.** Its zeroing was "Gap-P P1", a component of a bundle that
shipped **+38.7 ±27 Elo / 875 games**. Turning it on is an UN-FIX, not an untested lane — it re-credits an
obstruction `getPPIncrement` has already docked.

☠️ **The detector RETURNS its masks. It must never publish them as a side effect.** `getPPIncrement`'s
publication is what kills a naive pawn hash — a cache hit would skip the write and every downstream piece
evaluator would read stale masks. Same family as `eval-global-side-effects-are-skipped-by-a-cache-hit`.

---

## 2. KEEP / REMOVE / CHANGE / MOVE

### ★ KEEP — and flag as deliberately OURS

| item | why |
|---|---|
| ~~**Graded passed-ness** (`obstruction[]`)~~ | ☠️ **STRUCK 2026-09-12 by record-check.** The graded interpolation IS `ENABLE_PAWN_OBSTRUCTION_BLEND` (`passer_table_weight` over `PAWN_OBS_LO/HI`), it is **default FALSE**, and it is IN THE VALUATION GRAVEYARD: *"+7.33 val worse; a FORM SF doesn't have"*. v1 ships SF's BOOLEAN threshold (`ppIncrement >= 100`). I had written "do not flatten to SF's boolean" — backwards. ⚠️ What IS live is `(ppIncrement >> 3)` folded into the rank bonuses; the gate is not. ⇒ Re-introduce grading ONLY as part of the additive rewrite below, never as a new valuation knob |
| **SF candidate-passer detection** (lever / leverPush / blocked) | already SF-faithful; keep verbatim in the detector |
| **Rear-doubled exclusion** | a pawn with a friendly pawn ahead on its file can never promote. Correct, and SF agrees |
| **Blockade QUALITY** (secure minor on the stop square vs a major merely contesting) | our answer to `passer-defect-is-blockade-cost-blindness`. Keep, move to Layer C |

### ❌ REMOVE

| item | evidence |
|---|---|
| `pawn_chain_file_bonus[x]` (FILE-keyed chain bonus) | **confirmed central double-count**: the same loop calls `update_global_central_scores`. Harmful **6/6 corpora** once unmasked (−2.29%) |
| `STRUCT_OPPOSED_MG_PCT` / `_EG_PCT` post-hoc multipliers | harmful **6/6** (−1.38%). SF's `opposed` lives INSIDE the connected formula as `−bool(opposed)`, not as a percentage on an already-computed bonus |
| `PAWN_CLAMP_MID` / `_EG` | bounded-by-construction instead. The clamp's only job was hiding the two terms above |
| the 12 zeroed knobs (majority x5, closedness, obstruction blend, passer danger, passer resid, outposts) | dead code; re-derive from scratch if wanted, do not port switched-off machinery |
| `pawns_simd_initializer` + the SIMD pawn path | ~150 dead lines; the scalar path is the live one |
| `ENABLE_PASSER_V2` legacy arms, `!ENABLE_PASSER_V3` branches | shipped-on legacy pairs |
| `record_pawn_clamp` / `g_passer_{mid,end}_deferred` | probes and deferral machinery for a clamp that no longer exists |

### 🔄 CHANGE

| from | to |
|---|---|
| chain bonus keyed on FILE | **SF's rank-keyed connected term**: `Connected[r] * (2 + bool(phalanx) − bool(opposed)) + 21 * popcount(support)`, `Connected = {0,7,8,12,29,48,86}`, eg leg `v * (r−2) / 4` |
| `getPPIncrement` = 7 fused jobs with side effects | split: detection into Layer A, contest docking into `obstruction[]`, blockade/support pricing into Layer C |
| isolated / backward at 0 | build at SF magnitudes: isolated `S(5,15)`, backward `S(9,24)` ⇒ ~39/70 and 70/113 mp. ★ **Much smaller than intuition** — an isolated pawn costs SF ~4% of a pawn in the midgame |

### ➕ ADD — the three giant terms we do not compute AT ALL

`doubled` `S(11,56)` (**263mp in the endgame — SF's LARGEST pawn penalty**) · `weak-unopposed` `S(13,27)` ·
`weak-lever` `S(0,56)`. None has a knob or a line of code today.

### 📦 MOVE

| item | to |
|---|---|
| pawn attack spans | into `PawnEntry.attacks[]` — ★ **KS already needs them** for the SF zone's double-pawn-defended removal, so this is an immediate reuse win, not new cost |
| open / half-open file detection | detected here (`openFiles`, `halfOpen[]`), CONSUMED at rung 6 (rook files) |
| pawn → central / space contribution | ❌ **not replicated here.** Rung 3 owns central and space; pawns feed it through the shared attack maps, priced ONCE |
| pawn → OvD accumulators | dropped with the heat map; OvD is a late rung if at all |

---

## 3. ☠️ COLLINEARITY — the discipline, and the thing that will bite

**Pawn RANK is currently priced in at least four places**: the pawn PST, `default_midgame_pawn_rank_bonus`,
`passed_midgame_pawn_rank_bonus`, and the passer value — and **SF's connected term is ALSO rank-keyed**.
⇒ Porting SF's connected bonus on top of our PSTs **recreates the double-count in a new location**. This is
the single most likely way rung 2 fails, and it is the same error the chain bonus already committed.

### The rule: every concept gets exactly ONE owner

| concept | owner | everyone else |
|---|---|---|
| absolute square value (file + rank base) | **pawn PST** | may not add a rank ramp |
| relations between pawns (isolated/doubled/backward/connected) | **structure score** | expressed as a DELTA from the PST baseline, never as a second rank ramp |
| promotion proximity and its contest | **passer value** | the only place advancement is re-priced, and only for passers |
| centrality | **rung 3 central term** | structure prices NO file term (this is why the chain bonus dies) |

### ★ The new capability v1 never had: measure overlap BEFORE tuning

Because Layer A exposes every term's **firing set as a bitboard**, and the breakdown struct publishes them,
we can compute **pairwise firing-set overlap between any two pawn terms without tuning anything**. v1 could
not do this — its terms had no separable detector.
⇒ **Protocol: before assigning a single constant, dump the pairwise overlap matrix of the structural
masks.** Anything above ~50% overlap is two names for one signal and gets merged or dropped on the spot.
That is the concrete anti-collinearity tool, and it is cheap and deterministic.
⚠️ `bundling-is-refuted-components-cancel-26-percent` — components overlapping 50-64% cancelled ~26% of
each other's move changes. The overlap matrix is exactly the early-warning instrument for that.

---

## 4. EFFICIENCY

- ★ **The whole of Layers A+B runs once per pawn structure, not once per node.** Pawn structure changes on
  a small fraction of moves ⇒ high hit rate. This is the biggest single speed lever in the rung and it is
  available only because Layer A has no piece dependency.
- **Set-wise, not per-square.** isolated / doubled / phalanx / opposed / open-files are all file-fill and
  shift tricks over the two pawn bitboards. Only `obstruction[]` needs per-pawn work — keep it, inside the
  cached layer where it is paid once.
- Narrow the STORAGE (`int16_t` scores, `uint8_t` obstruction) for cache density; compute in `int`.
- ⚠️ Do not hand-optimise against `-Ofast -flto`; the structural wins above are the ones that pay.

---

## 5. ORDER OF WORK — structural first, per the owner

| step | content | gate |
|---|---|---|
| **2a** | `PawnEntry` detector + structure score: isolated, doubled, backward, connected/phalanx/supported, opposed. **No passers.** | overlap matrix, then §I, then colour symmetry |
| **2b** ★ | passer DETECTION — **the one genuinely-untested structural box.** We miss **~14% of SF's passers** and `ENABLE_PASSER_DETECT_SF` is still **false**. 🧰 `diagnostics/passer_detector_diff.py` measures the gap as a pure predicate count, no engine | detector-only: does it flag the right pawns? No scoring constant need exist |
| **2c** | `passer_value()` in **ADDITIVE** form under a single owner — the structural replacement, not a 5th valuation mechanism | §I + games at the rung 2-4 checkpoint |
| **2d** | KS-B shelter, keyed on `pawnKey` + king square | closes the one UNIVERSAL channel KS-A lacks |

★ 2a and 2b are separable precisely because the detector returns rather than publishes. That separation is
what makes the "is the detector right?" question answerable without any scoring constant existing yet.

---

## 6. OPEN — to settle before or during 2a

1. ✅ **DONE 09-12 — and it struck two items above.** ☠️ Also invalidates a claim made this morning:
   the **phantom-knob law** says there is NO master passer off-switch (`SCALE_PASSED_PAWN` = 18% of passer
   mass, `PASSER_MAG_SCALE` = 32%, **~68% under neither**) ⇒ **every past passer ablation measured a
   FRACTION; its null is not a null.** So "passers are worth +5.91%" (from `ENABLE_PASSER_V3=0`) and
   "−0.83%" (from `SCALE_PASSED_PAWN=0`, 18% of mass) are both under-reads of unknown size.
   ⚠️ Screening discipline from the same record: validate on the de-biased regret set + games, **never**
   `passer_corpus.csv under_fire` — tuning to it is anti-correlated with Elo (proven: MAG=150 read as a win
   on corpus and movematch while costing −54 STS).
2. ⏳ Do our pawn PSTs already encode a rank ramp steep enough that SF's connected rank keying double-counts?
   **Measurable**: correlate `Connected[r]` against our PST's per-rank pawn delta before choosing constants.
3. ⏳ `SCALE_PAWN_WALL` measured genuinely neutral (3/3 split). Does the concept survive as anything in 2a,
   or is it subsumed by connected + supported?
4. ⏳ Does `obstruction[]` survive contact with a correct connected/backward term, or was it compensating
   for their absence? ★ The honest risk to our one original: it may have been carrying a missing channel.

---

## 7. ▶️ 2a RESULT — the overlap matrix, run BEFORE any constant (2026-09-12)

🧰 **NEW: `diagnostics/_pawn_term_overlap.py`** — every structural predicate is a pure function of the two
pawn bitboards, so firing sets come straight from corpus FENs with **no engine, no build, no constants**.
✅ Self-tested 8/8 on hand-checked positions **and colour-symmetric 3/3** before any matrix was read.
4,000 positions / 39,738 pawns / `game_regret_set.csv`.

☠️ **Read LIFT, not raw overlap.** `opposed` fires on 71.75% of all pawns, so it co-occurs with everything
at ~72% for free. `lift = P(col|row) / base_rate(col)`; 1.0 = independent, >2.0 = structurally coupled.

| term | firing rate |
|---|---|
| opposed | 71.75% |
| phalanx | 30.56% |
| supported | 26.65% |
| stop_held | 26.59% |
| isolated | 16.62% |
| blocked | 14.04% |
| passed | 11.72% |
| backward | 11.44% |
| lever | 3.59% |
| **doubled** | **1.48%** |

### The three clusters
| cluster | members | lift | reading |
|---|---|---|---|
| ① **weak pawn** | isolated / backward / doubled | **2.47 - 2.85** mutually | land on the SAME pawns far above chance ⇒ price all three SMALL or triple-charge one weak pawn. ★ First direct evidence for SF's counter-intuitive magnitudes (isolated = 4% of a pawn) |
| ② **strong pawn** | phalanx / supported | **0.32** (ANTI-correlated) | a pawn beside you is not a pawn behind you ⇒ genuinely COMPLEMENTARY. Confirms SF feeding them as two inputs to ONE connected formula rather than two terms |
| ③ ☠️ **the trap** | isolated ↔ passed **2.57** · weak_unopp → passed **4.27** | highest in the matrix | an isolated pawn is 2.6x likelier than chance to be a PASSER ⇒ the isolated PENALTY and the passer BONUS fire on the same pawn, opposite signs, far above chance. **The components-cancel pattern, visible before either term exists** |

### Decisions for 2a
- ❌ **Drop `weak_unopposed`** — fires mostly on pawns the passer term will reward (4.27x). Revisit only if
  the passer term leaves residue.
- ❌ **`blocked` is not a priced term** — logical subset of `opposed` (raw 100%); it is an INPUT to
  `backward`'s definition, nothing more.
- ⚠️ **Isolated stays SMALL, and whether it applies to passers is now an EXPLICIT testable choice.** SF
  applies it with no exclusion — presumably BECAUSE it is small.
- ✅ **phalanx + supported into ONE connected formula** — confirmed complementary, not duplicative.
- ⚠️ `doubled` fires on 1.48% of pawns; at SF's 263mp eg that is ~4mp/pawn in aggregate. Worth building,
  but **invisible to any aggregate instrument** — do not expect §I to see it.
- ✅ Hard logical exclusions verified: passed ∩ {opposed, blocked, stop_held, backward} = 0;
  phalanx ∩ {isolated, backward} = 0; supported ∩ {isolated, backward} = 0.

★ All of this was decided with **zero eval code written and zero constants chosen** — which is the whole
point of the detector/scorer split, and something v1's architecture could never have supported.

---

## 8. SF -> Ethereal -> v1 -> v2 : the structural mapping (2026-09-12). PASSERS EXCLUDED (2b).

**Unit conversion** (SF11 `types.h:182`): `PawnValueMg = 128`, `PawnValueEg = 213`; ours = 1000.
⇒ **mg x7.81 · eg x4.69.** ☠️ Ratios transfer, scales never — convert, never copy.
⚠️ **We hold NO Ethereal source locally** (only `ethereal-search-brief`). Its `PawnConnected32` shape and KS
weights are recorded in `EVAL-V2-RUNG-PRIORS.md`; its isolated/backward/doubled constants are **UNKNOWN**
and are marked `?` below rather than guessed.

| concept | SF11 | SF15.1 (drift) | Ethereal | v1 | -> v2 (2a) |
|---|---|---|---|---|---|
| **isolated** | `S(5,15)` = **39/70 mp** | `S(1,20)` — mg nearly ZERO | ? | `ISOLATED_PAWN_PEN = 0` (built, zeroed) | build at ~39/70. ★ SF's tuner drove mg 5->1 over 4 years |
| **backward** | `S(9,24)` = **70/113** | `S(6,19)` — shrank | ? | `BACKWARD_PAWN_PEN = 0` (built, zeroed) | build at ~70/113 |
| **doubled** | `S(11,56)` = **86/263** | `S(11,51)` + NEW `DoubledEarly S(17,7)` | ? | ☠️ **NO KNOB, NO CODE.** Exists only as a passer EXCLUSION | build at ~86/263. ⚠️ fires on **1.48%** of pawns ⇒ ~4mp/pawn aggregate, invisible to §I |
| **connected / phalanx / support** | `Connected[RANK] = {0,7,8,12,29,48,86}` x `(2 + phalanx − opposed)` + `21*popcount(support)`; eg leg `v*(r−2)/4`. **NO FILE TERM** | `{0,3,7,7,15,54,86}` — flatter low, same top | `PawnConnected32` keyed **RANK *and* FILE**, mild centre tilt: rank 7 runs `108->214->216->233`, **d/e : a/h = 2.16x** | `pawn_chain_file_bonus = {10,15,100,150,150,100,15,10}` — **FILE ONLY, d/e:a/h = 15.0x, NO RANK TERM**; separate `EG_PHALANX=100`, `EG_SUPPORT=135`, `EG_DEFEND=115` | **rank-primary** (both references agree) + an **optional MILD file tilt** as an Ethereal-style refinement to test. See the correction below |
| **opposed** | inside connected as `−bool(opposed)` | same | inside its table | `STRUCT_OPPOSED_MG/EG_PCT` = post-hoc **percentage on an already-computed bonus** + `STRUCT_R_MG_PCT[rank]` array | inside the connected formula, per both references. ☠️ harmful 6/6 in its v1 form |
| **weak lever** | `S(0,56)` = **0/263** | `S(2,57)` | ? | absent | ⏳ defer — pure endgame term, and `lever` fires on only 3.59% |
| **weak unopposed** | `S(13,27)` = **102/127** | `S(15,18)` | ? | absent | ❌ **DROP from 2a** — lift **4.27x** to `passed`; it fires mostly on pawns 2b will reward |
| **"pawn wall"** | — | — | — | `pawn_wall_file_bonus = {75,50,60,75,75,60,50,75}` — U-shaped by file | ❌ **no reference has this.** Measured neutral (3/3 split). Drop unless it earns a place |
| **rank handling** | rank IS the connected key | same | rank is half the key | ☠️ rank tables live in the POSITIONAL half (`default_midgame_pawn_rank_bonus`, `endgame_pawn_rank_bonus_base = {0,90,210,360,750,990,1260,1560}`); structure gets only a rank PERCENTAGE (`STRUCT_R_*_PCT[]`) | rank owned by the **PST** (§3 one-owner rule); connected expresses a DELTA, not a second ramp |

### ☠️ CORRECTION to §2 — "structure prices NO file term" was OVER-READ
This morning's clamp probe showed v1's chain bonus harmful **6/6 corpora** once unmasked, and I concluded
"build SF's rank-keyed term, no file term." ⚠️ But the thing measured was **file-only, 15x tilt, zero rank
component** — the outlier against BOTH references, not "the Ethereal design". A refuted extreme does not
refute a **mild file tilt layered on a rank base**, which is exactly what Ethereal ships and what
`EVAL-V2-RUNG-PRIORS.md:204` explicitly warned against settling by appeal to SF.
⇒ **2a builds rank-primary connected (SF/Ethereal agree; it is the axis v1 omits entirely). A mild
Ethereal-style file tilt (~2x centre:edge, not 15x) is a SEPARATE, TESTABLE refinement** — and the overlap
instrument can screen it against the rung-3 central term before it is ever tuned.
★ The real diagnosis of v1's chain bonus is sharper than "double-count": **it is mispriced on the axis both
giants agree is primary (rank, absent) and over-priced 7x on the axis they treat gently (file).**

---

## 9. ✅ Ethereal column FILLED from source (2026-09-12) — and it overturns two cells

🔗 `https://github.com/AndyGrant/Ethereal` `src/evaluate.c` (now recorded in [[reference-engine-sources]];
⚠️ it is NOT on disk, unlike the Stockfish trees).
**Ethereal `PawnValue = S(82,144)`** ⇒ our conversion **mg x12.20 · eg x6.94**.

```c
PawnIsolated[FILE]  mg: -13, -1, +1, +3, +7, +3, -4, -4    eg: -12,-16,-16,-18,-19,-15,-14,-17
PawnStacked[2][FILE] mg: +10..-7 (mostly >= 0)             eg: -29..-20  /  -17..-9
PawnBackwards[2][RANK] mg: 0,0,+7,+6,-4 / 0,-9,-5,+3,+29   eg: 0,-7,-7,-18,-29 / 0,-32,-30,-31,-41
PawnConnected32 rank7: S(108,35) S(214,45) S(216,70) S(233,61)
                rank6: S( 45,40) S( 36,64) S( 58,74) S( 64,88)
                rank5: S(  8,14) S( 21,17) S( 31,23) S( 25,18)
                rank4: S(  6,-1) S( 20, 1) S(  6, 3) S( 14,10)
PawnCandidatePasser[2][RANK]  -- exists; a 2b item
```

### ☠️ ISOLATED: THE TWO REFERENCES DISAGREE IN SIGN (midgame)
| | mg (mp) | eg (mp) |
|---|---|---|
| SF11 `S(5,15)` | **−39** (penalty) | −70 |
| Ethereal, e-file `S(7,−19)` | **+85** (BONUS) | −132 |
| Ethereal, a-file `S(−13,−12)` | −159 | −83 |
| v1 `ISOLATED_PAWN_PEN` | **0** | **0** |

⇒ Ethereal charges isolation on the WINGS and **pays for it in the centre**; SF charges it flatly
everywhere. They agree only that the ENDGAME is negative. Per
`adopt-reference-methods-only-if-universally-superior`: **where they disagree, ours is legitimate and
theirs is a CANDIDATE.** ☠️ **`ISOLATED_PAWN_PEN = 0` may be a defensible compromise, not an omission** —
§8 was about to "fix" it to SF's −39/−70 on SF's authority alone. **Do not.** Build it FILE-INDEXED, or
endgame-only where both agree, and screen the midgame leg as a candidate.

### ★ Ethereal's file tilt is a RANK-7 PHENOMENON, not a centre preference
d/e : a/h ratio by rank — **rank 7 = 2.16x · rank 6 = 1.42x · ranks 2-5 = NOISE** (`6, 20, 6, 14`).
⇒ v1's `pawn_chain_file_bonus` applies at **EVERY rank, at 15x**, what Ethereal applies **only at the top,
gently**. Wrong on both axes independently. ★ This vindicates rank-primary more strongly than the clamp
probe did, and narrows §8's "optional mild file tilt" to: **a file tilt, if any, belongs at ranks 6-7 only.**

### ✅ UNIVERSAL (both references, same direction) ⇒ build per the adoption rule
| term | SF11 | Ethereal | reading |
|---|---|---|---|
| **doubled** | mg −86, **eg −263** | mg ~0/positive, **eg −201..−145** | ☠️ **an ENDGAME term.** Both agree eg ≈ −200..−263mp and midgame is contested/near-zero. Build eg-weighted; do NOT carry a large mg leg |
| **backward** | mg −70, **eg −113** | mg mixed (incl. **+29**), **eg −222..−285** | same shape: endgame-negative, midgame contested. Ethereal ~2.5x SF in eg |
| **connected** | rank-keyed, no file | rank-keyed, file tilt only at top | rank is the primary key in BOTH |

### ⚠️ Both references use TABLES; v1 uses SCALARS
Ethereal indexes isolated by FILE, backward and candidates by RANK, stacked by FILE x a binary condition.
SF indexes connected by RANK. v1 has `ISOLATED_PAWN_PEN` / `BACKWARD_PAWN_PEN` as **bare scalars** (at 0).
☠️ **A scalar cannot express "penalty on the wings, bonus in the centre"** — so v1's representation could
not have captured Ethereal's shape even if the constant had been non-zero. ⇒ v2 builds these as small
tables from the start, not scalars.

---

## 10. ☠️ CORRECTIONS from the owner (2026-09-12) — and the FINAL 2a spec

### Two claims of mine were wrong, both from scanning by NAME rather than by VALUE

1. ☠️ **"doubled has no knob and no code" — FALSE.** It is LIVE with mg/eg scaling, as hardcoded literals:
   `:958` / `:1095` `125 * (popcount(BB_FILES[x] & own pawns) > 1)` midgame, `:3261` / `:3374` `150 *`
   endgame. ⚠️ **Scan 2 enumerates `Config::` knobs and is therefore BLIND TO HARDCODED MAGNITUDES.**
   Second time the knob-only view misled (first: `EG_CLAMP_*` being a VALUE, not a flag).
   ⇒ **Scan 2 amended: enumerate hardcoded magnitudes as well as knobs.**
   ★ The real defect is the TAPER, which neither of us had seen:

   | | mg | eg | eg/mg |
   |---|---|---|---|
   | v1 | 125 | 150 | **1.2x** |
   | SF11 | 86 | 263 | 3.1x |
   | Ethereal | ~0 | −201 | infinite |

   Ours is nearly FLAT across phases where both references make it overwhelmingly an ENDGAME term, and our
   midgame value is the LARGEST of the three.

2. ★ **"pawn wall is unique to us" — FALSE, and the truth is better.** `:998-999`
   `structural_bonus += pawn_wall_file_bonus[x] * (left != 0)` where `left`/`right` are **same-rank adjacent
   pawns = PHALANX**; while `pawn_chain_file_bonus` (`:1053`) fires inside the pawn's ATTACK loop on own
   pawns = **SUPPORT**. ⇒ **v1 has BOTH of SF's connected inputs, split into two file-keyed terms:**

   | | SF11 | v1 |
   |---|---|---|
   | form | `Connected[rank] * (2 + phalanx − opposed) + 21 * support` | `chain_file[f] * support` **+** `wall_file[f2] * phalanx` |
   | key | ONE term, **rank** | TWO terms, **file**, no rank |

   ⇒ **The wall is not dropped — it is MERGED** as the `phalanx` input; chain becomes the `support` input.
   ✅ Consistent with the measured `phalanx` vs `supported` lift of **0.32 (anti-correlated)**: they are
   complementary halves of one relation, which is exactly why SF makes them modifiers of one term.

### ★ The additive-vs-2D point (owner's question, answered)

v1 scores connected as `rank_table[r] + chain_file[f]` — **additive and therefore SEPARABLE. A separable
form CANNOT express an interaction**, i.e. cannot say "file matters at rank 7 but not rank 3".
Ethereal's `PawnConnected32[rank][file]` is a full 2D table and says exactly that (2.16x @r7, 1.42x @r6,
noise @r2-5). ⇒ **v2 uses a 2D rank x file table for connected.** The owner's file concept SURVIVES; only
its representation changes, and that change is what makes the uniform 15x tilt impossible to express.

### ★★ "Doesn't Ethereal's file boost pollute central scoring?" (owner's contention) — RESOLVED

**The one-owner rule is about the same QUANTITY being priced twice, not about two terms sharing an axis.**
Central prices *control of central squares* (attack-derived). A connected file term prices *where a chain
sits*. Those correlate; they are not the same number, and correlation is not double-counting.
⇒ **Enforce the rule by MEASUREMENT, not by banning a functional form** — we now have both instruments
(firing-set lift, and contribution correlation per `_ks_channel_collinearity`'s method).
☠️ And v1's chain measured harmful at **15x, at every rank**. That condemns THAT magnitude and shape, not
file-sensitivity itself. §8's "structure prices NO file term" was over-read from a refuted extreme — the
same over-generalisation I made about the clamps earlier the same day.

### ▶️ FINAL 2a SPEC

| term | form | seed |
|---|---|---|
| **connected** | ONE term: `Connected2D[rank][file] * (2 + phalanx − opposed) + K * popcount(support)` | rank from SF `{0,7,8,12,29,48,86}`; file tilt from Ethereal's SHAPE (~2.16x @r7, ~flat @r2-5) |
| **doubled** | keep ours | retaper mg 125 -> ~86, eg 150 -> ~263 |
| **isolated** | FILE table | eg leg only (~−70..−132); **mg leg = 0** (references disagree in SIGN) |
| **backward** | RANK table | eg-primary (~−113..−285); mg contested |
| weak-unopposed | ❌ dropped | 4.27x lift to `passed` |
| weak-lever | ⏳ deferred | 3.59% firing, pure eg |
| clamp / opposed-% | ❌ removed | harmful 6/6 unmasked |

### ▶️ 2a EXPERIMENT LIST

| | experiment | arms |
|---|---|---|
| E1 ★ | connected FORM | 2D rank x file vs rank-only vs v1 additive |
| E2 ★ | file tilt MAGNITUDE | 1.0x / 2.16x (Ethereal) / 15x (v1) |
| E3 ★ | doubled TAPER | 125/150 vs 86/263 vs ~0/201 |
| E4 | isolated mg leg | 0 vs −39 flat vs Ethereal centre-positive file table |
| E5 | backward present / absent | |
| E6 | wall+chain MERGED vs separate | confirms the merge empirically |
| E7 | pawn hash | byte-identity cached vs uncached, AFTER E1-E6 settle |

★ E1-E3 are experiments that would NOT have existed yesterday: I would have built rank-only connected, no
file term, doubled from scratch, and isolated at SF's flat penalty.

### ✅ A correctness gate we do not normally get

`diagnostics/_pawn_term_overlap.py` computes these exact predicates in Python, **validated 8/8 on
hand-checked positions and colour-symmetric 3/3**. ⇒ Use it as a **mask-for-mask ORACLE for the C++
detector** over the corpus. A detector bug and a scoring bug are otherwise indistinguishable from outside.

---

## 11. CONVENTIONS for v2 (owner ruling, 2026-09-12)

### Function headers: Doxygen, ABOVE the signature
v1 places its header block **inside** the function body, after the opening brace, in
`Parameters:` / `Returns:` form -- i.e. exactly where a Python docstring goes, transliterated. The owner's
own house style already specifies column-0 `/* ... */` **above** the signature, so v1 does not follow it.
Owner: *"I made those docstrings while working with python docstrings anyway... perhaps following language
conventions is more important."*

The four-part CONTENT structure is unchanged -- it is the part that makes the code read as the owner's:
**WHAT + units/sign -> WHY (citing the reference source line) -> gating/byte-identity -> cost.**
Only the scaffolding changes: `@param` / `@return` instead of `Parameters:` / `Returns:`, and the block
moves above the signature. Emoji anchors are kept.

### Adopted for v2 without further discussion
`'
'` over `std::endl` (v1 has **148** flush-per-call sites) - `constexpr` for every compile-time table -
`const&` for struct parameters - `std::array` over raw C arrays for typed tables - `[[nodiscard]]` on pure
query functions - `noexcept` where it holds - narrow the STORAGE, compute in `int` - **no mutable globals**
(already the v2 contract).

### Kept as-is by owner ruling
**C-style casts stay in bitboard arithmetic.** Owner's rationale was speed/correctness.
Correction on the premise, recorded so it does not propagate: **C-style and `static_cast` compile to
identical code -- there is no speed difference.** The real difference is that a C-style cast silently falls
back to `reinterpret_cast`/`const_cast` where `static_cast` would refuse to compile. For bit arithmetic
that is a non-issue, so the ruling stands; `static_cast` is used only where a cast could hide a type error.

### Scope
✅ **v2 and all NEW code.** ☠️ **`placement_and_piece_eval_v1` is NOT touched** -- it is the frozen
control for every A/B and its `250 / 35,310,778` byte-identity is the gate. Any v1 convention pass is a
separate, byte-identity-verified change, ideally after v2 wins or is abandoned, so a convention edit and an
eval change are never debugged together.
★ **Owner: treat this as a slow, general migration** -- the same conventions apply if/when the search
side is rebuilt. *"as we move towards a proper engine being reborn out of the old one."*

### Amended cache sequencing (owner ruling)
Layer A is built and validated **without** the pawn hash: the detector is checked mask-for-mask against
the Python oracle first, because a detector bug and a cache bug are indistinguishable from outside and
`getPPIncrement`'s side-effect hazard is exactly how that goes wrong.
★ **The hash is then folded in AFTER 2b** -- it applies to passer DETECTION too, so building it once
after both halves exist is cheaper than retrofitting it twice -- and **before** the games / final
verification runs. Gate: byte-identity between cached and uncached (E7).

---

## 12. RUNG 2a RESULT (2026-09-12) -- three terms ship, connected is PARKED

### SHIPPED: `PS_V2_MAG=100` with `PS_V2_CONN_MAG=0`
**-0.66% mean, worst -0.00% -- negative or zero on ALL SIX corpora.**

| arm | mean% | worst% |
|---|---|---|
| doubled + isolated + backward | **-0.66** | **-0.00** |
| only_doubled | -0.33 | -0.02 |
| only_isolated | -0.22 | +0.11 |
| only_backward | -0.16 | -0.00 |
| everything incl. connected | +2.66 | +8.37 |

### PARKED: connected / support -- harmful at EVERY magnitude and BOTH shapes
| probe | result |
|---|---|
| rank-keyed, MAG sweep 3..100 | monotone, no positive optimum |
| rank-flat (FORM 2), FLAT 30..250 | WORSE than rank-keyed (+2.06% at 30) |
| support isolated, 10..80 | monotone to zero; support OFF is best |
| base30 + sup10 | -0.33%, still worse than dropping it |

☠️ Decomposition: `flat_90` +2.85% vs `flat_90_nosupport` +0.35% ⇒ **`PS_V2_SUPPORT` was ~71% of the
damage.** And the reason is already on record from the same day: **v1's `pawn_chain_file_bonus` IS the
support term** (it fires inside the pawn's attack loop on own pawns). Our PSTs and structure already price
supported pawns, so SF's `21 * popcount(support)` -> 164mp per supporter is a straight double-count, up to
328mp on a doubly-supported pawn against a base of 90.

★ The ladder protocol's next escalation step after "tune its own constants" is "retune the NEIGHBOURS",
which it marks as a RED FLAG -- the fitted-around-neighbours signature the whole rebuild exists to escape.
So we stop and park rather than tune outward. **Re-try after 2b**: a feature can look worthless before its
companion exists, and the isolated<->passed lift of 2.57x says passers are the missing companion for the
whole pawn layer.

### ★★ THE GENERALISABLE RULE, and it is not about pawns
Three predictions failed in this rung, all in the same direction -- assuming a reference-derived term works
once rescaled: SF's doubled taper would win (it did not; v1's was marginally better), a magnitude optimum
existed (monotone to zero), rank-flat would fix it (worse than rank-keyed).
But rung 1 PAID, at ~+101 Elo. The difference is not shape and not scale:

**KS priced something the core had NO representation of. Connected/support prices what the PST ALREADY
carries** -- both answer "is this pawn on a good square with friends".
⇒ **The predictor of whether a rung pays is whether the core already REPRESENTS that information, not
whether the giants have the term.** ⚠️ The overlap matrix cannot see this: it compares terms WITHIN a rung,
never against the PST. A pre-rung check against the existing core is the missing instrument.

### ✅ The structural/positional CAP question (owner, 2026-09-12) -- answered from source
**Neither giant caps pawn scores.** SF11 adds `score += pe->pawn_score(WHITE) - pe->pawn_score(BLACK)` RAW
(`evaluate.cpp:788`); the only `min`/`max` in `pawns.cpp` pick the best king-shelter square and a distance
minimum, not score bounds. Ethereal likewise: bounded tables, no cap.
They achieve the same goal by keeping constants small and letting values be large only where large is
CORRECT -- SF genuinely lets a connected 7th-rank pawn reach ~2 pawns. ★ **A cap cannot distinguish
"correctly large" from "mis-scaled large".**
⇒ **No cap in v2**, and today is the argument: connected was ~13x too large and the cap-free design
surfaced it in ONE run, where v1's clamp had hidden the equivalent for years (chain/struct masked 14x/23x
with their signs INVERTED). A cap bounds the OUTPUT instead of fixing the INPUT.
⚠️ The owner's underlying concern is still real and is met three targeted ways instead: small a-priori
reference constants, the E-series loop (which just caught a 13x error), and per-term magnitude knobs so any
single term can be bounded alone. ★ And the real pathology in v1 was never "a clamp" -- it was a clamp on a
**SUM OF TWO COMPETING QUANTITIES**. Bounding one term is defensible; bounding a sum is what made it
pathological.

---

## 13. CORRECTION (owner, 2026-09-12) -- the rule in section 12 was OVER-STATED

### A PST prices a SQUARE. It cannot price a RELATION.
Section 12 concluded "connected/support prices what the PST already carries". ☠️ **That cannot be right.**
A PST says "a pawn on e4 is worth X"; it structurally cannot say "a pawn on e4 DEFENDED BY d3 is worth
more than one that is not". It captures only the correlated part (supported pawns tend to sit on decent
squares). **The measurement stands; the explanation does not**, and it was written into the log and memory
as if settled. Owner caught it.

### ★★ The better candidate: TERM EXCLUSIVITY. The references DISAGREE.
Ethereal's pawn terms are an **if / else-if CHAIN** -- candidate-passer, else backward, else connected --
so its categories are MUTUALLY EXCLUSIVE, and it **explicitly denies the connected bonus to PASSED pawns**:
```c
else if (pawnConnectedMasks(US, sq) & myPawns)
    pkeval += PawnConnected32[relativeSquare32(US, sq)];
```
SF11 instead applies `Connected[]` to EVERY connected pawn including passers, and its `passed()` reads
rank / king proximity / path safety / blocker defence / file but NOT support or phalanx -- so in SF a
connected passer collects BOTH bonuses **additively**.

| | a connected PASSER receives |
|---|---|
| SF11 | passer bonus + connected bonus, ADDITIVE |
| Ethereal | passer bonus ONLY -- connected excluded by the else-if |

⇒ ★ The owner's intuition (connected passers, and phalanxes of passers, are worth more) is **backed by SF
and contradicted by Ethereal**. Per `adopt-reference-methods-only-if-universally-superior` that is a genuine
two-way split, so it is OURS TO SETTLE BY MEASUREMENT at 2b -- not a closed question.

### ⚠️ The exclusivity test run today was a NO-OP, and that is itself a result
`PS_V2_CONN_EXCL=1` (connected excludes backward) produced output **identical digit-for-digit** to the
additive arm on all six corpora. Reason: **`backward` and `connected` are ALREADY mutually exclusive by
construction** -- backward requires NO friendly neighbour at or behind, connected requires exactly such a
neighbour. Their intersection is empty, as is `isolated` with `connected`.
✅ So our detector already matches Ethereal's else-if semantics for those pairs for free.
☠️ But it means **the exclusivity hypothesis is UNTESTED, not refuted** -- I chose the one pair that could
not discriminate. The stacking that actually occurs in 2a is `isolated + backward` (lift 2.85x) and
`doubled + anything`; and since `best_no_conn` with both stacking is our BEST arm, that stacking is not
hurting. ⇒ **The real test requires passers. It belongs to 2b.**

### Connected remains PARKED -- nothing beat `best_no_conn`
| arm | mean% | worst% |
|---|---|---|
| `best_no_conn` | **-0.66** | **-0.00** |
| `conn_EXCL_m20` | -0.35 | +0.72 |
| `conn_EXCL_rank_sup0` | +0.36 | +4.55 |
| `conn_EXCL_flat90_sup0` | +0.35 | +2.42 |

### ▶️ What 2b must now settle (added by this correction)
1. **Connected x passed**: does a connected/phalanx PASSER earn extra (SF) or nothing (Ethereal)?
2. **Exclusivity for real**: with passers present, do our terms want to be a chain or a sum?
3. The parked connected term gets its re-try inside BOTH of those, not as a standalone retest.
