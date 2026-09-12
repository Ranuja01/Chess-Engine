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

Consumes `passed[]` + `obstruction[]`. Prices blockade quality, king proximity, rook-behind, path safety.

☠️ **The detector RETURNS its masks. It must never publish them as a side effect.** `getPPIncrement`'s
publication is what kills a naive pawn hash — a cache hit would skip the write and every downstream piece
evaluator would read stale masks. Same family as `eval-global-side-effects-are-skipped-by-a-cache-hit`.

---

## 2. KEEP / REMOVE / CHANGE / MOVE

### ★ KEEP — and flag as deliberately OURS

| item | why |
|---|---|
| **Graded passed-ness** (`obstruction[]`) | `ppIncrement` is a CONTINUOUS obstruction score, not a boolean. **No giant has this** — SF's `passed` is boolean, priced afterwards. Ours separates one distant stopper from three near ones; `cpp_bitboard.cpp:893` records a pawn at 75 (below the 100 threshold) worth about as much as one above it. Do NOT flatten to SF's boolean |
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
| **2b** | passer DETECTION into the entry: graded `obstruction[]` + SF candidates + rear-doubled | detector-only test: does it flag the right pawns? No scoring yet |
| **2c** | `passer_value()` — blockade quality, king proximity, rook-behind, path safety | §I + games at the rung 2-4 checkpoint |
| **2d** | KS-B shelter, keyed on `pawnKey` + king square | closes the one UNIVERSAL channel KS-A lacks |

★ 2a and 2b are separable precisely because the detector returns rather than publishes. That separation is
what makes the "is the detector right?" question answerable without any scoring constant existing yet.

---

## 6. OPEN — to settle before or during 2a

1. ⏳ **Record-check the passer graveyard** (`passer-law-multiplicative-vs-additive-and-valuation-graveyard`,
   `passer-defect-is-blockade-cost-blindness`) against the Layer C design. NOT yet done.
2. ⏳ Do our pawn PSTs already encode a rank ramp steep enough that SF's connected rank keying double-counts?
   **Measurable**: correlate `Connected[r]` against our PST's per-rank pawn delta before choosing constants.
3. ⏳ `SCALE_PAWN_WALL` measured genuinely neutral (3/3 split). Does the concept survive as anything in 2a,
   or is it subsumed by connected + supported?
4. ⏳ Does `obstruction[]` survive contact with a correct connected/backward term, or was it compensating
   for their absence? ★ The honest risk to our one original: it may have been carrying a missing channel.
