# Capgains + KS collapse session (2026-07-17, after SESSION-HANDOFF-2026-07-17)

Follows KS v1 ship. Investigated the "material over-read" collapse class; it dissolved into capgains-phantom +
KS-under-count + positional, with SF11-static exposed as an invalid yardstick. Built 4 levers (all gated
default-off, byte-id **248 / 42272840** at default). **Nothing new committed** beyond KS v1 (`73c7cad` +
`0140f7b`). Games (the only remaining gate for most of it) deferred — user gaming, 4 cores unavailable.

## The 4 levers (state)
| lever | knob (default) | deterministic result | verdict |
|---|---|---|---|
| **Pin** | `ENABLE_CAPG_PIN` (off) | drops illegal pinned capture from capgains; WAC 248→**250**, nodes −6.4%, but **STS −34**; reach **3/64** | correct-but-low-reach; **game-pending** |
| **Tempo** | `ENABLE_CAPG_TEMPO` (off) | resolution-loop: an *evasion* forfeits the evader's own pending capture (opponent gets the defensive tempo). Correct on P1-orig (material+capg → 0). WAC −3 / STS −10; reach **+4/64** | correct-but-low-reach; **game-pending** |
| **KS DEF** | `KS_SAFE_CHECK_DEF` (3→**5**) | defensive-asymmetric safe-check weight (side-to-move king only, via `king_safety_danger(...,defensive)` + `turn` in `king_safety_score`). **STS +16**, P2b −0.00→−1.44 ≈ SF11 −1.40, KS-term symmetry 0.0000, byte-id default. WAC −5 (fixed-node artifact — offense-only −4 AND defense-only −5 both cost it) | **clean positional win** |
| **KS_DEF_MAG** | `KS_DEF_MAG` (100) | percent multiplier on side-to-move king's *final* danger. Deepens P2b cleanly (−1.44→−4.32 at 300) with **0 extra calm leaks**, BUT **STS 1606→1473 (MAG150) / 1535 (MAG200)** — blunt amplification over-penalizes moderate king-danger → paranoia | **NO-GO, shelved at 100** |

## Conclusions (general)
1. **SF11-static is an invalid ground truth for attacking positions.** Its "Material" trace = `psq_score()`
   (PSQT, not raw count); it is blind to attacks only search resolves (P2 hanging rook: static Δ+0.28 vs
   search Δ+1.94; P3: static KS 0.00 while it's mate-in-5; P2b: static −1.40 while SF18 −6.12). ⇒ the
   collapse corpus's SF11-static over-read is **contaminated** — some "over-reads" are us being *right*.
   Use **SF18-search or games** as truth for attack positions. [[eval-collapse-diagnosis-method]] amendment.
2. **Deterministic suites can't judge capgains-*reduction* fixes.** capgains is **load-bearing for move
   selection**, so removing even *illegal/incorrect* credit (pin, tempo) reads as an STS regression (pin −34)
   while helping only ~7/64 collapses. Their net is a **games** question, not a suite question.
3. **WAC (tactical) ≠ STS (positional) — check both.** DEF=5 is WAC −5 but STS +16 (the matching metric).
   Fixed-node WAC penalizes eval-accuracy changes; STS is the right screen for positional/KS changes.
4. **The capgains-material class is low-reach** (pin+tempo concretely help **7/64**, all correct-direction,
   none over-corrected). The **KS-attack class is the bigger, cleaner lane** (DEF=5 = +16 STS).
5. **Our KS is ~10× under SF11's magnitude.** SF `king()`: safe-checks weighted **635–1080** units, quadratic
   (`kingDanger²/4096`), `>100` threshold (its deadzone, analog of our `KS_FLOOR`) → **one safe check ≈
   1.5–2.8 pawns** in SF. Ours: `KS_SAFE_CHECK=3` on a 0–80 scale → a fraction of a pawn. That under-magnitude
   is why a real counter-attack doesn't offset our (capgains-hot) material → collapses.
6. **"Counter capgains with strong KS" is the right shape — but it must be CONDITIONAL.** Keeping capgains at
   full power (preserves its STS benefit) and letting KS offset it *where it over-reaches* beats *reducing*
   capgains. BUT a **blunt** magnitude boost (`KS_DEF_MAG`) fails — it can't tell a decisive attack (should
   offset material) from a moderate one (shouldn't), so it over-penalizes everything (STS drop). The correct
   vehicle is **Item 3: realizability × counter-attack** — scale *material* down only in the co-occurrence
   (materially-ahead AND real attack), not blanket-amplify all king-danger.
7. **Item 2 tempo caveat (already satisfied):** the forfeit sits inside `can_evade(...)`, so if the threatened
   piece is *indefensible* (no safe escape / too many attackers), no evasion happens and capgains still fires.

## Nighttime game battery plan (when cores free)
Arms (each paired A/B vs SF@2400 vs the default baseline):
- **DEF=5 SOLO — priority arm.** It's the only clean deterministic win (STS +16), so isolate its game-level
  contribution first/foremost.
- **pin solo** and **tempo solo** — the bench-loss-risky capgains levers (pin STS−34, tempo STS−10); solo for
  attribution since the suites can't net them.
- **all-3 together** (pin + tempo + DEF=5) — the accumulated bundle.
Categorical verdict is **two-sided**: (a) target class shrinks AND (b) **KS-v1 class does NOT resurface** (a
resurfaced KS collapse = collapse-fix regression). Use `ks_collapse_attribute`. Judge by class, never raw total
(SF stronger ⇒ fixing one class exposes the next). If cores are tight, run **DEF=5 solo first** — likeliest win.

## NIGHTTIME GAME BATTERY RESULTS (2026-07-18, 200g each vs SF@2400, seed 0, conc3)
| arm | score | collapses | vs baseline |
|---|--:|--:|--|
| baseline (default) | 37.5% | 99 | — |
| **DEF=5 solo** | **47.8%** | **79** | **+10.3% / −20 collapses** ← DECISIVE WIN (~3 SE) |
| all-3 (pin+tempo+DEF5) | 40.8% | 84 | +3.3% / −15 |
| pin solo | 40.5% | 81 | +3.0% / −18 |
| tempo solo | 39.2% | 88 | +1.7% / −11 |

**⚠️⚠️ FINAL 3-SEED (600g) VERDICT — DEF=5 is NEUTRAL, NOT a win.** The seed-0 +10.3% was noise.
| metric | seed0 | seed1 | seed2 | 600g |
|---|--:|--:|--:|--:|
| DEF=5 score | 47.8% | 38.8% | 30.8% | **39.1%** (baseline 39.8% → **−0.7%**) |
| DEF=5 Δcollapses | −20 | −11 | **+26** | **−5 (FLAT)** |

**`KS_SAFE_CHECK_DEF=5` = game-NEUTRAL over 600g on BOTH score and collapses.** The "consistent collapse
reducer" read was 2-seed noise; seed-2 reversed it (score 47.8→38.8→30.8 = 17-pt range; +26 collapses).
**STS +16 did NOT transfer to games** — another deterministic-suite-≠-game-Elo case (cf. mobility +20 self-play).
**DO NOT SHIP DEF=5 on this evidence.** **pin/tempo:** net-negative in combination (all-3 dragged DEF5 down),
low-reach — DROP. `KS_DEF_MAG` — NO-GO. **Net: NONE of the 4 levers shows a confirmed game gain.**
- Every arm cut collapses (99→79-88) but only DEF=5 converted it to score. Categorical attribution
  (`ks_collapse_attribute`) is CONTAMINATED here — it classifies KS-caused by SF11-static KS≤−1.5, the invalid
  yardstick, so DEF=5's real defensive-KS wins land in "other" (−22, the biggest). KS-caused flat (8→10) ⇒
  **no KS-v1 resurfacing** (the two-sided criterion passes). Raw score + total collapses are the reliable signals.
- Seed-0 baseline (37.5%) ran below the KS-v1 gate's 42.2% ⇒ seed-1 confirmation run of baseline+DEF5 queued to
  guard against an openings fluke inflating +10.3%.
- **RECOMMEND (for user): flip default `KS_SAFE_CHECK_DEF` 3→5 and commit (narrow, KS-v1 style).** Not committed
  autonomously (commit-only-when-asked). pin/tempo stay gated default-off (or delete — game-confirmed no-go).

## Next build (next session)
**Item 3** (`REALIZ_KS_*`): couple the material edge / `PV_BOOST` ([cpp_bitboard.cpp:6011 realizability_factor],
[6949 PV_BOOST]) to the leader's king-danger; prereq = expose per-king danger in the search path (today
`g_ks_units_*` set only under the breakdown flag). Game-gated; guard with an "up-material-and-winning" control
set against over-damp. It's the conditional form of "counter capgains with KS."

## Tooling added this session (diagnostics/)
CAPG_DEBUG_DUMP (capture-sequence dump in `approximate_capture_gains`), ks_material_3fen(_sf), ks_pin_phantom,
ks_capg_dump, ks_shortfall, _ks_calm_safe, _ks_safecheck_sweep, _ks_sym_isolate, _ks_defmag_sweep,
_capg_corpus_reach, _capg_p123 / _capg_p2diff / _capg_p2sub / _capg_p1mod (SF11/SF18 position analysis).
