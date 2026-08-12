# What a pawn is actually worth — SF18 ground truth, first trustworthy pass

Answers the redesign brief's design questions with data instead of argument. Method and instrument
validation: `diagnostics/pawn_truth_generator.py` (+ `pawn_truth_analyze.py`). 2,817 samples, SF18 d14,
ranks 5-7, files c/e, piece and kings-and-pawns backdrops, balanced-baseline conditioning.

**Unit: marginal value of ONE test pawn** = eval(position with it) − eval(same position without it), in
White-POV cp. Cell figures below are means; medians run ~20% lower with the same ordering everywhere.

## ★★★ THE HEADLINE — OBSTRUCTION DOMINATES STRUCTURE, AND RANK DOMINATES BOTH
| obstruction (all ranks) | mean | median |
|---|---|---|
| passed | **+341** | +254 |
| contested | +205 | +174 |
| piece_blocked | +173 | +137 |
| blocked (by a pawn) | +89 | +65 |
| opposed | **+71** | +49 |
A **4.8× spread from obstruction alone.** By contrast the whole strong-vs-weak axis is worth **+25 (r5),
+29 (r6, unresolved), +8 (r7, unresolved)**, and one rank of advancement is worth **+109 (5→6)** and
**+300 (6→7)**.
⇒ **What is in front of the pawn matters several times more than what is beside it.**

## ✅ Q1/Q3 — IS A WEAK PAWN ON THE 6TH BETTER THAN A STRONG PAWN ON THE 5TH? **YES, DECISIVELY**
| contrast | result |
|---|---|
| weak@6 vs strong@5 | **+80 ±33 — the WEAK ADVANCED pawn wins** |
| weak@7 vs strong@6 | **+316 ±75 — the WEAK ADVANCED pawn wins** |
⇒ The two axes do **not** cancel. One rank of advancement beats the entire structural premium, by 3× at
r5→6 and by 10× at r6→7. **Structure is a second-order correction to a rank-dominated value.**

## ✅ Q2 — DOES THE RANK GAP DEPEND ON WHICH RANKS? **YES, STRONGLY NON-LINEAR**
`5→6 = +109 ±20` · `6→7 = +300 ±45`. The gradient nearly **triples** in one step. Per backdrop:
kp `+115 / +315`, pieces `+83 / +220` — same accelerating shape, ~30% flatter with pieces on.
⇒ A linear or gently-curved rank table cannot represent this; the top two ranks are a different regime.

## ✅ Q4 — WHAT ELSE DECIDES IT
- **Blockade type matters, and NOT in the obvious direction.** A pawn blocker (**+89**) is far worse for the
  pawn than a piece blocker (**+173**). A blockading pawn is permanent and cannot be driven off; a minor can
  be challenged. ★ Our `passer_block_quality` reasons about *which piece* blockades but the biggest
  distinction in the data is *piece vs pawn*.
- **A blockaded 7th-rank pawn is worth +227 to +399** vs **+589 to +701** free. **Blockade costs ~300cp and
  rank alone never buys a rook** — the owner's prediction, confirmed. (Mine, +100 to +250, was too low.)
- **Phase:** kp **+218** vs pieces **+137**. Pawns are worth ~60% more with the pieces off.
- **File:** c **+212** vs e **+188** — small, and swamped by everything above.
- **Structure matters most exactly where the pawn is stuck:** the strength premium is significant only in
  the `blocked` column (**+66 ±44**) and unresolved everywhere else.

## 📐 WHAT THIS IMPLIES FOR THE REDESIGN (implications, not yet measured as changes)
1. A file × rank table indexed on **rank × obstruction** captures far more of the truth than rank × file.
   File is the weakest of the four factors measured; obstruction is the strongest.
2. The brief's worry about **overvaluing early pushes is not supported at r5-r7** — advancement really is
   worth that much there. The risk lives at r2-r4, which this run did not cover.
3. **Withholding beats penalising**, consistent with SF: a blocked 7th-rank pawn still outscores a free
   rank-5 pawn, so blockade should reduce a large bonus, never drive the pawn below an ordinary one.
4. Structure is a ~25-30cp correction. That is **below the ~20-40 Elo game floor on its own** — so pawn
   structure terms should be expected to pay only as part of a bundle, which is consistent with
   [[pawn-structure-penalties-fail-on-four-venues]] being inconclusive rather than negative.

## 🚨 INSTRUMENT VALIDATION — three defects, each of which produced confident wrong answers
| defect | symptom | fix |
|---|---|---|
| `unsupported` neighbours placed on r+1 | the test pawn **defends** them ⇒ strongest shape labelled "weak" | moved to r+2; then removed from the WEAK group entirely (still a latent supporter) |
| dropped samples with `\|value\| > 400` | **filtering on the dependent variable**; rank-7 passer read **+153** | removed; mates clamped to ±2000, not dropped |
| sparse backdrops | an extra pawn was decisive, not marginal; rank-2 pawn read **+453** | 3-5 pieces + 3-6 filler pawns per side |
| `blocked` could not exist on the 7th | the 7th was sampled ONLY as a free runner ⇒ looked unconditionally huge | added `piece_blocked` (a minor on the stop square), valid at every rank |
✅ Anchors now reproduce: rank-2 pawn ≈ a pawn (median +55..+124); free rank-7 passer ≈ 5-9 pawns.
⚠️ **Means are tail-dominated** (a balanced baseline still leaves decisive samples) — read medians.
⚠️ `unsupported` is NOT a weak context; a genuinely weak-but-advanced construction still needs building.
⚠️ Ranks 2-4 not yet measured; the early-push risk cannot be judged from this run.

## ▶️ NEXT
- Extend to ranks 2-4 for the full curve and the early-push question.
- Build a true `backward` context (neighbours behind AND stop square controlled) for a clean weak anchor.
- Cross-check the rank × obstruction table against SF11/SF15.1 classical term values and Ethereal's, to see
  which of these effects a hand-written eval can actually carry.
