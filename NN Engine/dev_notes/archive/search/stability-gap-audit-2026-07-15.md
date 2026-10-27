# Stability-gap audit — why the SAME techniques are stable in strong engines, not ours (2026-07-15)

Reframe (user, 2026-07-15 eve): we are NOT missing the machinery (RFP/LMR/null/futility/razoring/LMP/TT all
present). Strong engines (Obsidian/Ethereal/Caissa — briefs in dev_notes/{obsidian,ethereal,caissa}-search-brief-
2026-07-14.md) run the SAME items but are STABLE; we have "a lot of random collapses". Find WHY the same item is
safe there and collapse-prone here. This is a comparative-conditioning audit, not a feature hunt.

## Ranked hypotheses (test via the collapse autopsy distribution + this audit)
1. **Flat vs conditioned prune guards.** Ours: LMP/futility/RFP/SEE flat-gated (move-count/fixed margins). Theirs:
   history- + improving-conditioned (Ethereal exempts high-hist quiets from futility/LMP, softens SEE by hist/128,
   cont-prunes on min(cmhist,fmhist); Caissa scales futility/SEE margins by statScore). SAME prune, richer guard.
2. **Eval reliability feeding static-eval prunes.** RFP/razoring/futility prune on static eval; ours over-reads ⇒
   same prune fires on bad info. (Re-test post-falsification — do not assume.)
3. **Ordering quality as a prune-safety MULTIPLIER.** EBF ~2 (theirs) ⇒ best move first ⇒ tail-prune/reduce safe.
   EBF 3.2 (ours) ⇒ tail sometimes holds the refutation ⇒ same LMR/LMP throws it away.
4. **⭐ UNIQUE-TO-US: non-negamax hand-mirrored min/max asymmetry.** Strong engines = negamax (ONE symmetric path).
   Ours = minimizer()/maximizer() hand-mirrored ~900 lines with DOCUMENTED asymmetries (min futility cur_depth>1
   gate :2529; NULLMOVE_CURDEPTH 3 vs 4; cur_depth==1 root-child branch has NO RFP/null/IIR/singular vs interior).
   Divergent prune conditions by side-to-move ⇒ inconsistent behavior ⇒ candidate for RANDOM collapses.
5. **Concrete bugs** (search-architecture-map-2026-07-15): root-razoring `break` (drops good root moves on a stale
   bound), root-child TT depth over-trust +1 (:2134/:2250), TT refuses score==0/mate, history pollution on
   fail-low best_move, live std::cout trap. Tested engines lack these.

## The "random" clue
Truly random (not position-type-systematic) collapses argue AGAINST #2 (systematic eval holes ⇒ predictable
position types) and TOWARD #4/#5 (asymmetry/bugs) or #3 (ordering variance). Weight the audit accordingly, but let
the autopsy LABELS decide.

## Method
A. **Autopsy distribution** (collapse-autopsy-design-2026-07-15.md) gives the mechanism histogram — the primary
   evidence for which hypothesis dominates.
B. **Comparative-conditioning table.** For EACH shared technique, extract from the briefs (+ further source reading
   of Ethereal/Caissa/SF which are open-source) the EXACT guards they wrap it in, side-by-side with ours:
   | technique | THEIR guards/conditions | OUR guards | gap |
   | RFP       | depth-scaled margin, improving, eval-reliability | flat 1500mp/ply, rd[1,6] | improving+scaling |
   | LMR       | log-based, history/statScore adjust, re-search | DEPTH_REDUCTION+statScore, full re-search | (close) |
   | futility  | improving, history-exempt | flat margins {200,450,650,950} | improving+hist-exempt |
   | LMP       | improving-scaled count, history-exempt | i>=1+rd^2, flat | improving+hist-exempt |
   | null      | verification search, zugzwang guard | table R, NO verify | verification |
   Fill this from the briefs + source; the GAPS are candidate stability fixes (each still gates fixed-TIME).
C. **Non-negamax asymmetry audit** (unique): diff minimizer vs maximizer prune conditions line-by-line; list every
   asymmetry; classify documented-parity vs latent-bug. If the autopsy shows side-to-move-correlated collapses or
   the BUG bucket is heavy, this is the lever. (Consider: could a negamax refactor or a shared-prune-helper remove
   the asymmetry class entirely? Big, but it is the structural root if #4 dominates.)

## Asymmetry remediation LADDER (if #4 proves load-bearing) — do NOT jump to negamax
The enemy is the hand-mirrored DRIFT (min/max prune conditions diverge as we edit one side), not "not-negamax".
Remediation in increasing cost/risk — take the cheapest that works, EVIDENCE-GATED:
1. **Asymmetry diff (cheap, diagnostic):** line-by-line minimizer vs maximizer prune conditions; classify
   documented-parity vs latent-bug. Measures how much asymmetry exists + whether load-bearing. DO THIS FIRST.
2. **Shared prune helpers (the 80% fix):** extract RFP/null/futility/LMP/razoring/LMR into ONE function each,
   called by both sides with a perspective/sign param. Kills the asymmetry-bug CLASS (one code path can't drift)
   AND gives write-once for every future lever (ends the mirror-twice tax). Incremental + flag-gated per device
   for byte-identity ⇒ safe/reversible. ~10% of a negamax rewrite's cost/risk. Keeps eval framing (Black-positive
   single root flip) untouched. PREFERRED remediation.
3. **Full negamax rewrite (LAST resort):** only if (a) autopsy confirms asymmetry/bugs are a MAJOR collapse
   source AND (b) shared helpers can't cleanly capture it. Separate branch. Forces per-node sign negation (bug
   risk). Validate: WAC/STS move-match PARITY (faithful refactor ⇒ ~identical) + fixed-TIME SPRT ≥ neutral
   (refactor, not feature) + MEASURED collapse-rate drop. Neutral-strength + fewer-bugs still counts IF a bug
   class is confirmed. Not worth it otherwise.
- **Pre-minimizer is SEPARABLE** from all of this (orthogonal to negamax — a negamax engine can still pre-search
  or use IID). Strong engines use internal iterative deepening/reduction instead of a separate pre-search, so it
  COULD fold in — but the engine "strongly depends on it" (clean removal needs real rework), so it's its own
  project on its own A/B, NOT a rider on the negamax change.

## ASYMMETRY DIFF — PRUNE GATES (done 2026-07-15 eve; single-core, no engine)
Min (get_score_for_minimizer) vs max (get_score_for_maximizer) prune-gate conditions:
- **LMP** (:2476 vs :2822): IDENTICAL ⇒ symmetric.
- **RFP** (:3459 vs :4079): correctly sign-mirrored (+margin<=alpha vs -margin>=beta), same depth/window guards ⇒ symmetric.
- **Futility formula** (:2531 vs :2877): both eval_by_mode(FUTILITY_EVAL_MODE); sign-mirrored comparison ⇒ symmetric.
- **ASYMMETRY 1 — null-move onset**: min cur_depth>=3 (NULLMOVE_CURDEPTH_MINI) vs max cur_depth>=4 (MAXI). Null
  fires one ply EARLIER on min nodes.
- **ASYMMETRY 2 — futility depth gate**: min has extra `cur_depth > 1` (:2529); max lacks it (:2875) ⇒ at
  cur_depth==1, min skips futility, max wouldn't. (max rarely at cur_depth==1 in normal flow, so maybe moot.)
- Documented (architecture map): root-child cur_depth==1 min branch lacks RFP/null/IIR/singular; root-razoring
  `break`; min PVS `i==0||rd==1` vs max `i==0`.
**PRELIMINARY CONCLUSION (prune gates only):** the mirror is TIGHT — gates nearly identical, only 2 small
deliberate-looking diffs. ⇒ WEAK evidence that asymmetry drives random collapses ⇒ a full NEGAMAX REWRITE is
likely NOT justified by asymmetry. The small asymmetries are testable by ALIGNING KNOBS (NULLMOVE_CURDEPTH_MINI=4,
neutralize the futility cur_depth>1 gate) + rebuild + measure collapse rate — a cheap experiment, not a rewrite.
CAVEAT: only PRUNE GATES checked. Asymmetry may still live in LMR reduction amounts, re-search/PVS conditions,
extensions, TT/ordering per side — extend the diff there before a final negamax verdict. But prune-gate tightness
+ the autopsy's bug/side-correlated bucket together decide it; so far both point AWAY from a rewrite.

## Sequencing
Autopsy FIRST (labels the collapses) → then this audit targets the dominant bucket. If EVAL-MIS-RANK/DRIFT heavy
⇒ eval calibration (+ the reopened eval bricks). If PRUNE-culprit heavy ⇒ the conditioning table (#1). If BUG/
asymmetry heavy ⇒ #4/#5 (correctness). Don't guess — the distribution picks the lane. Cross-engine research
continues as needed (further source reading of the open-source engines for exact guard formulas).
