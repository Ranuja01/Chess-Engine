# Collapse autopsy — per-losing-move attribution taxonomy + method (2026-07-15)

Goal: convert "eval or search?" into a LABELED DISTRIBUTION over real losing moves (from vs-Mediocre + vs-SF
collapses), attributing each to a mechanism. Reuses the prune-verification infra ([[prune-verification-methodology]]:
ENABLE_PRUNE_LOG / prune_verify.py) and the fact that EVERY prune is env-gated, so we can toggle-and-re-search.

## Step 1 — find the BLUNDER ply (not the peak)
collapses.csv dumps the peak→drop run-up window. The peak is a CALM position (useless — we agree with SF there).
Walk the window move-by-move; the blunder ply = where SF's eval swings against us by > T (e.g. 150cp) across ONE
of our moves. Autopsy the position BEFORE that move (we-to-move, still ~equal/better by SF).

## Step 2 — ground truth + our answer at the blunder position
- SF deep (d18-22): best move + eval = ground truth.
- Our engine (shipping config): our move + eval + PV.
- Classify the top fork:
  - **our move == SF best** → NOT a move blunder here → CATEGORY E (eval calibration): compare |our_eval − SF_eval|.
    Large over-optimism ⇒ we misjudge but play the right move; the loss is strategic drift — CONTINUE walking to
    the next ply (the real move error may be later, or it's death-by-accumulation of calibration error).
  - **our move != SF best** AND SF-best is much better ⇒ a MOVE-CHOICE error → Step 3 (why did we miss it?).

## Step 3 — toggle-recovery matrix (the culprit finder)
Re-search the blunder position with each lever INDIVIDUALLY changed; check if our move flips to SF-best (or our
eval of SF-best rises above our chosen move). The lever that RECOVERS it is the culprit.
| toggle (env knob)                    | if it recovers the move ⇒ category |
| ENABLE_RFP=0                         | SHALLOW static-eval prune cut the disproof (RFP)      |
| ENABLE_RAZORING=0                    | SHALLOW static-eval prune (razoring) — "earlier pruning" |
| ENABLE_FUTILITY=0                    | move-level futility cut the refutation                |
| ENABLE_LMP=0                         | late-move-count prune skipped the saving move         |
| ENABLE_NULLMOVE=0                    | null-move pruning (zugzwang/threat blindness)         |
| LMR_EXTRA=-N (or reduce)             | DEEP reduction under-searched the refutation — "deeper pruning" |
| deeper (MAX_DEPTH+2/+4)              | HORIZON: beyond effective depth (bushy EBF 3.2 ⇒ soft depth) |
| (root-only: root-razoring `break`)   | ROOT BUG: dropped good root moves on a stale bound    |
Decision:
- exactly one prune toggle recovers ⇒ that prune (shallow RFP/razoring/futility vs deep LMR vs LMP/null).
- only DEPTH recovers ⇒ HORIZON / ordering-efficiency (EBF) — the "search cuts important nodes via soft depth".
- NOTHING recovers (all prunes off + deeper still picks our move) ⇒ CATEGORY V: EVAL MIS-RANKS — our eval
  genuinely prefers the losing move ⇒ an eval feature/calibration hole (the pure eval move-choice error).
- root-only anomaly ⇒ CATEGORY BUG (root-razoring break / TT depth over-trust :2134/:2250).

## Step 4 — aggregate ⇒ the actionable map
Run Steps 1-3 over ~20-40 collapse blunders. The DISTRIBUTION decides the lever:
- heavy on RFP/razoring/futility ⇒ CONDITION the shallow static-eval prunes (but note: the RFP mechanism harness
  already found RFP prunes 99.7% correct on WAC/overread — so if collapses light up RFP, it is a DIFFERENT
  population than that harness sampled ⇒ re-open with the new labeled set).
- heavy on LMR ⇒ reduction is too aggressive on refutations (ordering can't support it — matches the LMR_EXTRA
  gauntlet lesson, but now MECHANISM-labeled).
- heavy on HORIZON/EBF ⇒ ordering/efficiency work (compress the tail so effective depth ~ nominal depth).
- heavy on EVAL MIS-RANKS ⇒ eval feature/calibration (this is where mobility/imbalance/KS bricks would help —
  and it explains why they help; validate the surviving bricks REDUCE this category's count).
- BUG hits ⇒ fix the concrete bugs (root-razoring break, TT depth over-trust) — cheap, correctness.

## Build (overnight)
`diagnostics/collapse_autopsy.py`: input a collapses.csv (or a list of blunder FENs), for each run the SF-truth +
toggle matrix + depth sweep via run_one/fen_vs_sf with the knob sets, emit per-position category + culprit, and a
summary histogram. Reuse fen_vs_sf.py's SF-compare + run_one. Single-core, deterministic (OMP=1). Prototype on
the mediocre_prelim collapse (game 5) then run on the 40-game mediocre_base collapse set.
NOTE the subtlety: a collapse is often NOT one move — it can be calibration drift (CATEGORY E repeated) with no
single "blunder ply". The walk in Step 1 must handle "no single >T swing" ⇒ label the whole game CATEGORY E-drift
(slow strategic loss from persistent over-optimism), which is itself a key finding (points to eval, not a prune).
