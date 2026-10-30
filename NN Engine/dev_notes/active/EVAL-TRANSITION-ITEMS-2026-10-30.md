# Eval items for the transition arc — where the gap is and what the reference engines do (2026-10-30)

The eval leg of the transition arc after gen 6.4. Owner's direction (10-30): narrow the items with the data, comparing term
breakdowns with SF11 only where SF11's static read agrees with SF18's searched value; then see how the reference engines do each
item. Time box ~1-2 weeks of idle box time for items with no signal; a positive item earns its follow-through (memory
`time-box-is-soft-promising-items-earn-time`).

## 1. Where the gap is (OPTIMIZATION_LOG 10-30, "WHERE THE EVAL GAP IS")
SF18 WDL label, 6,000 own-play quiet rows, gen 6.4, `_triangulate_sf11.py MODE=egshare STAGES=1 AGREE=<pts>`.

- **Endgame (eg stage)**: the largest share of our excess at every threshold (41% at AGREE 10). SF11's scaling and initiative
  close it. Scaling HURTS one stage earlier (egl +6.6 … +8.2) ⇒ gate to light endgames.
- **Middlegame**, where SF11 is reliable (45% of mg rows at AGREE 10): 26% of the excess, about half of it king safety
  (−12.4%). On all mg rows SF11's king safety would hurt (+3.1) ⇒ a subset signal.
- **Early endgame (egl)**: 16%; passed pawns close a little (−2.6).
- Ours is better on material, imbalance, placement, mobility and (since gen 6.4) threats.

## 2. Reference contrasts (three engine-contrast reads, code only; Ethereal / Weiss are SF-descended ≈ one data point)

### 2a. Endgame scaling and initiative
- All references gate scale factors by MATERIAL CLASS; phase enters only as the eg weight. SF11 / SF15.1 / Ethereal apply the
  scale to the eg LEG; ours scales the blended total × eg share (`win_scale_adjust`, eval_v2.cpp).
- Our pawn ramp `27 + 34·pawns` reaches 64 (no scaling) at 2 pawns; SF11 `36 + 7·pawns` and Ethereal `96 + 8·pawns` at 4.
- SF11 / SF15.1 / Ethereal initiative is an ABSOLUTE soft threshold on the eg leg: an edge smaller than the drawishness becomes
  exactly 0. A multiplicative scale cannot zero a small edge unless f = 0, and 97% of our eg excess is on rows SF18 calls level.
  Our `WIN_V2` port lost ≈ −24 Elo because it also ADDED when complexity > 0 (discontinuous at a level score); the damp-only
  form (SF11's own mg-leg shape) is untried.
- Ranked candidates: (1) damp-only drawishness threshold on the eg leg, light endgames only, pure pawn endings excluded (code,
  small) · (2) impure-OCB cap `POT_V2_WIN_OCBX` (4/4; knob) · (3) queen-vs-none cap `QNOQ` (knob; may want a phase gate) ·
  (4) pawn ramp to 4 pawns, eg-gated (code for the gate) · (5) lone minor cannot win `LONEMINOR` (knob; fires 0.16%) ·
  (6) rule-50 damping (SF only; needs the halfmove clock in eval).
- Queue #119 games caps 2, 3, 5 + ROOKE at SF15.1 values. Static, caps on: eg-stage MSE 290.8 → 273.9 (−5.8%), egl −0.5%,
  mgl +0.9%, mg flat; our excess over SF11 −9.5%; the eg scaling swap falls from −12.7% to −2.4%.

### 2b. Middlegame king safety (owner hypothesis: tuning, since we have the inputs)
- MIXED. 12 of ~16 SF11 king-danger inputs exist with the same definitions, 10 fitted non-zero, and the 10-03 joint fit already
  moved toward SF11's weights but was under-determined (+6 ± 9 calibrated) ⇒ plain refits have had their chance.
- What weights cannot express: (1) shelter does not feed danger — KS-B is a separate direct score (eval_v2.cpp, "not a feeder");
  SF11 subtracts mg shelter inside kingDanger; 3/4 references couple them; never measured in v2. (2) The eg leg is 66% of the same
  saturating curve; SF11's eg leg is linear in danger (a different function), and the 09-21 diagnosis said to condition it on
  attacker presence, not phase. (3) Pinned defenders count as full defenders (`KS_V2_PIN_DEF=0`); our fit flipped BLOCKERS to −20
  (SF11 +98) — the symptom. Built, never run. (4) Saturating vs unbounded curve: tried three ways, leave it.
- Ranked: (1) `KS_V2_PIN_DEF=1` + BLOCKERS refit (no code) · (2) shelter → danger coupling, one knob from 0, fitted jointly with
  the KS knobs (code) · (3) eg leg as its own function conditioned on attackers (code).

### 2c. Phase taper and the early endgame (owner: split mg/eg pairs, or better formulas?)
- Pairs are already done: every shipped term except KS-A (eg = 66% of the mg curve), Kaufman (flat) and the POT scale (on the
  blended total) carries its own (mg, eg) legs, blended per site by phase256 — equal to one end blend within 3 mp. v2 has no
  phase cliff (the 3-way boolean cliff is v1-only). No reference has a distinct egl value either: egl is linear interpolation.
- ⇒ the egl gap is SHAPE, not pairing: the scale on the blended total, the KS eg leg, and (smallest) passers.
- Ranked: (1) static per-stage sweep of the existing eg-leg knobs `THREAT_V2_EG_PCT`, `MOB_V2_EG_PCT`, `KS_V2_EG_PCT` on egl rows
  (no code; static only, games decide) · (2) KS two-leg shape (= 2b item 3) · (3) the caps (= queue #119) · (4) `EVAL_V2_PAIR=1`
  with the scale moved onto the eg leg — SF's exact object (small code; the pair accumulator passed non-inferiority 09-25).

## 2d. Closed reference items worth a re-open, by the threats lessons (record check 10-30)
Owner (10-30): items from the reference engines that failed may deserve a new look now that threats passed. Threats' five
lessons as a checklist: (1) closed on an instrument we now distrust (cp label, d10 / depth-target screens, STS, void `Knob_`
matches, v1-only); (2) our port changed the form (continuity, leg, mg/eg split, phase vs class gating); (3) never tested with a
tighter margin or the search consumer the reference has; (4) tested where a search defect blinded the region, or on a much older
v2; (5) magnitude fitted on the wrong label / depth. Of ~30 reference items on record, 14 meet ≥ 2 conditions:

| item | closed on | conditions | status |
|---|---|---|---|
| WIN_V2 initiative (signed, additive) | −24.9 / −22.9 Elo, SF18 @250k fresh seeds; fits on d6 outcomes and on the cp label grew edges | 2, 3, 5 (+1) | scheduled: damp-only threshold (§3 item 3) |
| **Material taper `EVAL_V2_PAWN_MG` ≈ 550-650** (Ethereal's ratios) | largest static effect of the rebuild (§I −6.59%), d7 regret / depth target null; **never gamed** | 1, 3, 4, 5 | **unscheduled**; register: games with the margins co-swept |
| `improving` as a pruning modulator (9/9) | v1, LMR-only, ≈ −7 ± 20 | 1, 2, 3, 4 | SEARCH V2 |
| Null-move eval gate + eval-R (9/9) | v1 node-adder, sign-unstable | 1, 2, 3, 4 | SEARCH V2 (null reform) |
| Corrhist CORR2 | quarter weights +5.5 ± 14 over 2,000 g; margins tuned without it | 3, 4, 5 | final retune |
| Space (SF11 gated form) | proxies only; never gamed in v2 | 1, 3, 4 | dynamic lane after the transition |
| KS-B cells re-priced on WDL + shelter → danger | SF18 −13 vs self-play +16 (drifted judge) | 1, 2, 5 | coupling scheduled (§2b) |
| PX cells (rook / queen behind a passer, stop-square ladder) | d10 screens; games ≈ −1 ± 7 | 1, 3, 4 | unscheduled; judge on eg_critical with FIX_LMR_EG on |
| Singular extensions · IIR | v1, outside the consensus margins / wrong node population | 1, 2, 4 | SEARCH V2 |
| Queen-vs-none (was additive MCL, mg-weighted) | −2.4 ± 25 | 1, 2 | queue #119 (QNOQ cap) |
| Mobility cells / eg leg · KPROT / KFL / Kaufman depth re-fit / PST pair | instruments disagreed, shared-baseline artefact | 1, 3, 5 | scheduled (10-29 proposed additions) |
| v1 pruning forms (quiet SEE prune, RFP depth cap, LMP improving-halving) | v1, harsher than consensus | 1, 2, 4 | margins re-sweep / SEARCH V2 |

Owner (10-30) on the search rows: they are re-done in the search arc (parts of the search may be rewritten), but with the eval
fresh in mind. `improving`, the null-move eval gate, RFP at the node, futility in its parent-eval form and corrhist all consume a
NODE-ENTRY static eval, which our search does not compute today (futility evaluates the child after the move; v1's `improving`
needed extra eval calls, and the cheap material + PST trend was null in games, −7 ± 20 over 1,658). ⇒ for the search arc: one
node-entry static eval, lazy / staged (cheap terms first, the expensive attack-based terms only when a consumer needs them) and
reused through the TT or eval cache, is the shared foundation these consumers wait on — the owner's lazy / incremental eval idea.
Transition-side preparation: the planned re-profile on gen 6.4 should report per-term eval cost, and how well a cheap partial
eval tracks the full eval's ply-to-ply trend.

Stay closed (clean, none of 2-5): rook files, weak-unopposed, tempo, Kaufman verbatim tables (superseded), KR-vs-minor (tablebase),
`MOB_V2_SAFE` / `EXCL_QUEEN`, flat bishop pair, quiet checks in qsearch, `FUTILITY_EVAL_MODE` / `QSTANDPAT_EVAL_MODE`, root razor
margins, `PE_PRUNE_GATE`, `NULLMOVE_PROGRESSIVE`, threats 100/100.

## 3. Order (proposed 10-30; games decide every item)
1. Caps — queue #119 (running). 2. Search: FIX_LMR_EG — queue #120 (chained). 3. Damp-only drawishness threshold (code).
4. `KS_V2_PIN_DEF=1` + BLOCKERS refit, then shelter → danger. 5. Scale onto the eg leg (`EVAL_V2_PAIR=1`), pawn ramp to 4.
Each positive item gets a tighter-margin (crank) arm, a fresh-seed replication and the ship checks.
6. Lazy eval exit — a dedicated item once the eval items are done and retuned (owner 10-30), before the margins re-sweep;
   research and measurement plan in `LAZY-EVAL-RESEARCH-2026-10-30.md`.
