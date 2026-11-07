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
| PX cells (rook / queen behind a passer, stop-square ladder) | d10 screens; games ≈ −1 ± 7 | 1, 3, 4 | unscheduled; judge on eg_critical (FIX_LMR_EG failed 10-30, so on the shipped search) |
| Singular extensions · IIR | v1, outside the consensus margins / wrong node population | 1, 2, 4 | SEARCH V2 |
| Queen-vs-none (was additive MCL, mg-weighted) | −2.4 ± 25 | 1, 2 | queue #119 (QNOQ cap) |
| Mobility cells / eg leg · KPROT / KFL / Kaufman depth re-fit / PST pair | instruments disagreed, shared-baseline artefact | 1, 3, 5 | ✅ already in the engine: fitted jointly in the 10-11 structural retune (gen 2). Only the mobility eg leg stays open (§2c item 1) |
| v1 pruning forms (quiet SEE prune, RFP depth cap, LMP improving-halving) | v1, harsher than consensus | 1, 2, 4 | margins re-sweep / SEARCH V2 |

Owner (10-30) on the search rows: they are re-done in the search arc (parts of the search may be rewritten), but with the eval
fresh in mind. `improving`, the null-move eval gate, RFP at the node, futility in its parent-eval form and corrhist all consume a
NODE-ENTRY static eval. Ours computes one only at null-window nodes with 1-6 plies left, for RFP (`rfp_want_eval`; corrhist
corrects that value); futility evaluates the child after the move, and v1's `improving` made its own extra eval call (the cheap
material + PST trend was null in games, −7 ± 20 over 1,658). ⇒ for the search arc: one
node-entry static eval, lazy / staged (cheap terms first, the expensive attack-based terms only when a consumer needs them) and
reused through the TT or eval cache, is the shared foundation these consumers wait on — the owner's lazy / incremental eval idea.
Transition-side preparation: the planned re-profile on gen 6.4 should report per-term eval cost, and how well a cheap partial
eval tracks the full eval's ply-to-ply trend.

Stay closed (clean, none of 2-5): rook files, weak-unopposed, tempo, Kaufman verbatim tables (superseded), KR-vs-minor (tablebase),
`MOB_V2_SAFE` / `EXCL_QUEEN`, flat bishop pair, quiet checks in qsearch, `FUTILITY_EVAL_MODE` / `QSTANDPAT_EVAL_MODE`, root razor
margins, `PE_PRUNE_GATE`, `NULLMOVE_PROGRESSIVE`, threats 100/100.

## 3. Order (proposed 10-30; games decide every item)
1. Caps — **PARKED 10-31** (pooled ≈ +6 ± 14; with RFP 500 +7.0; 60+0.6 −8.7; trigger in the parked register). 2. Search:
FIX_LMR_EG — #120 FAILED in its venue (−19.8 ± 12.5 on eg_critical; the bypass stays, see OPTIMIZATION_LOG). 3. Damp-only
drawishness threshold — built (`WIN_V2_DAMP` / `_PHASE_MAX` / `_SKIP_PAWNEND`), static eg −2.9%; #124 alone −5.9 ± 19.9, **with
RFP 500 +15.3 ± 19.1** (crank ≈ 1.5σ) ⇒ next: fresh-seed replication + RFP 500 alone control. Passer exemption (UI game 2) on
eg_critical +4.5 ± 10.6, not a ship ⇒ search arc. **Closed 11-01 (#125-#127):** RFP 500 alone matched the crank, damp on top of
500 −20.9 head to head ⇒ damp PARKED; RFP 500 over three runs +2.3 ± 10.9, margin flat 400-600 ⇒ 600 stays. Lesson: the eval
leg's two endgame items improved static eg accuracy (−5.8%, −2.9%) and neither became Elo — the eval payoff is opening /
middlegame (memory `eval-payoff-is-opening-midgame-not-endgame`) ⇒ item 4 (middlegame king safety) next.
4. `KS_V2_PIN_DEF=1` + BLOCKERS refit, then shelter → danger. **Started 11-01.** Record check: PIN_DEF never measured
   anywhere (built 09-27, left out of every fit as a mode); BLOCKERS −20 is a bundle coefficient (10-03 joint fit), never isolated;
   the coupling had no knob. Done: PIN_DEF rebuild only when a non-pawn defender is pinned (result-identical, 250 / 36,176,814
   both ways; its ~5% bench cost gone) · `KS_V2_SHELTER_K` built, 0 = byte-identical, K 12 live (251 / 38,767,998; SF11's 6/8
   converts to K ≈ 11, a few % of the units). Games: #129 pin and pin + BLOCKERS 0 vs gen 6.4 (pin binary both sides), #130
   replicates the better if > 0. **Decision after #129/#130 (owner 11-01):** a positive pin arm ⇒ refresh the depth-fit base
   (our d10 on the 9,771 mg rows is from 10-01/10-04, four gens stale) and refit BLOCKERS + K lean on it; both flat ⇒ pin
   defenders and the coupling park, no fit infrastructure built for them. Other subsystems are not re-fitted now (structural
   retune closed; the full retune is the final one).
   **Read 11-01 (#129): pin +2.8 ± 19.0, pin + BLOCKERS 0 +0.7 ± 18.7 ⇒ pinned defenders PARKED**, no replicate (under the
   owner's +8 bar), no depth-fit refresh. The coupling gets its own read instead of parking with them: #132 runs
   `KS_V2_SHELTER_K=12` on UHO (and a replicate at the same bar). Caps and damp are re-judged on the transition starts (#131;
   replicates in #132 at +8).
   **Read 11-02 (#131 / #132, transition starts):** caps +12.9 ± 14.5 then +5.2 ± 13.9 ⇒ pooled +9.0 ± 10.0 (z 1.77), the
   first positive read anywhere but under 2σ ⇒ targeted capped-ending venue in #134 (ship checks only if that clears 2σ) ·
   damp −0.4 ± 13.9 (stays parked).
   **Read 11-02 (#134, capped-ending starts):** caps +9.7 ± 13.4 (Ptnml middle 60%, inside −5 … +30), replicate −1.0 ± 14.0 ⇒
   pooled +4.3 ± 9.7 (z 0.88): no ship checks. **Owner 11-02: caps PARKED** (the unselected replicates pool to +2.1 ± 9.9).
   **11-02 status of the eval leg:** every re-open from §2d with a games verdict has failed (damp, caps, shelter coupling,
   pinned defenders); the material taper worsens gen 6.4's static WDL (retune); the 10-07 cells are already in gen 2. Left:
   item 5 (endgame, low prior after caps / damp), the lean retune, item 6 (lazy eval exit), and two live defects with no fix
   built (queen-vs-minor compensation; KR vs K+minor read +1.5 pawns).
   **Read 11-02 (#132): shelter coupling K 12 −6.3 ± 19.9 ⇒ PARKED.** Item 4 closed with nothing shipped.
   5. Scale onto the eg leg (`EVAL_V2_PAIR=1`), pawn ramp to 4. **PARKED 11-07 (owner: "sure"), not run:** low prior after caps and damp, and it goes with the focused eval arc after search v2 (`EVAL-V2-PARKED-REGISTER`, last section).
Each positive item gets a tighter-margin (crank) arm, a fresh-seed replication and the ship checks.
Forecast registered 10-30 (self-play, vs gen 6.4; check it at the end of each leg): rest of the eval leg incl. the lean retune
and the lazy eval exit +15 … +40; the search arc +50 … +150, top end raised to +250 after the owner's argument (EBF: each step
toward SF's tree shape is several plies at equal time, IF the pruning is effective — prune more where safe, less where lines matter)
(two-thirds of the SF11 gap is node efficiency; STS300 at equal nodes 1,944 vs SF11 2,374). External gains expected at roughly half the self-play figure. Search arc start ≈ 2026-11-07 — **a soft time box on the eval
re-check so it does not run too long, not a deadline (owner 11-04): close-out items are done in full, never skipped to meet it.**
6. Lazy eval exit — a dedicated item once the eval items are done and retuned (owner 10-30), before the margins re-sweep;
   research and measurement plan in `LAZY-EVAL-RESEARCH-2026-10-30.md`.

**Close-out order of the transition (owner 11-04: "the big retune was part of closing off the transition first before going
into search, due to the fact that we added some new stuff").** The eval was last fitted before threats (gen 6.4) and before the
search changes of gen 6.5 / 6.6, so the retune closes the transition, not the search arc:
1. the open correctness items (the mate-score bundle #150 → gen 6.6; `ENABLE_TT_FLAG_FIX`, dead since 08-22) and banked speed
   (batch 9 + F15's real-play read);
2. **the retune** — the eval re-fitted on top of threats and the current search, at the play depth;
3. the static ladder re-check (owner 11-04: after the full retune steps; last run gen 6.4);
4. lazy eval exit (item 6), then the margins re-sweep last;
5. then the search arc.
**Search v2 scope (owner 11-04):** a long arc, more than rewriting the search — also tuning its knobs the way the eval was
tuned, with EBF as the target; correctness measured as nodes kept (pruning that is right where it prunes); and the technical
changes that support it (SEARCH-TRANSITION-PREPLAN §3 negamax migration, §4 speed, §5 the fair node-efficiency instrument).
