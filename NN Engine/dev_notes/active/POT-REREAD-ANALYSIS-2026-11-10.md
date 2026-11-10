# POT reread: analysis and proposals (2026-11-10)

Answer to `POT-REREAD-HANDOFF-2026-11-10.md`, written in the research chat (Fable 5.2). Nothing was built or run; no
engine code was touched. Every proposal is for review by the owner and the working chat before anything is built.

Tags: **[record]** = read in the dev notes or memory (file and section given); **[code]** = read in the source (stable
symbol given); **[reasoning]** = my inference; **[general]** = chess-engine knowledge from memory, not checked against a
source here.

## 1. The short version

1. **POT has two halves, and each was tested with an instrument that cannot see it.** Magnitude ("how unresolved") is a
   statement about how far the eval can be *wrong*, in either direction. Every fit on record asked whether a POT feature
   moves the score the right way on average. A feature can predict the size of the error perfectly and still have zero
   correlation with its sign, and then every one of those fits reads null. The magnitude half has never been tested on
   its own terms.
2. **The direction half ("who benefits") as a static score is the part I think is unlikely to work.** Three independent
   reads agree on about zero, including the one that looked at the realised future. Who benefits from a break is decided
   by concrete moves, which is the search's job.
3. **So the natural split is: magnitude tells the search how much to trust the eval here; direction becomes a class of
   moves the search should not throw away.** Neither adds a number to the eval. This fits the owner's own rules better
   than any eval form did ("score the option, not the execution"; "a nudge toward what search would find").
4. **Two cheap offline tests can stop the whole idea before any engine work** (§5, tests A and B). Both use data already
   on disk. If structure does not predict the size of our error, or the value of extra depth, search-side POT has nothing
   to stand on and should be closed.
5. **Two eval items still deserve a test, both in the form that has worked (a multiplier on an existing edge):** "the
   leader cannot create a passer", and its generalisation "the leader cannot change the structure at all". They belong to
   the focused eval arc after search v2, not now.

## 2. Why the eval fits read null (reasoning, with the record it rests on)

### 2.1 Magnitude is a second-moment quantity

- **[record]** The owner's 10-15 decision: "Complexity = the reliability of the other systems' verdict ... unresolved ⇒
  the regular terms read UNRELIABLE, not 0 ... It never adds score on its own" (POT brief §5, item 2).
- **[reasoning]** "Unreliable" means the spread of the error is large, not that its average is off. A term that
  estimates spread cannot lower the mean squared error of a score unless it is used to shrink the score toward zero,
  and shrinking helps only where the eval is over-confident.
- **[record]** That one use was tested (the complexity scaler, POT brief §5k) and at depth 12 there was nothing left to
  shrink: the candidate was already slightly over-confident by +0.9 on average and the middlegame set was
  *under*-confident by −2.0 (§5l). So the only eval-side consumer of "how unresolved" was dead on arrival at play depth.
- **[record]** The middlegame features were fixed as "Black − White" (POT brief §5l), and the 10-01 screen used "net
  White − Black gates" (`POT-TYPE-DEFINITIONS-2026-09-30.md`, depth screen). **[reasoning]** Netting destroys magnitude
  by construction: a position where both sides have a live break is the most unresolved kind and nets to zero.

So the magnitude half is not refuted and not unreadable either: it was never asked the right question. The right
question is "do these features predict the *size* of our error?", with the features **summed** over both sides.

### 2.2 Direction as a static score

- **[record]** Middlegame fit (§5m): −0.46% beyond a calibration control; the realised-future targets (SF18 at t+16 and
  t+24) −0.50% and −0.55%; move test z −0.77; pending-break features about zero weight.
- **[record]** Boosted trees (§5n, §5o): with or without the POT features the result is the same (−10.17% without them
  against −9.96% with), and after the full-size check on SF's own win model "POT's hand features still add nothing".
- **[record]** Race terms (§5b, §5i): the race "knows WHO wins the race, not WHICH MOVE does" (52.5% pair ordering,
  about blind), and its edge is gone at depth 12 (§5f).
- **[reasoning]** These agree for a reason. Whether a break works is a concrete fact (does the recapture hold, who gets
  the file first). A static feature can say a break *exists*; the sign of its value averages out near zero across
  positions, and whatever average remains is already carried by space, mobility and the piece tables.

### 2.3 Where I agree and disagree with the prior reading (handoff §8)

- **Agree:** the middlegame fit added a root-position term to a stored depth-12 score (§5l: "nested on the d12 score").
  That is the proxy the record says reads dynamic terms backwards (memory
  `root-delta-depth-proxy-is-biased-against-dynamic-terms`). No POT middlegame feature has ever been inside the eval
  during a real re-search, because no C++ was built.
- **Disagree on the label:** the material-scaled cp label matters most where material varies across rows (endgame
  classes: the complexity scaler, the C64 cells). It matters little for the middlegame fit, which had a calibration
  control, and not at all for the pair move tests, which compare sibling moves.
- **Disagree on "feeder form untested":** a feeder into an *additive* leg (pawn structure, passers) is an added score
  under another name. The form distinction is real only where the consumer is non-linear: the winnability multiplier,
  or king danger. So M1 and M3 as eval terms were in effect tested by §5m ("v2-valued levers now / after a push, by who
  can act" is M1's outcome-resolved feature, and it took about zero weight).

## 3. Question 1: refuted, or only unreadable?

| piece | my verdict | why | cleanest re-test, if any |
|---|---|---|---|
| Winnability, additive | **Refuted (the form)** | games on fresh seeds; no label involved | none |
| Winnability, multiplicative; C64 | **Positive, stands** | games; `ONEMINOR` is the open half (WDL says mis-calibrated, games say it helps) | the final retune on WDL, as registered |
| 10-01 middlegame types as added scores | **Resolved as a signed score; never asked as magnitude** | netted gates; signed target | test A below (summed features, size of error) |
| King-path race (direction, move test) | **Refuted as move discrimination** | sibling comparison, label-robust; edge gone at d12 | none as an eval direction term. Its other use (a truer static buys pruning) keeps its parked trigger: race + margins at equal time, after S4 |
| Complexity scaler | **Unreadable on the label, and moot at d12** | endgame rows span material classes, where the label error is largest; but the bias it targeted was gone at d12 | not worth repeating. Replaced by test A |
| Middlegame POT, linear | **Close to refuted as a signed static score; unreadable as an in-search term; never asked as magnitude** | §2.2 and §2.3 | not as an eval term first. If the search-side detectors get built (§6), an eval leg at weight 0 can then be read by a real depth-10 re-search almost for free |
| Boosted-tree ceiling | **Stands at half size** | −7.26% on SF's win model at full size; "still mostly MAGNITUDE" (§5o) | none here. Note it supports §2.1: what is learnable is how confident to be by position type, not who is better |

## 4. Question 2 and 5: the search-side form

### 4.1 What each half becomes

| POT half | search object | what reads it |
|---|---|---|
| Magnitude, U (one number per node) | how much to trust the static eval at this node | reverse futility margin, futility margin, null-move gate or reduction, razoring, base late-move reduction, aspiration width, time |
| Direction (a class of moves) | "transformation moves" the search should not discard early | quiet ordering, late-move reduction (reduce less), late-move pruning and futility (exempt) |

### 4.2 The mechanism it would have to fix (hypothesis, not verified)

**[reasoning]** A slow plan such as game 7's …Rg7, …Rg8, …gxh5 is a chain of quiet moves. Each sits late in the move
list because history has no reason to like it yet, so each is reduced. Three such moves in a row, each reduced by two or
three plies, need a nominal depth far beyond 18 before the line is searched to the point where king safety turns
"kinetic" and the eval sees it. The eval is not wrong at the end of the line; the search never gets there with enough
depth left.

**[general]** Our search today prunes little (branching factor flat at 1.9, search v2 plan §6.1). Search v2's S5 is meant
to bring that toward Stockfish 11's 1.46 by reducing late quiet moves much harder. That makes this blind spot **larger**,
not smaller. If the mechanism is real, the exemption is worth more after S5 than today.

**A discrepancy to settle first [record]:** `pot_owner_cases.txt` annotates game 7 as "⇒ EVAL" (depth 13 and depth 18 both
miss and read +1.1), while POT brief §5m says it "now reads as a DEPTH / pruning case". The triage tool's own definition
(`_case_triage.py`: EVAL means our static disagrees with SF at the end of SF's line) decides which. The planned re-run on
gen 7.1 (search v2 plan §7b) should report g7a and g7b explicitly, because game 7 is the only concrete case behind
"pending-break protection".

### 4.3 Why the standard parts do not already cover it

| standard part | what it reads | why it misses this |
|---|---|---|
| history, killers, counter moves | moves that already caused cut-offs in this search | a break that pays off 10 plies later has never caused a cut-off, because it was never searched deep enough. Chicken and egg |
| continuation history | the same, keyed by the previous move | same limit |
| pawn-keyed history **[general]** (Stockfish 16.1 and later index a history table by pawn structure) | which moves worked in this pawn structure | the closest standard relative: it is the learned version of "transformation knowledge per structure". Still learned only from lines already searched; not a prior |
| `improving` | static eval trend along the line | blind by definition: potential is what the static eval does not show yet |
| \|correction\| (Reckless candidate 2 **[record]**) | how wrong the eval has lately been for this pawn key and others | the closest competitor for magnitude, and it is learned, so it adapts. Also horizon-bound: it only knows about errors the search has already uncovered |
| check and singular extensions | checks; a TT move far better than the rest | unrelated to quiet build-up |
| null-move threat detection **[general]** | what the opponent does with a free move, at reduced depth | sees one-move tactical threats, not three-move plans |

**[reasoning]** So POT's only honest niche in search is as a **prior**: a cheap structural guess about where the search
has not yet looked. Precedent on our own record: threat-aware ordering failed through history (under-fill) and worked as
a static term (`THREAT_ORDER_W`, +21 twice, shipped gen 6.3). **[record]** (parked register, search phase rows.)

**Honest counterweight [general]:** Stockfish 11 reaches its branching factor with none of this, and no strong
hand-written-eval engine I know of has a lever move class. The large gains in search v2 are the standard parts. POT in
search is a possible few-Elo refinement of selectivity, and a distinctive one; it is not a substitute for S4-S5.

### 4.4 Can "unresolved" be computed at every node, and where would its precision come from?

**Cost: yes, cheap enough for every node.**
- **[code]** `PawnEntry` already holds `lever[2]`, `blocked[2]`, `attacks[2]`, `openFiles`, `halfOpen[2]`, `passed`,
  `candidate`, and lives in the pawn hash (`PawnHashEntry`, `pawn_hash_probe`). `passer_potential` exists.
- **[reasoning]** What would be added on a pawn-hash miss only: per side, the pawn pushes that create contact
  (`lever_push`), and the files a lever exchange could open (`break_files`, 8 bits). Per node: a few ANDs and popcounts
  against king files and heavy pieces. Per move: two ANDs (is this a lever push; does this rook or queen land on a break
  file). My guess is well under 1% of node cost; to be measured with the instruction-count yardstick.
- One dependency to check: the search needs the pawn entry at node entry, before or without a full eval. In search v2
  that is natural at S4 (node-entry static eval). The lazy-eval exit must not skip the pawn probe at nodes that use U.

**Precision: this is the weak point, and the record already says so.**
- **[record]** The loose gates fire in 70.2% of middlegame positions; lever reach in 44-60%. "Unresolved structure is
  the normal middlegame state" (`POT-TYPE-DEFINITIONS`, coverage study).
- **[record]** Lever reach did not even predict that files open: lift 0.97× for T1 and 0.94× for T3. Only the flank
  majority predicted its event (1.65×).
- **[record]** "Guard cost scales with breadth, benefit with marker precision" (memory
  `shadow-auc-evidence-map-and-protect-killers`: protecting killers cost +1.16% nodes, protecting the PV blanket +151%).
- **[reasoning]** So a yes/no "unresolved" flag is useless: it is on most of the time. U has to be graded, and only its
  top tenth or so can matter. And the move class has to be narrow, a few moves per node, which means restricting it by
  relevance: breaks that would open a file next to a king, or create a passer.
- **[reasoning]** Precision can come from three places, and they should be compared, not assumed:
  1. structure alone (the owner's rule; cheapest; weakest on record);
  2. structure × piece placement (heavy pieces behind the break, king on or next to the file);
  3. the search's own signals: |TT score − static eval|, eval swing between plies, |correction|.
  The owner's rule says U is judged from structure, never from the owning term's output. Source 3 does not break that
  rule (it reads the search, not king safety's score), but if source 3 alone does the job, then the standard parts
  already cover POT's magnitude half and the structural version adds nothing. That is a stopping point, stated in §5.

## 5. Proposals, in order, each with its stopping rule

### Test A: does structure predict the SIZE of our error? (offline, no engine)

- **What:** on the 26,244 middlegame trajectory starts with our depth-12 score and SF18's win model labels (queue #82
  data, POT brief §5o), regress the absolute error |win%(SF18) − win%(ours d12)| on the §5l features **summed over both
  sides**, plus a king-relevance version (breaks on files next to either king).
- **Controls:** |d12 score| (errors grow with the score), phase and material, and |static − d12| (the search-derived
  competitor that is available offline). Five folds by game, shuffled-feature null, as in `_pot_mg_fit.py`.
- **Second target:** |SF18(t+16) − SF18(t)|, the realised change. This is "how much did the position still change" with
  no engine of ours in the loop.
- **Keep:** beyond the controls, held-out positions in the top fifth of predicted U have at least 1.25× the mean absolute
  error of the bottom fifth, in at least 4 of 5 folds.
- **How it can mislead:** an absolute error is partly a function of the score itself (hence the control); trajectory
  starts are tactical (§5n says so), so repeat on quiet rows before believing it.
- **Cost:** an afternoon of Python on existing files. Adjust `_pot_mg_fit.py`; do not write a new probe.
- **My prior:** moderate that raw structure predicts error size; low to moderate that it survives |static − d12|.
- **Stop rule:** a miss here closes the structural magnitude half. Search may still use its own uncertainty signals
  (|correction|), which are already in the plan.

### Test B: is extra depth worth more where U is high? (offline, existing passes)

- **What:** on a set with our moves at two depths and SF18 multi-PV labels (the regret sets; the depth-10, 12 and 14
  passes on record), compute the regret removed by the extra depth, split by U fifths.
- **Keep:** the top fifth gains at least 1.5× the bottom fifth's regret reduction, replicated on a second set ("cross-set
  or it didn't happen").
- **Why it matters:** it is the direct justification for spending search effort or clock time by U. If depth is worth the
  same everywhere, there is nothing for U to steer.
- **Cost:** Python only if the passes exist for one set at two depths; otherwise one fixed-depth pass (safe during the
  owner's games).

### Test C: the plan-move audit on game 7 (small, engine, diagnostic only)

- **What:** at g7a and g7b, follow SF18's line. For each of Black's plan moves record, in our search: its index in the
  move list, the reduction it received, and the depth left when it was searched. Also the ply at which our static eval
  first agrees with SF along that line.
- **Reads:** if the plan moves sit late and reduced while our static agrees at the line's end, §4.2's mechanism is
  confirmed on the one case that motivated it. If our static still disagrees at the end, game 7 is an eval case (king
  safety's reading of the opened file) and pending-break protection loses its only example.
- **Cost:** part of the triage re-run already proposed in the search v2 plan §7b.

### Test D: do POT markers separate wrong pruning decisions? (needs a diagnostic field, working chat)

- **What:** **[code]** `ENABLE_PRUNE_SHADOW` / `ENABLE_SHADOW_EVENTS` and `_shadow_auc.py` already label wrong reductions
  and prunes and rank decision-point signals by AUC. Add fields: U at the node; "move is a lever push"; "move puts a heavy
  piece on a break file"; "the break file touches a king zone". Run on a **quiet middlegame** set, not WAC (WAC is tactical
  and says nothing about slow plans).
- **Keep:** a POT marker reaches AUC ≥ 0.60 with at least 200 wrong events (standard error about 0.02), and it still
  separates within strata of the existing leaders (move index, depth left, history). The record's rule: about 0.6 means a
  guard is buildable; nothing separating means a guard cannot work.
- **How it can mislead [record]:** every AUC is conditional on the population (the killer marker fell from 0.750 to 0.499
  once killers were protected); a better wrong-rate has not been shown to convert to Elo; and today's search is not search
  v2. Read it now as a feasibility check, and again at S5.
- **Cost:** a small diagnostic patch behind the existing flag, one fixed-depth pass of an hour or two.
- **Stop rule:** no POT marker above 0.55 on a quiet set, today and again at S5, closes the move-class half.

### Search-side candidates, only if A-D pass

| candidate | changes | stage | judged by | keep |
|---|---|---|---|---|
| S-1 lever term in quiet ordering | a static bonus for relevant lever pushes and for heavy pieces moving behind them, beside the threat term | S3 (the static quiet terms slot) | time to depth; then games | not slower to depth; games not negative; then it rides the S3 stack |
| S-2 reduce transformation moves less; exempt from late-move pruning | the sibling of `ENABLE_PASSER_PRUNE_EXEMPT` **[code]** | S5, as an arm of the reduction stack | `transform_critical.csv` (382) and `transform_trap.csv` (1,146) **[record]**, never engine-passed: finds SF's move more often on the first, no more premature breaks on the second; then games | node cost under +3%; critical set up, trap set not worse; self-play not negative, SF18 at equal nodes > −1σ (narrow-item rule) |
| S-3 U scales the pruning margins | reverse futility and futility margins widen with U; null move more careful | S4, beside \|correction\| as a 2×2 | games at equal time | U must add on top of \|correction\|, not replace it; otherwise drop U |
| S-4 U in time management | more time when root U is high | S6 | clock games | only if test B passed |
| S-5 race term + tightened margins | the parked trigger **[record]** | after S4, endgame critical venue | equal-time games where it fires | as registered in the parked register |

Set aside: quiet lever pushes inside the quiescence search. **[reasoning]** Pawn-takes-pawn is already a capture, so
live tension is resolved there today; adding quiet pushes widens the part of the tree that already takes a third of
search time, and a one-ply look does not reach a three-move plan anyway.

## 6. Question 3: dependencies in search v2

| item | needs | earliest |
|---|---|---|
| Tests A, B | nothing | now |
| Test C | triage re-run | S0, now |
| Test D | the shadow logger (today's search); a port of it to search v2 for the second read | now, and again at S5 |
| Pawn-hash masks (`lever_push`, `break_files`) | eval v2's pawn entry; result-identical at weight 0 | with test D |
| S-1 | the staged picker or the static-term slot in our move source | S3 |
| S-3 | node-entry static eval; pawn entry available at node entry; corrhist option for the 2×2 | S4 |
| S-2 | the per-ply stack (to know the previous own move, for "plan continuity" if tried) and the new reduction table | S5 |
| S-4 | the time manager re-check | S6 |

The history redesign matters in one way: if search v2 adds a pawn-keyed history table, measure S-1 and S-2 **with it on**.
It is the standard part most likely to absorb them.

## 7. Question 4: the untested eval items

| item | verdict | reasoning |
|---|---|---|
| **E1** leader cannot create a passer → winnability | **Worth a test; first in line for the focused eval arc** | It is a multiplier (the form that worked), structural, pawn-hash resident, and search cannot rescue it: "nobody can progress" only ends by repetition. **[record]** about a third of SF-drawn K+P positions read decisive at our depth 10; the scale is inert at two or more leader pawns. **[reasoning]** `POT_V2_WIN_PAWNEND=64` now *raises* confidence in pawn endings, so a false edge in a dead-drawn one is amplified: E1 is the missing guard for a cell already shipped. Judge where it fires: drawn pawn-ending starts against SF, pass = holds or accepts the draw; labels from SF's win model or tablebases. **[record]** caveat: caps and the damp both gained static accuracy and no Elo, so expect little Elo and judge it as a correctness item |
| **E1b** (new) the leader cannot change the structure at all | **Worth one offline look** | The owner's "meeting point" (preplan framing notes: "no potential left ⇒ drawish"). U = 0 for the side ahead (fully rammed, no lever reach, no candidate) feeds the winnability multiplier, graded by pieces. Check first in Python on the WDL rows: at equal edge, is the draw rate higher when the leader has no lever reach, beyond the existing cells? |
| **M2** closed-file openability correction | **Worth a cheap re-read, eval arc** | −5.0σ beyond controls on record, and it is a *subtractive* correction (the record: only subtractive king-safety changes have won). But it was measured on the static residual with the old label. Re-run that one screen on SF's win model before anything else |
| **M1** lever outcome as an eval feeder | **Unlikely as an eval term** | §2.3: in effect tested in §5m with about zero weight. Its test sets move to S-2, where they are the right venue |
| **M3** majority → passer, middlegame feeder | **Unlikely** | beyond-controls +0.3σ; the passer terms and search price it; middlegame passer legs repriced alone lost 22 Elo **[record]** |

## 8. What I think is unlikely to work, plainly

- Any additive middlegame POT score in the eval, in any gating.
- Direction terms for endgame races beyond what is parked.
- A yes/no "unresolved" flag of any kind.
- Structural U as a replacement for the search's own uncertainty signals. At best it adds to them.

What has a real chance: a narrow, king-relevant or passer-relevant transformation-move class in ordering and reductions
(S-1, S-2), and E1. What is genuinely unknown and cheap to find out: tests A and B.

## 9. Things in this note that are not verified

- §4.2's reduction-chain mechanism for game 7 (test C exists to check it).
- Every cost figure; none was measured.
- The **[general]** items: Stockfish's pawn-keyed history, null-move threat detection in older engines, the absence of a
  lever move class in other engines.
- Whether the #82 WDL labels and the depth passes are still on disk at the paths the brief gives.
- I did not read `TEXEL-C3-DETECTORS-DESIGN-2026-09-27.md` §8 / §17 directly; the concept is taken from the handoff, the
  preplan §0 and memory `ovd-is-the-owners-long-term-pressure-concept`, which quote it.
