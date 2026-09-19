# Eval v2 — SLICE 3 DESIGN: central / space / threats / Kaufman + pairs

@author: Ranuja Pinnaduwage (maintained with Claude)

★ Read order: this §0 (record-check — what is ALREADY closed, and by which instrument) → §1 five-engine contrast (from source)
→ §2 design → §3 measurement plan. Slice 2's doc (`EVAL-V2-SLICE2-MOBILITY-DESIGN.md`) is the template; its §2.1b/§2.1c show the
form bake-off and the parked-with-trigger pattern this slice must follow.

Standing rules this slice runs under (unchanged, owner's charter): best definition per term from any engine or ours · one owner
per concept · efficiency (no second attack pass; reuse `SideAttacks` / `MobAcc`) · knobs default off and byte-identical ·
detector + oracle before any magnitude · ladder BOTH anchors · regret vs a neutral measured ON THIS BASE · games only when two
instruments that can resolve the term agree.

Base for every slice-3 measurement: **SHIP + placement E**, fingerprint WAC d10 `250 / 61,352,373 / EBF 4.114`.
Neutrals on that base: **primary 49.9 · `_v2` 51.1**.

---

## 0. RECORD CHECK (2026-09-15, before any design)

### 0.1 Verdict table
| term | verdict class | what that means for slice 3 |
|---|---|---|
| **space** | v1's FLAT form tried on the contaminated harness (non-monotonic, net-flat) ⇒ **UNREADABLE, not refuted**. SF's GATED form **never built** | ✅ legitimate build. ⚠️ the "died 3×" headline is a count of one unreadable null plus a proxy |
| **central** | v1's `central_score` is load-bearing but AMBIGUOUS (paired nulls −0.2..−1.6pp; `CENTRAL_BOUNDED_MODE=2` the only consistently negative arm) | ⚠️ **v2 has no feeder**: no heat map, no `central_score` global. Any v2 central is a NEW mechanism, not a re-run — but see the one-owner flag below |
| **threats** | v1 ship **+45.0 ±40.6** (388 games) — but on `gate`'s fixed openings, CI barely clears 0, and never attributed. SF-form extensions (`thr_corner` +1.7/+1.7/+0.5, `thr_hanging` +0.9/+0.8/+0.5) are sign-consistent 3/3 but ~1σ ⇒ **UNRESOLVED LEADS** | ✅ build, but as SF's FORM family, not a constants grid ("do not re-propose a constants grid without a new mechanism") |
| **Kaufman imbalance** | **WEAKLY POSITIVE on games** (6 seeds × 200g: +1.15% score, ≈0.85 SE, wins 4/6) — the only genuinely positive record of the four. STS jagged (−79) | ✅ port the census-product FORM; ☠️ NOT the tables (fit to v1's SF11 residual with v1's other terms live) |
| **bishop / knight pair** | **NEVER MEASURED ALONE** in v1 (the flat pairs are dead code behind `ENABLE_KAUFMAN_IMBALANCE`); `MOD_PAIR_OPEN` built, no result found | ✅ genuinely open. Prior v2 note "Kaufman owns pairs, no separate pair term" is a DESIGN choice, not a measurement |

### 0.2 ☠️ One-owner flags to settle BEFORE building (charter: never recreate v1's collinearity)
1. **central vs PST + mobility.** SF has no "central" term at all — it prices centrality once, via PST / mobility / space. v2's
   rung-0 PSTs and shipped mobility (attacked-square counts) already price central squares. A v2 "central" term is a candidate
   SECOND OWNER of what we already pay for. ⇒ The collinearity gate must run central against mobility and PST BEFORE any ladder.
2. **space vs mobility's area.** SF's space `safe` mask is `~enemy pawn attacks`; v2's mobility area excludes exactly that.
   Overlap unmeasured. ⇒ same gate, before the ladder.
3. **threats/hanging ownership CHANGED.** In v1, Hanging was ~87% a subset of capture gains (r = 0.604). **v2 has no capture
   gains**, so that argument does not transfer — in v2, Hanging would have no other owner. This is a changed premise, not a re-run.
4. **`RestrictedPiece`** (squares they attack that we also attack) sits adjacent to mobility's area ⇒ gate it.
5. **SF's `KnightOnQueen` / `SliderOnQueen`** sit next to v2's shipped **weak queen** (placement E) ⇒ gate it.
6. **Kaufman vs slice 4.** Kaufman's pawn-count and bishop-pair products touch the same census as slice 4's endgame scale /
   winnability (OCB, pawn count). Flag the slice 3/4 overlap at the checkpoint.

### 0.3 Instruments that are INVALID for anything here (do not cite them as closures)
- Every STS / per-theme / move-match delta from 06-20 → 08-13 (harness history contamination).
- The +45 threats SPRT and the +20.8 bundle: `gate`'s hardcoded `--seed 0` fixed openings ⇒ signs probably right, magnitudes suspect,
  and the bundle was never attributed to its components.
- The 09-08 global-null win% screen (superseded by paired nulls).
- Every corpus-fit ranking (07-05, 07-16, 08-03) — corpus fit is anti-correlated with Elo.
- `_v2` §I ablations that show "switching a v1 term OFF helps" (`ENABLE_THREATS=0` +2.65%, `ENABLE_KAUFMAN=0` −4.50%/+7.22%):
  switch-off winners are suspect BY RULE, and they measure v1's degenerate regime, not a clean v2 term.

### 0.4 What genuinely remains open (the slice-3 candidate set)
- **space**, SF's gated form (safe-square area, behind-pawn doubling, piece-count weight, npm gate) — never built here.
- **threats**, SF's form family, specifically the six structural omissions recorded against v1: defended-piece minor leg ·
  pawn targets · `ThreatByPawnPush` · `RestrictedPiece` · `Knight/SliderOnQueen` · a stronger `stronglyProtected`.
- **bishop pair** (and knight pair), flat and openness-conditioned — never measured alone anywhere.
- **Kaufman**, census-product form with OUR magnitudes.
- **central**: build ONLY if the collinearity gate says it is not a second owner of PST + mobility. Otherwise it is a
  measured "already owned", which is itself a result worth recording.

---

## 1. FIVE-ENGINE CONTRAST (from source: SF 1.1 / 11 / 15.1 local · Ethereal @0e47e9b · Weiss @c735b8f)

**Units + comparator.** Ratio = a term's mg value ÷ THAT engine's own knight PST rim-vs-centre mg spread (a4→d4), the
scale-free comparator slice 2 used: SF1.1 **63** · SF11 **84** · SF15.1 **84** · Ethereal **11** · Weiss **26**.
Pawn mg/eg: SF1.1 204/256 · SF11 128/213 · SF15.1 126/208 · Ethereal 82/144 · Weiss 104/204.
⚠️ Slice 2's comparator cells were never written down; reconstructed here as d4−a4 (reproduces slice 2's 84 for SF11). If slice 2
used other cells every ratio rescales UNIFORMLY per engine — the cross-engine ordering is unaffected.

### 1.1 Space — **3/5 have it** (SF11, SF15.1, Ethereal; SF1.1 and Weiss have NONE)
| axis | SF11 / SF15.1 | Ethereal |
|---|---|---|
| square set | own camp, files c-f × ranks 2-4 (development room) | shared `CENTER_BIG` c3-f6 (16 squares) |
| "safe" | `~own pawns & ~enemy PAWN attacks`, **plus a second count** of squares ≤3 ranks behind an own pawn not attacked at all ("counted twice") | `~attacked[THEM] & (attacked[US] \| friendly)` — ALL enemy attacks, and we must attack or occupy |
| weight | SF11 `bonus × (pieces−1)²/16`; SF15.1 `× (pieces−3+min(blocked,9))²/16` | LINEAR, `count × S(3,0)` |
| gate | npm ≥ 73.6% (SF11) / 69.6% (SF15.1) of start | `minors + 2·majors > 12` |
| endgame leg | **0** | **0** |
Ratios: SF11 ≈ 0.17×/square at 16 pieces (falls to ~0.09× at 12); a typical 12-square count ≈ **2.0×** total. Ethereal 0.27×/sq,
ceiling ≈ 4.4×. ★ **eg = 0 in 3/3** — space is a midgame term everywhere that has it.
⚠️ Ethereal also carries a NEGATIVE half inside its space term (`SpaceRestrictPiece/Empty`, S(−4,−1)/S(−4,−2), ungated, whole
board) — the same construct SF prices in THREATS as `RestrictedPiece`. Same idea, two different owners depending on the engine.
☠️ Collinearity, stated by the sources themselves: SF's `safe` shares the enemy-pawn-attack exclusion with `mobilityArea`, so a
minor's mobility count and Space count the SAME squares in our own camp.

### 1.2 Central control — **0/5. No reference engine has a standalone central term.**
SF1.1, SF11, SF15.1, Weiss: none. SF's only uses of `Center`/`CenterFiles` are inside Space (own ranks 2-4, i.e. development
room, NOT the d4/e4/d5/e5 complex) and `LongDiagonalBishop` (a bishop term). Ethereal's `SpaceCenterControl` is the nearest
thing and lives INSIDE its space term, gated on material, mg-only.
⇒ ★★ **The concept v1 calls `central` has no counterpart in any reference.** They price centrality ONCE, in PST + mobility.
This is the adoption rule's clearest case: 0/5 is not a split, it is a consensus AGAINST a separate owner.
☠️ Correction to a standing claim: "our `central` IS SF's Space" is WRONG at the source — SF's Space is own-camp development
room; Ethereal's centre-control half is the only central-ish reference term.

### 1.3 Threats — **4/5 have a threat family** (not SF1.1, whose only threat is a mate-threat term inside king safety)
| sub-term | count | the split |
|---|---|---|
| pawn attacks a non-pawn piece | **4/4** | SF gates the ATTACKER (`safe` pawn); Ethereal and Weiss ungated (victim-side penalty in Ethereal) |
| safe pawn-push threat | **4/4** | SF single+double, push square `~enemy pawn attacks & safe`; Weiss single, no safety test |
| minor attacks piece, table by TARGET | 3/4 | SF pays `defended \| weak` (even a pawn-defended target); Ethereal buckets by ATTACKER class |
| rook attacks piece | 3/4 | SF `weak` only; Weiss excludes pawn-defended only; Ethereal gated `poorlyDefended` |
| king attacks piece | 3/4 | not Weiss (king removed from targets) |
| **hanging** | **2/4 — SF only** | Ethereal has `Overloaded` (attacked once, defended once) instead; Weiss none |
| restricted squares | 2/4 | SF `RestrictedPiece`; Ethereal's equivalent sits in SPACE |
| queen-specific | 4/4 in **four different forms** | SF `Knight/SliderOnQueen` on safe squares (SF15.1 ×(1+queenImbalance)); Ethereal `QueenAttackedByOne`; Weiss via table |
| pawn TARGETS | 3/4 | SF eg-weighted `S(6,32)`/`S(3,44)`; Ethereal penalises OWN weak pawns; Weiss excludes pawns |
★ **Shape agreement across all four: pawn-attacks-a-piece is the LARGEST single constant** — SF11 S(173,94) = 2.06× · SF15.1
1.99× · Weiss 3.1× · Ethereal 5.0× (victim-side). That is the transferable fact; the absolute scales are not.
⚠️ Defence-gate definitions differ in WHOSE view they take: SF's `stronglyProtected` is the enemy's view; Ethereal's
`poorlyDefended` is the victim's and lets pawn support override. Gate coverage: SF 5 of 7 · Ethereal 4 of 10 · Weiss 1 of 4.
☠️ Self-declared collinearity: SF's `RestrictedPiece` uses the same attack maps as mobility's area; `Knight/SliderOnQueen` sits
beside `WeakQueen` — **which v2 already ships in placement E**.

### 1.4 Bishop pair / knight pair / imbalance — **5/5 pay a bishop pair**; the FORM splits three ways
| engine | form | value | ratio (mg) | condition | census coupling |
|---|---|---|---|---|---|
| SF1.1 | flat + 3 separate Kaufman linears | 100/100 | **1.59×** | 2 bishops | knight×(pawns−5); rook redundancy |
| SF11 | quadratic MATRIX, applied flat to both phases | 89.9 | 1.07× | pair = pseudo-piece | pair × OWN pawns **+2.5/pawn**; N×pawns +15.9; N×N −3.9; R×R −13 |
| SF15.1 | same matrix, phase-split cells | 88.7 / 90.9 | 1.06× | same | same shape |
| Ethereal | flat, **eg-heavy** S(22,88) | | 2.0× | ⚠️ requires OPPOSITE COLOURS | none |
| Weiss | flat, eg-heavy S(33,110) | | 1.27× | two bishops | none |
★ Transferable: the pair is worth **≈ 1-2× the knight PST spread in mg in 5/5**, and 2/5 make it strongly eg-heavy.
☠️ Two v1 ideas have NO reference support: a flat **knight pair** bonus is **0/5** (it exists only as NEGATIVE redundancy inside
SF's matrix: −3.9 per pair), and **openness scaling is 0/5**. Worse for the folklore: SF scales the pair **POSITIVELY with its
OWN pawn count** (+2.5/pawn) — the opposite of "bishops like open boards". v1's `MOD_PAIR_OPEN` inverts SF's sign.
Rook redundancy 3/5 · knight-likes-pawns 3/5 (SF1.1 linear; SF11/15.1 a matrix cell).

### 1.5 What the adoption rule makes of this
- **CORE (adopt the shape):** pawn-attacks-a-piece as the dominant threat constant · a safe pawn-push threat · target-indexed
  piece threats · a bishop pair at ≈1-2× our knight PST spread.
- **CANDIDATE FORMS (references split ⇒ knobs, ours legitimate):** space square set / safe mask / weight / gate · threat defence
  gates (SF's attacker-side vs Ethereal's victim-side) · queen-threat form (4 ways) · pair phase split and pawn-count coupling ·
  flat vs matrix imbalance.
- **NOT PLANNED (0-2/5):** ☠️ a standalone **central** term (0/5 — consensus against) · flat knight pair (0/5) · openness
  scaling (0/5) · `WeakQueenProtection` (1/5) · `Overloaded` (1/5) · opposite-colour pair requirement (1/5).
- **OPEN QUESTIONS from the fetch** (not blocking): Weiss's PSQT orientation (would halve its spread and double its ratios);
  whether Ethereal pays an attacker-side threat bonus inside `evaluateKings`.

## 2. DESIGN (proposed 2026-09-15, after §0 + §1; owner's call on scope before building)

### 2.0 ☠️ What slice 3 does NOT contain, and why
| dropped | evidence | what would re-open it |
|---|---|---|
| **a standalone `central` term** | **0/5 references** (§1.2) — a consensus AGAINST, not a split. v2 already prices central squares TWICE (rung-0 PST cells + mobility's attacked-square counts) | ▶️ **replaced by a sharper test (owner, 2026-09-16), see §2.0b** — not "is central unpriced?" but "are its OWNERS sized right where centrality decides?" |

#### 2.0b ★ THE CENTRALITY QUESTION, REFRAMED (owner, 2026-09-16) — the reason we are NOT building `central`
Owner's framing: *"the centre is not some magic zone — the point is what you can DO with it. For high-accuracy
definitions we don't want blanket scores for things with nuance; let the scores that are inherently bettered by the
geometry of the centre shine through."* That is exactly what 5/5 references do: centrality is priced ONCE by PST (where
a piece stands) and ONCE by mobility (what it therefore controls), and the engines differ only in HOW THEY SPLIT IT
(Ethereal's PST spread is 11 with a mobility table ~10× that; SF11's PST spread is 84). Nobody adds a third owner.
⇒ v1's `central_score` — an accumulator re-summing the PST and heat cells the pieces already scored — is the
second-owner pattern the charter exists to prevent.
**BUT the owner's second point is the live one:** those owners were each sized on corpora where central positions are a
minority, so a term can be right on average and wrong exactly where the concept decides. ⚠️ And v2's knight PST
rim-vs-centre spread is ~30 mp against SF11's 84 units (~0.66 of their pawn) — our centrality signal may simply be
UNDER-SCALED, which is the same denominator error that mis-sized mobility.
**Test instead of build (🧰 `_position_class.py`, built 2026-09-16, no engine):** classify every corpus position by
pawn structure — `centre_tension` (contact on d/e, the fight live) · `centre_locked` (rammed centre, ≤1 open file) ·
`centre_open` · `centre_cleared` · `other` — then run §I and the d7 regret gate PER CLASS. The candidate that follows is
a **rung-0 PST SCALE ladder** (one knob on the existing owner), not a new term. Also emitted: a `pin_dense` tag, which is
the named trigger the parked `MOB_V2_PIN` knob has been waiting for — same machinery, second question answered free.
⚠️ Register classes and expected direction BEFORE reading: slicing many ways invites a chance finding.

**Class sizes (47,653 positions over the four standard corpora, 2026-09-16):** `centre_tension` 4,117 (8.6%) ·
`centre_cleared` 8,433 (17.7%) · `centre_open` 13,749 (28.9%) · `centre_locked` 1,652 (3.5%) · `other` 19,702 (41.3%) ·
tag `pin_dense` 7,310 (15.3%). All are readable on §I; `centre_locked` is the thinnest, so a regret read there will be
noisier than the ~2-2.5pp cross-set bar and must not be quoted alone.
☠️ **The first run of the classifier was WRONG and the counts caught it:** `centre_cleared` fired 0 times (unreachable —
`centre_open` was tested first and swallowed it) while `centre_open` reached 45.9%, i.e. a corpus rather than a class.
Fixed by ordering the narrower class first and bounding `centre_open` by total pawns. ★ Same failure shape as a detector
that never fires: **an unreachable class passes every check vacuously** — which is exactly why the tool prints counts.

**REGISTERED PREDICTIONS for the per-class reads (2026-09-16, before any per-class number exists):**
1. The v2-vs-SF11 accuracy gap is **largest in `centre_tension`** — if our centrality signal is under-scaled, that is
   where it should show, because the centre is still contested there.
2. `centre_locked` is where a blanket central bonus would LIE, so I expect our current terms to be **least wrong** there —
   mobility already collapses correctly when the position is fixed.
3. A rung-0 **PST-scale ladder** (one knob on the existing owner) moves `centre_tension` more than `centre_open`.
4. ⚠️ Prediction I expect to be WRONG if my recent record holds: that any of this clears the regret bar at all. Two
   §I-positive mobility candidates and (so far) the bishop pair have all come back null on the move-level instrument;
   the honest prior is that per-class analysis re-describes the same sub-bar effects rather than revealing a new one.
5. `pin_dense`: if `MOB_V2_PIN` is genuinely unreadable rather than worthless, its regret there should be ABOVE its
   whole-corpus reading (−0.1 / +0.6). If it is flat there too, the parked knob is a measured null, not an unread one.

✅ **PER-CLASS §I (4 arms × 5 class corpora, 2026-09-16; negative = better). Base MSE per class:** tension 285.98 ·
locked 297.83 · open 346.95 · cleared 283.59 · pin_dense 413.74.
| arm | tension | locked | open | cleared | **pin_dense** | mean | WORST |
|---|---|---|---|---|---|---|---|
| `pin` | −0.33 | −0.47 | −1.46 | −1.32 | **−4.75** | −1.67 | −0.33 |
| `exlow` | **−2.39** | −1.21 | −0.64 | −0.54 | −1.40 | −1.24 | −0.54 |
| `bp40` | −0.19 | −0.05 | −0.18 | −0.14 | −0.20 | −0.15 | −0.05 |
★★ **The owner's hypothesis is confirmed on this instrument: whole-corpus averages WERE hiding class structure.**
- `pin` is **9× larger in `pin_dense` (−4.75) than its whole-corpus mean (−0.51)**, and better on ALL five classes ⇒ on
  these corpora it clears the both-better rule it failed globally. This is exactly the registered prediction (5) shape:
  the earlier null was **unreadable, not worthless** — our standard corpora average over positions where absolute pins
  are rare. ⚠️ §I is still corpus fit; the regret read on the class decides, and it needs a NEUTRAL measured ON THE CLASS
  (the 49.9 / 51.1 bars are whole-corpus and do NOT transfer).
- `exlow` is strongest exactly where the centre is contested (**tension −2.39**, vs −0.54 in `cleared`) — an area rule
  about rank-2/3 pawns SHOULD matter most while the centre is unresolved. Its slice-3 bundle ride is now better motivated
  than "it won the global ladder".
- `bp40` is flat everywhere (−0.05..−0.20, no class above its global −0.11) ⇒ **no hidden regime**. Combined with its
  regret null on `_v2` (−0.1pp), the reading is that v2's PST + mobility ALREADY price what the pair is worth.
**Prediction scorecard:** ✅ (5) pin reads far above its whole-corpus value in `pin_dense`. ⚠️ (1) WRONG as stated — the
largest base error is `pin_dense` (413.74), not `centre_tension` (285.98, the SMALLEST after `cleared`); centrality is not
where our eval is least accurate. ✅ (2) `centre_locked` is where the arms move least (pin −0.47, exlow −1.21, bp −0.05).
❌ (4) my "expect everything to be sub-bar" hedge is at least premature: pin's class effect is an order of magnitude
above its global reading. (3) untested — the PST-scale knob does not exist yet.
▶️ **Regret for `pin` on `ks_sets/classes/pin_dense.csv` (7,310 positions), verdict PENDING its class neutral:**
`pin_pindense` **49.4% win, delta +0.087 (WORSE), 2,623 changed / 35.9%**. Slice split: opening **55.3%** (delta −0.543,
better) vs midgame **46.6%** (+0.326, worse) — opposed signs INSIDE one corpus. `cr4_CRITICAL` n=10 (unreadable; the +22.5
delta there is one or two positions). ⏳ neutral on this corpus running; 49.4 is uninterpretable until it lands.
⚠️ **Provisional read, stated before the neutral so it cannot be retro-fitted:** this does NOT look like corroboration.
§I said −4.75% on this very corpus; the move-level instrument says +0.087 the wrong way. If the neutral is ≈50, `pin` is
another **§I-only effect**, and the per-class §I result will have re-described corpus fit at higher resolution rather than
revealing a real term. ☠️ My own framing two messages earlier ("the owner's hypothesis is confirmed on this instrument")
was premature: §I agreeing with itself more loudly is not a second instrument.
★ The method still worked — the CLASS instrument is what made the question askable at all, and a §I/regret DISAGREEMENT on
a 7,310-position class is a sharper, more useful null than the global one it replaced.

★★ **VERDICT — `MOB_V2_PIN` IS A NULL ON ITS OWN CLASS. The provisional read above was right.**
Class neutral (`ASPIRATION_DELTA=300` on `pin_dense`, same base, same corpus): **50.0%** (2,282 changed / 31.2%, delta
+0.112). Candidate: **49.4%** ⇒ **−0.6pp**, inside the ~2-2.5pp bar, on a 35.9% footprint.
⇒ §I read −4.75% on this corpus — **9× its global value — and the move-level instrument on the SAME 7,310 positions says
nothing.** The parked verdict stands unchanged; pin stays built, default-off.
☠️ **The lesson is about instruments, not about pins: a per-class §I win is NOT a second instrument.** Slicing by class
raised §I's resolution, and §I is corpus fit — so a louder §I is a louder reading of the same thing. The corroboration rule
needs two instruments that RESOLVE the term, and here the second one refused. (Note the delta metric flatters the candidate
— +0.087 vs the neutral's +0.112 — which is exactly why win% is the honest statistic: the delta's null is not zero.)
★ What the class machinery DID buy, and keep: (1) the question was askable at all, (2) the null is now bounded on the
population where the term fires most, which is a far stronger statement than a whole-corpus null, and (3) `pin_dense` is
reusable for any future pin/tactics work. The owner's "park it, someone may need it one day" stands — this measurement
narrows WHERE it could still matter (not in ordinary play at d7), it does not close the idea.

**CLASS NEUTRALS measured on this base (the bars for any class read; whole-corpus 49.9 / 51.1 do NOT transfer):**
`pin_dense` **50.0%** (2,282 changed / 31.2%) · `centre_tension` **50.4%** (1,478 changed / 35.9%).
⚠️ Note how close both sit to 50 while the whole-corpus bars sit at 49.9 and 51.1 — the bar is a property of the CORPUS, not
a constant, which is the whole reason it has to be re-measured per class.

★★ **VERDICT — `MOB_V2_EXCL_LOWRANK` IS ALSO A NULL ON ITS OWN CLASS.** `exlow_tension` **49.0%** vs the class neutral
**50.4%** ⇒ **−1.4pp, the WRONG direction**, on the largest footprint of any arm here (1,650 changed / 40.1%). Its §I read on
this same corpus was −2.39% (4× its global value). ⚠️ `.endgame` shows 21.4% on **n=14** — quoted only to note it is
unreadable, not as a signal.
⇒ **BOTH parked mobility candidates are now confirmed §I-ONLY, on the very classes where their accuracy effect was largest.**
That is the second independent confirmation of the instrument lesson: pin (−4.75% §I → −0.6pp regret) and exlow (−2.39% §I →
−1.4pp regret) both had class-local ACCURACY structure and neither had class-local MOVE structure.
▶️ **Consequence for the slice-3 bundle: `exlow` no longer rides it.** The earlier plan (09-15: "`exlow` rides slice 3's
bundle" on its 6/6 global §I win) is WITHDRAWN — a term that is null on the whole corpus AND null where it fires hardest has
no measured case, and riding a bundle is not a place to hide one. It stays built, default-off, with the same trigger as before
(a later re-scale of mobility's area or magnitude). ☠️ This is the discipline the drift risk demanded: the bundle bar (≥ −10
Elo) would have accepted it silently.
⇒ Slice 3's bundle will contain only terms with their OWN measured case: **space** and **threats** (both never properly built
here), plus **Kaufman** if it earns one.
| **flat knight pair** | 0/5 — exists only as NEGATIVE redundancy inside SF's matrix (−3.9/pair) | nothing on the record; would need an ours-first case |
| **openness scaling of the pair** (v1 `MOD_PAIR_OPEN`) | 0/5, and SF couples the pair POSITIVELY to its OWN pawn count (+2.5/pawn) — v1's knob inverts SF's sign | measure SF's sign first; the folklore version is not a candidate |
| `WeakQueenProtection` · `Overloaded` · opposite-colour pair test | 1/5 each | — |

### 2.1 SPACE (new; the term §0 showed was never actually refuted here)
- **Core (the 3/3 that have it agree):** count SAFE squares in a region, weight by piece count, **endgame leg = 0**.
- **Forms as knobs** (the references split, so each is a candidate, ours included):
  `SPACE_V2_REGION` 0 = SF own-camp c-f × ranks 2-4 · 1 = Ethereal shared c3-f6.
  `SPACE_V2_SAFE` 0 = `~own pawns & ~enemy PAWN attacks` (SF) · 1 = `~all enemy attacks & (we attack or occupy)` (Ethereal).
  `SPACE_V2_WEIGHT` 0 = `(pieces−1)²/16` (SF11) · 1 = `(pieces−3+min(blocked,9))²/16` (SF15.1) · 2 = linear (Ethereal).
  `SPACE_V2_BEHIND` = SF's second count of un-attacked squares ≤3 ranks behind an own pawn ("counted twice").
  `SPACE_V2_GATE` = npm threshold as a percent of start (SF11 74 / SF15.1 70) · `SPACE_V2_MAG` = magnitude, mg only.
- **Efficiency:** every input already exists — enemy pawn attacks (mobility's area computes them), both sides' attack maps
  (`SideAttacks`), own pawns. ⇒ a few masks and popcounts inside the existing pass. ☠️ No second attack pass.
- ☠️ **One-owner risk, stated by SF itself:** its `safe` mask shares the enemy-pawn-attack exclusion with `mobilityArea`, so
  own-camp squares are counted by BOTH space and mobility. **The collinearity gate runs BEFORE the magnitude ladder**, not after.

#### 2.1a SPACE build status (2026-09-16)
✅ **BUILT and compiling.** Knobs `SPACE_V2_MAG` (mp per reference unit = SF11's shape at the start weight × 12 safe
squares = raw 169) · `SPACE_V2_REGION` 0 SF own-camp c-f × ranks 2-4 / 1 Ethereal c3-f6 · `SPACE_V2_SAFE` 0 SF pawn-attack
safety / 1 Ethereal all-attacks · `SPACE_V2_WEIGHT` 0 SF11 `(pieces−1)²/16` / 1 Ethereal linear · `SPACE_V2_BEHIND` (SF's
double count of un-attacked squares behind our pawns) · `SPACE_V2_GATE_PCT` (non-pawn material as % of start; SF11 74).
Applied **midgame-only** (×`phase256`/256), since all 3/3 references that have space give it a zero endgame leg.
★ Rides the EXISTING attack build — the dispatch gate widened to `KS_V2_MAX > 0 || mob_on || space_on`, so space can run
without KS or mobility and still costs no second attack pass.
⚠️ **NOT built, recorded rather than approximated:** SF15.1's `(pieces−3+min(blocked,9))²` weight needs the pawn entry,
which the dispatch builds AFTER the attack maps this term rides on.
☠️ **Build failure worth remembering:** `space_mp` was placed above the slice-2 placement constants and referenced
`PL_CENTRE_FILES` from below its own declaration point. Fixed with a local `SPACE_CENTRE_FILES` rather than reordering a
shipped, oracle-verified block. The `build` sub deletes the `.so` first, so a failed build leaves NO engine.
✅ **Byte-identity on the space build (both arms, space OFF): SHIP+E `250 / 61,352,373 / EBF 4.114` · v1
`250 / 35,310,778 / EBF 3.784` — EXACT.** So the term and its widened attack-build gate are inert at `SPACE_V2_MAG=0`.
✅ **Byte-identity re-verified on the PROBE build (shipped path, space off): `250 / 61,352,373 / EBF 4.114` — EXACT.**
Adding a diagnostic probe changed `eval_v2.cpp` and `ChessAI.pyx`, so the earlier fingerprint no longer described the binary;
re-running it is the cheap way to prove the probe is not in the search path.
✅ **Probe added (2026-09-16): `space_probe` / `ChessAI.space_counts`** — 6 slots (per-side safe counts incl. the BEHIND
double count, per-side piece counts, the Black-positive score, phase256), matching the `mobility_probe` / `placement_probe`
pattern. It recomputes the counts rather than instrumenting `space_mp`, so the hot path carries no diagnostic branch and a
divergence between the two is itself a finding. The oracle now compares **counts AND score** — a score-only check would pass
on a compensating pair of errors (region too large, weight too small), which was the hole in its first version.
✅ **SPACE ORACLE PASSES BOTH FORM FAMILIES** (🧰 `_space_detector_oracle.py`, counts AND score compared):
`REGION=0 SAFE=0 WEIGHT=0 GATE=74` (SF) — 2,004 positions, **0 mismatches**, fired **36.5%** ·
`REGION=1 SAFE=1 WEIGHT=1 BEHIND=1 GATE=70` (Ethereal, every axis flipped at once) — **0 mismatches**, fired **34.3%**.
⇒ both regions, both safe masks, both weights, the behind-pawn double count, the material gate and the midgame taper are
verified against an independent python-chess rebuild.
✅ **Symmetry with space ON** (`MAG=40 REGION=0 WEIGHT=0 BEHIND=1` — the colour-DEPENDENT parts: own-camp rank masks and the
behind-pawn shifts): **colour swap 0/800 violations**, file mirror at the pre-existing 21 @ 5 mp baseline.

⏳ **§I LADDER RUNNING — 11 arms across BOTH anchors and all four form axes:**
`ship · sp20 · sp40 · sp60 · sp100 · sp300 · sp700 · sp40r1 (Ethereal region) · sp40s1 (Ethereal safe) · sp40w1 (linear
weight) · sp40b1 (behind double count)`. Rule unchanged: better than `ship` on BOTH mean and worst, ±0.05 floor.
**REGISTERED PREDICTIONS (2026-09-16, before any number):**
1. ☠️ **Nothing clears the both-better rule.** Every positional term measured on this base — mobility magnitudes, endgame
   share, the bishop pair at every size — has helped the four general corpora and hurt `lichess_ks_labelled`. I expect space
   to do the same, and I am registering that pessimism BECAUSE my opposite prediction for the pair ("a genuinely new term is
   exempt") was wrong yesterday.
2. The mean optimum sits at **20-60**, not at the pawn-conversion end: SF's space at 12 squares is ~1.3 of THEIR pawns, so
   `sp700` is the pawn anchor and should be clearly worse on the worst column (the mobility/eg-share pattern).
3. `sp40b1` behaves like a magnitude increase (more counted squares) rather than a distinct shape ⇒ close to `sp60`.
4. `sp40r1` (Ethereal's shared c3-f6 region) differs from the SF arms MORE than `sp40s1`/`sp40w1` do, because it changes
   WHICH squares count rather than how they are filtered or weighted.
5. If anything survives to regret, it will read null there like everything else this week — and the per-class read on
   `centre_tension` is where space has its best chance, since that is where the contested squares are.

✅ **§I LADDER RESULT (11 arms; negative = better; positive = WORSE):**
| arm | mean% | WORST% | | arm | mean% | WORST% |
|---|---|---|---|---|---|---|
| sp20 | +0.01 | +0.02 | | sp700 (pawn anchor) | **+0.43** | **+0.81** |
| sp40 | +0.02 | +0.04 | | sp40r1 (Ethereal region) | +0.04 | +0.06 |
| sp60 | +0.03 | +0.07 | | sp40s1 (Ethereal safe) | +0.04 | +0.09 |
| sp100 | +0.05 | +0.11 | | sp40w1 (linear weight) | −0.00 | +0.00 |
| sp300 | +0.17 | +0.33 | | sp40b1 (behind count) | +0.03 | +0.07 |
☠️ **SPACE AS BUILT IS A MEASURED NEGATIVE ON §I AT EVERY MAGNITUDE AND IN EVERY FORM** — worse on BOTH columns, monotone in
size, with no interior optimum: smaller is merely less harmful. Only `game_regret_set_uho` improves (−0.01..−0.21), i.e. one
corpus out of six, which is the corpus-fit signature rather than a signal.
**Prediction scorecard:** ✅ (1) nothing clears the both-better rule — and stronger than predicted, since it is not even a
mean/worst TRADE, it is worse on both. ❌ (2) there is no 20-60 optimum; the curve is monotone harmful, so "optimum" was the
wrong shape to expect. ✅ (3) `sp40b1` ≈ `sp60` (0.03/0.07 vs 0.03/0.07) — the behind count behaves as a magnitude increase,
not a distinct shape. ❌ (4) `sp40r1` does NOT differ more than `sp40s1` — region and safe-mask changes land within 0.03 of
each other.
☠️ **MY LADDER HAD A DESIGN FLAW, and `sp40w1` is the tell, not a result.** Ethereal's LINEAR weight makes the multiplier 1
where SF11's quadratic gives `(16−1)²/16 ≈ 14.1` at full material, so `sp40w1` is really a **~2.8 mp** arm — an order of
magnitude below the others and below §I's ±0.05 resolution. Reading it as "the linear form is neutral" would be false: it was
never tested. ⇒ **A form comparison must be SCALE-NORMALISED before it means anything** (the same lesson as the mobility
table bake-off, where every table was rescaled to a common knight range — I applied it there and forgot it here).
▶️ Next, in ONE run: the linear form at **parity (`MAG=560`)**, plus a PER-CLASS read of `sp40` / `sp40b1` on the structure
classes — prediction (5) says `centre_tension` is space's best chance, and the class instrument exists to make exactly that
question askable. ⚠️ Remembering that per-class §I is still §I: it can bound or motivate, never corroborate.

✅ **PER-CLASS §I + LINEAR-FORM PARITY (base MSE: tension 285.98 · locked 297.83 · open 346.95 · cleared 283.59):**
| arm | tension | **locked** | open | cleared | mean | WORST |
|---|---|---|---|---|---|---|
| sp20 | +0.02 | **−0.04** | +0.01 | 0.00 | 0.00 | +0.02 |
| sp40 | +0.05 | **−0.07** | +0.02 | +0.01 | 0.00 | +0.05 |
| sp40b1 | +0.09 | **−0.11** | +0.02 | +0.01 | 0.00 | +0.09 |
| **sp560w1** (linear @ parity) | **−0.02** | **−0.13** | +0.01 | 0.00 | **−0.04** | **+0.01** |
❌ **PREDICTION (5) WAS WRONG, AND INVERTED.** Space does not help where the centre is CONTESTED — it hurts there
(+0.02..+0.09) and helps where the centre is **LOCKED** (−0.04..−0.13), monotonically in magnitude in BOTH directions.
★ That has a mechanism, which is why it is worth pursuing rather than filing: **when the centre is locked, mobility
collapses** (few safe squares, blocked sliders), so safe-square ROOM behind a fixed structure carries information mobility
cannot. Space and mobility are complements on the structure axis, not duplicates — and my "space's best chance is where the
fight is live" intuition had it exactly backwards.
★ **The scale-parity fix mattered:** `sp560w1` — Ethereal's LINEAR weight at a magnitude matched to SF's quadratic — is the
first space arm that is not uniformly harmful (mean −0.04, worst +0.01, i.e. inside the ±0.05 floor, better on tension AND
locked). Had I left the un-normalised `sp40w1` reading in place, the linear form would have been recorded as "neutral, not
worth pursuing" on an arm that was never actually tested.
▶️ **Next: a GLOBAL ladder for the linear form across magnitudes** (`280 / 560 / 840 / 1120`, plus linear+behind), because the
class run omits `lichess_ks_labelled` — the KS-critical corpus that has decided the worst column for every positional term on
this base. **Registered predictions:** (1) the linear form's mean optimum sits near 560-840; (2) its worst column turns
positive before 1120, because every positional magnitude on this base eventually taxes KS-critical accuracy; (3) linear+behind
≈ a magnitude increase, as it was for the quadratic form; (4) if the worst column stays inside the floor at the mean optimum,
this is the FIRST slice-3 term with a genuine §I case — and it then owes a regret read on the whole corpora AND on
`centre_locked`, whose neutral is not yet measured.

✅ **GLOBAL LINEAR-FORM LADDER (6 corpora; negative = better):**
| arm | mean% | WORST% | | arm | mean% | WORST% |
|---|---|---|---|---|---|---|
| sp280w1 | −0.00 | +0.01 | | sp1120w1 | −0.02 | +0.02 |
| sp560w1 | −0.01 | +0.01 | | sp560w1b1 | 0.00 | +0.04 |
| sp840w1 | −0.01 | +0.02 | | | | |
☠️ **VERDICT: the linear form is INERT globally — every arm sits inside §I's ±0.05 resolution floor on BOTH columns.** The
class-local −0.13 on `centre_locked` does not reach the aggregate because that class is **3.5% of positions**: 0.035 × 0.13 ≈
0.005, i.e. a tenth of the floor. Nothing here is harmful, and nothing is readable.
**Prediction scorecard:** ⚠️ (1) no resolvable optimum — the mean improves weakly all the way to 1120 but never leaves the
floor, so "optimum" was the wrong shape to look for (the same error I made on the quadratic ladder). ⚠️ (2) the worst column
IS positive at every magnitude as predicted, but at +0.01..+0.02 it is unreadable, so the prediction is right in sign and
meaningless in magnitude. ❌ (3) WRONG — `sp560w1b1` (mean 0.00 / worst +0.04) is worse than both `sp560w1` and `sp1120w1`,
so the behind-pawn double count is NOT equivalent to a magnitude increase in the linear form; it adds something harmful.
❌ (4) the worst column did stay inside the floor — but so did the MEAN, so this is a null, not "the first genuine §I case".
★ **What is real and worth keeping:** space's only measurable effect is class-local (`centre_locked` −0.04..−0.13, monotone
in magnitude), with a mechanism — locked centres are where mobility collapses, so space carries what mobility cannot. ⚠️ But
`centre_locked` is 1,652 positions, the thinnest class, which I flagged when the classes were built as too thin for the
~2-2.5pp regret bar. ⇒ **Before spending the move-level instrument on it, ENLARGE the class** by classifying the two corpora
the first pass omitted (variant, lichess-KS), written to a SEPARATE output dir so a re-run cannot overwrite a corpus a
measurement is reading.

☠️ **ENLARGEMENT FAILED, AND THE FAILURE IS THE FINDING** (`ks_sets/classes6`, all six corpora, 61,153 positions):
`centre_tension` 6,489 (10.6%) · `centre_locked` **1,974 (3.2%)** · `centre_open` 15,541 (25.4%) · `centre_cleared` 9,740
(15.9%) · `other` 27,409 (44.8%) · `pin_dense` 10,372 (17.0%).
Adding **13,500 positions from two entire corpora grew the locked class by 322 rows** and left its SHARE unchanged (3.5% →
3.2%). ⇒ **Locked centres are intrinsically rare in our position pools, not under-sampled by corpus choice.** More of the
same material cannot thicken this class; only purpose-built positions could (closed openings — King's Indian, French,
Closed Sicilian — mined or generated).
★ **This is a structural limit on what we can measure, and it belongs in the instrument map, not in space's ledger:** any
term whose regime is a few percent of played positions is unresolvable by our corpora at the ~2-2.5pp regret bar, no matter
how real its mechanism. The same arithmetic retires the aggregate case for space (0.032 × 0.13 ≈ 0.004 = a tenth of §I's
floor) and explains pin (`pin_dense` is larger at 17%, and even there the move-level read was null).
**Bar stability on a thin class (measured, 2026-09-16):** the neutral on the OLD 1,652-row locked class reads **49.2%** on
only **607 changed moves** (delta +0.247; `.midgame` 54.0% on 219). ☠️ **A bar drawn from ~600 changed moves carries a
standard error of roughly 2pp on its own** — the same size as the entire effect we would be trying to detect. This is the
quantitative reason the thin-class read cannot settle anything, and it is why the enlarged class (1,974 rows) gets its own
neutral rather than borrowing this one. (This run is superseded by that one; kept as the bar-stability datapoint.)
**Enlarged-class neutral (the bar for the bounded read): `centre_locked` @ 1,974 positions = 49.4%**, on **644 changed
moves** (delta +0.255). ⚠️ Only 37 more changed moves than the 1,652-row version gave (607) — **enlarging the corpus by 322
positions did NOT thicken the measurement**, because the changed-move subset is what the statistic is built from. The bar
therefore still carries ~2pp of its own standard error. ⇒ The read below is deliberately run at the LARGEST magnitude
(`MAG=1120`, linear) so the class effect has its best chance of exceeding that noise; anything smaller could not be
distinguished from the bar even in principle.
★★ **SPACE VERDICT (2026-09-16): PARKED — mechanism supported, effect unresolvable, nothing shipped.**
Bounded read, `MAG=1120` linear on `classes6/centre_locked.csv`: **50.9%** vs the class neutral **49.4%** ⇒ **+1.5pp, the
right direction**, delta −0.106, on **467 changed moves (23.7%)**.
☠️ **But it cannot resolve:** a win% on 467 changed moves carries ≈ **±2.3pp** of standard error, and the bar itself carries
≈ ±2pp (644 changed moves) ⇒ total uncertainty ≈ 3pp against a 1.5pp effect. ⚠️ The sub-slices are worse than useless —
`.middlegame` reads 100.0% on **n=4** — quoted here only so nobody later mistakes them for a signal.
**Final ledger for space:** ✅ correctly built and verified (oracle 0 mismatches on both form families incl. every axis
flipped · colour symmetry 0/800 · byte-identical with the knob off, on both the term build and the probe build) ·
❌ globally INERT in the linear form (all arms inside §I's ±0.05 floor) and globally HARMFUL in SF's quadratic form ·
★ class-local positive on `centre_locked` with a MECHANISM (locked centres are where mobility collapses, so safe-square room
carries what mobility cannot) · ⚠️ that class is 3.2% of positions and **cannot be thickened from our pools** (+13,500
positions bought 322 rows and 37 changed moves).
⇒ **PARKED, built and default-off. Named trigger: a PURPOSE-BUILT closed-centre corpus** (King's Indian / French / Closed
Sicilian structures, mined or generated), not another pass over these pools. It does NOT ride the slice-3 bundle — same
ruling as `exlow`: a term without a resolvable case is not carried by a bundle bar that would absorb it silently.
★ **The general limit this established, which outlives space:** our corpora resolve a term only if its REGIME is a large
enough share of play AND it changes enough moves — **the changed-move subset is the real sample, not the corpus**. Enlarging
`centre_locked` by 20% moved the changed-move count by 6%. Any term whose regime is a few percent of positions is
unresolvable here at the ~2-2.5pp bar however real its mechanism.

▶️ Plan, stated before the numbers (kept for the record): ONE bounded regret read on the enlarged locked class against a
neutral measured on THAT corpus, at a magnitude where the class effect is largest. If it does not clear, space is PARKED with its measured case —
correctly built (oracle + symmetry + byte-identity), globally inert, class-local positive with a mechanism, class too rare
to resolve — and its named trigger is **a purpose-built closed-centre corpus**, not a re-measurement on these pools.
☠️ **Oracle history, kept because both mistakes are instructive:** v1 of it read a probe that did not exist (exiting 2
rather than passing vacuously); v2 forgot that the probe fills the count slots even when the material gate CLOSES, which
would have made every gated position a false mismatch.

☠️ **Superseded note — `_space_detector_oracle.py` could not gate anything as first written:** It rebuilds the whole term from
python-chess (region, safe mask, behind double count, weight, gate, taper) but reads the engine's value through a probe
`eval_breakdown_space` **that does not exist**; it exits 2 with a message instead of silently passing. ⇒ Before space gets
a single magnitude number, add a `space_probe` alongside `mobility_probe`/`placement_probe` (diagnostic-only, never called
from search) and re-verify. ★ A tool that LOOKS like a gate but cannot fire is worse than no tool — the same class of
defect as the unreachable corpus class and the broken KPK oracle, both caught this week.

#### 2.2a THREATS build status (2026-09-16) — ✅ BUILT AND COMPILING, ☠️ NOT YET VERIFIED
Knobs: `THREAT_V2_PCT` (percent of SF11's pawn-converted value, the placement anchor, so it ladders against BOTH anchors) ·
`THREAT_V2_GATE` 0 = SF `stronglyProtected` (enemy's view) / 1 = Ethereal `poorlyDefended` (victim's view, pawn support
overrides) · five switchable legs where the references split: `HANGING` (SF only) · `RESTRICT` (SF only) · `KING` (3/4) ·
`PAWN_TARGETS` (3/4, eg-weighted) · `PUSH` (4/4).
Constants: SF11 evaluate.cpp:116-121 / :133-147, each leg converted by ITS OWN phase's pawn (mg /128, eg /213, ×1000) —
`ThreatBySafePawn` is the largest at 1,352 mp mg, matching the 4/4 cross-engine shape agreement.
★ Rides the EXISTING attack maps (`SideAttacks.by[]`, `.dbl` = SF's attackedBy2); dispatch gate widened to
`KS_V2_MAX > 0 || mob_on || space_on || threats_on`. No second attack pass.
✅ Byte-identity with threats OFF: **SHIP+E `250 / 61,352,373 / EBF 4.114` · v1 `250 / 35,310,778 / EBF 3.784` — EXACT.**
☠️ **ORACLE FIRST RUN FAILED ON MY OWN BUG — THE SAME BUG CLASS AS SPACE'S, ONE WEEK LATER.** Every mismatch had the SCORE
matching EXACTLY (−230/−230, 102/102, 152/152, 161/161) and differed only in COUNT slots, always with the oracle at 0.
Cause: `threats_probe` fills each leg's detector count UNCONDITIONALLY (before scoring), while the oracle filled a count only
when that leg's knob was on — and the run had HANGING / RESTRICT / KING / PUSH off. ⇒ Fixed in the ORACLE (counts
unconditional, scoring gated) and the probe's CONTRACT corrected in `eval_v2.h`, `eval_v2.cpp` and the pyx docstring, which
all wrongly claimed "counts as scored".
★ **The diagnostic rule worth keeping: when a probe and an oracle agree on the SCORE but disagree on COUNTS, suspect the
GATING CONVENTION, not the detector.** Matching scores across 2,000 positions is strong evidence the maths is right.
☠️ I wrote this exact lesson into the space oracle's toolkit row ("the probe fills the count slots even when the material gate
CLOSES") and then reproduced it four legs over. Knowing a failure mode is not the same as having a habit that prevents it.

✅ **ORACLE PASSES at the SF gate form after the fix: 2,005 positions, 0 mismatches, fired 38.6%** (counts AND score).
✅ Byte-identity re-verified on the probe build: SHIP+E `250 / 61,352,373 / EBF 4.114` · v1 `250 / 35,310,778 / EBF 3.784`.
✅ **ALL ORACLE ARMS PASS — threats is fully verified on every axis:**
| arm | positions | mismatches | fired |
|---|---|---|---|
| SF gate (`GATE=0`), core legs, XRAY=1 | 2,005 | **0** | 38.6% |
| Ethereal gate (`GATE=1`), ALL five legs, XRAY=1 | 2,005 | **0** | **82.3%** |
| SF gate, ALL legs, **XRAY=0** | 2,005 | **0** | **94.8%** |
⇒ both defence-gate definitions, all seven legs, and both occupancy settings agree with an independent python-chess rebuild on
COUNTS and SCORE. ★ The rising fire rate (38.6% → 82.3% → 94.8%) is the non-vacuity evidence: the switchable legs genuinely
activate rather than passing by never firing — the failure mode that made June's KPK oracle worthless.
✅ **Symmetry with threats ON and all five legs on** (Ethereal gate, the widest surface): **colour swap 0/800 violations**;
file mirror at the pre-existing 21 @ 5 mp baseline. Run on the FULL proposed stack (SHIP+E + pin + exlow + threats), so the
mirror is clean for the whole configuration, not just for threats in isolation.
⏳ **Collinearity gate RUNNING, extended for threats** (🧰 `_v2_term_collinearity.py` now reads `threats_counts`' seven leg
counts beside mobility and placement). ★ This is the gate slice 3 §0 demanded BEFORE any threats magnitude: SF's own comment
says `RestrictedPiece` reads the SAME attack maps as the mobility area. ⚠️ Under the ONE-OWNER rule a flagged `RESTRICT` is
REDEFINED or DROPPED, not laddered — so this runs before the §I ladder, inverting slice 2's order on purpose.
★ The probe's unconditional leg counts (see the corrected contract) are what let ONE run gate every leg at once, with only
`THREAT_V2_PCT>0` needed.

✅ **COLLINEARITY GATE PASSES — 10,000 positions, 21 terms, NO FLAGS.** The suspect the sources warned about is clean:
| term | VIF | | term | VIF |
|---|---|---|---|---|
| **`th_restricted`** | **1.10** | | `th_minor` · `th_rook` | 1.05 · 1.11 |
| `th_hanging` | 1.10 | | `th_king` | 1.03 |
| `th_safepawn` · `th_push` | 1.05 · 1.06 | | placement terms | 1.01-1.23 |
★★ **`RESTRICT` does NOT duplicate mobility's area** (VIF 1.10, no pair at \|r\| ≥ 0.70) even though SF's comment notes they
read the same attack maps. Sharing an INPUT is not sharing a SIGNAL — mobility counts squares a piece can reach, restricted
counts squares the ENEMY's pieces are denied, and on our corpora those move nearly independently. ⇒ Under the one-owner rule
there is nothing to redefine: every threats leg may be laddered as built.
⚠️ The four VIF flags are all MOBILITY'S OWN INTERNALS — `mob_table_eg` 20.10 · `mob_table_mg` 13.17 · `mob_R` 9.22 ·
`mob_B` 5.42 — i.e. per-type counts against their own table sums. That is the SAME detector by construction and is excluded
from the flag rule by design (the tool's `MOB` set). Quoted here so a later reader does not mistake them for a defect.
▶️ **Threats has now passed every gate it owes** (oracle on both gate forms + all legs + both occupancies · symmetry on the
full stack · byte-identity off · collinearity) ⇒ **the §I ladder is unblocked**, and it is the first slice-3 term to reach a
magnitude ladder with its detector fully verified first.

⏳ **§I LADDER RUNNING — 11 arms on the FULL proposed base (SHIP+E + pin + exlow):** magnitudes `th10 · th25 · th50 · th100`
(percent of SF11's pawn conversion) · the Ethereal defence gate `th25g1` · and each switchable leg alone at 25%:
`th25hang · th25restr · th25push · th25pawnt · th25king`. Rule unchanged: better than `ship` on BOTH mean and worst, ±0.05 floor.
**REGISTERED PREDICTIONS (2026-09-17, before any number — my slice-3 record on these is 2 right, 6 wrong or unreadable):**
1. ⚠️ **Some magnitude clears the both-better rule.** This is the first slice-3 term with a real prior in its favour: 4/4
   references carry a threat family, v1's version shipped (however badly measured, at +45 ±40.6), and unlike the pair/space
   it prices a relationship no shipped v2 term owns. I am registering an OPTIMISTIC call here having registered pessimism for
   space — if it fails, the honest summary is that v2's positional signal is saturated at this stage, not that I mis-sized it.
2. The mean optimum is at **25-50**, not 100: the pawn conversion puts `ThreatBySafePawn` at 1,352 mp mg, ~45× v2's knight
   PST spread, and every term laddered on this base so far has wanted well under its pawn-converted value.
3. `th25restr` is the most likely leg to HURT the worst column: it is a per-square count over a large set (11-16 squares per
   side in the oracle's sample), so it adds a broad, low-information term — the shape that has taxed KS-critical accuracy.
4. `th25hang` helps: v2 has NO capture-gains term, so hanging has no other owner here (the premise that made it redundant in v1
   does not transfer), and it fires on a small, high-information set.
5. `th25g1` (Ethereal's victim-side gate) differs from the SF gate by LESS than the magnitude steps do — a gate change
   re-labels which victims count, while a magnitude change rescales everything.

✅ **§I LADDER RESULT (11 arms, negative = better; base MSEs on the SHIP+E+pin+exlow stack):**
| arm | mean% | WORST% | | arm | mean% | WORST% |
|---|---|---|---|---|---|---|
| th10 | −0.74 | **+0.35** | | th25g1 (Ethereal gate) | −1.46 | +0.88 |
| th25 | −1.71 | +0.90 | | th25hang | **−2.38** | +1.61 |
| th50 | −2.91 | +1.85 | | th25restr | −1.68 | +1.09 |
| th100 | **−3.80** | **+3.90** | | th25push | −1.80 | +1.24 |
| | | | | th25pawnt | −2.01 | +0.83 |
☠️ **NOTHING CLEARS THE BOTH-BETTER RULE — my optimistic prediction (1) was WRONG.** But this is the sharpest and most
*coherent* trade of the slice: every arm improves the four general corpora AND the variant corpus strongly (`th100` reads
**−9.71% on `variant_regret_set`**, the largest single-corpus gain measured on this base) while taxing `lichess_ks_labelled`
in near-exact proportion. Mean and worst both scale monotonically with magnitude — a clean dose-response, not noise.
**Prediction scorecard:** ❌ (1) nothing cleared — recorded as a failed OPTIMISTIC call, the mirror of space's pessimistic one.
✅ (2) the useful magnitudes are at the low end (10-25), far below the pawn conversion. ❌ (3) WRONG — `th25restr` (+1.09) is
among the GENTLER legs on the worst column, not the harshest; its broad per-square count did not tax KS-critical as I argued,
which is consistent with its clean VIF of 1.10. ✅ (4) `th25hang` is the strongest leg on mean (−2.38), as predicted from v2
having no capture-gains owner. ✅ (5) the gate change (−1.46 vs −1.71) moves less than one magnitude step.
★ **What this trade probably IS, and why it is not a reason to stop:** threats prices *relationships between pieces*, which is
exactly what the KS-critical corpus measures with its own machinery (king-zone attacks). Two terms describing overlapping
facts in the same positions is the classic double-count shape — yet the collinearity gate says `th_*` × mobility/placement is
clean, and KS has NO count probe, so **the one overlap that could explain this is the one the gate cannot see.**
▶️ **NEXT, in this order (the KS probe is the honest prerequisite, but the move-level read is cheap and answers first):**
1. Regret for **`th10`** (worst +0.35, the only arm near the ±0.05 floor) on both corpora — ⚠️ against a NEUTRAL MEASURED ON
   THIS BASE. The 49.9 / 51.1 bars belong to SHIP+E, NOT to SHIP+E+pin+exlow, so they do not transfer.
   ✅ **New base neutral, primary corpus: 50.1%** (5,047 changed / 33.6%, delta +0.045) — vs 49.9% on the SHIP+E base.
   ★ Close to the old bar, but MEASURED rather than assumed: the drift is small here and was 1-2pp between other bases, so
   borrowing a bar is a coin-flip we do not need to take.
   ☠️ **`th10` on primary: 50.1% vs the bar of 50.1% — an EXACT NULL**, on 5,233 changed moves (34.9%), delta +0.020.
   Slices: `.endgame` +51.6% / `.midgame` 48.8% — opposed signs inside one corpus, i.e. noise, not structure. `cr4` 51.4% on
   n=39 (unreadable, quoted so it is not mistaken for a signal).
   ⇒ At the ONE magnitude whose worst column sat near the ±0.05 floor, threats changes a third of our moves and improves
   none of them. ⚠️ Larger magnitudes buy general accuracy but tax KS-critical monotonically, so there is no magnitude that
   is both readable on §I and non-harmful there. ⏳ `_v2` candidate + its own base neutral running (the 51.1 bar belongs to
   SHIP+E and does NOT transfer — I launched the candidate before its neutral, which is the gap that made pin's first class
   read uninterpretable; fixed by queueing the neutral immediately).
   ▶️ Provisional reading, stated before the `_v2` pair lands: threats is **verified, coherent on accuracy, and null on
   moves** — the same shape as the bishop pair. If `_v2` agrees, threats is PARKED (built, default-off) and the remaining
   question is whether its KS-critical tax is a genuine double-count, which needs the KS count probe to answer at all.

★★ **THREATS VERDICT (2026-09-17): PARKED — fully verified, coherent on accuracy, NULL on moves across BOTH corpora.**
The provisional reading above held exactly.
| corpus | `th10` | bar (measured on THIS base) | gap | changed |
|---|---|---|---|---|
| primary | **50.1%** | 50.1% | **0.0pp** | 5,233 (34.9%) |
| `_v2` | **49.7%** | 50.5% | **−0.8pp** | 4,335 (36.3%) |
⇒ Cross-set replication CONFIRMS the null rather than rescuing it: one corpus reads exactly the bar, the other 0.8pp under,
both far inside the ~2-2.5pp resolution, on footprints of 35-36% of all moves. A term that changes a third of our moves and
improves none of them is not under-measured — it is not adding move-level information at this magnitude.
**Final ledger for threats:** ✅ best-verified term of the slice (oracle 0 mismatches over BOTH gate forms × all 7 legs ×
both occupancies, fire 38.6→82.3→94.8% · symmetry 0/800 on the full stack · byte-identity exact off · collinearity 21 terms
no flags, `th_restricted` VIF 1.10) · ✅ coherent §I dose-response (up to −9.71% on the variant corpus) · ❌ no magnitude
clears both columns · ❌ regret null on both corpora at the only near-floor magnitude.
★★ **KS COUNT PROBE BUILT, AND THE DOUBLE-COUNT STORY IS NOT SUPPORTED (2026-09-17).** `ks_probe` / `ChessAI.ks_counts`
exposes six channels per king (attacker count · weighted attacker sum · weak zone squares · king-adjacent attacks ·
safe-check squares · the scored unit total), and `_v2_term_collinearity.py` now carries them — closing the gate's last
coverage hole, open since slice 2.
**Result (10,000 positions, 27 terms): NO cross-subsystem flag. Threats does NOT re-express king safety.**
| threats leg | VIF | | KS channel | VIF |
|---|---|---|---|---|
| `th_restricted` | 1.27 | | `ks_adj` | 2.45 |
| `th_hanging` · `th_rook` | 1.12 | | `ks_weak` | 2.17 |
| **`th_king`** | **1.10** | | `ks_units` | 1.69 |
| `th_minor` · `th_push` · `th_safepawn` | 1.05-1.09 | | `ks_checks` | 1.43 |
★ `th_king` is the leg that counts the king's OWN attacks on enemy men — the most obvious overlap candidate — and it reads
1.10. No pair reaches \|r\| ≥ 0.70.
⇒ ☠️ **The tidy explanation for threats' KS-critical tax is REFUTED on the one instrument that could see it.** Threats does
not double-count king safety; the tax is something else (mis-calibration in sharp positions, or the KS-critical corpus
simply rewarding a different balance). Elegance of explanation is not evidence — and this is the third time this week a
mechanism story of mine failed its own test (the pin/exlow "interference", the space "centre-tension" prediction, this).
⇒ **Reopening the KS rung has NO evidence behind it.** The owner's question ("have we infringed on KS's domain?") now has a
measured answer: not measurably. KS stays as shipped at +101 Elo.
☠️ **One flag did fire, and it was MY column choice, not a finding:** `ks_natt × ks_watt r=+0.93` — the attacker count and
the weighted sum over the IDENTICAL piece set, i.e. one detector twice, plus `ks_units` as their scored total. Fixed with a
`KS_SET` intra-subsystem exemption mirroring the existing `MOB` one. ⚠️ A gate that flags its own redundant columns trains
you to ignore it, which is worse than no gate.

★★★ **THE SHAPE MISMATCH — what the source read found that the collinearity gate STRUCTURALLY CANNOT SEE (owner's idea to
compare how the giants balance the two, 2026-09-17).** In SF and Ethereal, king danger is an **unbounded quadratic** while
threats is **linear**, so KS OVERTAKES threats ~2:1 in severe attacks. Ours SATURATES: `KS_V2_MAX=4000` caps king danger at
4.0 pawns while threats at `th100` reaches 4.2 ⇒ **in exactly the KS-critical regime our two terms are level where every
reference lets KS win.**
| engine | KS transform | KS:threats quiet → severe |
|---|---|---|
| SF11 / SF15.1 | `kd²/4096`, gate `kd > 100`, NO ceiling | 0.5 → 0.8 → **1.7-2+** (crosses 1 at kd ≈ 1150) |
| Ethereal | `−mg·max(0,mg)/720`, hard onset (needs ≥1 attacker + enemy queen) | 0.3 → 1.2 → **2.2** |
| Weiss | `attackPower·CountModifier/128`, linear, `CountModifier[0]=[1]=0` | 0 → 0.8 → 1.1 |
| **ours** | `4000·u'²/(u'²+600²)`, `u' = u − 450` ⇒ **asymptote 4.0p** | 0 → 0.7 → **0.85, never > ~1** |
★ This RECONCILES the two facts that looked contradictory: the detectors genuinely do not overlap (VIF ≤ 1.27, measured) AND
the tax is real — because **VIF measures co-movement of detector COUNTS, not the relative HEIGHT of the scored curves in the
severe regime.** Two different questions; I had been treating the gate's clean result as settling both.
★ Source facts worth keeping: **no reference has any data path between threats and king danger** (SF feeds only `mobility`
into kingDanger; Ethereal and Weiss feed KS from their own attack sets) · **none damps one when the other fires** — no shared
cap, no min/max, no "already counted" discount · **threats is never gated by material or phase in any of the four**, only the
mg/eg taper · all four DO double-count the ring-resident-weak-defender subset and accept it, which matches our VIF 1.05-1.27.
▶️ **2×2 RUNNING (the discipline's own rule: run the 2×2 first):** `ship · th100 · ksmax6000 · th100+ksmax6000 ·
th100+ksmax8000 · th100+kshalf400`. The `ksmax6000`-alone arm is the CONTROL — raising the ceiling may help or hurt on its
own, and without it a joint improvement is unattributable.
⚠️ **Constraints on acting on this, registered before the numbers:** (a) KS-A shipped at **+101 Elo** and its magnitude was
tuned with threats ABSENT ⇒ [[a-correctness-fix-into-absorbed-tuning-is-not-free]]; (b) v1's KS floor/knee shaping was
REFUTED on clean instruments ([[taper-and-ks-floor-knee-both-refuted-on-clean-instruments]]), so a ceiling change is a
candidate, not a fix; (c) any KS change rides GAMES, never the accuracy instrument alone.
⚠️ Also owed, and cheap: the gate's 10,000 positions were a GENERAL sample, while the tax lives on `lichess_ks_labelled`.
**Overlap is a property of a POPULATION** — `_v2_term_collinearity.py` now takes `SETS=` so the gate can be re-run on the
corpus that actually shows the tax. That is the honest way to close the overlap story rather than assuming the general
sample generalises.

✅ **2×2 RESULT (the shape hypothesis, tested rather than argued):**
| arm | mean% | WORST% (KS-critical) |
|---|---|---|
| `th100` | −3.80 | **+3.90** |
| `th100 + ksmax6000` | −3.97 | +2.96 |
| `th100 + ksmax8000` | −3.80 | **+2.13** |
| `th100 + kshalf400` | **−4.14** | +2.20 |
| `ksmax6000` ALONE (control) | −0.15 | **+0.54** |
★ **Raising the KS ceiling HALVES threats' KS-critical tax (3.90 → 2.13) while keeping the general gains** — the direction
the source contrast predicted, so the shape mismatch is REAL and accounts for roughly half the effect.
☠️ **But it does not rescue threats, for two reasons:** (a) +2.13 is still ~40× the ±0.05 floor, so no arm clears the
both-better rule; (b) **the control earns its keep** — `ksmax6000` alone is +0.54 on the worst column, so part of the
apparent "fix" is just raising a ceiling the KS-critical corpus mildly dislikes on its own terms. Without that arm the joint
improvement would have looked like clean attribution.
⇒ Acting on this would mean reopening a rung that shipped at **+101 Elo**, on ACCURACY evidence alone, for a term that is
move-null. That is exactly the trap registered before the run ⇒ **not done.** `KS_V2_MAX` stays 4000.

✅ **PER-CORPUS COLLINEARITY on `lichess_ks_labelled` (5,000 positions, 27 terms) — the overlap story is now closed on the
corpus that SHOWS the tax, not just on a general sample:** every threats leg VIF ≤ 1.30 (`th_king` 1.10 · `th_restricted`
1.30 · `th_minor`/`th_rook` 1.10 · `th_hanging` 1.16). Largest threats×KS correlations: `th_restricted × ks_natt` **+0.32**,
`th_restricted × ks_adj` +0.26, `th_king × ks_adj` +0.23 — all far below the 0.70 flag. (KS's internals correlate with each
other as expected: `ks_natt × ks_watt` 0.93, `ks_adj × ks_weak` 0.61, `ks_units × ks_watt` 0.62 — one subsystem, exempt.)
★ **This is the stronger version of the earlier result, not a repeat:** overlap is a property of a POPULATION, and the
detectors stay disjoint even in the regime where the tax appears. ⇒ The tax is a SCORED-CURVE SHAPE effect (half of it) plus
something still unexplained — never a detector double-count.

⇒ **PARKED, built and default-off.** Named triggers: (a) ~~a KS count probe~~ **BUILT — it refuted the DETECTOR-overlap
story on BOTH a general sample and the KS-critical corpus; the SHAPE mismatch explains ~half the tax but its fix requires
reopening a +101 Elo rung, so it is recorded and NOT acted on** — if the KS-critical tax IS a double-count with
king safety, then threats may be a net gain once KS is re-scaled, and that is testable rather than arguable; (b) **lazy eval**
(owner's §2.5b) — threats is the most expensive term in the slice, so if it is strength-neutral its cost matters more than
its score; (c) the SF-form omissions we did not build (`Knight/SliderOnQueen`, `WeakQueenProtection`).
☠️ **My optimistic prediction for this ladder was wrong, and the pattern across slice 3 is now the finding itself:** central
(not built, 0/5 references) · bishop pair (already owned) · space (globally inert) · threats (verified, null on moves). Four
concepts the references carry, and on top of v2's current KS + pawns + mobility + placement, none of them adds measurable
move-level information. The one thing that DID pay (+31 Elo) refined the AREA of a term we already own. ⇒ Working hypothesis
for the rest of the rebuild: **v2's positional signal is closer to saturated than its term COUNT suggests, and the remaining
gains are in the definitions of the terms we have, not in new concepts.** ⚠️ That hypothesis is now load-bearing, so it needs
its own test, not just a pattern of nulls — the KS probe is the cheapest one available.
2. If regret is positive, the KS-critical tax becomes the question, and answering it needs **a KS count probe** so the
   collinearity gate can finally see king safety — the coverage hole flagged since slice 2.

☠️ **NO FULLY VERIFIED ORACLE YET — no magnitude may be read.** Five legs × two gate forms is more interacting surface than space had, and
space's oracle needed two revisions when written blind. Next: a `threats_probe` (per-side leg COUNTS, not just the score, so a
compensating pair of errors cannot pass) + `_threats_detector_oracle.py`, then symmetry, then the collinearity gate for
`RESTRICT` vs mobility's area (the sources themselves flag the shared attack maps), and only then a §I ladder.

### 2.2 THREATS (new; SF's FORM family, not a constants grid)
- **Core (4/4):** pawn-attacks-a-piece (the largest constant everywhere), safe pawn-push threat, target-indexed piece threats.
- **Forms as knobs:** `THREAT_V2_GATE` 0 = SF (attacker-side `safe` for pawns; `defended|weak` for minors, `weak` for the rest)
  · 1 = Ethereal victim-side `poorlyDefended` (pawn support overrides). `THREAT_V2_QUEEN` 0 = SF `Knight/SliderOnQueen` ·
  1 = Ethereal `QueenAttackedByOne` · 2 = off. `THREAT_V2_PAWN_TARGETS` (3/4, eg-weighted). `THREAT_V2_HANGING` (SF-only 2/4 —
  ★ but v2 has NO capture gains, so v1's "87% a subset of capgains" argument does not transfer: here it would have no other owner).
  `THREAT_V2_RESTRICT` (SF `RestrictedPiece`) — ⚠️ gate it against mobility first; it is the same attack maps.
- **Magnitude:** one `THREAT_V2_MAG` percent over the whole family, ladder BOTH anchors. Ratios to size against (§1.3): the pawn
  leg is ~2× the knight PST spread in the SF lineage.
- ☠️ **One-owner risk:** SF's `Knight/SliderOnQueen` sits beside **our shipped weak queen** (placement E). Gate that pair.

### 2.3 BISHOP PAIR (new; v2 has no pair term at all) — ✅ BUILT 2026-09-16 (first slice-3 code)
Knobs: `BPAIR_V2_MAG` (midgame mp; 0 = off = byte-identical) · `BPAIR_V2_FORM` 0 flat (eg == mg, SF lineage) · 1 endgame-heavy
(eg = 3.5× mg; Ethereal 4:1, Weiss 3.3:1) · 2 flat + SF's own-pawn coupling (+2.8% of the pair per own pawn, POSITIVE sign).
`bishop_pair_mp()` sits with rung-0 material (a census term, not placement), per side then differenced so the colour mirror
swaps identical computations. Two or more bishops of any colour complexion count (4/5; only Ethereal requires opposite colours).
Sizing anchors for the ladder: reference ratios put it at **1-2× v2's ~30 mp knight PST spread (≈30-60 mp)**, while the pawn
conversion would say ~700 — ladder BOTH, which is slice 2's standing lesson.
✅ **Gates so far:** build clean · v1 control EXACT (`250 / 35,310,778 / EBF 3.784`) · colour-symmetry **0/800 violations** with
the pair ON at 40 mp in the PAWN-COUPLED form (form 2, the variant most able to break the mirror since it reads a per-side pawn
count); file mirror at the pre-existing 21 @ 5 mp baseline. **Shipped-config fingerprint with the pair OFF: `250 / 61,352,373 /
EBF 4.114` — EXACT** ⇒ the knob is genuinely inert at default, on both arms. All three build gates pass.
⏳ **§I ladder running, 10 arms across BOTH anchors:** `ship · bp20 · bp40 · bp60 · bp100 · bp300 · bp700 (pawn conversion) ·
bp40f1 · bp100f1 (endgame-heavy) · bp40f2 (pawn-coupled)`. Rule unchanged: better than `ship` on BOTH mean and worst.
**Registered predictions (before any number):** (1) the optimum sits in 20-100, NOT at the pawn conversion — `bp700` clearly
worse (the tempo/mobility lesson cuts both ways, and v1's `BISHOP_PAIR_BONUS=300` is already ~10× the reference ratio).
(2) `bp300` and `bp700` degrade the KS-critical worst column fastest, as every over-sized positional arm has. (3) The three
forms are within ±0.15 of each other at equal magnitude — phase shape should matter less than size for a census term.
(4) At least one arm clears the both-better rule: unlike slice 2's candidates this is a term v2 does NOT have at all, so it adds
signal rather than re-weighting an existing one.

✅ **§I LADDER RESULT (10 arms, N=2500 × 6 corpora; negative = better; WORST decides):**
| arm | mean% | WORST% | | arm | mean% | WORST% |
|---|---|---|---|---|---|---|
| bp20 | −0.06 | **+0.05** (at the ±0.05 floor) | | bp300 | −0.61 | +0.73 |
| bp40 | −0.11 | +0.09 | | bp700 (pawn conversion) | −0.69 | **+1.78** |
| bp40f2 (pawn-coupled) | −0.12 | +0.11 | | bp40f1 (eg-heavy) | −0.24 | +0.13 |
| bp100 | −0.25 | +0.24 | | bp100f1 | −0.53 | +0.32 |
☠️ **NO arm clears the both-better rule.** The shape is perfectly monotone: every magnitude helps the four general corpora and
hurts `lichess_ks_labelled` (the KS-critical column), and BOTH effects grow with size. The trade is the term's signature here,
not a magnitude artefact — exactly the "mean-only" shape that disqualified slice 2's eg-share and magnitude arms.
**Prediction scorecard:** ✅ (2) `bp300`/`bp700` degrade the KS-critical worst fastest (+0.73 / +1.78). ✅ (3) the three FORMS sit
within ±0.13 at equal magnitude (bp40 −0.11 · bp40f2 −0.12 · bp40f1 −0.24) ⇒ for a census term, SIZE matters more than phase
shape. ⚠️ (1) half right: the optimum is NOT at the pawn conversion on the WORST column (+1.78), but mean keeps improving all
the way to 700 — so "bp700 clearly worse" was wrong on mean and right on worst. ❌ (4) WRONG — no arm clears; a term being
genuinely NEW does not exempt it from the KS-critical trade.
▶️ Decision: do NOT ship on §I. Take the smallest arm whose worst is AT the floor to the move-level instrument — `bp40` gated on
both corpora vs neutrals (primary 49.9 · `_v2` 51.1).
**Regret:** `bp40` on `_v2` **51.0%** (−0.1pp; 3,059 changed / 25.6%) · primary **49.5%** (−0.4pp; 3,021 changed / 20.1%,
delta −0.008). **NULL on both corpora**, signs slightly negative, well inside the bar.
★★ **VERDICT — BISHOP PAIR: v2's PST + mobility ALREADY OWN IT.** Three instruments agree, which is why this is a finding
and not a shrug: (1) §I — every magnitude helps the general corpora and hurts the KS-critical one, so nothing clears the
both-better rule; (2) per-class §I — **flat in all five classes** (−0.05..−0.20, no class above its global −0.11), so there
is no hidden regime where it matters; (3) regret — null on both corpora on a 20-26% footprint.
⇒ The ONE-OWNER RULE, established by measurement rather than argument: the concept 5/5 references pay for is, in v2,
already priced by the bishop's PST cells and by the mobility a second bishop buys. ☠️ This is exactly the trap the charter
warns about — "5/5 references have it" is an argument for a CANDIDATE, never for a second owner ([[adopt-reference-methods-only-if-universally-superior]]).
⚠️ Bounded claim: this says the pair adds nothing ON TOP OF v2's current PST and mobility. If a later rung re-scales either
owner (e.g. the PST-scale question in §2.0b), the pair must be re-measured, not assumed dead. `BPAIR_V2_MAG` stays built and
default-off with that as its named trigger. Precedent: placement's trapped rook showed the same "good on mean, turns
harmful above a threshold" shape and shipped at the small magnitude (10%) where the harmful column stayed inside the floor.
⚠️ If regret is also null, the honest outcome is that v2's PSTs + mobility already price the pair's signal — which would be a
REAL finding (the one-owner rule, arrived at by measurement rather than by argument), not a failure to find something.
- 5/5 pay it, at ≈1-2× the knight PST spread in mg. `BPAIR_V2_MAG` + `BPAIR_V2_FORM` 0 = flat, mg=eg (SF lineage) ·
  1 = eg-heavy ≈3-4:1 (Ethereal/Weiss) · 2 = flat + SF's pawn-count coupling (+2.5/pawn equivalent, POSITIVE sign).
- Cheapest term in the slice: one popcount per side. No detector risk, but it still gets an oracle row for symmetry.

### 2.4 KAUFMAN IMBALANCE (port the FORM, not v1's tables)
- Census-product form (piece counts × piece counts, bishop pair as a pseudo-piece), 3/5 carry its core ideas (rook redundancy,
  knight-likes-pawns). ☠️ v1's ridge-fitted coefficients do NOT transfer — they were fit against v1's SF11 residual with v1's
  other terms live ([[a-correctness-fix-into-absorbed-tuning-is-not-free]]).
- ⚠️ Touches slice 4's census (OCB, pawn count) ⇒ defer the magnitude ladder to the slice 3/4 checkpoint, or build it LAST in
  this slice so the pair term's owner is settled first.

## 2.5 OWNER'S TWO PROPOSALS (2026-09-16) — the parked shelf, and lazy eval

### 2.5a ★★ BUNDLE THE PARKED TERMS AS A MEASUREMENT (owner) — CONFIRMED ON §I
Owner: *"marginal increases in things like outposts likely only make a difference in a few games, so I can see how it is quiet
and hard to measure. Bundling things when testing might help."* ⇒ Never tried for the PARKED terms: bundle E (placement) was
bundled and games-confirmed, but `MOB_V2_PIN`, `MOB_V2_EXCL_LOWRANK`, `BPAIR_V2_MAG` and `SPACE_V2_MAG` had only ever been
measured ALONE, each below its instrument's resolution.
✅ **§I additivity check (6 corpora; negative = better):**
| arm | mean% | WORST% | note |
|---|---|---|---|
| `all4` (pin + exlow + bpair40 + space560 linear) | **−1.70** | **−1.05** | better on ALL SIX corpora |
| `mobpair` (pin + exlow) | −1.59 | −0.92 | pin alone −0.51 + exlow alone −1.08 = **−1.59 — exactly additive** |
| `pairspace` (bpair40 + space560) | −0.12 | +0.10 | ≈ null, as both were alone |
| `all4big` (bpair60 + space1120) | −1.76 | −1.11 | marginally better; within noise of `all4` |
★★ **The owner's argument is right on this instrument:** the group's effect (~1.7% mean, ~1.05% worst) is **~30× §I's ±0.05
floor**, where each member alone sat at or under it. ☠️ **And my "pin and exlow INTERFERE" inference was wrong** — it came
from one `_v2` regret reading (pin +0.6 / exlow −0.4 / together −1.8); on accuracy they are exactly additive. A single
sub-bar regret reading was never evidence of interference.
⚠️ Honest attribution: `mobpair` carries −1.59 of the −1.70, so **the bundle is mostly the two mobility-area terms**; the pair
and space contribute ~0.1 and are along for the ride (they are also the two with their own measured nulls).
**Bound flipped vs a shipping bundle.** My earlier ruling (a "costs ≤ 10 Elo" bar absorbs unmeasured terms silently) stands for
SHIPPING; as a MEASUREMENT the question is the opposite. ⇒ **Clearance SPRT at elo0 0 / elo1 +10 — "does this group ADD?"**
Registered rule, before the games: **H1 ⇒ the group ships as a bundle** (then attribute by leave-one-out with a dilution
control). **H0 ⇒ all four are RETIRED**, not re-parked — a group that cannot beat 0 Elo collectively has no case left.

★★★ **CLEARANCE SPRT RESULT (2026-09-16): H1 ACCEPTED. `+248 −160 =101 of 509 (58.6%), elo +60.7 ±35.5, LLR +3.008`**
(A = SHIP+E + pin + exlow + bpair40 + space560-linear · B = SHIP+E · elo0 0 / elo1 +10 · LIGHTNING · seed 16).
The estimate was flat at ~+60 from game 181 (+1.03 LLR) through 509 — it did NOT drift the way mobility's did (+31 → +6).
⇒ **The owner's proposal is correct and is now the second-largest measured gain of the rebuild** (rung 1 KS +101 · THIS +60 ·
rung 2 pawns +60.4 · mobility ≈ +162 stands apart).
☠️ **This overturns four of my own judgements, and the pattern in them is worth more than the result:**
1. `MOB_V2_PIN` — "null on its own class, parked."  2. `MOB_V2_EXCL_LOWRANK` — "no measured case; **WITHDRAWN from the
bundle**" (the exact opposite ruling to what the games just showed).  3. `BPAIR_V2_MAG` — "already owned by PST + mobility,
three instruments agree."  4. `SPACE_V2_MAG` — "globally inert."
Each verdict was defensible ON ITS INSTRUMENT and each was wrong about what the terms are worth TOGETHER. The error was not
in any single reading — it was treating "individually unresolvable" as "individually worthless", which is a category mistake
about the INSTRUMENT, not about chess. ★ The arithmetic was always available: 4 terms × ~4 Elo needs ~25,000 games apiece
but only ~500 as a group, and I applied that reasoning to placement (bundle E) while failing to apply it here.
⚠️ What the result does NOT establish: (a) the magnitude — an SPRT bounded at ≥10 crosses fast when the truth is far above it,
so +60.7 ±35.5 is "clearly beats 10", not a point value; (b) attribution — §I says `mobpair` carries −1.59 of the −1.70, so
most of it is probably pin + exlow, but **games cannot attribute at this effect size** (a 2-term vs 4-term A/B is a ~0 Elo
question). ⇒ Attribute on §I leave-one-out WITH a dilution control, and treat the ranking as indicative, not proven.
▶️ **Proposing the whole four ship AS TESTED** (owner's call): shipping a SUBSET would ship a configuration no game ever saw.

✅ **§I LEAVE-ONE-OUT ATTRIBUTION, measured against the FULL bundle** (positive = removing it makes accuracy WORSE, so larger
= more load-bearing). Base MSEs are the `all4` arm's.
| removed | mean cost | WORST cost | reading |
|---|---|---|---|
| `exlow` | **+1.09** | **+1.65** | the largest contributor by ~2× |
| `pin` | +0.52 | +0.71 | second |
| bishop pair | +0.11 | +0.34 | near-inert |
| space | **0.00** | +0.02 | **exactly inert inside the bundle** |
★ **The dilution controls worked and the result is unusually clean:** the two members with their own measured nulls (pair,
space) are also the two that cost nothing to remove — space to two decimal places. This is what a dilution-controlled
leave-one-out is FOR ([[leave-one-out-on-a-bundle-needs-a-dilution-control]]): the +60 Elo is carried by the two
MOBILITY-AREA terms, which refine the definition of the largest term v2 owns (mobility, ≈ +162), not by the two new concepts.
⚠️ ☠️ **This cuts against my own "ship as tested" advice, and both sides have a real objection:** shipping all four carries two
terms with no measured contribution; shipping the subset ships a configuration no game has seen, and §I attribution is NOT Elo
attribution. ⇒ Resolved by MEASUREMENT rather than by argument (owner approved, 2026-09-16), chained overnight:
1. `s3_mobpair_confirm` — **pin + exlow ALONE vs SHIP+E**, elo0 0 / elo1 +10, seed 17. Reproduces ≈ +60 ⇒ ship TWO terms and
   retire space + the pair with a measured reason. Falls short ⇒ ship all four as tested.
2. `s3_mobpair_bracket` — same pairing, seed 18, **elo0 30 / elo1 50**: brackets the MAGNITUDE, which test 1 cannot (an SPRT
   bounded at ≥10 only establishes "> 10", and its point estimate is inflated by the stopping rule).
⚠️ Registered expectation, before the games: pin + exlow reproduce most of the gain (§I says ~94%), and the bracket run lands
H0 — i.e. the true value is nearer +20-30 than +60.

★★★ **BOTH OVERNIGHT RUNS DECIDED (2026-09-17). BOTH REGISTERED EXPECTATIONS HELD.**
```
s3_mobpair_confirm  seed 17  elo0 0  / elo1 10   +310 -222 =159 of 691  (56.4%)  elo +44.5 +/-30.4  LLR +2.980  H1 ACCEPTED
s3_mobpair_bracket  seed 18  elo0 30 / elo1 50   +202 -186  =99 of 487  (51.6%)  elo +11.4 +/-36.3  LLR -2.839  H0 ACCEPTED
POOLED (same arms, both seeds)                   +512 -408 =258 of 1178 (54.4%)  elo ~ +31
```
✅ **Composition answered: pin + `exlow` ALONE carry the gain** (H1 accepted on its own bound), which matches the §I
leave-one-out exactly — space cost 0.00 to remove and the bishop pair +0.11.
✅ **Magnitude answered: it is NOT ≥50.** The two accepted decisions bracket the truth at **10 < true < 50**, and the pooled
1,178-game estimate lands at **≈ +31**.
☠️ **The methodological lesson is the durable part: an SPRT's POINT ESTIMATE inflates at whichever bound it stops on.** The
same pairing read +60.7 (509 g, bound ≥10), +44.5 (691 g, bound ≥10), and +11.4 (487 g, bound ≥50). None is wrong as a
DECISION; all three are unreliable as MAGNITUDES. ⇒ Quote the pooled tally across seeds, never a single run's elo figure.
▶️ **PROPOSAL TO THE OWNER: ship `MOB_V2_PIN=1 MOB_V2_EXCL_LOWRANK=1` (≈ +31 Elo over 1,178 games, two seeds, H1 on the
composition test).** Retire `BPAIR_V2_MAG` and `SPACE_V2_MAG` from the shelf with a measured reason: §I attribution 0.00 /
+0.11 inside the bundle, no individual case, and the subset without them reproduces the gain. Both stay BUILT and default-off
with their existing triggers (Kaufman should own the pair; space wants a purpose-built closed-centre corpus) — retired from
the CANDIDATE list, not deleted from the code.
⚠️ Honest caveat on the 4-term result: its +60.7 is now best read as the stopping-bound inflation of the same ≈ +31 effect,
not as evidence that the pair and space added 30 Elo between them.
★ **Candidate fingerprint for the proposal (`SHIP+E + pin + exlow`): WAC d10 `250 / 59,549,832 / EBF 4.080`** — the SAME 250
solves as the shipped `250 / 61,352,373 / 4.114` in **1.8M FEWER NODES (−2.9%)**. ⇒ Independent corroboration from a different
instrument class: a slightly truer eval buying cheaper search is the [[a-truer-eval-buys-pruning-headroom-the-crank-result]]
pattern, and it is not something noise-only terms produce. (Node counts are a fingerprint, not an Elo claim — but a −2.9%
node drop at equal solves alongside ≈ +31 Elo is a coherent picture, where a spurious Elo result would not be.)

### 2.5b LAZY EVAL (owner) — the right disposition for "neutral but not free"
Owner: *"if they prove to be neutral, perhaps lazy passing might also be useful if we know the giants use them."*
✅ The giants do: SF bails out of `Eval::value()` to material + a cheap partial score when the position is already far outside
the window; Ethereal has an equivalent early-out. **v2 computes every term on every call.**
⇒ This is the correct answer to a strength-neutral term that costs NPS: stop paying for it where the score cannot matter,
rather than deleting knowledge that may pay later. ★ It also LOWERS THE BAR for quiet terms — from "must win Elo" to "must not
lose Elo" — which is exactly the situation slice 3 keeps producing.
▶️ **Lane recorded, NOT started** (it belongs with the v2 SEARCH program, per the owner's own sequencing):
- Threshold on material + PST vs the window, in OUR absolute millipawns — never SF's numbers.
- ☠️ It CHANGES the eval's value in bailed positions ⇒ search-visible, needs games, and couples to the margin re-sweep
  ([[eval-accuracy-payoff-is-pruning]]).
- ⚠️ Ceiling is bounded by eval being ~35% of node cost ([[eval-is-a-third-of-node-cost-not-half]]) — a fraction of that 35%.
- ★ Cheapest first read, no behaviour change: instrument what share of eval calls already have material + PST outside a
  plausible window. That answers "is there anything to win here" before any build.

## 3. MEASUREMENT PLAN (order is deliberate; predictions get registered BEFORE each run)
1. **Build all knobs default-off; verify byte-identity** against `250 / 61,352,373` and v1 `250 / 35,310,778`.
2. **Detector oracle per term + form** (independent python-chess rebuild), non-vacuity per term, colour-symmetry gate.
3. ★ **COLLINEARITY GATE FIRST — before any magnitude ladder.** New pairs to measure: space × mobility (own-camp overlap) ·
   restricted × mobility · queen threats × shipped weak queen · pair × Kaufman census. This inverts slice 2's order on purpose:
   there the terms were disjoint by construction; here §1 says three of them are NOT.
4. **§I ladder**, one change at a time, on SHIP+E; rule unchanged — better on BOTH mean and worst, ±0.05 floor.
5. **Regret** for survivors vs the neutrals measured on this base (**primary 49.9 · `_v2` 51.1**), cross-set mandatory.
6. **Games:** one regression-bundle SPRT for the slice (elo0 −10 / elo1 0), with **`MOB_V2_EXCL_LOWRANK` riding along** as agreed
   (slice 2's parked §I winner). Then the slice-end cumulative SPRT vs the pre-slice-3 ship, to catch drift from chained
   regression passes.
⚠️ Arithmetic that governs every "is it worth games" call here: 1,200 games ≈ ±23 Elo ⇒ resolving ±5 Elo needs ~25,000 games
(~9 days). Terms this small ride bundles; they do not get solo SPRTs.

---

## §3 KAUFMAN + PAIRS — design notes from reading v1's implementation (2026-09-17)

Source read while showdown run A played (no engine jobs allowed, so this is desk work only).
v1's term lives at `cpp_bitboard.cpp:8264-8300`, gated by `ENABLE_KAUFMAN_IMBALANCE` (shipped ON, `KAUFMAN_SCALE=100`).

### ✅ v1 ALREADY has the ownership architecture the owner described
`cpp_bitboard.cpp:8245`: `if (!Config::ENABLE_KAUFMAN_IMBALANCE) { ...flat BISHOP_PAIR_BONUS / KNIGHT_PAIR_BONUS... }`
with the comment "Kaufman imbalance (below) owns bishop-pair + knight-redundancy; skip the flat pair bonuses when it
is on." ⇒ **The owner's "Kaufman already should handle bishop pairs" is exactly v1's structure, and SF's too.**
v2 must inherit the STRUCTURE: one owner, the pair priced INSIDE the census product, never a second flat bonus.

### THE FORM (this is what ports)
Quadratic census product with a bishop-pair PSEUDO-PIECE at index 0; counts `0=pair(0/1) 1=P 2=N 3=B 4=R 5=Q`:
`scalar_white_pov = SUM over pt2<=pt1 of OURS[pt1][pt2]*(cw1*cw2 - cb1*cb2) + THEIRS[pt1][pt2]*(cw1*cb2 - cb1*cw2)`
⇒ one scalar re-pricing ALL material by the whole census. Antisymmetric by construction (swapping w/b negates it),
which is why it costs nothing on the colour-symmetry gate — worth noting given v2 is currently 0/4000 clean.

### ☠️ WHAT MUST NOT PORT: v1's FITTED TABLES (`kaufman_fit.py 800 50`)
The three canonical Kaufman results ARE present and correctly signed — **N x P = +178** (knights like own pawns),
**B x P = -106** and **R x P = -80** (bishops and rooks dislike them). But much of the rest looks like fit artifact
(`R x N = -108` vs `R x B = +98`, `Q x P = +163`, `R x R = -90`), and the coefficients are non-monotone in ways no
chess account supports. ★ **Diagnostic finding: v1's fit prices the BISHOP PAIR at only ~0.13 pawns**
(PAIR x PAIR 26 + B x PAIR 53*2 = 106, less small negatives) — against the flat term's **0.30** that it replaces, and
the literature's ~0.3-0.5. ⇒ Either the flat 300 was too high or the fit collapsed the pair; both cannot be right.
⚠️ This is a READ of the constants, not a measurement. It is a REASON to take reference coefficients, not ours.

### ★★ THE UNIT ANCHOR IS THE PAWN HERE — the one place the 09-13 lesson INVERTS
[[convert-reference-constants-by-positional-scale-not-by-the-pawn]] says positional constants must be scaled to v2's
5-35 mp positional spread, never pawn-converted (that error made tempo 8x too big). ☠️ **Kaufman is NOT a positional
term — it RE-PRICES MATERIAL.** Its natural unit is the pawn, so reference coefficients convert by the PAWN ratio
(ours 1000 vs SF11 mg 128), and scaling it to the positional spread would make it ~30x too small. ⇒ **Check which
quantity a term is denominated in BEFORE choosing the anchor.** Record this as the boundary case of that lesson.

### ▶️ AND A CONNECTION WORTH FOLLOWING: `pawn_closedness()` is built, unused, and may be the right OWNER for the
### signal SPACE was reaching for
`cpp_bitboard.cpp:7086-7099` already implements **Ethereal's closedness index (0 = wide open .. 8 = fully closed)**
from rammed pawns + open files, and its own comment says it "drives the knight/rook imbalance tables and the bishop
colour-complex modulator". It is gated OFF in v1 (`ENABLE_CLOSEDNESS=0`, `CLOSED_N4/R4=0`).
★ Three separate threads converge here: (1) slice 3's **space** term was globally inert and showed its ONLY effect
class-locally on `centre_locked`; (2) the owner's open-vs-closed framing ("corpora where central control is good, and
others where we shouldn't care"); (3) Ethereal conditions the KNIGHT/ROOK IMBALANCE on closedness — i.e. it prices
closedness through MATERIAL re-pricing, not through a space term. ⇒ **Hypothesis for the Kaufman slice: the closed-
position signal belongs to the imbalance term (knights gain / rooks lose as the structure closes), not to space.**
That would also explain why space measured null: it was competing for a signal that PST + mobility + material
re-pricing already own. ⚠️ HYPOTHESIS, unmeasured — and it must pass the collinearity gate (now 40 columns, all five
subsystems) against `ps_blocked`/`ps_opposed`... ☠️ except `ps_blocked` is exactly the column the gate CANNOT see
(W-B identically zero by construction, [[a-differenced-detector-count-carries-the-census]]). So closedness needs its
own reduction before it can be gated — a concrete instance of that blind spot mattering.

### §3.1 KAUFMAN BUILT 2026-09-18 — gates passed, §I ladder predictions REGISTERED BEFORE THE RUN

**Built:** `kaufman_mp()` in `eval_v2.cpp` (after `bishop_pair_mp`), SF11's quadratic census product with the
`QuadraticOurs`/`QuadraticTheirs` tables transcribed VERBATIM **from source** (`stockfish_11/src/material.cpp:33-53`,
read directly rather than via a relay — transcription is where the Ethereal isolated-table near-miss happened).
Knobs `KAUF_V2_MAG` (0 = off; **1000 == exactly SF's scale in our mp**, since SF divides by 16 and its mg pawn is
128 per `types.h:182` vs our 1000 ⇒ 1 cell unit = 0.488 mp) and `KAUF_V2_PAIR` (who owns the bishop pair).

**Gates:**
- ✅ **Colour symmetry with the term ON at MAG=1000: 0 violations / 4000, worst 0 mp.** ★ Predicted — the form is
  antisymmetric BY CONSTRUCTION (swapping colours negates the sum), so this was a check on the TRANSCRIPTION and the
  index mapping, not on the design. It is the cheapest possible catch for an off-by-one in the `cw`/`cb` build.
- Byte-identity at default (MAG=0) against `250 / 49,440,513 / EBF 4.031`: pending.

**Three hazards handled in code, all from the five-engine contrast:**
1. ☠️ **Index convention.** SF is `0=pair,1=P,2=N,3=B,4=R,5=Q`; `V2Context::cnt_*` is `0=P..5=K`. The local `cw`/`cb`
   vectors are built in SF order deliberately; `cnt_*` is never indexed against these tables.
2. ⚠️ **B x own-pawn stays POSITIVE (+104).** The contrast found ALL FOUR references keep bishop-vs-own-pawns OUT of
   the census and in the piece loop, because **a COUNT cannot see square COLOUR** — SF `N*(1+blk)`, SF15.1
   `N*(!defended+blk)` by file class, Weiss `N*blk`, Ethereal rammed-only. v2 already ships SF15.1's form (the most
   conditioned; won the form ladder 6/6). A negative cell here would double-own it AND mis-state the mechanism.
   ⇒ ★ **"Bishops dislike own pawns" is not a Kaufman effect at all.** Recorded because the folklore says otherwise.
3. ⚠️ **Double-pay guard** in `search_engine.cpp`: warns if `KAUF_V2_PAIR=1` and `BPAIR_V2_MAG>0` are both live.

☠️ **CORRECTION to §3's earlier note:** I wrote that v1's fitted cells contain "the three canonical Kaufman results,
correctly signed (N x P +178, B x P -106, R x P -80)". **WRONG, and the folklore is wrong too.** SF says B x own-pawn
is **+104 POSITIVE** and R x own-pawn is **-2 (~zero)**. The Kaufman direction survives only as RELATIVE — knights
gain ~2.4x more per own pawn than bishops. v1's -106/-80 CONTRADICT SF rather than agreeing with it.

**REGISTERED PREDICTIONS for the §I ladder (`KAUF_V2_MAG` 0/250/500/1000/1500/2000 x `KAUF_V2_PAIR` 0/1):**
1. §I mean improves up to roughly **1000-1500**, then worsens.
2. ☠️ **The WORST column will be `variant_regret_set`**, and the ladder will tax it while helping the mean. Mechanism:
   that corpus is built from piece-REPLACEMENT arrays (all N→B, one-swap, asymmetric), i.e. deliberately abnormal
   censuses — exactly what a census-quadratic will mis-price. ⇒ If true, Kaufman repeats the threats pattern
   (good mean, bad worst) and does NOT clear the both-columns rule.
3. `KAUF_V2_PAIR=1` beats `PAIR=0`, because SF prices the pair NOWHERE else and the CONDITIONING (rises with own
   pawns, **-189 per own queen**, falls with every enemy unit) is the part v2's "already owned" verdict never tested.
⚠️ Prediction 2 is the one that matters: it would mean the term is structurally unshippable on the current rule set,
not merely mis-scaled. Registering it so the ladder cannot be narrated after the fact.

### §3.2 KAUFMAN LADDER RESULT (2026-09-18): SF's tables are MONOTONICALLY HARMFUL in v2

| arm | mean% | WORST% |
|---|---|---|
| k250 | +1.00 | +2.03 |
| k500 | +2.53 | +4.08 |
| **k1000 (SF's exact scale)** | **+7.04** | **+8.14** |
| k1500 | +13.19 | +15.71 |
| k2000 | +20.61 | +25.05 |
| k1000 no-pair | +7.71 | +9.74 |
| k1000 no-pair + standalone pair 40 | +7.59 | +9.40 |

Positive = WORSE. **Every arm worse, on all six corpora, monotone in magnitude. No local optimum, no corpus that
likes it.** Base MSEs: 350.12 / 333.58 / 337.41 / 209.45 / 623.27 / 1937.37.

**PREDICTION SCORECARD: 1 of 3.** (1) "improves to ~1000-1500 then worsens" — **WRONG**, never improves.
(2) "the WORST column will be `variant_regret_set`" — **WRONG**: it is `lichess_ks_labelled` at low magnitudes and
`game_regret_set_v2` at high. (3) "pair inside Kaufman beats the alternatives" — **HELD** (7.04 vs 7.71 / 7.59).

**Two lying explanations ruled out before concluding anything:**
- **Sign:** not inverted. v1 does `total -= white_pov_sum`; ours returns `-sum` and the caller adds. Same sign.
- **Scale:** sane. A hand-computed pair+minor-swap census (W: P5 N1 B2 R2 Q1 vs B: P5 N2 B1 R2 Q1) reads
  **+0.29 pawns** at FORM 0 MAG=1000 — a plausible size, not an inflated one. And k250 is a QUARTER of that.

★★ **LIVE HYPOTHESIS — BASIS, not scale (and it is the owner's own framing: "imbalance factors should line up with
the ratios that work for our eval").** Imbalance cells are corrections layered on the PIECE VALUES they correct.
SF11's midgame pieces are ~2x steeper than ours (mg knight **6.10 pawns** vs our **3.25**; SF 1:6.10:6.45:9.97:19.83
vs ours 1:3.25:3.45:5:10), so SF's corrections assume a different underlying ratio set and cannot port at ANY scale.
★ **The test is cheap and decisive because v1 and v2 SHARE `Config::values[]`**: the same census form with v1's
FITTED cells is recorded as HELPING v1 ~7% on this instrument (`EVAL-V2-REBUILD-LOG.md:605`). If v1's cells help v2
where SF's hurt it, basis is confirmed and the route forward is "derive/fit against OUR values", not "port SF's".
⇒ Built `KAUF_V2_FORM` (0 = SF verbatim, 1 = ☠️ v1 fitted, DIAGNOSTIC ONLY — not a ship candidate), with each form
keeping its own native unit convention (SF /2048, v1 /1000) so "MAG 1000 = this form's native scale" holds for both.
Ladder launched: base · sf500 · v1f250/500/1000/1500 · v1f1000 no-pair.

**REGISTERED PREDICTION for that run:** v1's FORM-1 cells will read BETTER than SF's FORM-0 at matched MAG, and at
least one v1f arm will be NEGATIVE (an improvement) on the mean. ⚠️ If FORM 1 ALSO fails uniformly, the shared-piece-
values argument dies and the problem is the CENSUS FORM ITSELF in v2 — which would make Kaufman the fifth slice-3
concept to park, and would make the PATTERN (five for five) the finding rather than any single term.
