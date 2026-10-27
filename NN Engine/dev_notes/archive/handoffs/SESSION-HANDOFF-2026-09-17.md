# Session handoff — 2026-09-17: slice 2 closed, slice 3 closed, and a pattern worth more than either

@author: Ranuja Pinnaduwage (maintained with Claude)

★ Read order: this file → `EVAL-V2-CURRENT-CONFIG.md` §1 (shipped config) + §2 (decision register) + §5 (slice plan, the
collinearity gate, the games policy) → `EVAL-V2-SLICE3-DESIGN.md` (the whole slice-3 record, newest sections last) →
`EVAL-V2-SLICE2-MOBILITY-DESIGN.md` §2.1b/§2.1c (the mobility bake-off and the parked-with-trigger pattern) →
`EVAL-V2-REBUILD-LOG.md` (newest entries at the bottom) → `INSTRUMENT-MAP.md` §I2 and §F (four new gaps).
The previous handoff, `SESSION-HANDOFF-2026-09-15.md`, is superseded but kept — its live-queue sections record how the
placement SPRT and the mobility bake-off were sequenced.

## 0. NOTHING IS RUNNING. NOTHING IS COMMITTED.
HEAD is still `a11ab6c` on `NN-ENgine`. Every change from 09-15 → 09-17 is uncommitted working tree:
- engine: `eval_v2.cpp`, `eval_v2.h`, `search_engine.h`, `search_engine.cpp`, `ChessAI.pyx`
- diagnostics: `_position_class.py` (new) · `_space_detector_oracle.py` (new) · `_threats_detector_oracle.py` (new) ·
  `_v2_term_collinearity.py` (extended to 27 terms + `SETS=`) · `_mobility_detector_oracle.py` · `_placement_detector_oracle.py`
- corpora written: `ks_sets/classes/*.csv` (4 corpora) and `ks_sets/classes6/*.csv` (6 corpora) — generated, not tracked yet
- docs: this file (new) · `EVAL-V2-SLICE3-DESIGN.md` (new) · `EVAL-V2-CURRENT-CONFIG.md` · `EVAL-V2-SLICE2-MOBILITY-DESIGN.md` ·
  `EVAL-V2-REBUILD-LOG.md` · `DIAGNOSTICS-TOOLKIT.md` · `INSTRUMENT-MAP.md` · `SESSION-HANDOFF-2026-09-15.md`
⚠️ Commit only when the owner asks. **No footer.** Push only on explicit say-so.

## 1. SHIPPED THIS PHASE (both on the owner's sign-off)
| what | evidence | fingerprint after |
|---|---|---|
| **placement bundle E** (outpost SF11@100 · bad bishop **SF15.1**@100 · trapped rook@10 · weak queen@25 · minor-behind-pawn **Weiss**@25) | **≈ +13 Elo**, 2,655 games / 3 segments; segment 3 alone accepted H1 (+611 −548 =296 / 1,455, +15.1 ±21.0, LLR +3.039) | `250 / 61,352,373 / 4.114` |
| **mobility AREA** (`MOB_V2_PIN=1` · `MOB_V2_EXCL_LOWRANK=1`) | **≈ +31 Elo**, 1,178 games / 2 seeds; H1 at the ≥10 bound, H0 at ≥50 ⇒ **bracketed 10 < true < 50**; pooled 54.4% | **`250 / 59,549,832 / 4.080`** ← CURRENT |
★ Note the node count FELL 1.8M (−2.9%) at identical solves when the area terms shipped — a truer eval buying cheaper search.
☠️ **STS was NOT re-measured this phase** for either arm. Last known: v2 1588 (mobility in, pre-placement) · v1 1796. That is
an open gap, not a result.

## 2. THE HEADLINE RESULT, AND IT WAS THE OWNER'S IDEA
`MOB_V2_PIN` and `MOB_V2_EXCL_LOWRANK` were each measured ALONE and **each read as a §I-only null**. I parked both, and
explicitly **WITHDREW `exlow` from the slice-3 bundle** on the reasoning that "a bundle bar of ≥ −10 Elo would absorb an
unmeasured term silently".
The owner asked: *"have you tried putting those terms together?"* — which had never been done for the parked shelf.
```
§I additivity:  pin -0.51 + exlow -1.08  =  mobpair -1.59   EXACTLY additive (I had claimed they INTERFERE)
                all4 (pin+exlow+bpair40+space560lin)  mean -1.70 / worst -1.05, better on ALL SIX corpora
clearance SPRT, bound FLIPPED to a GAIN test (elo0 0 / elo1 +10) so it could RETIRE the shelf:
                +248 -160 =101 of 509 (58.6%)  elo +60.7 +/-35.5  LLR +3.008  H1 ACCEPTED
attribution (§I leave-one-out vs the full bundle, positive = removing it HURTS):
                exlow +1.09/+1.65 · pin +0.52/+0.71 · bishop pair +0.11/+0.34 · space 0.00/+0.02
composition test: pin+exlow ALONE  H1 accepted, 691 games, +44.5 +/-30.4
magnitude bracket: same pairing at elo0 30 / elo1 50  H0 accepted, 487 games, +11.4 +/-36.3
POOLED 1,178 games  ->  ~ +31 Elo
```
⇒ **"Individually unresolvable" is not "individually worthless."** That is a statement about the INSTRUMENT: ±5 Elo needs
~25,000 games alone but ~500 as a group of four. I had already applied that arithmetic to the placement bundle and failed to
apply it to the parked shelf — and my `exlow` ruling inverted the right conclusion. **A bundle is how such a term becomes
measurable; the fix is to ask the bundle for a GAIN, not for harmlessness.**
★ What actually earned the Elo: **both terms refine the AREA of mobility**, the largest term v2 owns (≈ +162). Neither adds a
concept. ⚠️ The owner's own framing correction, which the records now carry: the mobility FORM BAKE-OFF concluded **no
change**; these two are *additions to mobility*, a separate later result.

## 3. THE SLICE-3 PATTERN — four concepts the giants carry, none of them move-readable
| concept | reference support | outcome |
|---|---|---|
| central control | **0/5** — no reference has a standalone central term | **NOT BUILT.** They price centrality once, via PST + mobility, and differ only in HOW they split it (Ethereal PST spread 11 with a ~10× mobility table; SF11 spread 84). v1's `central_score` re-sums PST + heat cells = the second-owner pattern the charter exists to prevent |
| bishop pair | **5/5** carry it | **ALREADY OWNED** by v2's PST + mobility. Three instruments agree: §I (every magnitude helps general corpora, hurts KS-critical) · per-class §I **flat in all 5 structure classes** · regret NULL both corpora (−0.4 / −0.1pp) |
| space | 3/5 (SF11, SF15.1, Ethereal; SF1.1 and Weiss have none) | **BUILT, VERIFIED, PARKED.** SF's quadratic-weight form harmful at every magnitude; Ethereal's linear form **at scale parity** inert. Only real effect is class-local on `centre_locked` (−0.04..−0.13) **with a mechanism** — locked centres are where mobility collapses |
| threats | 4/5 (not SF1.1) | **BUILT, BEST-VERIFIED TERM OF THE SLICE, PARKED — move-NULL.** §I loved it (`th100` −9.71% on the variant corpus, the largest single-corpus gain on this base) but every magnitude taxes `lichess_ks_labelled` in proportion; regret primary **50.1 vs bar 50.1 (0.0pp)** · `_v2` **49.7 vs 50.5 (−0.8pp)** on 35-36% footprints |
⇒ **Working hypothesis (now a memory, and load-bearing):** v2's positional signal is nearer saturation than its term COUNT
suggests. The marginal CONCEPT adds accuracy without adding move-level information; refining the DEFINITION of an existing
owner still pays. ⚠️ A pattern of four, not a law — it needs its own test before it steers scope.

## 4. THE KS INVESTIGATION (owner's question: "have we infringed on KS's domain?")
Answered in three layers, and the third is the one that matters.
1. **Built the KS count probe** — `ks_probe` / `ChessAI.ks_counts`, six channels per king (attacker count · weighted attacker
   sum · weak zone squares · king-adjacent attacks · safe checks · scored units). This closed the collinearity gate's last
   major hole, open since slice 2.
2. **The double-count story is REFUTED, twice.** Gate now runs **27 terms**: no cross-subsystem flag on a general 10,000-position
   sample AND on `lichess_ks_labelled` itself (5,000, via the new `SETS=`, because **overlap is a property of a POPULATION**).
   Every threats leg VIF ≤ 1.30; `th_king` — the most obvious overlap candidate — reads **1.10**. ⇒ **Reopening the KS rung has
   no evidence behind it. KS stays as shipped at +101 Elo.**
3. ★★★ **The shape mismatch — found by the owner's idea of comparing how the giants BALANCE the two, and invisible to the gate.**
   In SF and Ethereal king danger is an **unbounded quadratic** while threats is **linear**, so KS overtakes threats ~2:1 in
   severe attacks. Ours **saturates**: `KS_V2_MAX=4000` caps KS at 4.0 pawns while threats at th100 reaches 4.2.
   | engine | KS:threats, quiet → severe |
   |---|---|
   | SF11 / SF15.1 | 0.5 → 0.8 → **1.7-2+** (crosses 1 at kd ≈ 1150) |
   | Ethereal | 0.3 → 1.2 → **2.2** |
   | Weiss | 0 → 0.8 → 1.1 |
   | **ours** | 0 → 0.7 → **0.85, never > 1** |
   Tested as a 2×2: `th100+ksmax8000` cut the tax **+3.90 → +2.13** while keeping the general gains.
   ☠️ **NOT acted on:** +2.13 is still ~40× the §I floor; the `ksmax6000`-ALONE control is itself +0.54 on that column (so part
   of the "fix" is ceiling-raising the KS-critical corpus dislikes anyway); and acting would reopen a +101 Elo rung on accuracy
   evidence alone. `KS_V2_MAX` stays 4000.
★★ **Method, and the limit that cost most this slice: VIF measures co-movement of detector COUNTS, not the relative HEIGHT of
the scored curves.** A clean gate means two terms don't measure the same thing — NOT that they coexist well at their chosen
magnitudes. Curve balance needs a source comparison or a 2×2.

## 5. NEW INSTRUMENTS (all in `DIAGNOSTICS-TOOLKIT.md`, each with its own recorded defects)
| tool | what it answers |
|---|---|
| 🧰 `_position_class.py` | **pawn-structure classes**, pure python-chess, no engine: `centre_tension` 4,117 · `centre_locked` 1,652 · `centre_open` 13,749 · `centre_cleared` 8,433 · `other` 19,702 + tag `pin_dense` 7,310 (4 corpora; a 6-corpus run is in `classes6/`). Writes `fen,stratum,…` WITH the source SF18 label columns, so §I and the regret gate consume them unchanged |
| 🧰 `_space_detector_oracle.py` | rebuilds the WHOLE space term (region · safe mask · behind double count · weight · gate · taper) and compares COUNTS **and** score against `space_counts` |
| 🧰 `_threats_detector_oracle.py` | the slice's widest surface — **2 gate forms × 7 legs** — counts and score vs `threats_counts`. Refuses to run at `THREAT_V2_PCT=0` (would pass vacuously) |
| 🧰 `ChessAI.ks_counts` | the KS count probe (above) |
| 🧰 `_v2_term_collinearity.py` | now **27 terms** (mobility + placement + 7 threats legs + 6 KS channels), `SETS=` for per-corpus runs, and intra-subsystem exemptions (`MOB`, `KS_SET`) |
★ All three new probes were added WITHOUT moving any fingerprint — that is the check proving they are not in the search path,
and it is worth re-running after adding one.

## 6. NEXT STEPS
**Immediate (agreed with the owner, in order):**
1. ~~**Pawn-structure count probe**~~ ✅ **DONE 2026-09-17 (later the same day) — and NO probe was needed.** A record-check
   found `pawn_entry_probe` / `pawn_masks` had exported every Layer A mask since 09-12 for the rung-2 oracle, so the "hole"
   closed as a ~30-line column-set edit to `_v2_term_collinearity.py`: no C++, no rebuild, no fingerprint risk. The gate now
   spans **all five scoring subsystems at 40 columns** and is CLEAN on two populations (general 10,000, strongest cross pair
   `mob_table_mg × ks_natt` −0.41; `lichess_ks_labelled` 5,000, `th_restricted × ks_natt` +0.33). ☠️ It refuted the
   prediction written into the tool: the two pairs wired to a SHARED MAP (`ps_pattacks × mob_*`, `ps_halfopen ×
   traprook_units`) read **≤ 0.07** ⇒ third instance of "sharing an input is not sharing a signal". Two new §F gaps:
   `blocked` is invisible to White−Black differencing (symmetric by construction), and a differenced count carries the
   CENSUS (three pawn columns are mostly material) so a `ps_npawns` control is now required. See the REBUILD-LOG entry.
   ⚠️ Remaining sliver: `attacks2` (double pawn attacks) is still unexported — a probe change + rebuild, only if needed.
2. ▶️ **The checkpoint — MARGIN RE-SWEEP DONE (fixed-depth half), 2026-09-17.** First sweep ever under `EVAL_ARM=1`.
   **`RFP_MARGIN=1000` recovers −17.0% nodes at IDENTICAL WAC solves (250)**; `FUTILITY_MARGIN_SCALE=70` adds a free
   −2.0%; `QDELTA_PERMOVE_MARGIN` is a non-lever; `DELTA_MARGIN` is DEAD at defaults; razor is not eval-denominated.
   ☠️ **~19% total is BELOW the ~35% Elo-visibility bar ⇒ the handicap is real but SMALL (a few Elo, not tens)** — the
   showdown is de-risked, so play it twice as the rule requires but expect a narrow spread. ⚠️ Still OPEN for v2:
   `ASPIRATION_DELTA`, `VERIFY_MARGIN`, and the 2×2 to check RFP+futility compose. ⚠️ **Needs a QUIET WINDOW:** a margin
   change is a node-saver ⇒ judged at FIXED TIME, so the decision is a timed SPRT of v2 @1500 vs @1000 (and @1250).
   Full detail in the REBUILD-LOG's two 09-17 checkpoint entries and CURRENT-CONFIG §5. Remaining original text:
   margin re-sweep for v2 (tempo folds in here — ⚠️ **but against a CORRECTED channel list**: two of the three thresholds
   its mechanism story named are dead/off), then
   an NPS pair in a quiet window, then the **v1-vs-v2 showdown played TWICE** (on v1's margins and on the re-swept ones) per
   the owner's fairness rule. ⚠️ v2's EBF is **4.080** vs v1's **3.784** with every margin fitted to v1's eval scale — the
   re-sweep is not optional politeness, it is the difference between measuring v2's eval and measuring v1's tuning.
3. **Kaufman + pairs** — the last named slice-3 item. ★ **It should OWN the bishop pair** (owner's point; SF does exactly this,
   as a pseudo-piece row in the imbalance matrix) ⇒ the standalone `BPAIR_V2_MAG` becomes redundant if Kaufman ships.
   ⚠️ Port the census-product FORM only; v1's ridge-fitted tables were fitted against v1's residual with v1's other terms live.
**Named triggers on parked items:**
| item | trigger |
|---|---|
| threats | a **lazy-eval** lane (it is the slice's most expensive term ⇒ a strength-neutral term's COST is the live question) · the unbuilt SF legs (`Knight/SliderOnQueen`, `WeakQueenProtection`) · any independent re-opening of `KS_V2_MAX` |
| space | **a purpose-built closed-centre corpus** (King's Indian / French / Closed Sicilian), NOT another pass over these pools |
| bishop pair | when Kaufman is built, or if PST/mobility are re-scaled |
| mobility forms (tables, eg share, SAFE, EXCL_QUEEN) | only if the §I mean/worst trade changes — e.g. after a KS re-tune |
| tempo · material taper · rook files · pawn hash · LATENT_V2 | unchanged from §2 of the config doc |
**Owner's lazy-eval lane (recorded, not started):** SF bails out of `Eval::value()` to material + a cheap partial score when
already far outside the window; Ethereal has an equivalent; **v2 computes every term on every call.** ★ It lowers the bar for
quiet terms from "must win Elo" to "must not lose Elo", which is exactly the situation slice 3 keeps producing. Cheapest first
step is pure instrumentation: what share of eval calls already have material + PST outside a plausible window. ⚠️ Ceiling is
bounded by eval being ~35% of node cost, and it CHANGES the eval's value in bailed positions ⇒ search-visible, needs games.

## 7. OPS RULES THAT BIT THIS PHASE
- ☠️ **`build` deletes the `.so` BEFORE compiling** ⇒ a failed build leaves NO ENGINE. It happened twice (a duplicate
  `mob_reset` from a comment-move edit; `space_mp` referencing `PL_CENTRE_FILES` from above its declaration). Fix forward
  immediately; never leave a failed build unresolved.
- ☠️ **PowerShell quoting:** one dropped closing `"` on a `bash -lc` string produced "The string is missing the terminator" and
  the run did nothing. Check the tail of a long launch line before trusting an empty output.
- ☠️ **Never `Get-Content` a task output** — I did it once for an SPRT progress read. Use `Read`; the rule exists because shell
  reads prompt and bypass the file-state tracking.
- ⚠️ **Launch the NEUTRAL with (or before) the candidate.** I launched a `_v2` candidate whose bar did not exist yet; the
  whole-corpus bars (49.9 / 51.1) belong to a specific BASE and do not transfer. Measured bars this phase: SHIP+E+area primary
  **50.1** · `_v2` **50.5** · class bars `pin_dense` 50.0 · `centre_tension` 50.4 · `centre_locked` 49.4.
- Unchanged: no `$` inline across PowerShell→WSL (use a script file) · conc ≤ 4 and RAM caps at ~2 engine loads · never rebuild
  while an engine job runs · `wac`/`sts` take the tag first and knobs last · `EVAL-V2-CURRENT-CONFIG.md` is CRLF (single-line
  Edit anchors) · read the echoed `[toggles]` before trusting any result · owner games ~9pm-midnight (fixed-depth safe, timed not).

## 8. INSTRUMENT RESOLUTIONS (updated; see `INSTRUMENT-MAP.md` §I2 and §F)
| instrument | resolution / how it lies |
|---|---|
| §I accuracy | ±0.05%; **worst column decides**. ⚠️ Slicing by position class MAGNIFIES (pin 9× on `pin_dense`) but is still ONE instrument — §I-by-class and §I-global are one vote |
| class reads | **the CHANGED-MOVE subset is the sample, not the corpus**: +13,500 positions bought 322 class rows and **37 changed moves**; a win% on ~600 changed moves carries ±2pp by itself. Compute `share × effect` vs 0.05% BEFORE running |
| d7 regret | ~2-2.5pp cross-set bar; neutral must be measured per base AND per class; quote `n_crit` (24-39 this phase = unreadable) |
| games (SPRT) | **decides, does not measure.** One pairing read +60.7 / +44.5 / +11.4 across three sound runs; pooled 1,178 games = ≈ +31. Quote the pooled tally; bracket a magnitude with TWO bounds |
| collinearity gate | flags \|r\| ≥ 0.7 / VIF ≥ 5 on detector COUNTS. ☠️ **Cannot see whether two terms coexist well** — that is curve height, not co-movement |
| WAC d10 | fingerprint only (±5-6 solves) |
| STS | ±150 arm-vs-arm; blind to mobility magnitude and to tempo's mechanism |

## 9. ⚠️ WEIGHT MY EXPLANATIONS — where I went wrong this phase
**Judgement errors (the expensive ones):**
(a) Treated "individually unresolvable" as "individually worthless" for four terms, and **withdrew `exlow` from a bundle** on
reasoning that was exactly backwards. The owner's question recovered ≈ +31 Elo I had written off.
(b) Claimed pin and `exlow` **interfere**, from ONE sub-bar regret reading. They are exactly additive on §I.
(c) Predicted space would help where the centre is **contested**. It helps where the centre is **locked** — the opposite.
(d) Predicted our largest error would be in central-tension positions. It is the **smallest** of the classes (286 vs pin-dense 414).
(e) Registered an OPTIMISTIC prediction that some threats magnitude would clear the both-better rule. None did.
(f) Told the owner the bake-off "yielded a stronger mobility"; the owner corrected the framing — the bake-off was NO CHANGE and
the two winners are additions to mobility's area.
**Tooling errors (same bug class twice in one week):**
(g) Wrote the space oracle against a probe that **did not exist** (it exited 2 rather than passing vacuously).
(h) Then wrote the space oracle to fill count slots only when the gate was open — and **reproduced that exact bug in the threats
oracle four legs over**, after documenting the lesson in the toolkit row. Signature to remember: **scores match exactly, counts
differ ⇒ suspect the GATING CONVENTION, not the detector.**
(i) Shipped a classifier whose `centre_cleared` class was **unreachable** (0 of 47,653) while `centre_open` swallowed 45.9%.
(j) Then had that same classifier **drop the SF18 label columns**, making its output unusable by the tools it exists to feed.
(k) Built the collinearity gate's KS columns so that it flagged its **own redundant pair** (`ks_natt × ks_watt r=+0.93`).
A gate that flags its own columns trains you to ignore it.
(l) Left a stray `ic` typo at the top of the collinearity tool; a one-line syntax error.
**Prediction record this phase: 2 right, 6 wrong or unreadable** across the mobility, space, bishop-pair and threats ladders.
That is why predictions are registered in the design docs BEFORE each run, and why the scorecards are kept.

## 10. ★ LOAD-BEARING OWNER CONTRIBUTIONS
- *"Have you tried putting those terms together?"* → the +31 Elo ship, and the refutation of four of my verdicts.
- *"If they prove to be neutral, perhaps lazy passing might be useful if the giants use them."* → the lazy-eval lane, and the
  reframe that a neutral term's COST is the live question.
- *"Does this mean we have infringed on KS's domain and need to relook at that rung?"* → the KS count probe, the closed gate
  hole, and a measured "not measurably" instead of a plausible story.
- *"We can do the comp to the giants that have a threats definition to see how they balance things."* → the shape mismatch,
  which no instrument we own could have found.
- *"Kaufman already should handle bishop pairs."* → the correct ownership architecture (and it matches SF's matrix).
- The centrality theory: *"the centre is not some magic zone… let the scores that are inherently bettered by the geometry of
  the centre shine through"* → matched the 0/5 source finding exactly, and produced the position-class instrument.
- *"Corpuses where central control is good, others where we shouldn't care; open vs closedness."* → `_position_class.py`.
- *"I'm not so sure we should immediately jump to the showdown"* + the EBF point → the checkpoint sequencing, and the reminder
  that v2's 4.080 EBF against margins tuned to v1 is a fairness problem, not an eval problem.
- Insisting docs and memory be current BEFORE the handoff → this document, and five doc/memory files brought up to date.
