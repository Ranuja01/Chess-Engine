# Fable consult — dynamic (realizability-conditioned) king safety + a sanity check on recent failures (2026-07-09)

*Self-contained briefing. You (Fable) have no access to our repo — everything you need is below.*

## 0. What the engine is
- Custom C++ **hand-crafted-eval (HCE)** chess engine, pre-NNUE by choice (we want to exhaust HCE first).
  Alpha-beta with the usual machinery (TT, LMR, null-move, futility, RFP, qsearch). **Non-negamax**: separate
  `minimizer`/`maximizer`, eval is **absolute, Black-positive**, single root sign-flip. Scale = **millipawns**
  (pawn = 1000).
- We play/evaluate at a **fixed node budget** (250k nodes/move) for A/B determinism.
- Reference engines available with per-term eval output: **SF18** (NNUE, gives `eval`), **SF11** (classical
  HCE, prints a labeled per-term eval breakdown), **SF1.x**.

## 1. Our measurement-venue trust hierarchy (READ THIS — our "it failed" claims depend on it)
Hard-won and non-obvious. In order of trust:
- **External gauntlet = TRUTH.** OUR engine (250k nodes) vs **native SF18 throttled to a fixed low node
  count** (~400 nodes ≈ 50% score = our anchor). SF18's NNUE eval is maximally different from ours →
  **blind-spot-immune** adjudication. **≥2 seeds mandatory** — the baseline itself swings ~±6% by seed
  (44.7%↔56.2% over 160 games), so single-seed deltas lie. We measure **score%** and a **"collapse rate"**
  (see §2).
- **node_ab** (fixed-node self-play A/B, our engine vs itself with/without a change): low-variance but
  **ANTI-PREDICTIVE** for us — candidates that won node_ab have LOST real games. Self-play is also blind to
  shared-blind-spot changes. We use it only as a cheap screen, never to ship.
- **A "WDL-cploss compass"** (move-agreement vs SF over a corpus): we built it, found it **TT-context-fragile
  and anti-predictive** (sign-flips with corpus context) → **retired**.
- **WAC/STS** (tactical/positional test suites): **≠ strength** (STS penalizes pruning by construction). Floors
  only.
- We also have a **byte-identical bench signature** (deterministic node count on a fixed suite) to prove a
  change is off-by-default / no accidental behavior change.

**Implication:** please weigh our results as "gauntlet says X"; treat node_ab/compass/WAC as unreliable.

## 2. The root-cause finding (measured)
- Static eval **well-calibrated ON AVERAGE** vs SF (mean per-position gap ≈ 0, per a large sample — `our_static
  ≈ SF_static ≈ SF_search` at typical positions).
- BUT at our over-optimistic **peaks**, our SEARCH backs up **~+3.9 pawns of "phantom"** over its own static
  eval (it prunes/reduces the opponent's refutations at shallow depth). **~22% of gauntlet games "collapse"**:
  our eval peaks ≥ +2 pawns, then the game draws or loses.
- So: calibrated *mean*, but **fat conditional tails** — specific position classes where we're wrong by
  multiple pawns, and those are where games are lost.

## 3. The recent failure chain (what we tried against the collapse, with numbers)
1. **Singular extensions** — built fully (both node types, dedicated per-path extension cap, instrumented;
   fires at a healthy 4.11% of eligible nodes). Result: **node_ab neutral** (+6 ±62 Elo); **gauntlet 2-seed
   SIGN-FLIP** (+11.0 / −10.6 at one margin, +7.5 / −7.8 at another) = **net ≈ 0**; **collapse rate unchanged**
   (~22%). Banked default-off.
2. **Then we TRIAGED the collapse positions** — re-searched each collapse decision-FEN at **depth 18** + SF
   compare, classifying EVAL-hole vs HORIZON(depth-fixable) vs PRUNING. **~70% EVAL** across two seeds. "EVAL"
   here means: **our own depth-18 low-prune search STILL plays the move and our eval stays far above SF** (e.g.
   we read **+14** in a position SF calls **0.0**; a literal **K+P-vs-K draw reads +5.2**). These are genuine
   eval-FUNCTION over-reads that our own deep search *shares*, not horizon effects.
3. **Consequence for correction-history / verification:** standard **correction history** trains
   static→(backed-up search value); **OTV** (our bespoke "re-search deeper when search ≫ static") re-searches.
   Both need SEARCH to *disagree with* / see *through* the static error. In our EVAL collapses, depth-18 search
   **agrees** with the inflated static → the training/trigger signal is ~0 exactly where we need it →
   **corrHist and OTV are structurally blocked for this failure mode.** (OTV was independently built and
   shelved earlier on node_ab −17.8; never gauntlet-tested, but the triage predicts it's the wrong lane.)
4. The EVAL over-reads cluster into **(a) drawn-endgame over-reads** (KPK/fortress/opposite-bishop scored
   +5..+8; we're fixing these separately with a tunable draw-scale factor — not your question) and **(b)
   middlegame KING-SAFETY / attack over-reads**: we value our attack/position at +2..+9 when the attack
   fizzles or the opponent's counterplay wins. That second class is this consult.

## 4. King-safety specifics (years of tuning, mostly nowhere)
- We've tuned KS for a long time via **Texel** (fit eval params to game outcomes and to SF) and **SPSA/PACE**
  (self-play parameter search). It goes **mostly nowhere**.
- **Structure today:** the shipping king-danger term is a **FLAT presence sum** (attacker presence near the
  king minus defenders), with no realizability gating active. A newer **attack-units** term exists (sum of
  per-attacker-type units + weak squares + open files + storm + safe checks → a **non-linear danger curve**
  `danger = clamp(units)²/divisor` → × a master magnitude `KING_SAFETY_MAG`) but it's **off by default**.
- **Realizability hooks exist but are all parked (default 0):** a **per-king DYNAMIC magnitude** (`realness =
  attacker_count × (open_files + weak_squares) − pivot`, fed through a clamped 0.5×..2× gain — scales a
  genuinely-attacked king UP and a safe king DOWN), a **material-backing discount** (damp KS when the
  attacker is under-backed on material = the "fantasy attack" killer), a **control-edge** scale, and a
  super-linear "coffin" co-occurrence term.
- **KEY DATUM:** setting the flat magnitude to a scalar (e.g. `KING_SAFETY_MAG=1500`) made the **collapse
  stratum WORSE (25→48 collapses).** Evidence that a flat scalar magnitude cannot be context-right: raising it
  helps real attacks and worsens fantasy attacks (our exact failure) in equal measure.

## 5. Our thesis + proposed method (what we want you to pressure-test)
**Thesis (engine author):** the KS *magnitude* should never have been a scalar. It should be **its own
realizability function of cheap board detectors** — a per-position dynamic magnitude — giving the term the
**degrees of freedom to be right in context** instead of one averaged-out compromise.

**Proposed method to SET that function (this is the new part — prior tuning used bad signals/venues):**
1. Swap the flat term out for the attack-units term (retire the flat presence sum).
2. Harvest a **labeled failure corpus** = the collapse / over-read positions where we KNOW our KS is wrong
   (SF says ~0, we say +4). Per position, dump SF's eval AND our cheap detectors (attacker-count, backing
   edge, control edge, open files, weak squares — the eval already computes these).
3. **Fit the realizability function `magnitude = f(detectors)` to the SF per-position residual** on that
   corpus (train/holdout split by game), i.e. learn which detector co-occurrence explains the over-read and
   set the dynamic-magnitude coefficients from that — instead of SPSA/Texel on a flat scalar.
4. **Gate at the GAUNTLET** (collapse rate + score, ≥2 seeds), not node_ab/compass.

## 6. Questions
1. **Methodology / overfit.** Is fitting a dynamic term-magnitude = f(detectors) to SF *per-position residuals*
   on a small *labeled failure corpus*, then gauntlet-gating, a sound way to escape the "flat-scalar-averages-
   to-nothing" trap? How do we keep the extra degrees of freedom from curve-fitting the corpus — regularization,
   holdout/transfer-ratio structure, sane cap on # detectors, corpus size? (Our failure corpus is on the order
   of hundreds–low-thousands of positions.)
2. **Teacher choice + the scale confound.** We can target (a) SF18-NNUE *total* eval, (b) SF11 classical HCE's
   *king-safety term* (apples-to-apples HCE), or (c) SF shallow-*search*. We have a prior lesson that fitting to
   SF's *absolute magnitude* is confounded by a strength-neutral global scale (argmax-invariant) — "SF is a
   feature library, not a magnitude target." Which teacher, and how do we fit the **conditional STRUCTURE**
   (where our residual co-varies with detectors) rather than absolute magnitude?
3. **Where the realizability multiplies.** Our danger is `f(units)` through a non-linear (quadratic-then-linear)
   curve, then × magnitude. Should the realizability factor multiply the **units** (inside the curve — changes
   the effective attacker count) or the **final danger** (outside the curve — pure scale)? And is per-king
   `realness = attacker_count × (open_files + weak_squares)` the right functional form, or do you expect a
   richer / different detector combination to carry "is this attack real"?
4. **Sanity-check our diagnosis of the recent failures.** Given §2–§3: singular came back neutral and we argue
   corrHist/OTV are *structurally blocked* because our own depth-18 search shares the eval over-read (no
   search−static signal). Is that reasoning correct — i.e. for a "conditional eval hole our own search
   believes," the fix must be in the eval function (or its conditioning), and search-side verification/learning
   levers can't help? Are we missing a lever, or misreading the ~70%-EVAL triage?
5. **Degrees of freedom vs curve-fit (the core tension).** The engine author's thesis is that a scalar
   magnitude *can't* be context-right and f(detectors) supplies the needed DoF. Do you agree in principle, and
   what's the right *amount* of DoF for a single eval term before "data-driven conditioning" becomes
   "overfit"? Any structural priors (monotonicity, sign constraints, detector orthogonality) you'd impose to
   keep it honest?

*(Send order suggestion: Q1, Q2, Q4 first — they shape the build; Q3, Q5 refine.)*

## 7. Primary records in the repo (you have access — verify / go deeper, don't take §1–§4 on faith)
- **Failure chain + triage:** `dev_notes/SESSION-HANDOFF-2026-07-08.md` — the `SINGULAR VERDICT` section (the
  neutral/sign-flip numbers) and the `pt.6 COLLAPSE TRIAGE` section (EVAL-dominated split + the worst over-read
  FENs). Raw triage output: `selfplay/games/g_base_s0/triage.csv` (+ `g_base_s1`).
- **Memory (`memory/…md`):** `singular-banked`, `realizability-conditioning-architecture` (our guiding
  `value × f(detectors)` vision — the architecture your KS answer plugs into), `external-gauntlet-calibrated`
  (the venue), `compass-context-fragility` (why the compass is retired), `search-soundness-phantom-pv`,
  `eval-accuracy-payoff-is-pruning`, `sf11-texel-scale-invariance` (the scale-confound in Q2),
  `ks-detection-rebuild` + `holistic-eval-pivot` (the prior KS attempts that died).
- **KS code:** king-safety knobs incl. the parked realizability hooks (`KING_SAFETY_MAG`, `ENABLE_KS_REPLACE_LT`,
  `KS_DYN`/`KS_DYN_PIVOT`/`KS_DYN_SHIFT`, `MOD_KS_BACKING`, `MOD_KS_CONTROL`, `KS_INTERACT`) live in
  `search_engine.h` (~search around `KING_SAFETY_MAG`); the term itself is `king_safety_danger` /
  `king_safety_score` and the flat incumbent is `get_latent_threat_score`, all in `cpp_bitboard.cpp`.
- **Tooling for the proposed fit:** `diagnostics/mine_overreads.py` (harvest), `diagnostics/bench_split.py`
  (game-keyed train/holdout), `selfplay/tune_corpus.py` (per-FEN SF-eval + detector corpus),
  `selfplay/tune_cond.py` (detector-conditioner fitter), `diagnostics/bias_profile.py` (ours-vs-SF static gap).
