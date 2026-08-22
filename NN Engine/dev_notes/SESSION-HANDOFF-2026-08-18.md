# SESSION HANDOFF 2026-08-18/19 — (1) KS output-gate + (2) FOUR search value bugs + (3) a game-positive search candidate

## ✅ OVERNIGHT 2026-08-19 — RESULTS (both queued lanes resolved to NULL; box left idle by choice)
1. ✅ `wsl.exe --shutdown` → fresh build → byte-id reverify **250 / 36,651,879 / EBF 3.751**. Clean.
2. ☠️ **C1 SPRT RESOLVED = CONFIRMED PRACTICAL NULL.** Added a clean 400g segment (fresh WSL, conc 3,
   segmented to prevent the OOM): **+139 −151 =110 / 400 = −10 Elo**, OPPOSITE the prior run's sign.
   **POOLED 296-282-212 / 790g = +6 Elo, 95% CI [−15,+27]** — firmly straddles 0. The +23 was
   regression-to-mean (over-call, as the handoff warned). ⇒ the 4 value fixes are correct but Elo-neutral =
   the productive-bug pattern realized. Flags stay GATED. Full: [[search-value-bugs-and-the-productive-bug-pattern]].
3. ☠️ **Eval-calibration aggregate = NULL.** `collapse_term_attribution.py CLASS=positional` (n=171 vs
   SF11+SF15c): NO resolvable systematic over-read — `capture_gains` −0.08pw (IDENTICAL in quiet control ⇒
   not collapse-specific), `passed_pawn_support` −0.07pw, `king_safety` −0.01. The −6.75/−4.55 live blowups
   are RARE per-position TAIL events, not a mean bias the aggregate can see. Full: [[live-game-losses-are-single-term-blowups]].
4. ⏸️ **No overnight game run launched.** Deliberate: no candidate clears the ~20-40 Elo floor, and the only
   untested directions (additive-KS magnitude = 0-for-9 prior; `CAPG_KS_DAMP` = marginal) are the prior-null/
   sub-floor directions memory warns against burning games on. Idle box > misleading negative SPRT.

## 🌅 MORNING — the real decision is YOURS (direction, not execution)
The two lanes I could resolve autonomously both came back null. The genuinely-open threads, ranked, each with
its risk:
- **(A) Per-position TAIL structure from REAL owner games** — the ONLY live eval thread. The aggregate proved
  the blowups are tail, not mean; they need the actual FENs. **Owner: paste the 2 loss PGNs (or any new
  loss)** → `_pgn_walk.py SIDE=<us> PGN=<file>` → `FENS=<file>` per-position attribution → look for a
  STRUCTURAL (bounded/re-shape) fix, not a magnitude tune. ⚠️ expect search-absorption even here.
- **(B) The big search lever: a stronger capture-searching qsearch** — capgains is a crutch for the current
  weak qsearch (Arc 2). This is the real headroom but it's a large, risky rebuild, not a knob.
- **(C) KS magnitude deficit** (0.45 of SF) — additive direction, 0-for-9 in games. Low priority.
- Byte-id baseline REVERIFY every build: **250 / 36,651,879 / EBF 3.751 / STS 1771**.

## ⚡ AFTERNOON 08-19 — SEE_PRUNE_CAPTURES: the arc's best screen candidate (game gate in flight)
Owner's idea: test default-off search knobs on the bug-fixed ("honest") landscape — do the fixes UNBLOCK any?
FUNNEL: node-triage (fixed-depth NODES) → fixed-time BOTH-instrument screen → 2-knob tune → game gate.
- **Node triage on honest base (`TT_FLAG_FIX=1 NULL_MATE_CLAMP=1`, fixed-depth NODES vs ref 36,475,047):**
  `ENABLE_SEE_PRUNE` (quiets) +0.8% nodes (dead) · **`SEE_PRUNE_CAPTURES` −3.3% (only real cutter)** ·
  `NULLMOVE_EVAL_R` −1.2%/−6 solves (weak) · `IIR` −1.2%/holds-solves (but the known STS mirage).
- **Fixed-time screen (the both-instrument gate):** SEE-captures gains **~+0.3 ply on BOTH** wac_timed_depth
  and sts_timed_depth, **HOLDS STS ≥ baseline** (1661/1773 & 1728 vs fresh base 1669 — NOT the IIR mirage
  that sagged to 1613), WAC solves +3-5. First both-instrument non-regression the arc produced. Additive with
  the fixes (works solo +0.195 depth; fixes stack ~+0.12 more), NOT a pure unblock.
- **Tune:** `SEE_PRUNE_MAX_DEPTH=3` is a clean interior PEAK (2→14.513, 3→14.568, 5→14.432 depth). `MARGIN`
  flat within noise; **`SEE_PRUNE_CAPTURE_MARGIN=1000` chosen** — nominal-best (259 solves, STS 1728) AND
  safest (only prunes captures losing >1 pawn ⇒ structurally cannot prune a real sacrifice = lower game
  variance). Mechanism: search_engine.cpp:3570/3947 (prune losing capture at rd≤MAX_DEPTH, not in-check/extend).
- **GAME GATE — BANKED PARTIAL (stopped midday for research, RESUMABLE by pooling):** `gate
  'ENABLE_TT_FLAG_FIX=1 ENABLE_NULL_MATE_CLAMP=1 SEE_PRUNE_CAPTURES=1 SEE_PRUNE_CAPTURE_MARGIN=1000' seecm1k
  seecm1k_s1 400 5 3` vs default. **Banked: 62-48-44 / 154g = score 0.545 ≈ +32 Elo, CI ~[−20,+84]
  (UNRESOLVED, spans 0).** Oscillated +33→+10→+32 but stayed PERSISTENTLY POSITIVE (never dropped below ~0
  after game 130) — more encouraging than C1 at the same stage, but a ~+30 effect needs ~1000+ games to
  resolve. ⚠️ Throughput was throttled (~68 g/hr at conc3 — a day of bench runs grew VmmemWSL; `wsl.exe
  --shutdown` before resuming). To RESUME: relaunch same cfg, ADD W/L/D to the 62-48-44. ⚠️ Prior cautionary:
  C1's +0.13 depth→null; SEE-captures was +9.5 n.s. on the BUGGY landscape. Best search shot of the arc but
  unproven.
- 📌 CORRECTION: honest-base bundle is **−0.48% nodes at fixed depth** (36,651,879→36,475,047), FAR less than
  the "−3.7%" cited for TT-fix-alone — either the bundle differs (NULL_MATE_CLAMP adds nodes) or the −3.7% was
  measurement-specific. Needs TT-fix-ALONE re-measure to correct the memory cleanly. C1's node-reducer claim
  was thinner than believed ⇒ reinforces its null.

## STATE: all work GATED, byte-identical at default, NOT committed (commit pending owner confirm — memory+docs
ARE the transfer, so a commit isn't required for the new window). Gated flags added this session (all default-off):
`KS_ONSET_MODE`, `KS_PIN_ATT`, `KS_ADJACENCY`, `KS_SQPRUNE_MODE`, `CAPG_KS_DAMP`(+`_PIVOT`), `ENABLE_QUIET_PROBE`
(diagnostic), `ENABLE_QSTANDPAT_SEED`, `ENABLE_TT_FLAG_FIX`, `ENABLE_NULL_MATE_CLAMP`. New diagnostics:
`_ks_auc.py`, `_ks_calibration.py`, `_ks_bench_liveness.py`, `_ks_dblpawn_coverage.py`, `_pgn_walk.py`,
`_capg_orient_check.py`, `_qquiet_agg.py`, `_collapses_from_selfplay.py`, `_qquiet`/`_searchbug` stderr probes.

**This session has THREE arcs. Arc 2/3 (search internals + the game-positive candidate) are the bigger finding — read first.**

---

# ⚡ ARC 2 — SEARCH INTERNALS: capgains is a qsearch crutch, and the search has 4 real value bugs

## The capgains mystery, solved
capgains (`approximate_capture_gains`, a static both-sides SEE melee folded into the leaf eval) is **not a
qsearch substitute** — it's the static-accuracy input that makes the shipped **+53.8 per-move qsearch
futility** safe. `QSTANDPAT_EVAL_MODE=0` puts capgains INTO the qsearch stand-pat, so 72% of qsearch
terminals are stand-pat cutoffs (measured: `ENABLE_QUIET_PROBE`, 8.1M terminals: horizon 0%, 72% stand-pat,
77% still have pending captures, only 23% genuinely quiet). **capgains is what LETS qsearch not descend to
quiet positions** — remove it and cutoffs mis-fire (+15-70% nodes). It's architecturally LOCKED IN (coupled
to +53.8); removing it needs a stronger capture-searching qsearch (the big lever), not a small fix.
Full: memory [[capgains-is-a-pruning-accuracy-term-not-a-qsearch-substitute]].

## FOUR real fail-soft VALUE bugs (fable audits vs LOCAL SF11/15.1 + measurement + gated fixes)
Full: memory [[search-value-bugs-and-the-productive-bug-pattern]].
1. **qsearch missing stand-pat seed** — init `best=±mate`, returns below achievable stand-pat (14% of nodes).
2. **qsearch phantom-mate** — all noisy moves pruned ⇒ returns fake ±mate (714k/WAC). Fix (both):
   **`ENABLE_QSTANDPAT_SEED`** (seed best=static_eval + pruned moves contribute their bound). Verified → 0/0.
   WAC 250→252, **+4% nodes** (removes fraudulent cutoffs), balanced STS −86 (scattered/noise).
3. **null-move UNPROVEN mate** — `ENABLE_NULL_MATE_CLAMP` (built). 141k/WAC but **root_adjacent=0 = LATENT**
   (never reaches root ⇒ the resign/freeze danger doesn't manifest). Cheap insurance.
4. **TT EXACT mislabel** — PVS re-search flags vs the entry window not the DRIFTED window ⇒ stores bounds as
   EXACT (61k/WAC). Fix **`ENABLE_TT_FLAG_FIX`** (built). Verified → 0. WAC 250→246 (−4), **−3.7% nodes**.

## ☠️ THE PRODUCTIVE-BUG PATTERN + the SEARCH-vs-EVAL objective (the durable strategy)
Every correctness fix is WAC-neutral-to-NEGATIVE because the search is TUNED AROUND its own bugs (the
phantom mates / mislabelled-EXACT gave aggressive-but-fraudulent cutoffs that HELP fixed-depth WAC). ⇒
**correctness ≠ strength on static benches; only fixed-TIME games decide.**
★ **Owner's framing (the objective):** a SEARCH change should REDUCE nodes / increase depth (lower
time-to-depth) while HOLDING accuracy — accuracy-per-position is EVAL's job. By that lens:
- **`ENABLE_TT_FLAG_FIX` (−3.7% nodes) is the real SEARCH candidate** — cheaper ⇒ deeper at fixed time; the
  −4 WAC (fixed depth) is plausibly recovered by the extra depth. **TEST AT FIXED TIME** (`wac_timed_depth`
  / `sts_timed_depth`: mean depth + accuracy), NOT fixed depth. (in flight at handoff time.)
- **`ENABLE_QSTANDPAT_SEED` (+4% nodes) is a CORRECTNESS fix, not a search-efficiency play** — don't bundle
  it with the search objective; judge it on its own (games).

## ★ SEARCH-KNOB SCREEN — RESULT (measured fixed-time; supersedes the predictions below)
Ran the fable-designed bundles at FIXED TIME (`wac_timed_depth`/`sts_timed_depth`). **Noise: mean DEPTH
stable ±0.001; SOLVES ±4; STS-SCORE ±30-60 — repeat STS before trusting.**
| config | WAC depth | STS score (2 runs) | verdict |
|---|---|---|---|
| baseline | 14.252 | 1618, 1648 | — |
| **C1 = TT_FLAG_FIX + NULL_MATE_CLAMP** | **14.383 (+0.13)** | 1714, 1654 (~+51, ~1.5σ) | ★ **the candidate** — real node-reducer, holds accuracy; modest, mostly tactical; Elo unresolved |
| +QSTANDPAT_SEED (no qcheck) | 14.250 (0.00!) | — | seed CANCELS the depth gain — correctness-only, exclude |
| +IIR (C6) | 14.518 (+0.27) | **1613** | ☠️ WAC mirage — STS below baseline |
| +IIR+margin750 (full stack) | 14.693 (+0.44) | **1617** | ☠️ bigger WAC mirage — STS below baseline |
| +QCHECK_DEPTH0 (C2) | 14.402 | 1682 | ☠️ −4 mates, STS < C1 |
| +QCHECK_FULL (C4) | 12.042 (−2.3!) | 1706/d11.64 | ☠️ q-tree explosion |

**GAME CANDIDATE (search): C1 only.** ⚠️ **METHODOLOGY LESSON: fixed-time TACTICAL (WAC) alone is a trap for
node-reducers — IIR/margin gained huge WAC depth by discarding positionally-important nodes (STS below
baseline). STS is the mandatory cross-check; only a both-instruments winner is real.** IIR reviving on the
honest-TT landscape PROVED the "bugs mis-calibrate knobs" thesis (its old −2.9% harm WAS the flag bug) — but
the revival is tactical-only, so no shippable bundle beyond C1. Full detail: memory
[[search-value-bugs-and-the-productive-bug-pattern]].

## (superseded predictions) COMPENSATED-BUNDLE SEARCH SCREEN (fable-designed)
⚠️ **`NODES` EXCLUDES the q-tree** (`qnodes=` is a separate counter) ⇒ judge q-tree levers on FIXED-TIME
depth (`wac_timed_depth`/`sts_timed_depth`) + `qnodes`, NEVER on `NODES`. My isolated QCHECK_DEPTH0 test was
INVALID — DEPTH0 is only SOUND *with* the seed fix (a q-node that drops checks + prunes all captures must
return stand-pat, not the old fake-mate; SF ships them as ONE unit). Landscape base = `ENABLE_TT_FLAG_FIX=1
ENABLE_NULL_MATE_CLAMP=1` (null-clamp rides free everywhere).
- **C1 (floor, MEASURED):** base only. Fixed-time WAC 253/depth 14.383 vs baseline 254/14.252 — +0.13 depth,
  accuracy held. The in-flight SPRT candidate; everything must beat it.
- **★ C2 (FLAGSHIP):** `+ ENABLE_QSTANDPAT_SEED=1 ENABLE_QCHECK_DEPTH0=1` — SF-qsearch-policy bundle. Seed
  ADDS main nodes (+4%, correctness), QCHECK_DEPTH0 shrinks the q-tree to pay it, TT-fix funds it. Both
  halves are SF11's SHIPPED design. Predicted: qnodes ↓ sharply, fixed-time depth ↑, accuracy held.
- **C3:** C2 `+ QDELTA_PERMOVE_MARGIN=750` (SF-scale ~0.75pawn vs our 2× loose; sound now the seed makes
  pruned captures contribute their bound). ⚠️ per-move knob, NOT the dead node-level `DELTA_MARGIN`.
- **C4 (C2 alt):** `+ ENABLE_QSTANDPAT_SEED=1 ENABLE_QCHECK_FULL=1` — keep checks at all plies but use the
  fast detector (tree-identical, NPS↑). Pays the seed in wall-time not nodes; head-to-head vs C2.
- **C5:** C2 `+ SEE_PRUNE_CAPTURES=1` (main-tree compensator; +9.5 n.s. standalone, now on sound q-values).
- **C6:** base `+ ENABLE_IIR=1 IIR_MIN_DEPTH=6` — re-test; its −2.9% harm plausibly WAS the flag bug (IIR
  writes TT entries that were stored mislabelled-EXACT). Biggest reducer if it now holds.
- C7 `+ ENABLE_NULLMOVE_EVAL_R=1` (clamp enables aggressive null) · C8 `+ ENABLE_PROBCUT=1` (speculative, last).
- **GENUINELY DEAD** (fable, don't re-test): `ENABLE_QSEE_RESORT`, `ENABLE_CONT_HIST_2PLY`, corrhist(+qsearch)
  (harm = coarse pawn-key keying, untouched by the fixes), `ENABLE_HIST_PRUNE`, node-level `DELTA_MARGIN`.
- ⚠️ Fixed-time is NOISY (~10%); establish the floor (repeat runs), demand a CONSISTENT depth gain, then ONE
  game gate on the surviving bundle — NOT per-knob SPRTs. Search is 0-for-13; C2 is the best-justified shot.

## STATE (arc 2): all gated, byte-id at default (250 / 36,651,879 re-verified). New flags:
`ENABLE_QSTANDPAT_SEED`, `ENABLE_TT_FLAG_FIX`, `ENABLE_NULL_MATE_CLAMP`, `ENABLE_QUIET_PROBE` (diagnostic).
New tools: `_qquiet_agg.py`, `_capg_orient_check.py`. Overnight: fixed-time screen + SPRT the TT-flag search
candidate; game the seed fix as a separate correctness question. NOT shipped, NOT committed.

---

# ARC 1 — KING SAFETY (earlier 2026-08-18)

## TOP BLOCK — read this first

**The reframe.** The 2026-08-17 diagnosis said KS is a phase-1 FEEDER (detection) problem and that quiet
positions are over-detected. **Measured on 5,481 SF-labelled positions, both halves are wrong.**

- **Quiet is NOT over-detected**: 98% of positions where SF sees no king danger already read exactly 0 for
  us. The "~5 units where SF nets 0" came from a 37-position hand-built counter set, not real data.
- **Our feeders are much better than assumed**: floor-free they give a nonzero reading on **96%** of SF's
  danger positions (AUC 0.9113). Of the 51% of real dangers we miss at defaults, **~47 points are discarded
  by the OUTPUT GATE and only ~4 are genuine detection failure.**
- **No built feeder helps.** A full sweep (adjacency, SQC, aim, both pin halves, weak-att2, weak-val) at
  `KS_FLOOR=0` produced **no resolvable gain** — every excess-over-volume ≤0.005 vs SE 0.007, and
  `KS_WEAK_VAL_MODE` was WORSE. The volume control landed at ~0 excess, validating the arithmetic.

⚠️ **BUT the feeder lane is NOT closed** — see §Calibration. AUC is scale-free and therefore **structurally
blind** to a uniform magnitude deficit, which is exactly what the known feeder defects would produce. We sit
at **0.45 of SF's KS magnitude**. That dimension is untested and is where feeder work should now aim.

## ★ THE CANDIDATE — `KS_ONSET_MODE=1 KS_FLOOR=6`

The floor does TWO separable things: it silences small readings (its purpose) AND charges
`ks_safety_table[KS_FLOOR]` as a STEP on crossing (an accident — the table already reads 42 at unit 13, so
the gate costs ~1.26 pawns of eval discontinuity that search feels through RFP/futility/stand-pat).
`KS_ONSET_MODE=1` subtracts the gate's own table value: silencing kept, step removed, everything above the
gate priced LOWER (net-subtractive, so it does not fight the 0-for-9 additive record).

| gate | result |
|---|---|
| byte-identity at gated default | 250 / 36,651,879 / EBF 3.751 ✓ |
| AUC (n=5,481) | 0.7365 → **0.8865 (+0.150)**; danger missed 51% → 16% |
| magnitude | 0.916 vs base 0.929 — slightly BELOW base |
| colour symmetry | identical to same-build control (11/800, 14/651) ✓ |
| balanced STS | 1661+1715 = 3376 vs base 1771+1613 = **3384 (−8)** |
| **D7 full-sample (15k)** | **HELDcrit −0.7028**, HELD regret 2.7042→2.6239 (generalises) |
| SPRT | launched 2026-08-18, 1200 games, elo1=5 — **RESULT PENDING, see below** |

**Floor 6 is an INTERIOR OPTIMUM on D7** (floor 4 = −0.046, floor 9 = −0.373) even though AUC is monotone
and wants 0 ⇒ **AUC cannot choose this knob; D7 is the tuner.** It also coincides with the independently
derived "lowest gate that does not raise magnitude" criterion.
Compare the PARKED V2 safe-check arm: same-size crit gain (−0.73) but bought with STS −181; this costs −8.

## INSTRUMENTS — the real story of the session

☠️ **The archetype bench is weak and misled most of the day.** At the default floor only **35%** of its 82
positions are live; `A4_other`/`B1_defended_crowd`/`B3_shelter` have **ZERO** live positions (they read
0.000 in every config, forever); `A3`/`A5`/`B2` are **ONE position each**. It is also NETTED between kings,
so a floor artifact on one king moves the other's number and a SUBTRACTIVE change can look like it RAISES
over-production. **Always run bench arms with `KS_FLOOR=0`** (liveness 35%→72%). Tool: `_ks_bench_liveness.py`.

🧰 **NEW — `_ks_auc.py`** (discrimination) over `ks_sets/diverse_corpus_wide.csv`, 23,113 rows carrying SF's
per-term `target_ks`; midgame n≈5,500, SE≈0.007. ☠️ **Run with `KS_FLOOR=0` and ALWAYS pair with the volume
control `KS_ATTACK_COUNT=2`.** The floor manufactures mass ties (both classes at 0) so volume alone inflates
AUC: the control gained **+0.107 floored vs +0.007 floor-free**. That control caught a false "adjacency is a
detection win" and is the single most important habit to carry forward.

🧰 **NEW — `_ks_calibration.py`** (magnitude), the dimension AUC cannot see. Ratio ours/SF per SF-|KS| band:

| SF \|KS\| band | n | base ratio | base zeros | candidate ratio | zeros |
|---|---|---|---|---|---|
| 0.05-0.25 | 36 | 2.24 | 56% | 2.08 | 8% |
| 0.50-1.00 | 247 | 0.64 | 49% | 0.58 | 8% |
| **1.00-2.00** | **1529** | **0.22** | **75%** | **0.29** | **24%** |
| 2.00-4.00 | 1161 | 0.54 | 26% | 0.49 | 8% |
| **OVERALL** | 3200 | **0.45** | 51% | **0.44** | **16%** |

Two findings: (1) the deficit is NOT uniform — there is a **HOLE at SF |KS| = 1-2 pawns** (ratio 0.22, 75%
silent) and that band is **48% of all positions where SF sees any king safety**; it is the floor's
fingerprint. (2) **The candidate is a COVERAGE fix, not a CALIBRATION fix** — zero-reads 51%→16% and
Spearman 0.575→0.608, but the magnitude ratio is unchanged at ~0.44. We remain at under half of SF's
magnitude everywhere. **That residual is the real target for feeder work.**

## 🎮 TWO OWNER-PLAYED LOSSES — and neither was king safety

Post-mortems via the new `_pgn_walk.py` (walks a pasted PGN, SF-evals every ply, ranks plies by cp lost on
OUR move) + `breakdown_fen` + ablation. **Both losses = ONE term hallucinating several pawns and FLIPPING
THE SIGN of the position:**

| game | term | evidence |
|---|---|---|
| lightning | **`capture_gains` −6.75** | ours −4.86 vs SF11-static +2.73. `SCALE_CAPTURE_GAINS=0` → ours +4.52, gap −7.59→**+1.79** ⇒ a **9.4-pawn swing** from one term |
| standard | **`passed_pawn_support` −4.55** | ours −2.88 vs SF11-static +0.64 / SF18-search +2.45. Three "passers" (a5/b5/c4) that are UNREALIZABLE — knight trapped on a2, enemy bishop pair + attack |

KS read +1.05 (correct side!) in the first and exactly 0.00 at every worst ply in the second. Also a strong
BEHAVIOURAL signal: an aimless rook shuffle (`Rf8→Rf7→Rf8`) cost 110+158 cp — no-plan, not an eval term.

☠️ **PHANTOM KNOB — no master off switch for passed pawns:** `SCALE_PASSED_PAWN=0` removes only **18%** of
the term, `PASSER_MAG_SCALE=0` only **32%**; ~68% under neither ⇒ **≥3 independent magnitude sources**, and
any past passer ablation via the obvious knob measured a fraction of the term (its null is not a null).
Detail + the reopening of the "passer VALUATION CLOSED" verdict: `PAWN_MODEL.md` §4b.

## STATE

HEAD on NN-ENgine = the shipped +20.8 bundle. **Nothing committed.** New GATED knobs, all default-0 and
byte-identical: `KS_ONSET_MODE` (1 = continuous onset, 2 = onset + rescale — mode 2 measured near-inert, the
rescale normalizes against `KS_CAP` which is far above the live operating range), `KS_PIN_ATT` (executes,
AUC-flat — pinned enemy attackers on king zones are rare), `KS_ADJACENCY` (unproven: +0.005 excess over
volume, under SE), `KS_SQPRUNE_MODE` (modes 1-4, all measured, none resolvable).
New diagnostics: `_ks_auc.py`, `_ks_calibration.py`, `_ks_bench_liveness.py`, `_ks_dblpawn_coverage.py`,
`_pgn_walk.py`. Patched: `_ks_detect_dist.py` (KEY=VAL argv), `_ks_bench_score.py` (ratio line),
`_ks_regret_score.py` (CONFIGS = the onset sweep).
📐 Canonical doc: **`dev_notes/KING_SAFETY_MODEL.md`** — pipeline vs SF11/SF15.1/Ethereal with provenance,
instrument-resolution table, and a REFUTATION RECORD. Read it before any KS work.

## NEXT

1. **Read the SPRT result** (below / `selfplay/games/sprt_onset6/`). Ship only on the games.
2. **The magnitude lane** — feeder work aimed at the 0.45 calibration ratio, not at AUC.
3. **`capture_gains` and `passed_pawn_support`** — where the live-game evidence actually points. Use
   `collapse_term_attribution.py` (has a quiet-position CONTROL set) rather than building a new probe.
4. **The second discontinuity**: the `isEndGame` 71%→0 cliff at `phase_score` 64→65 (`KS_EXTEND_EG`), same
   defect class as the floor step we just fixed.
5. `wac_speed` A/B on an idle box — the 2026-08-18 reading was taken under gaming load and is unusable.
