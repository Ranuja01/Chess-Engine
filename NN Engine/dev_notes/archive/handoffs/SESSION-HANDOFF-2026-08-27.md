# SESSION HANDOFF 2026-08-27 — LMR product schedule SHIPPED (+20.7/2800g, first search win); presearch CLOSED at the mechanism level; KS reframed as a SHAPE defect

## ⏱️ READ FIRST — state in one screen

- 🏆 **SHIPPED + COMMITTED `379a5c3`: `LMR_SHAPE=1 LMR_PRODUCT_DIV=64`.** +20.7 ±15.1 Elo / 2800 diverse-UHO
  games. **New register: `251 / 34,318,987 / EBF 3.785 / STS 1795`.** `LMR_SHAPE=0` reverts EXACTLY to
  `243 / 31,764,817 / 3.729 / 1703`. **This is the first SEARCH win in the project's history** (search was
  0-for-13 before it).
- ☠️ **The presearch lane is CLOSED in all three forms** — remove, replace, and reduce — each for a
  *mechanistic* reason, not a tuning miss. See §2. Every prior test of it was confounded.
- 🔬 **King safety is a SHAPE problem, not detection or magnitude** — and the cause is an off-by-one that
  has been live for the engine's whole history. See §4. This reframes twelve failed KS attempts.
- 📏 **Instrument resolutions now measured** (§5). Most of what we screen is below what we can see, which is
  the real reason the cheap-screen well keeps coming up dry.
- ▶️ **Next lane, ranked, in §6.** Top item is a knob sweep with no code.

---

## 1. THE SHIP — `LMR_SHAPE` (the only thing that changed in the engine)

`reduced_search_depth` combined remaining depth and move number **additively**, and derived its base from the
**ITERATION** depth (`DEPTH_REDUCTION[depth_limit]`) — so it never read `cur_depth` and applied the same
constant reduction at every node of an iteration. Near the horizon that constant consumes the child's last
ply: the header records wrong-reduction at **~1% at L1-L5 but 4.1% at L6 and 6.2% at L8**.

Mode 1 ports the classical Stockfish schedule: `reduction = log(remaining) x log(move number)`, keyed on the
node's own remaining depth. `log(remaining)` collapses the reduction toward zero at the horizon by itself,
and amplifies it where depth remains.

**SF's constants do not transfer** — `LMR_PRODUCT_DIV` had to be calibrated here:

| `DIV` | 1024 | 512 | 256 | 128 | 64 |
|---|---|---|---|---|---|
| WAC | 249 | 248 | 249 | 251 | 251 |
| nodes | 50.0M | 37.1M | 33.7M | 35.3M | 34.3M |

Node floor ≈33.7M (the `rem-2` clamp saturates below ~128 and LMR re-searches add nodes back), so work can
never be matched to the old 31.8M. Fixed depth at DIV=64: **+8 WAC / +92 STS for +8% nodes**.

**Games (the decider):** 7 segments x 400, varied seeds, `openings_uho.txt`.

| seed | 7 | 23 | 41 | 59 | 77 | 93 | 109 | **pooled** |
|---|---|---|---|---|---|---|---|---|
| elo | +28.7 | +29.6 | +24.4 | **−15.6** | +16.5 | +39.3 | +22.6 | **+20.7 ±15.1** |

⚠️ **Fixed-TIME benches read FLAT** (+6 WAC but unstable across two passes: +2 then +10; −3 STS), and mean
depth *falls* ~0.2 ply. So the gain is not depth. Best explanation is fewer catastrophic mis-reductions near
the horizon — a blunder-rate effect that games see and a pass/fail suite does not. **That is a story fitted
to the data, not a measurement.** Testable: the advantage should concentrate in longer games; `summary.csv`
carries ply counts and termination reasons for all 2800 games. Never checked.

⚠️ **Untuned and open:** `LMR_PRODUCT_K=2480` was never swept (only DIV). The phase-dependent `scale`
(1.5-2.25) and the deep-endgame disable (`phase_score >= 117 -> return base`) were deliberately preserved to
isolate the form change. Both are now open questions on a live mechanism.

---

## 2. PRESEARCH — CLOSED, and every earlier test of it was confounded

`reorder_legal_moves`/`pre_minimizer` runs a full-width shallow search of every root move each iteration.
`SearchData` has exactly three fields, so its outputs are bounded and enumerable:

| output | consumer | measured failure without it |
|---|---|---|
| `moves_list` **order** | root loop; **root LMR keys on move INDEX** | `nopre+norazor+ROOT_LMR` = **181/300** |
| `scores[i].top_score` | root razoring + the `scores[0]` threshold adjust | **124/300**; static-eval substitute → **39/300** |
| `scores[i].second_moves` | `alpha_beta` indexes it **unguarded**, per root move | must be synthesised |
| `scores[i].second_scores` | `minimizer`'s `ascending_sort` → **ply-2 ordering** | silently falls back to heuristic order |

**Two of those four are irreducibly the product of having searched those moves.** Previous-iteration data
gives sentinels for fail-lows; static eval is zero-ply. Neither manufactures a score for a move nobody looked at.

**(a) The 124/57 collapse was ROOT RAZORING, not lost ordering.** At defaults `razorable = i < synthetic_from`
is purely positional and the evidence test (real score AND recent AND deep enough) only runs under
`ENABLE_ROOT_TABLE`, which is **off**. Reused previous-iteration scores are therefore razored as if current,
and `alpha - stale_score` is inflated because alpha grows with depth. Razor-off → 242, table-on → 237.

**(b) Static-eval razoring is structurally impossible — COMMENSURABILITY.** Built `PRESEARCH_OFF_FILL`
(gated, byte-id). Filling every root move with a static eval measured **39/300**, nodes *below* base, lowest
EBF ever seen — it razors the root away, because alpha is searched to iteration depth and a static eval is
zero-ply. **SF razors on static eval ONLY at `depth < 2`, where the two ARE commensurate.**
★ Modes 1/2 (opposite polarities) came back **byte-identical** — a liveness result showing the fills were
already excluded by `synthetic_from`, and locating the razorable population as the REUSED entries.

**(c) Making it cheaper also fails.** `PRESEARCH_TAIL_MODE=2 CHUNK=4` = 249 solves but **+9.4% nodes** and
**STS −98**; `ENABLE_PRESEARCH_SUBSET=1` = +5.5% nodes (07-30). Degrading it costs more downstream than it
saves upstream.

**(d) The SF-like alternative loses:** no presearch, no root pruning = **242 / 45.03M** vs base
**243 / 31.76M** — same strength, **+42% nodes**.

**Decomposition:** presearch+razor 31.76M · presearch only 39.34M · razor-defanged 43.27M · neither 45.03M.
⇒ razoring is worth ~19%, the presearch's ordering ~13%, and **its own cost is already inside every one of
those numbers.**

▶️ **THE REFRAME THAT MATTERS.** "The giants don't need a presearch" is true but not transferable — *they
also never prune a root move*. Ours trades a shallow full-width pass for the right to razor aggressively at
the root, and on this engine that trade pays. **The presearch is compensating for a weaker ORDERING STACK.**
So the route to not needing it is not attacking it — it is strengthening ordering until it stops paying.
🧰 **Progress meter:** re-run `ENABLE_ROOT_PRESEARCH=0 ENABLE_ROOT_RAZOR=0` after each ordering change and
watch **45.03M** walk toward **31.76M**. When they meet, removal is viable and we will have seen it coming.

---

## 3. ORDERING-STACK GAPS vs the giants (the lane that feeds §2's meter)

Source-read Weiss, Ethereal, SF11, Caissa.

- 🐛 **Capture history has no VICTIM dimension.** Ours is `captureHistory[2][64][64]` = `[side][from][to]`;
  Ethereal `[piece][threat_from][threat_to][to][captured]`, Caissa `[stm][piece][captured][to]` —
  **both include the victim and both DROP the from-square.** We do the exact inverse. `Rxe5` taking a pawn
  and taking a queen share a slot. ▶️ **A complete diff is drafted** (every read/write/decay site verified,
  funnelled through one `capture_hist_ref` helper so the two keyings cannot drift; both arrays kept so
  defaults stay byte-identical; new table is 24.5KB vs 32KB, i.e. smaller AND denser).
- 🐛 **`clearSearchTables` never clears `captureHistory`** (verified, `search_engine.cpp:2489-2507`). Every
  bench we have ever run has capture ordering contaminated ACROSS positions — a hole in the 08-14
  contamination fix. **Fix this SEPARATELY and FIRST: it moves bench numbers, so it must not ride along
  with a feature.**
- **Pawn history is absent entirely** (Weiss: `pawnHistory[pawnKey][piece][to]`, gravity div 8663).
  ⚠️ Blocked on instrumentation: it accumulates ACROSS moves in a game, and our benches clear per position,
  so no bench we own can see its mechanism. Needs a game-walk instrument first. ⚠️ Prior is poor — corrhist
  died of sample starvation on a SMALLER key space.
- ☠️ `ENABLE_PIECE_CONTHIST` (a separate piece×to term, not a re-keying) measured neutral twice: 249 (07-27)
  and 240 (08-26). Closed.
- ☠️ `ENABLE_NULLMOVE_EVAL_R` (SF11's eval-scaled null R) is live and correctly scaled but needs
  `eval − beta ≥ 1920` mp to add one ply, so it almost never fires. 242 / +0.5% nodes. Null.

---

## 4. KING SAFETY — a SHAPE defect, and an off-by-one that has always been live

Two agents: a three-phase audit of ours (FEEDERS → SUBSYSTEM → TRANSFORMATION) and a source-read of
SF11/Ethereal/Weiss in the same structure.

### ☠️★★★★ THE FINDING
`rebuild_ks_tables` builds `ks_safety_table` as **`u²/4` up to `KS_KNEE=12`, then affine `36 + 6(u−12)`** to
`KS_CAP=80`. But **`KS_FLOOR=13`** — *one unit above the knee* — and anything below the floor returns 0.
⇒ **The quadratic band has never been reachable. Every king-safety reading this engine has ever produced
rides the affine tail at 6 danger per attack-unit.**
Real attacks accumulate ~13-51 units (median ~18): u=18 → 72 danger → **2.16 pawns**, which is exactly the
**−3.09** ceiling observed in collapse positions where **SF11 reads −14.41**.
Also the source of the floor discontinuity: crossing 13 charges `table[13]=42` at once = **0 → 1.26 pawns**,
the fingerprint of the documented "SF |KS| = 1-2 pawn hole" where 75% of positions read exactly zero.

### ✅ IT IS NOT A CLAMP
Traced the whole Phase-3 stack: per-king danger caps at `table[80]`=444, and `KING_SAFETY_MAG=3000` gives
`3000*444/100` = **±13.3 pawns**. We CAN express what SF11 expresses. The deficit is Phase-2 arithmetic — a
flat sum feeding a linear tail. ⚠️ And raising magnitude through the non-discriminating map is already
refuted: `KING_SAFETY_MAG=6000` cost **−222 balanced STS**.

### 🔬 WHAT THE GIANTS DO
- **None of them CAPS king safety.** SF11's −14.4 is architecturally unbounded; the only guard is a
  `kingDanger > 100` dead zone. **Strong HCEs control the BOTTOM (gates), never the top.**
- **SF11 is effectively QUARTIC in attacker count:** `kingDanger += kingAttackersCount * kingAttackersWeight`
  (a PRODUCT — each attacker scales the whole weight sum) `+ 3*kingFlankAttack²/8`, then
  `score -= kingDanger²/4096`. danger 200 → 9 cp; danger 2000 → 976 cp. **Silent on phantoms, explosive on
  real attacks — the discrimination is entirely SHAPE.**
- Ethereal: linear accumulator, `mg²/720` map, hard gate (≥2 attackers unless a queen is present).
- Weiss: linear with a cliff at the 2nd attacker — the weakest of the three, and **the shape ours most
  resembles**.

### 🎮 LIVE-GAME CORROBORATION (owner played, both as Black, both lost)
Walks + `probe_fens.py` triangulation. In four evaluable positions **SF11-static tracks SF18-search while we
are 2-4 pawns off** — the *statically fixable* signature, not search-absorbed. Corpus-wide
(`sf11_collapse_gap.py`), in the nine worst collapse positions SF11 attributes the gap almost entirely to
king safety (−14.41, −7.89, −6.10, −5.87, −4.04) where ours reads −3.09, −1.56, or is absent from the top ten
— with 4, 6, 7 and **9** enemy king-ring attackers present. Our win% reads 71% where SF11 reads 9%.
⇒ **Material over-valuation is a SYMPTOM: we price the attack at ~nothing, so material decides the score.**
⚠️ Both games had us as Black, so "optimistic about Black" vs "optimistic about ourselves" is not separated
(symmetry harness argues the latter — colour violations run 1.4%).
📁 `diagnostics/_loss_berlin_2026-08-27.pgn`, `_loss_giuoco_2026-08-27.pgn`, `_loss_worst_fens.txt`.

### ☠️ ARMS CLOSED THIS PASS
- **`KS_SHELTER_MAG` is DEAD at defaults** — byte-identical at 25; it only reads under `ENABLE_KS_V2=1`.
  The memory claim that it is "the ONLY live duplicate channel never measured alone" is **FALSE**.
- **Adjacency redistributive pair: NULL.** 2×2 (base / `ADJ=2` / `COUNT=0` / both): WAC 251/247/250/250,
  nodes 34.32M/33.07M/**36.54M**/**32.96M**, STS 1795/1773/1691/**1780**.
  ✅ The pair is **SUPER-additive** (+111 STS over the additive prediction) — redistribution works exactly as
  the model doc theorised, the opposite of the anti-additive `ROOT_LMR` corner. ☠️ But it redistributes to
  **NEUTRAL**. Plateau over `ADJ` 1/2/3/4 = 1690/1780/1695/1785, **ragged, none reaching baseline 1795**,
  whole spread inside the ±150 STS floor.
- ⚠️ `KS_ATTACK_COUNT=0` alone COSTS **+6.5% nodes** — proximity does real pruning work.
- ⚠️ Corrections to the record found by the audit: additive KS is **0-for-11**, not 0-for-9;
  `KS_ATT_PRODUCT` is already refuted (**−8.8 ±17.3**); `KS_BATTERY` was wired 08-16 so the "phantom"
  housekeeping note is stale; the open-file predicate's "own pawns only" is **defensible**, not a bug (storm
  is a separate term) — its real weaknesses are that it is binary rather than graded, and unconditioned on
  the enemy having a rook or queen to exploit the file.

---

## 5. INSTRUMENTS — measured resolutions, and why most screens are futile

| instrument | resolution |
|---|---|
| D7 regret @MAXN=6000 | **±0.08** — deterministic but the knob→regret map is JAGGED |
| WAC solves | **±5-6 solves** (240-249 observed across null configs) |
| STS | **±150** |
| fixed-TIME benches | non-deterministic, 14.6% NPS spread; **called `LMR_SHAPE` flat when games said +20.7** |
| games | ±40 per 400g segment; ±15 at 2800 |

- ★★★★ **±0.5 ply is Elo-NEUTRAL** (measured by moving `VERIFY_RESEARCH_REDUCTION` in both directions at
  fixed time). ⇒ **a speed change must be worth ≳0.5 ply (~35%+ NPS) to be MEASURABLE.** That closes the
  entire micro-speed class with a number: movegen's best idea was ~0.1 ply.
- ★★★★ **THREE DEAD KNOBS FOUND IN ONE DAY** (`DELTA_MARGIN` — dead but PRINTED in the toggles dump while
  the live `QDELTA_PERMOVE_MARGIN` is hidden; `CONT2_GRAVITY_DIV`; `KS_SHELTER_MAG`). ⇒ **before any sweep,
  run one arm at an extreme value and prove the node count moves.** A both-directions-identical result is a
  LIVENESS result, not a null.
- ★★★★ **A feature measured in the regime where it is redundant looks worthless** — now **five** instances
  (TT-move, the root table's 14 arms, singular, piece-conthist, and SF's `RootMove` structure judged as a
  razoring guard when its real jobs are ordering and aspiration). **Ask which REGIME a closure was measured in.**
- ★★★ **A catastrophic ablation (124, 57, 39, 181) is a BUG REPORT, not a verdict.**
- ★★★ **Don't chain one measurement onto another to skip a test** — I inferred `LMR_SHAPE`'s node cost was
  free from the 0.5-ply bar; the direct fixed-time test disagreed.

---

## 6. ▶️ WHERE TO GO NEXT (ranked)

1. **`captureHistory` clear-gap** — one-line class of fix, but it MOVES BENCH NUMBERS, so it must land
   before anything else is bench-screened or it contaminates every comparison. Do this first.
2. **Capture-history re-key** to `[side][piece][captured][to]`. Diff drafted, all sites verified, table
   smaller and denser, two independent references converge on the design, and a deferred note
   ("revisit after capture-hist turn-on") is now due. Ordering-stack work ⇒ feeds §2's progress meter.
3. **KS joint `KS_FLOOR`/`KS_KNEE`/`KS_CAP` curve sweep — specifically configurations with FLOOR < KNEE**,
   so the quadratic band is live for the first time. Never swept jointly (model doc). Knob sweep, no code.
   ⚠️ `KS_FLOOR=0` and `ONSET_MODE=1 FLOOR=6` were separately games-nulled, but **never with the knee moved
   to keep a working convex region above them**. ⚠️ The adjacent `KNEE→40` test failed with a stated
   mechanism ("you cannot compound units that don't discriminate"), so the prior is mixed.
4. **`LMR_SHAPE` follow-ups on a now-live mechanism**: sweep `LMR_PRODUCT_K` (never swept), and revisit the
   phase `scale` and the deep-endgame disable that were preserved to isolate the form change.
5. **Adaptive aspiration** (plan complete: `score_two_ago` is the only missing state; size the window from
   score volatility instead of a fixed 500; our fail rate is ~51% against a healthy 10-30%).
   ⚠️ Predicted interaction: tighter windows → more root fail-lows → more UNPROVEN entries feeding root
   razoring's provenance guard. Watch `g_root_razor_stale_skips`.
6. **`RootMove`-style `prev_score` root ordering** — SF's actual use of a structure we ported and then only
   ever judged as a razoring guard.
7. **Per-opening / per-game-length analysis** of the 2800 `LMR_SHAPE` games — does the advantage concentrate
   in longer games, as the blunder-avoidance story predicts? Pure CSV analysis, no engine. Also tests whether
   the same opening families recur as losses across candidates (a persistent weakness class would outrank
   everything above).

**Deferred as a CLASS:** time management. `MOVE_TIMES` is a per-depth soft schedule with a `TIME_LIMIT` hard
cap and no stability input. ☠️ **Unmeasurable in our harness at any preset** — under a per-move budget,
extending on instability makes the candidate spend more wall-clock than the baseline, so any win is
confounded with "had more time". LIGHTNING has ~0.25s of headroom against iterations that take far longer.
Needs a **game clock** (e.g. 10s+0.1s) to become testable at all. Owner's call: strength first, allocation later.

---

## 7. Housekeeping

- Committed this session: **`379a5c3`** (search_engine.h + search_engine.cpp only — the ship plus the gated
  `PRESEARCH_OFF_FILL`). Everything else in the tree is still uncommitted, including pre-existing scratch
  and large CSV datasets. Diagnostics prune (~50 dead scripts) still pending owner confirmation.
- New diagnostics: `_passer_miss_profile.py` (with its control), `_loss_*.pgn`, `_loss_worst_fens.txt`;
  `_ks_regret_score.py` gained an env `ARMS=` override so it need not be forked per subsystem.
- `KING_SAFETY_MODEL.md` **needs updating** — it predates the 08-20/21/24 game verdicts and the §4 finding.
- ☠️ **Never rebuild while games or a long diagnostic run** — it swaps the `.so` under the running process.
  All 7 game segments used one binary; that is what makes the pool valid.
