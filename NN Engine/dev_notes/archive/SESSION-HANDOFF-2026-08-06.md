# Session handoff — 2026-08-05/06: the pawn subsystem, fitted three ways, 0-for-3 in games

**Read this top block first.** Prior: `SESSION-HANDOFF-2026-08-05.md`. Canonical pawn reference:
[`PAWN_MODEL.md`](PAWN_MODEL.md) (new, registered in `NN Engine/CLAUDE.md`).

---

## STATE — engine unchanged, nothing committed
Default still **`250 / 35,791,173 / EBF 3.804`**, verified byte-identical after every build today. All new
knobs gated off. Nothing shipped, nothing committed.

## ★★★ THE HEADLINE — three corpus wins in a row, zero Elo
| change | corpus val | games |
|---|---|---|
| pawn structure penalties (earlier) | −7.3 | ~0 |
| iteration 1 — fitted rank/file tables | −13.98 | **−15** (232g) |
| iteration 2 — + endgame surface, `opposed`, tunable caps | **−17.4** | **~+4** (324g, cut) |
| **`ISOLATED`/`BACKWARD` ALONE (120/120)** | — | **+12.4 ±30.2** (700g, inconclusive) |
| *(contrast: capped threats, shipped)* | −5.2 | **+45** |

## ⚖️ THE ONE ARM THAT DID NOT DECAY
`ISOLATED_PAWN_PEN=120 BACKWARD_PAWN_PEN=120`, tested alone for the first time:
**W280 L255 D165, 51.8%, elo +12.4 ±30.2, LLR −0.013, inconclusive at the 700 cap.**
Ten progress checks read **+7, −3, +8, +8, +11, +17, +17, +20, +14, +9** — a flat non-negative band.
Compare `pawn2` (+44 → +4, monotone decay) and iteration 1 (+45 → −15). A zero effect wanders symmetrically;
this did not.
⚖️ `_venue_power.py` predicted this exactly: resolving +10 Elo needs **~4,344 games**, 700 buys **±30**.
⇒ **"Not ≥25 Elo, point estimate +12.4"** — NOT a failure. The owner's thesis is **mildly supported and
unresolved**, and resolving it needs ~4,000 games or a higher-power venue.

**The proxy improved monotonically. Elo never moved.** This is a stronger statement than any single null,
and it is the main result of the session. ★ Mechanism already on file:
[[most-eval-error-is-move-neutral]] — 52% of our eval error is large but does NOT change the move we play.
Pawn terms shift standing evaluations more than they flip candidate moves.
⇒ **For pawn changes, a corpus gain is not a reason to spend games.** Prefer candidates with a mechanism
argument independent of the corpus, and go to games earlier.

## WHAT WAS BUILT (all gated, all byte-identical off)
- **Per-rank pawn tables** `RANK_DEF/PSD/EG_R2..R7` and **per-file** `CHAIN_F_A..H`, `WALL_F_A..H`. Whole-table
  `SCALE_*` could only rescale a hand-picked SHAPE; these let the descent choose entries. First time the
  tables were ever FITTED.
- 🚨 **The endgame structural literals became knobs** — `EG_PHALANX/SUPPORT/DEFEND/LATENT` were hardcoded and
  **no knob reached them** (`SCALE_PAWN_WALL/_CHAIN` only rebuild the file tables the MIDGAME path reads).
  Endgame pawn structure was flat in file, flat in rank, and had **never been fitted once**.
- **Tunable caps** `PAWN_CLAMP_MID/EG`; **per-phase structural rank curves** `STRUCT_R_MG/EG_R2..R7`
  (two evaluators ⇒ the phases can carry different SHAPES, which SF's `S(mg,eg)` pairs cannot);
  **`opposed`** modulator; **`PASSER_R_MAX`**; **`ENABLE_PASSER_DEFER_ON_FLAG`**; the six `PP_*` constants.
- ☠️ **`ENABLE_PAWN_OBSTRUCTION_BLEND` — DEAD**, keep off (+7.33 val worse, and it damaged `contested`, our
  best-calibrated context). It was **my invention, not an SF port**.

## WHAT THE DESCENT CHOSE (and what it overturned)
- ☠️ **Both caps want to go DOWN** — 225→175, 175→125 — *even with new terms competing for the headroom.*
  Against the brief's "raise the clamp" and against the intuition that added terms need room.
- ✅ **`opposed` pays**: both phases pinned to the grid minimum **60%**. `getPPIncrement` detects it and
  throws it away via an early `return 0`. ⚠️ `opposed` ⊂ "not passed".
- ↩️ **`PASSER_R_MAX` declined more headroom** (offered 512, kept 384) ⇒ "the architecture blocks a passer
  valuation it wants to pay" was **overstated**.
- `ISOLATED_PAWN_PEN` 80→120, `BACKWARD_PAWN_PEN` 120, chain up then down, all structural rank curves → 80.

## 🚨 INSTRUMENT LESSONS (the expensive ones)
- **Manufactured positions were wrong by 2.6× on magnitude.** Retracted: "our curve is flat 1.5× vs SF 7.6×"
  (real: **1.9× vs 2.1×**) and "we underpay rank 7 by 83" (real: **overpaid by 88**). ✅ Survives: a
  near-uniform level error, ours **+58 cp/pawn vs SF18 −6** (1,621 real removals). **Validate a manufactured
  instrument against real positions before believing a magnitude.**
- **Four generator defects**, each producing a confident wrong answer: context pawns on r+1 were DEFENDED BY
  the test pawn; a `|value|>400` cut **filtered on the dependent variable**; sparse backdrops made a pawn
  decisive not marginal; `blocked` cannot exist on the 7th so that rank sampled only free runners.
- **Count the resolvable effect size before recording a null as a failure.** `ISOLATED/BACKWARD`'s "four
  failing venues" were each ±35-67 Elo wide — inconclusive, not negative. 🧰 `_venue_power.py`.
- 🐛 **The `ps` sub did not match `sprt.py`**, reported a live SPRT as dead, and a duplicate was launched on
  top of it. ✅ Pattern fixed. **A negative from a monitoring tool is a claim about the tool.**
- 🐛 **Never edit a shell script while it is executing** — bash reads incrementally; editing
  `overnight_runner.sh` mid-run corrupted the running instance and faked a failure on a successful descent.

## 🆕 ITERATION 3 (2026-08-06 daytime) — shipped regime, widened corpus, new surface
- **Corpus widened 2,713 → 4,987 rows** (bank labelled to 4,925/4,987, ~4 pos/sec at d13). Snapshots
  `*_pre0806.csv`. 🚨 **No `val` from before today is comparable.** New shipped-default baseline
  **train 230.555 / val 234.881**.
- **`PPS_OWN_BLOCK/ENEMY_BLOCK/OWN_ATTACK/ENEMY_ATTACK`** exposed — the passed-pawn support magnitudes
  (`y*75`, `y*100`, `y*60`, `y*50`), **never tuned before**. `PPS_OWN_ATTACK 60→35` was the largest single
  move of the descent.
- **First descent in the SHIPPED regime** (`pawn_fit_shipped.py`) — earlier ones optimised inside
  `ENABLE_KS_CHECK_V2=1`, which is not the default, so their winners were never directly applicable.
  Result **val 234.88 → 213.03 (−21.85)**, all nine guards held.
- **SPRT stopped at 389 games for context transfer**: `+161 −141 =89, LLR +0.556, elo ~+18` — volatile
  (−6, +6, +4, +2, +2, +8, +18) and **not converged**.
- ⚡ **Speed re-established**: pinned **peak 445,330 NPS** vs register 446,218 ⇒ nothing today cost
  measurable NPS. One unconditional cost (the `opposed` mask, computed per pawn even at default) was found
  and short-circuited. **Byte-identity does not detect added work.**
- 🧰 New: `pawn_fit_shipped.py`, `blend_corpora.py`, `pawn_truth_casebook.py`, `_endgame_passer_doublepay.py`.

## ▶️ NEXT — my recommendation, for you to accept or reject
**Owner's direction (2026-08-06):** keep building these items plus new ideas into ONE joint shippable
candidate rather than testing terms individually. That is the right call given the measurement floor —
four pawn arms have each landed inside ±30 Elo, so a bundle large enough to sum past ~30 is the only unit
games can resolve.

1. **Build the remaining terms** so the joint candidate has real new surface, not just retuned constants:
   SF's `WeakUnopposed`, `BlockedPawn[]`, phalanx-count modulation of the connected bonus, the king-support
   term currently buried in `advanced_endgame_eval`, and a true rank × file table for chain/wall.
2. **Expand the corpus ONCE to ~25k and re-baseline** (`build_position_bank.py` → `add_sf18_labels.py` →
   `build_diverse_wide.py`). Labelling is ~4 pos/sec, so this is 1-2h, not a day. ⚠️ Batch it — every
   corpus change invalidates all prior optima, so dribbling costs a re-baseline each time.
   ⚠️ It will NOT close the proxy→Elo gap: val tracks train at 0.89-0.98, so we are not overfitting. That
   gap needs a different OBJECTIVE (e.g. weight positions by whether the error changes the best move).
3. **One joint descent over EVERYTHING in the shipped regime** — pawns, KS, attack layer, placement,
   imbalance, capgains, Kaufman — with `ENABLE_KS_CHECK_V2` as a knob the descent chooses rather than an
   assumption baked into the seed. Selection rule: include a component only if it is **non-negative on the
   proxy AND has a mechanism argument**; exclude anything merely "never measured".
4. **Then one SPRT on the bundle.**

## ☠️ SIBLING-SPREAD TEST — RUN, AND IT REFUTES THE LEADING HYPOTHESIS (2026-08-06 evening)
🧰 `_sibling_spread.py`, 800 real corpus positions, ~32 children each, two seeds, **zero Stockfish**.
Delete a whole term group, re-take the argmax — **deletion is the CEILING on what retuning could do.**
`threats` carried as the control: the one recent eval change that won games (+45 Elo).

| group | sibling spread | flip >0 | flip ≥25 cp |
|---|---|---|---|
| `pawn_nonmat` | 49.4 cp | **14.0% / 16.6%** | **7.1%** |
| `threats` (control) | 101.2 cp | 10.0% / 11.5% | 5.1% |
| `capture_gains` | 609 cp | 38.1% | 33.9% |
| `piece_place` | 124 cp | 35.2% | 22.1% |

⇒ **Pawn scoring reorders candidates MORE often than the term that won +45 Elo.** It is NOT arithmetically
locked out of move choice. **Do not retire the pawn lane on that argument.**
★ The mechanism was real and over-generalised: with `QUIET_ONLY=1` pawn spread collapses **49.4 → 6.7 cp**
— among PIECE moves the term is nearly constant, exactly as claimed. But the candidate set is not piece
moves, and the variation lives in **pawn moves and captures**, which is where a pawn term should act.
☠️ Also killed: "threats convert because they attach to pieces that candidates relocate" — threats reorder
*less* than pawns.
▶️ **Now the leading explanation:** the effect is **genuinely ~10-20 Elo, under our ±30 floor** (SF18
prices the whole structural axis at +4..+27 cp; `isobwd` measured +12.4 ±30.2).
▶️ **The sharp unrun follow-up:** deletion bounds the SUBSYSTEM; it says nothing about the tuned DELTA.
Dump each position's best move under the default and under iteration 3 and diff. 🐛 Blocked once: **the
iteration-3 knob string was passed on the `gate` command line and persisted nowhere** — record arm configs.
⚠️ Instrument limit: a ONE-PLY STATIC argmax. It bounds the leaf score and static ordering, not the played
move.

▶️ **Still unrun — the other cheap diagnostic:**
   - **Every-ply audit** — is `decision_fen` even the mistake ply? A lot of collapse work assumes it is.
⚖️ Also open: [[collapse-leverage-map-replicated-two-samples]] — **~54 pts forfeited/200g, 60-69%
POSITIONAL, ~67% OPENING/EARLY-MID**. Nothing in two days of pawn work touched that region.
