# Session handoff — 2026-08-09: capgain lane closed, eleventh symmetry defect found and fixed (gated)

> ## 🌙 OVERNIGHT 08-09 — FRESH COLLAPSE CORPUS + PROGRESS CHECK
> **Old corpus was stale by two shipped bundles** (mined on `246 / 35,089,668 / STS 1746`). Refreshed on
> the current build: batch A 200g + batch B 400g, `vs_sf 2400`, defaults. All three snapshots preserved
> under `selfplay/games/_archive/` (`_preserve_vssf.py`, new).
>
> ✅ **WE GOT BETTER:** score vs SF@2400 **44.5% → 48.3%** (600g, +3.8pp ≈ +26 Elo, ~1.9σ); points
> forfeited **30.8 → 27.1 per 100 games**. Score is outcome-based ⇒ no eval-shrink confound.
> ☠️ **BUT THE FAILURE MIX IS UNCHANGED** — 70% opening/early-mid, 62% positional, 61% ply 40-79, all
> replicated on 400g; mean over-read still **+2.01 pawns vs SF11**. The ten symmetry defects were real
> and fixing them did NOT change what kind of position we lose.
> 🎯 **67% of positional collapses are STATICALLY FIXABLE.** Term gaps vs SF11: Material **+0.674**,
> **Space/`central` +0.392 (NEW #2, never swept)**, KS +0.192. ★★★ In the worst cases **`place`
> dominates (+3.4..+5.6), larger than material.**
> 🏆 **Progress check vs the pre-Claude engine** (`selfplay/old CE`, matched LIGHTNING, .so-vs-.so via the
> new `oldce` runner sub): **+92 −6 =12 of 110, 89.1%, Elo +364.8 ±76.3.**
> ▶️ Owner logic pass queued — see [[collapse-leverage-map-refreshed-400g-2026-08-09]] for the FENs.
>
> ## 🎯🎯🎯 THE SPRT CANDIDATE — RUN THIS FIRST
> `PV_BOOST_MAG=0 ENABLE_ROOT_LMR=1 ENABLE_QCUT=1 SEE_PRUNE_CAPTURES=1`
>
> | | baseline | combined |
> |---|---|---|
> | balanced tactical | 495 | 488 (−7) |
> | balanced positional | 3372 | **3324 (−48, inside ±150)** |
> | wac nodes | 35,426,396 | **−9.2%** |
> | eval footprint | — | **6.8% = 74% of the 9.2% control** |
>
> ★★★ **The halves have INVERSE node profiles and pay for each other** — `PV_BOOST_MAG=0` alone is
> −33 / **+13.3% nodes**; the 3 search arms alone are −42 / **−13.5% nodes**; together −48 / −9.2%.
> **Better than additive on BOTH axes.** The eval change stops paying for itself in depth.
> ✅ Search arms carry measured game Elo (+9.5 / +6.8 / +4.7, each n.s.). ⚠️ That ~+21 is NOT additional
> to the node saving — the saving is the MECHANISM; do not double-count.
> ⚠️ **RISK:** `PV_BOOST_MAG=0` is a UNIFORM SHRINK, the −85.6 Elo signature, was a corpus-fit winner
> pinned at a grid edge, and has **NO PLATEAU** (sweep: 0 → −33 · 5000 → −177 · 10000 → 0; the term is
> LINEAR in mag so the non-monotonicity is search chaos). Justified only by SF11 having no analogue and
> `piece_value_boost` = +0.46 self-favouring bias in real collapses. ☠️ Half the bundle is search (2-for-15).
> ☠️ The TARGETED alternative (`MOD_PVBOOST_COMP/MOB` — dampers whose comments name our exact failure)
> **saturates at 2.0% footprint, identical at strength 128 and 512** ⇒ right shape, CANNOT REACH.
> ▶️ Keep `QUEEN_PST_FILE_SYM_MODE` gated through the SPRT so the baseline stays fixed.
> See [[pvboost-plus-search-is-the-sprt-candidate]].
>
> ## 🚨 READ THIS BLOCK FIRST
>
> **Nothing was shipped this session.** Defaults are unchanged except `ENABLE_CAPG_LAZY_PIN=true`, which
> is byte-identical. The register fingerprint still stands:
>
>     250 / 35,426,396 / EBF 3.800 / STS 1631   ·   MIRROR 245 / 35,727,805 / STS 1741
>     BALANCED  tactical 495   positional 3372
>
> ### ☠️ RESOLVED 08-08 — THE CAPGAIN SELECTION LANE IS CLOSED
> Both open questions were answered and the answer is **no candidate**. Footprints vs the
> `ENABLE_THREATS=0` control (9.2% = the +45 Elo change), balanced four-suite:
>
> | arm | balanced tactical | balanced positional | footprint | nodes |
> |---|---|---|---|---|
> | baseline | 495 | 3372 | — | — |
> | `NET_SELECT + PROMO_CREDIT` | 494 (−1) | 3307 (**−65**) | 3.8% | −3.1% |
> | `+ LVA_STATIC` | — | 3219 (**−153**) | ~4.6% | — |
> | `+ LVA_STATIC + FILE_INVARIANT_TIEBREAK` | 488 (−7) | 3134 (**−238**) | 5.5% | −3.3% |
>
> ★★★ **Each added knob costs ~−85 balanced STS and buys ~+0.85 points of footprint — near-linear.**
> In this family footprint is PURCHASED WITH ACCURACY, not found. −238 is past the ~150 band, so unlike
> most numbers here it IS readable: the big-footprint arms are genuinely worse.
> 🐛 **Both symmetry knobs close the orig/mirror gap by dragging the MIRROR branch DOWN** (3-knob
> +6/−159; 4-knob −46/−192; the 4-knob arm cut the baseline's 110-point gap to −36). A symmetry fix can
> be a real defect repair AND a strength regression at once — check WHICH DIRECTION the gap closes.
> ↩️ `FILE_INVARIANT_TIEBREAK`'s −116 was stale in the OPPOSITE direction to what was assumed: the real
> cost is larger, not obsolete.
>
> ⇒ **Best arm is the 2-knob bundle: −65 (unresolvable), 3.8% footprint = 41% of control. Do NOT spend
> games on it.** Only the **−3.1% node saving** is bankable, and node counts are exact where NPS lies.
> ▶️ NEXT: **stop testing small knobs — find LARGE-footprint content.** Leads: the `pt_queens` 5 mp
> defect · collapse hunting (large-footprint by construction) · the zero-compute re-triage of the 13
> dead search arms against the now-known noise floor. See
> [[capgain-symmetry-fixes-symmetrize-toward-the-worse-branch]].
> ✅ The three missing `CAPG_*` `[toggles]` entries are now in the dump.
>
> ### 🐛 ELEVENTH SYMMETRY DEFECT — FOUND, FIXED, FREE, GATED (08-08)
> **The queen PST's files are not mirrors in two cells**, 5 mp each, in BOTH colour tables
> (`whitePlacementLayerBase`/`blackPlacementLayerBase`, index 4): file A rank 7 = 15 vs H = 20, and
> file B rank 6 = 25 vs G = 20. Rows are FILES. Every other table in the array is file-symmetric.
> ⇒ **13 of the 14 file-mirror violations: 2.2% → 0.2%** (survivor = the 194 mp capgain tie-break).
>
> | | baseline | `QUEEN_PST_FILE_SYM_MODE=3` |
> |---|---|---|
> | balanced tactical | 495 | 493 (−2) |
> | balanced positional | 3372 | 3347 (**−25**, inside ±150) |
> | wac nodes | 35,426,396 | 34,149,917 (−3.6%) |
> | file-mirror violations | 14 | **1** |
> | colour-swap violations | 11 | 11 (untouched) |
>
> Mode 0 byte-identical. Footprint **0.0%**, and the control `SCALE_PLACE_QUEEN=0` (whole table zeroed)
> is only **0.8%** ⇒ **the entire queen-PST lane is capped below the resolvable floor.** Yet mode 3 swings
> STS orig +96, so its effect runs through pruning/reduction downstream, not static ordering.
>
> ### ☠️★★★ THE TWO FREE FIXES CANNOT SHIP TOGETHER
> | arm | balanced positional | Δ |
> |---|---|---|
> | capgain 2-knob alone | 3307 | −65 |
> | queen PST mode 3 alone | 3347 | −25 |
> | **both** | **3169** | **−203** (additive predicted −90) |
>
> **Interaction −113**, past the ±150 band ⇒ readable and bad. Leave-one-out on ONE build reproduced
> capgain's 3307 EXACTLY, so this is interaction, not drift. Queen PST moves STS orig **+96 alone but
> exactly 0** on top of capgain. ⇒ **SHIP AT MOST ONE.** ▶️ Recommend **queen PST mode 3 alone** (better
> balanced STS, better nodes, true invariant violation, zero footprint risk — but all inside noise, so
> the tiebreak is defect quality, not the gap); keep capgain gated.
> ★★★ Third bundling surprise in one session: pair RESCUED, 4-knob COMPOUNDED, this COMPOUNDED.
> **Measure the bundle you intend to ship.**
>
> ▶️ **SHIP CANDIDATE awaiting an owner call** — same basis as `KING_ZONE_SYM_MODE=2`. ⚠️ All three repair directions land inside the band (−25 / −77 / −81), so the
> bench CANNOT choose the direction; mode 3 is chosen on STRUCTURE (both cells → 20, table stays
> monotone). ⚠️ And today's capgain result is the standing warning that a correct symmetry fix can still
> cost strength. See [[eleventh-symmetry-defect-queen-pst-file-cells]].
>
> ★★★ **Method:** three 5-valued KNOBS were refuted before the answer turned out to be a 5-valued
> TABLE. When knob ablation keeps refuting, **ablate the TABLE a scale knob scales** —
> `SCALE_PLACE_QUEEN=0` localised it in one run.
> ☠️🚨 **New wiring hazard:** `rebuild_scaled_placement/pawn_tables/ks_tables` run at
> `search_engine.cpp` ~1525-1527, ~400 lines BEFORE most `env_int` registrations. A knob those rebuilds
> bake in must be registered ABOVE them or it registers, dumps, and does nothing — the first run of this
> fix was byte-identical for exactly that reason.

---

## 1. Overnight result — EG clamps PARKED
SPRT `egclamp_h500`: **+455 −450 =295 of 1200, elo ~+1.4 ± 23.1, INCONCLUSIVE**. Per the agreed tree:
park, do not extend, do not ship. `EG_CLAMP_*` stay 0.
★★★ **The bench said −59; games said +1.4 ± 23.1.** With the near-inert arm's −151, two independent
methods now agree: **|balanced STS| < ~150 is UNRESOLVABLE.**
☠️ **NEVER-PEEK PROVEN:** at 91 games this run read **elo ~ −31**. See
[[balanced-sts-noise-floor-confirmed-by-games]].

## 2. The new instrument — `diagnostics/_d1_move_attribution.py`
Depth-1 move choice vs the three-way reference, in **win%**, with **no filter**. SF is an ORACLE OVER
MOVES only; our breakdown is used solely to attribute our OWN choice. Never places an SF term beside one
of ours — term names are not 1-to-1 (our king-zone attacker−defender was once TRIPLE-counted).
Each position carries a signed weight `our_winpct_err − sf11_winpct_err`; tactical positions
self-suppress because both errors are large and the DIFFERENCE is small, so nothing needs filtering.

**400 corpus positions:**

| | count |
|---|---|
| both statics found SF18's move | 45 (11.3%) |
| **NEITHER did — search's job** | **294 (73.5%)** |
| SF11 found it, we did not (fixable) | 26 (6.5%), win% weight 158.7 |
| WE found it, SF11 did not (guard) | 35 (8.8%), win% weight 20.2 |

★★ **Three quarters of positions are search's job** ⇒ only ~27% are statically decidable at all, a hard
ceiling on what ANY eval term can earn. ★ **We beat SF11 on move choice more often than it beats us
(35 vs 26)** — but the weights invert it: our failures are severe (158.7), our wins marginal (20.2).
★★★ Of the 26 fixable cases, **14 (54%) have the truth move at rank 2** and 81% at rank ≤3 ⇒ **not
missing detectors — near-misses.** And search recovers rank-2 moves, which is a candidate explanation
for why so many correct eval fixes measure ~0 Elo.
🐛 **Known flaw, unfixed:** the guard table is structurally empty — for a guard case our move IS the
truth move, so attributing "our pick vs truth pick" compares a position to itself. It should attribute
against SF11's pick instead. Also the blame sums are outlier-dominated; use median / frequency-as-top-
offender instead.

## 3. ☠️ THE CAPGAIN ROOT CAUSE — fictional compensation
Surfaced from `6r1/nk1q4/1p6/3p1n2/p2Pp3/4P2P/PB1Q1Pp1/2RK3R w`: we prefer `c1c8` (hangs a rook to THREE
attackers) over `h1g1` (saves one). SF18 d16: **−10.78 vs −4.34**.

The trace shows capgain books Black's `g2xh1` (rook, 5000) AND White's `c8xg8` (rook, 5000) and **nets
them to zero** — so hanging a rook reads as free. The compensation is fictional: Black moves first and
the c8 rook is attacked 3× with no defender.
✅ **Root cause verified by independent code read:** both Black captures are worth 5000 (a tie);
`capg_less` sorts the cheaper attacker LATER and the consumer pops `back()`, so the **pawn** capture is
chosen, leaving White's rook alive. Picking `Rxc8` deletes White's reply entirely.

🎯 **Minimal repro, 6 pieces:** `2R3r1/8/8/k7/8/8/6p1/3K3R b - - 0 1`

| | `capture_gains` | ours | SF18 d10 |
|---|---|---|---|
| both Black captures available | **0** | **+6.77** | **−9.62** |
| only the rook capture | −3.50 | +3.39 | 0.00 |
| only the pawn capture | −2.00 | −9.22 | −9.83 |

Each capture alone is booked correctly; together they cancel — a **16-pawn** error.
⚠️ **`ENABLE_CAPG_TEMPO` cannot catch it**: it is on the evasion path inside a branch requiring
`opp.value_gained > cur.value_gained` — STRICTLY greater. Equal-value trades (which is what creates the
tie) can never reach it. Design gap, not a wiring bug.

### A second, independent defect: promotion uncredited
`promo_gxR` and `plain_gxR` (identical geometry one rank apart) both booked `capture_gains = −2.00` —
winning a rook AND queening scored the same as winning a rook.
⚠️ **Why an unguarded fix would be catastrophic:** `apply_basic_capture` updates OCCUPANCY ONLY — no
piece-type overlay — so a recapture on the promotion square is priced as a PAWN. Crediting +9000 with the
reply valued at 1000 is far worse than today. Hence the guard: credit only when the square is undefended.

## 4. What was BUILT (all gated, nothing shipped)

| knob | default | verified |
|---|---|---|
| `ENABLE_CAPG_LAZY_PIN` | **true** | byte-identical; defers 2 `slider_blockers` calls to first use. **No measurable speed** — keep as an additive. |
| `ENABLE_CAPG_NET_SELECT` | false | repro fixed (picks `Rxc8`); 3.0% move-change; alone: −100 STS, +0.72% nodes, −7.5% NPS |
| `ENABLE_CAPG_PROMO_CREDIT` | false | repro fixed (`vg` 5000 → 14000); 0.8% move-change; ~free |

**The rule:** `net(c) = vg(c) − best_opponent_reply_that_survives(c)`; a reply dies iff
`reply.from == c.to`. 1-ply, computed once against the initial stacks.
★ **Bundling mattered.** `NET_SELECT` alone was −100 STS / +0.72% nodes / −7.5% NPS. With
`PROMO_CREDIT` added: −65 STS / **−3.1% nodes** / NPS neutral. Neither component predicted that.

## 5. 🚨 METHOD FINDINGS — these cost real time, do not relearn them
- ☠️☠️ **FIVE code-reading hypotheses were refuted by ablation this session**: `BISHOP_MOB_SECONDARY`,
  `CHEAP_QUEEN_MOB_MG`, the attacking layer, the evasion-polarity fix, and "capgain is silent". ★ The ONE
  that survived (LVA tie-break) was the one isolated with a REPRO FIRST, then verified in code.
- 🚨 **A PRECONDITION BOUNDS WHERE A RULE *CAN* ACT, NOT WHERE IT DOES.** The net rule's preconditions
  hold in **50.8%** of positions; realised move-change is **3.0%** — 17× lower. Do not quote a
  precondition as a footprint.
- 🚨 **A TIE-ONLY KNOB UNDER-BOUNDS A GENERAL RULE.** `ENABLE_CAPG_INVARIANT_ORDER` moves 1.7% (exact
  ties only); the net rule reorders untied captures too. I quoted 1.7% as the addressable set and was
  wrong by an order of magnitude in the other direction.
- 📏 **THE NPS RULER CANNOT RESOLVE CAPGAIN MICRO-OPTIMISATIONS.** Three separate changes that provably
  do less work — early-exit gate (abandoned at a 10.2% fire rate), lazy pin, double-sort removal — all
  measured FLAT. ⇒ "make capgain cheaper" is not reachable by increments; only structural change shows.
- ⚠️ `probe_fens.py` uses argparse and **cannot take `KEY=VAL` knobs**; use `_capg_trace_pair.py` or
  `_d1_move_attribution.py`, which set env from argv.
- ⚠️ A grep filter of `=[0-9]+ ` eats CAPG trace lines (`from=41 `). Filter on `CAPG|===`.

## 6. ▶️ NEXT, in order
1. **Bundle move-change footprint** — `_move_change_arms.py ARM=ENABLE_CAPG_NET_SELECT=1,ENABLE_CAPG_PROMO_CREDIT=1`
   against the `ENABLE_THREATS=0` control (9.2%). Decides games-testability.
2. **The 5-item bundle** — add `CAPG_FILE_INVARIANT_TIEBREAK` + `CAPG_LVA_STATIC`. ⚠️ Their −116 was
   measured against the OLD selection rule and is stale: net-select reorders BEFORE any tie-break is
   consulted, so it changes what those knobs even do.
3. **If shipped:** flip defaults, rebuild, re-record ALL FOUR suites, `refresh_bank_ours.py` to relabel
   the corpus (never fit stale), update the register.
4. **Free side-quests, no compute:** the search re-triage (13 arms, many killed on sub-noise bench
   readings that we now know were noise) and the `ENABLE_CAPG_TEMPO` four-suite bench.

## 7. ⏸️ Still open, owner decisions
- Ship the capgain symmetry pair on the invariant, or fold it into the 5-item bundle (recommended).
- Collapse hunting: `vs_sf` hardcodes its tag `vssf_<elo>` ⇒ a re-run **OVERWRITES** the existing corpus
  and PGNs. Preserve first, or run at a different elo, or use `gauntlet` with a fresh tag (but it passes
  no `--win-threshold`, so likely no collapse dump).
- The uniform **5 mp file-mirror class** is localised to `pt_queens` in the midgame queen evaluator,
  7-piece repro `2b2k1r/Q5b1/2q5/8/8/8/8/6K1 b`, all 5 non-king pieces load-bearing. Three suspects
  ablated dead; next candidate is the xray `values[type] >> 6` (unscaled — it survives
  `SCALE_ATTACK_LAYER=0`, which drops the term but leaves the gap at EXACTLY 5).
