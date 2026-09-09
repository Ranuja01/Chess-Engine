# EVAL TERM REVIEW TABLE — 2026-09-08

Built to decide what to retire, re-shape, consolidate or keep. Every claim below carries a `cpp_bitboard.cpp`
line unless marked otherwise (`h:` = `search_engine.h`). Read-only audit; nothing here is a recommendation.

## 0. HUB FACTS
- `phase_score = 128*(24-phase)/24` (`:7359-7364`); midgame branch `<= 64` (`:7384-7840`), endgame `> 64`
  (`:7842-8128`), common tail (`:8130-8593`).
- Blend zone 41..64 applies to **pawns** (`:7408`), **rooks** (`:7501`), **kings** (`:7598`, again `:7955`).
  **Queens never blend** — their blend is commented out (`:7556-7582`), so queens STEP at 64.
- 🐛 `isNearGameEnd` is hard-initialised `true` (`:7338`) ⇒ `advanced_endgame_eval` **replaces `total` in
  every endgame**, not just `>96`. Deliberate (comment `:7330-7337`) but a live design decision.
- `total` is Black-positive; White terms subtract.

## 1. ☠️ DUPLICATION CLUSTERS — with REAL / APPARENT verdicts
Judged against DESIGN INTENT, not merely shared inputs.

> ☠️☠️ **CLUSTERS 1 AND 2 ARE SUPERSEDED (2026-09-09) — MEASURED, NOT ARGUED.**
> `_ks_channel_collinearity.py MODE=heat` (744 positions): mean|contribution| heat **433.3mp** ·
> central **215.9mp** · OvD **22.7mp**; r(heat,central) **+0.389** · r(heat,OvD) **+0.420** ·
> r(central,OvD) **+0.131** — **all under the 0.5 duplication threshold ⇒ the channels carry DISTINCT
> information ⇒ de-dup would SHED SIGNAL.**
> ⇒ **`central` is NOT a "pure re-sum"**: removing it costs up to −1.6pp on v2, and it is only r=0.39 with
> the heat. ⇒ **OvD is free to remove because it is TINY, not because it is redundant.**
> ⇒ The whole "stop paying for the same cells three times" plan is **REFUTED**.
> 📄 [[heat-map-is-load-bearing-and-its-channels-are-NOT-collinear]] · `SESSION-HANDOFF-2026-09-09.md`
> ✅ The rest of this document — the line-numbered term inventory, the heat consumer list, the dead code,
> the never-fitted literals — **stands**.

| # | cluster | verdict | evidence |
|---|---|---|---|
| 1 | **heat → `central`** vs the per-piece heat/PST reads | **REAL — a pure re-sum** | `central_score` is accumulated at 22 sites (`:947 … :2997`) from the SAME PST cells and heat cells already added to each piece's `total`; only the saturating knee differs. The code says so: `:7164-7170`. |
| 2 | **heat → `ovd_imbalance`** vs the per-piece heat reads | REAL at the feeder, APPARENT in shape | same cells, but netted per side and saturated at 300 (`:7144-7162`). |
| 3 | **pawn shelter ×3** | **REAL** (same signal, same horizon) | `evaluate_kings_midgame` +185/+75 and ×4/×2 base (`:3129-3147`) · `KS_SHIELD=2`×shield pawns (`:5632`) · `setAttackingLayer` pawnShield `−kinc>>1` (`:9524`, `:9549`). `h:1350-1363` acknowledges it; `ENABLE_KS_V2` "re-homes" it and is default OFF. |
| 4 | **bishop scope ×3** | **REAL** | per-square 20 (`:2020`) · `CHEAP_BISHOP_MOB` 6/sq (`:1802`) · `get_latent_bishop_activity_score` (`:1606-1607`). Plus `CHEAP_BISHOP_KING=12` overlapping the heat king-ring AND `KS_ATTACK_COUNT`. |
| 5 | **material axis ×4** | REAL on the axis, different shapes | `values` inside every piece loop · Kaufman (`:8171-8203`) · `piece_value_boost` (`:8386-8416`) · `EG_EXIST_*` (`:3488` etc.). ⚠️ `piece_value_boost` and `MOD_KS_REALIZ` read material **after** `approximate_capture_gains` mutates it (`:7704-7707`, `PIECEVAL_RECOMPUTE_LATE=false`). |
| 6 | **rook file openness ×2** | REAL partial | open-file increment (`:2427`) and the file-direction mobility popcount (`:2556-2561`). |
| 7 | attack-set cluster: capg ↔ threats ↔ passer path ↔ KS | mixed | capg↔threats mitigated by `THREATS_STANDING_ONLY` (`h:826-829`); passer path REAL partial (`:6381-6387` vs `:6558-6562`); capg/threats ↔ KS **APPARENT** (different targets). |

## 2. ⭐ THE OvD FINDING — the concept is intact; the implementation is not
**Owner's intent:** `ovd_imbalance` = LONG-TERM positional pressure (king opened early, infiltration
later); `evaluate_king_safety` = SHORT/MEDIUM immediate danger traded against material. Two horizons.
☠️ **The code does not compute that.** The O/D accumulators are built exclusively from heat cells read at
each piece's **current exact attack squares** (writers `:944-945`, `:1029-1030`, `:1380-1381`, `:1418-1419`,
`:2334-2335`, `:2464-2465`, `:2826-2827`, `:3122`, `:3131`, `:3134`, `:3137`, `:3143`, `:3146`). So OvD
moves move-to-move exactly as KS does. Its breadth comes from the heat table's **distance-to-king
weighting**, the **×5 open-square multiplier** (`ATTACK_OPEN_MULT`, `:9521`), board-wide non-king heat, and
**netting against own-king defence** — not from a slower time constant.
✅ KS genuinely uses a different feeder: `king_safety_danger` (`:5339-5892`) reads `attack_bitmasks` and has
**zero heat reads**.
▶️ ⇒ the opportunity is to make OvD *actually* long-term (structural king exposure, pawn-cover holes,
infiltration squares independent of where pieces stand this move). That realises the intent AND removes it
from the heat cluster — strictly better than retiring it.

## 3. 🔥 THE HEAT TABLE — the widest-read feeder
Built `setAttackingLayer` `:9383-9650`. Two 8×8 planes (one per king). Base tables literal: MG peak 65
(`:9437-9458`), EG peak 35 (`:9414-9435`). King fan-out `kinc = increment × KS_ZONE_ATTACK_PCT(50)/100`
(`:9483`); open squares `+kinc×ATTACK_OPEN_MULT(5)` (`:9521`); pawn-shield squares `−kinc>>1` (`:9524`).
⚠️ The builder's own comment (`:9479-9482`) records that the king-zone slice **"double-counts (it also feeds
OvD)"** — which is why the percentage was cut to 50.
**In the MIDGAME the heat reaches `total` through three channels**: per-piece credit (every piece, weighted
by type), `central` (re-sum), `ovd` (per-side re-sum). **In the ENDGAME evaluators there are NO O/D or
central writes** — placement credit only.
✅ NOT consumers: `king_safety_danger`, `threats_by`, `evaluate_passers`, `approximate_capture_gains`.

## 4. ☠️ DEAD OR UNREACHABLE AT DEFAULTS (deciding line)
- **`priced_passer`** — declared `:546`, cleared `:7307`, written `:6714`, **ZERO reads**. Its comment
  (`:6654`) calling it "the single source of truth capgains reads" is **false**.
- **The entire pressure/support machinery** — `pressure_*`, `support_*`, `num_attackers/supporters`
  (`:462-468`), `update_pressure_and_support_tables` (`:4794`), `handle_batteries` (`:4815`),
  `adjust_for_pins` (`:4886`): **every writer call site is commented out** (`:1015`, `:1404`, `:2047`,
  `:2450`, `:2523`, `:2905`, `:3105`, `:3310`, `:4699`, `:7633`, `:7986`). The readers return zeros.
- **Rook PST plane** — never indexed; `rebuild_scaled_placement` hard-codes its scale 100 as "dead code"
  (`:344-347`). Only `cheap_eval` (`:7223`) reads all six planes, and it is off the default path.
- **SIMD `pawns_simd_initializer`** (`:1203-1352`) — entirely inside a `/* */` block; call site `:7446` also
  commented.
- `get_latent_threat_score` (`:6072-6337`) — skipped under `ENABLE_KS_REPLACE_LT=true` (`:7745`).
- All `!ENABLE_PASSER_V3` per-piece passer bonuses (~20 sites) — dead under V3.
- `pair_bonus` (`:8148`) — skipped when Kaufman is on. Endgame KS arm (`:8106`). Queens' phase blend
  (`:7556`). `g_passer_*_deferred` (written, never read).
- **~28 KS sub-knobs at 0/false**: `KS_ZONE2`, `KS_BATTERY`, `KS_FLANK_MODE`, `KS_PIN_MODE`, `KS_ADJACENCY`,
  `KS_SQPRUNE_MODE`, `KS_OVERLOAD`, `KS_DEFENDER`, `KS_ATT_PRODUCT`, `KS_ACCUM_MODE`, `KS_ONSET_MODE`,
  `ENABLE_KS_V2`, `KS_CONSOLIDATE`, `KS_COORD_GATE_MODE`, … (h:1356-1605).
- `ks_safety_table`'s quadratic band — `KS_FLOOR 13 > KS_KNEE 12` (`:435`).

## 5. ☠️★★★★ NEVER-FITTED LITERALS — the un-tunable surface
**The heat read weights are the biggest one.** Every `>>1`, `>>2`, `>>3`, `/3` at ~96 per-piece sites, every
O/D shift, and the `central` multipliers (×2, ×3/2, ×1, ×½ by piece type at `:947`, `:1032`, `:1421`,
`:2007`, `:2467`, `:2869`) are **bare numbers, not Config knobs**. ⇒ **the widest-influence surface in the
eval cannot be reached by any tuner we build.**
Also literal: the placement/heat/threat/Kaufman **tables** themselves (`:205-339`, `:9414-9458`,
`:5981-5984`, `:8172-8187`); doubled-pawn 125/150; knight 20/5; bishop 20/5; rook mobility cap 225 and the
per-rook ±5600 clamp (`:7530`); all x-ray shifts and `values>>6/7/8`; king shelter 185/75 and its ×4/×2;
central phase steps 20/31/45 and caps 400/350/300 (`:7177-7180`); endgame mop-up 2000/200/45 (`:5099-5149`);
the `piece_value_boost` formula itself (`:8388`).

## 6. 🎨 OURS, WITH NO SF ANALOGUE (preserve through any consolidation)
The heat table and its channels · **`ovd_imbalance`** · `approximate_capture_gains` (SF does no static
exchange resolution in eval) · latent bishop/rook activity · cheap bishop colour-complex · multiplicative
passer **R** with blockade docking · `piece_value_boost` · the endgame mop-up · `EG_EXIST_*` ·
`PASSER_ENEMY_CREDIT` · pawn wall/chain/latent-support.
⚠️ `MOD_KS_REALIZ` (material-as-realizability conditioner) shipped inside the **+36.7 Elo** bundle — the
strongest evidence that our own conditioning ideas can win.

## 6b. 🐛 TWO REAL BUGS FOUND AND GATED (2026-09-08)
Both behind default-off knobs; fingerprint re-verified **250 / 35,310,778 / 3.784** with them off.
1. **`ENABLE_ROOK_LATENT_RAY_FIX`** — `get_latent_rook_activity_score`'s second-order scan used
   `BB_DIAG_ATTACKS` (DIAGONAL rays, copy-pasted from the bishop version) for a ROOK (`:2285`), while the
   same function computes rook rays correctly at `:2237/:2242`. ⇒ our latent rook activity has been
   measuring bishop moves.
2. **`ENABLE_CAPG_ROOK_SQVAL`** — the rook MIDGAME loop never writes `square_values` (`:7534`, commented
   out) while the rook ENDGAME loop does (`:7921`) and every other piece writes its own. After the per-eval
   `fill(0)` a midgame rook reads as **value 0**, so `get_least_valuable_attacker` picks it as the CHEAPEST
   attacker in the capture-gains gather (`:9031` — live, `ENABLE_CAPG_LVA_STATIC=false`). The code's own
   comment at `:9020-9030` flags this family as a known unfixed asymmetry.
**Measured (primary set, null band 49.9-50.4):** ray fix **48.7%** (n=1410, SE 1.33 ⇒ ~1σ);
sqval fix **49.2%** (n=2056, SE 1.10 ⇒ ~1σ). **Both UNRESOLVABLE, both leaning slightly negative.**
⚠️ ★ **A CORRECTNESS FIX CAN READ NEGATIVE IN A TUNED SYSTEM** — the latent rook's `10`/`5` were hand-set
against the wrong ray, and the gather's ordering has been wrong throughout; the constants absorbed the
defects. Unlike the 7-fix symmetry bundle (harmful in expectation, with a mirrored control), these have
neither property.
▶️ **RECOMMENDATION: keep gated; ship WITH a re-fit of the affected constants, not in isolation.**
⚠️ Also note the ray-fix arm's MEAN reads −0.1353 ("better") while its win% reads 48.7% ("worse") — a clean
live demonstration of why the mean is the misleading statistic here.

## 6c. ✅ VERIFIED CODE FACTS (2026-09-08 targeted read) — answers to specific questions, with lines

**Latent activity is NOT a duplicate of inline mobility (the suspicion was reasonable and wrong).**
`PROF_BLOCK` is a **no-op in production** (`cpp_bitboard.h:605-607`; `-DEVAL_PROFILE` only under
`PROFILE_EVAL=1`), so `get_latent_bishop_activity_score` at `:2075` **is** live and ungated. But it iterates
a **disjoint square set** — squares reachable only after removing own blockers from the rays, minus the real
attacks (`:1592`) — at weight `BISHOP_MOB_PAWN_ATTACK=15` vs the inline `20`. ⚠️ **The real overlap is with
the X-RAY block** (`:2049-2072`): squares behind an own first blocker are paid twice, `>>2` there and `/3` in
latent. Same for rooks (`:2526-2527` vs the latent function).

**Mobility gating, per piece (several assumptions here were wrong):**
- Knights: the expensive nested scan at `:1430-1462` **IS live** (`ENABLE_CHEAP_KNIGHT_MOBILITY=false`).
- Queens: the nested scan at `:2881-2896` **IS live** (`ENABLE_CHEAP_QUEEN_MOBILITY=false`).
- Rooks: the inline nested scan `:2479-2514` is **DEAD** (`ENABLE_CHEAP_ROOK_MOBILITY=true`); what runs is
  the cheap popcount form at `:2556-2559`, capped 225.
- Bishops: cheap colour-complex path live (`ENABLE_CHEAP_BISHOP_COMPLEX=true`).

**⚡ `eval_by_mode` — modes 1 and 2 are UNREACHABLE at defaults ⇒ there is NO light eval in production.**
`FUTILITY_EVAL_MODE`, `QSTANDPAT_EVAL_MODE`, `RFP_EVAL_MODE` are all **0** (`search_engine.h:1938/1939/1956`);
`g_eval_light` is assigned only at `search_engine.cpp:1196-1198` and so is never true. `cheap_eval`'s other
callers are `static_eval_for_improving` (needs `ENABLE_IMPROVING`, false) and `log_prune_fire`
(`ENABLE_PRUNE_LOG`, false). ▶️ **A real, unexploited NPS lever — the giants all have some form of it.**

**⚡ Material popcounts are computed THREE ways per midgame eval:** (1) the piece loops accumulate
`whitePieceVal`/`blackPieceVal` as a side effect; (2) `:7659-7670` rebuilds both with **12 popcounts**
(`ENABLE_MATERIAL_COUNT_FIX=true`, correcting the blend-path double count); (3) `:8189-8195` takes **12 more**
for Kaufman's census — **ten of which are the same quantities**, with bishops counted twice within Kaufman
itself. Plus `king_safety_danger` recomputes identical attacker popcounts at `:5551-5554` **and again** at
`:5573-5576`. ▶️ Compute once at entry, reuse where the state is unmutated.

**⚡ Four dead locals in the KS zone scan, accumulated every square for a zero multiplier:** `overload_sum`
(`:5487-5488`, consumed at `:5561` × `KS_OVERLOAD=0`), `breakthrough_sq` (diagnostic-only, `:5448`),
`weak_val_sum` (`:5507`, needs `KS_WEAK_VAL_MODE≠0`), and the `defenders_sq` popcount at `:5562`
(× `KS_DEFENDER=0`). The scan itself and its live accumulators are sound.

**`square_values` is NOT dead** — `get_least_valuable_attacker` is off the SEE path
(`ENABLE_SEE_INCREMENTAL`/`ENABLE_SEE_FIX` both true ⇒ the `_static` variants) but **is** on the live
capture-gains gather (`:9031`, `ENABLE_CAPG_LVA_STATIC=false`). That is where the midgame-rook-priced-at-0
defect bites. `approximate_capture_gains1` (reads it at `:8821/:8829`) is dead — referenced only from a
commented `cout`.

**`priced_passer` — intended purpose confirmed.** It was to be the per-passer value capgains consumes
instead of `pawn_rank_bonuses`; blocked by **ordering** (capgains `:7678` runs before `evaluate_passers`
`:7741`), and `passer-doubled-hce-comparison-2026-07-22.md:180` records that obstacle with two proposed
fixes, neither done.

## 7. UNRESOLVED (where the answer lives)
`getPPIncrement` internals (`:9730-9901`) · `approximate_capture_gains` body (`:9010-9323`) and whether
`square_values`/`get_least_valuable_attacker` is on the live SEE path under `ENABLE_SEE_FIX` (`h:1689`) ·
`king_safety_danger` zone-scan head (`:5429-5478`) · `CHEAP_QUEEN_MOB_EG` gate (`:4485-4500`) ·
`eval_by_mode` default (comment-based, `search_engine.cpp:1182-1202`) · the PST plane VALUES (`:205-339`).
