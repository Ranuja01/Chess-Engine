# Eval architecture & degeneracy map

**Canonical doc** — the measured system as read from `cpp_bitboard.cpp` / `search_engine.h` on 2026-08-10
(branch `NN-ENgine`). Scope: `placement_and_piece_eval` (cpp_bitboard.cpp:6824) and everything it feeds.
Every claim cites file:line; every redundancy claim states what would falsify it. This is a READING of the
code — no measurement in this doc is new; where a claim needs an ablation to be trusted, that is said
explicitly (per the "ablate, don't read, to explain" rule this doc only *finds*, it does not *explain
measurements*).

Convention reminders: `total` is **Black-positive**. `phase_score` 0=open..128=deep endgame
(`phase-score-convention`). Piece values `values = {0, P1000, N3250, B3450, R5000, Q10000, K12000}`
(cpp_bitboard.h:141).

---

## 0. Execution skeleton at defaults

Midgame branch (`phase_score <= 64`, cpp_bitboard.cpp:6970):

1. `setAttackingLayer(5, false)` — rebuild the attack table around both kings (6975, impl 8909).
2. Per-piece loops → `total` (pawns 6987, knights 7037, bishops 7056, rooks 7075 with mg/eg blend,
   queens 7130, kings 7176 with blend). Side effects: `whitePieceVal/blackPieceVal`, O/D scores,
   `central_score`, `attack_bitmasks`, `square_values`, `pawn_rank_bonuses`, passer bitboards.
3. `ENABLE_MATERIAL_COUNT_FIX` (default **true**, search_engine.h:1731) — recompute both PieceVals from
   popcounts (7244-7257), because the phase-blend calls BOTH evaluator variants per square and the
   `+= values[]` side effect double-counts (comment 7239-7243).
4. `approximate_capture_gains` (7262-7274, impl 8504) → `total += cg` (scaled by
   `capg_conditioned_scale`, LIVE: `ENABLE_CAPG_COND=true`, 10%→100% ramp on tension 0→3,
   search_engine.h:701-705). **Side effect: mutates `whitePieceVal/blackPieceVal`** (8815/8836).
   `PIECEVAL_RECOMPUTE_LATE` default **false** (search_engine.h:1741) ⇒ the mutation is what every later
   consumer sees.
5. `threats_term_scaled` (7299-7301, LIVE: `ENABLE_THREATS=true`, `SCALE_THREATS=75`,
   `THREAT_PER_TARGET_CAP=800`).
6. Passer terms (7304-7313): `boost_pieces_for_supporting_passed_pawns` (LIVE) + `evaluate_passers`
   (LIVE: `ENABLE_PASSER_V3=true`).
7. Latent threat (7316-7328) — **DEAD**: `ENABLE_KS_REPLACE_LT=true` skips it.
8. `evaluate_king_safety` (7338-7343) — LIVE: `KING_SAFETY_MAG=3000`, `MOD_KS_REALIZ=128`.
9. `central_add` from `central_score`, phase-stepped clamp ±300..400 (7354-7364, `SCALE_CENTRAL=100`).
10. PV-boost trigger latch on `total` (7367-7371, `PV_BOOST_TRIGGER=1500`).
11. O/D imbalance, `IMBALANCE_SCALE=3` (7386-7400).

Endgame branch (7403-7661): same shape minus latent/KS/central/imbalance, plus
`advanced_endgame_eval` (7656, impl 5070) which **rewrites** `total` (mate-drive + edge-drive when
|total|>2000; its own passer block is skipped under V3, guard 5164). The `isNearGameEnd` gate carries the
documented UB-history: it is deliberately pinned `true` so AE fires in ALL endgames (6916-6924).

Common tail (both branches): Kaufman imbalance (LIVE: `ENABLE_KAUFMAN_IMBALANCE=true`, 7705-7737 —
flat pair bonuses 7682-7698 therefore DEAD), then ~10 gated-off conditioners (7743-8127), then
`piece_value_boost` (7920-7950, LIVE) — which despite its position in this list runs BEFORE
pawn-majority/pawn-struct/outpost/space (all gated off) — and the diagnostic breakdown publish (8130).

---

## A. Accumulator map

### A1. `whitePieceVal` / `blackPieceVal` — the "material" that is three different numbers in one eval

Globals (cpp_bitboard.cpp:82). Timeline within one eval at defaults:

| phase | value held | writer |
|---|---|---|
| after piece loops | **corrupted** (blend double-count) | `+= values[]` in every evaluator, both variants (917/1054, 1353/1461, 1935/2070, 2310/2550, 2799/2925, 3070/3139 + eg twins) |
| after COUNT_FIX | **true board material** | popcount recompute 7245-7256 / 7573-7584 (`ENABLE_MATERIAL_COUNT_FIX=true`) |
| after capgains | **hypothetical post-exchange material** | `blackPieceVal -= value_gained` 8815, `whitePieceVal -= value_gained` 8836 (`PIECEVAL_RECOMPUTE_LATE=false` leaves this in place, 7279-7292 inert) |

Readers, in execution order, with which value they see at defaults:

| reader | line | live? | sees |
|---|---|---|---|
| capgains realizability | 7271 | OFF (`ENABLE_CAPG_REALIZ=false`) | (true) |
| latent-threat backing | 7323, 5595 | DEAD (host skipped) | — |
| `evaluate_king_safety` `MOD_KS_REALIZ` damp | 5614 | **LIVE** (=128) | **hypothetical** |
| O/D imbalance realizability | 7389/7397 | OFF (`REALIZ_MAT_K=REALIZ_PHASE_K=0`) | — |
| `MOD_PIECES_LEVEL` mat gate | 7744 | OFF | — |
| NPEDGE damp | 7816 | OFF (recomputes own popcounts anyway) | — |
| `piece_value_boost` | 7920-7950 | **LIVE** | **hypothetical** |
| breakdown `material` / `det_*_pieceval` | 8133, 8168-8169 | diag | **hypothetical** |

**Crux answered:** the memory line "`material` IS blackPieceVal − whitePieceVal, mutated inside the
capture loop" is correct **for the live consumers**. Capgains itself starts from TRUE material (the
COUNT_FIX recompute runs immediately before it), but both live downstream consumers — the whole-budget
king-safety damp and the material-domination boost — and the diagnostic `material` field read the
**post-capture-simulation** value. This is by design and known: the comment at search_engine.h:1734-1740
says flipping `PIECEVAL_RECOMPUTE_LATE` is "a retune (sweep MOD_KS_REALIZ on top) rather than a straight
A/B" because MOD_KS_REALIZ=128 was tuned against the mutated values.

**Currency contamination:** `value_gained` is SEE material **plus** the positional
`pawn_rank_bonuses[]` when a non-pawn captures a pawn (8794-8799 / 8819-8824, clamped ±`CAPG_PAWN_RANK_CLAMP=275`).
That combined number is subtracted from the *material* accumulators (8815/8836), so a passed-pawn
positional credit leaks into what PVB and the KS damp treat as material.

### A2. `attackingLayer[2][8][8]` — one table, four total-reaching lenses

Written: rebuilt EVERY eval by `setAttackingLayer` (8909): a hardcoded central-heat base table
(8940-8984, phase-variant) plus a king-directed 2-ring boost `kinc = increment * KS_ZONE_ATTACK_PCT/100`
(9009; increment 5 mg / 10 eg, PCT=50 ⇒ kinc=2 mg) with open-square multiplier `ATTACK_OPEN_MULT=5`
(9046-9052) and pawn-shield discount. `SCALE_ATTACK_LAYER=100` global scale (8896-8907).
`KING_ZONE_SYM_MODE=2` shipped 08-08.

Read — the same cells reach `total` through FOUR distinct channels:

1. **Direct into per-piece `total`** — occupancy square + every attacked square (knights 1359-1360 /
   1397-1398 at >>1; bishops 1941-1942; rooks 2313-2314 at >>1/>>2; queens 2805-2806 at >>2/>>3; kings
   3105/3175 full; pawns route it via `positional_bonus` 923-924/1008-1009 into the
   `PAWN_CLAMP_MID=225` clamp 1048).
2. **O/D accumulators** → imbalance term ×3 (writes at 928-929, 1364-1365, 1946-1947, 2318-2319,
   2810-2811, 3106/3176 etc.; read 7386-7399).
3. **`central_score`** via `update_global_central_scores` (4768-4776: ×2 inner / ×1.5 outer center) —
   called with attackingLayer-derived increments at 1016, 1405, 1991, 2451, 2853 etc.; read 7354-7364.
4. **King-shelter arithmetic** in `evaluate_kings_midgame` multiplies `attackingLayer[def][x][y]` by
   1×/2×/4× depending on shielding (3111-3132 / 3178-3198) — both into `total` and into the defensive
   scores.

### A3. `whiteOffensiveScore/whiteDefensiveScore` (+black) — a scaled shadow of A2

Globals (81), reset 6876-6879. Written ONLY from `attackingLayer` cells (per-piece shifts differ:
pawns >>1/>>2, minors/rooks full, queens >>1/>>2, kings full offense + shelter-multiplied defense
including a NEGATIVE defensive write 3130/3196 for an unshielded non-back-rank king). Holds the
**absolute** board fact (no capture simulation touches it).

Readers: **O/D imbalance** (7386-7400, LIVE, one-sided `off > def` gates ×`IMBALANCE_SCALE=3`);
`MOD_KS_CONTROL` (5605, OFF); `MOD_PIECES_CONTROL/DEFEND` (7763/7781, OFF); `MOD_PVBOOST_COMP/MOB`
(7927-7947, OFF); diagnostics (8164-8167). So at defaults this accumulator exists for exactly one term —
the imbalance — and that term is a re-projection of the same table already in `total` via channel 1.

### A4. `central_score` — a third projection of the same two tables

Global, reset 6886. Written only through `update_global_central_scores` (4768) whose `base_increment`
arguments are, at every call site, either a `*PlacementLayer` cell (pawns 931/1068, knights 1368/1475,
bishops 1949/2084, queens 2813/2939) or an `attackingLayer` cell (pawn attacks 1016/1151, knight attacks
1405/1515, etc.). It contains NO independent board information — it is (placement + attack tables)
restricted to 12 center squares and re-weighted ×2/×1.5. Read once: `central_add` 7354-7364 (midgame
only, phase-stepped, clamped, `SCALE_CENTRAL=100`).

### A5. `square_values[64]` — eval magnitudes reused as a capture currency

Reset 6907. Written: `abs(per-piece eval result)` — midgame: knights 7044, bishops 7063, queens 7137,
kings 7207; **NOT rooks** (commented, 7120) and **NOT pawns** in the midgame; endgame: all types
(7430-7539). Also a stray `square_values[r] = 1000` for pawns at 1275 (inside the pawn evaluator's
neighbor scan).

Read (live): `get_least_valuable_attacker` in the capgains gather (8566, since
`ENABLE_CAPG_LVA_STATIC=false`) ranks attackers by these EVAL magnitudes. Consequences documented in
code (8554-8564): mirror asymmetry; and — reading the write set above — **midgame rooks and (mostly)
pawns carry `square_values = 0**, so the "least valuable attacker" scan sees them as free`. The `see()`
call itself was fixed (`ENABLE_SEE_FIX=true`, cpp_bitboard.h:1671, search_engine.h:1590); the gather's
LVA was not (that is the parked `CAPG_LVA_STATIC` knob). Two currencies, as memory says — and the
square_values currency is additionally *phase-inconsistent* (written for different piece sets in mg vs eg).

### A6. `attack_bitmasks[64]`

Reset 6889. Written by every evaluator (attacker-bit OR per attacked square: 997, 1386, 3087, …).
Absolute fact. Readers: capgains gather + `see` + `can_evade` (8538, 8306-8324), `king_safety_danger`
zone scan (5327), passer pricing (V3 realizability, per comment 6891), `ENABLE_MOBILITY` loop (7797,
OFF), PVB mobility conditioner (7913, OFF), breakdown mobility detail (8174). The one accumulator with
many consumers that is NOT a redundancy problem: each consumer extracts a different physical fact from it.

### A7. `g_capg_tension`

Written at 8708 (count of pending SEE≥0 captures, both sides). Readers: `capg_conditioned_scale`
(464, **LIVE** — note the self-reference: capgains' own weight this eval is a function of its own output
this eval, fine because it is written before the return is scaled); threats tension gate (5713, inert:
`THREATS_QUIET_PCT=100`); NPEDGE quiet gate (7841, OFF); `rook_tension_scale` (509, OFF); winnability
(6684, OFF). Stale under `g_eval_light` or `SCALE_CAPTURE_GAINS=0` (comment 5703-5705).

### A8. `whitePlacementLayer/blackPlacementLayer` (PSTs)

Static after init scaling (327-355). Read: per-piece `total` (920, 1356, 1938, 2802, 4669 + black
twins — note midgame rooks do NOT use a PST; kings only in the endgame evaluator), `central_score`
re-read (A4), `cheap_eval` (6792-6822, search's improving heuristic, not part of this eval), and the
pawn evaluator's neighbor `placement_val` scratch (1234-1259).

### A9. Dead accumulators (cleared every eval, never meaningfully used)

`pressure_white/black`, `support_white/black` (6899-6902), `num_attackers/num_supporters` (6904-6905):
their sole writer `update_pressure_and_support_tables` (4778) is called ONLY from commented-out sites
(all 24: 1000, 1135, 1389, …, 4732; plus the commented `adjust_pressure_and_support_tables_for_pins`
7219/7547). Their sole consumer `approximate_capture_gains1` (8331) is referenced only from a commented
line (7953). **This is ~6 arrays of per-eval `fill(0)` cost plus a whole dead capgains implementation.**

### A10. Smaller live scratch

- `pawn_rank_bonuses[64]` (local, 6932): written by pawn evaluators (947 etc.); read by capgains
  (A1 currency contamination) and `boost_pieces_for_supporting_passed_pawns` (6061: enemy-side passer
  credit is `min(-pawn_rank_bonuses[r], black_adjustment)`).
- `g_passer_mid/end_deferred`, `priced_passer` (6892-6894): V3 stash; `evaluate_passers` pays (6365).
- `g_rook_file_bonus` (500): written 2412/2645 (capped rook open-file/7th edge, midgame); read only by
  the OFF rescale (8123-8126). Computed-but-unread at defaults.
- `g_rook_file_bonus`'s parent bonuses themselves are IN `total` via the rook evaluator — the global is
  a *copy* for conditioning, not a channel.

---

## B. The degeneracy table

"Same accumulator" ⇒ genuinely collinear (a fit cannot separate them on any corpus).
"Same fact, different signal" ⇒ redundant in concept but separable in data.
LIVE = contributes at current defaults.

| physical concept | channel | where | live? | underlying signal | verdict |
|---|---|---|---|---|---|
| **King-zone pressure** | (1) unit-KS `evaluate_king_safety` | 7341, 5296-5628; MAG=3000 | **LIVE** | `attack_bitmasks` over king zone (attackers/weak/shield/open-file/storm/safe-check units) | owner candidate |
| | (2) attackingLayer king slice → per-piece `total` | kinc 9009-9155 → channel A2.1 | **LIVE** (PCT=50) | table geometry around kings | REDUNDANT w/ (1) — same fact, different signal |
| | (3) same slice → O/D → imbalance ×3 | A2.2, 7386-7400 | **LIVE** | **same accumulator as (2)** | COLLINEAR w/ (2) |
| | (4) same slice → central_score | A2.3 (only where king zone ∩ center) | LIVE (partial) | **same accumulator as (2)** | COLLINEAR (partial) |
| | (5) `get_latent_threat_score` | 5726; skipped 7316 | DEAD | — | — |
| **King shelter** | (1) flat 185/75 | 3116/3119, 3183/3186 | **LIVE** (`KS_V2`,`KS_CONSOLIDATE` off) | pawn-in-front-of-back-rank-king | three live channels for one pawn |
| | (2) attackingLayer ×4/×2/×1 shield multipliers | 3111-3132/3178-3198 (same loop) | **LIVE** | same detection, table-valued | REDUNDANT w/ (1) |
| | (3) `KS_SHIELD=2` units + `KS_OPEN_FILE=2` | 5378-5389 | **LIVE** | own-pawn popcount in shield mask | REDUNDANT w/ (1)(2) — same fact, different signal |
| **Material advantage** | (1) `values[]` inside per-piece `total` | 916, 1352, … (the ONLY direct add — there is NO standalone material term) | **LIVE** | census | owner |
| | (2) `piece_value_boost` | 7920-7950 (trigger 7367 on `total`≥1500) | **LIVE** | matDiff/leaderMat ratio of **capgain-mutated** PieceVals | re-pricing of (1) + double-count of (5) |
| | (3) Kaufman quadratic imbalance | 7705-7737 | **LIVE** | census products | re-pricing of (1) (deliberate; separable — quadratic) |
| | (4) flat pair bonuses | 7682-7698 | DEAD (Kaufman on) | — | correctly exclusive |
| | (5) `capture_gains` | 7262-7273 | **LIVE** | pending SEE exchanges (hypothetical) | complementary to (1) — but leaks into (2) via A1 |
| | (6) AE mate/edge-drive | 5083-5135, keyed `total`>2000 | LIVE (endgame) | king geometry, gated on winning | complementary |
| **Piece placement** | (1) PSTs → per-piece `total` | A8 | **LIVE** | static tables | owner |
| | (2) attackingLayer base heat → `total` | A2.1 | **LIVE** | central-heat table | near-duplicate of (1) in the center; separable at edges |
| | (3) `central_score` | A4, 7364 | **LIVE** | **literal re-read of (1)+(2)** on 12 squares | COLLINEAR |
| | (4) O/D imbalance | A3 | **LIVE** | **re-read of (2)** whole-board | COLLINEAR w/ (2) |
| | (5) `square_values` → capture LVA choice | A5 | **LIVE** | abs(per-piece result) | not additive — a selection side-channel; contaminating, not collinear |
| **Central control** | subsumed by placement (2)(3) — no independent detector exists | | | | |
| **Mobility** | (1) bespoke in-evaluator scans (knight second-order 1414-1456, bishop/queen surrogates, `CHEAP_*` variants) | per-piece | **LIVE** | reachable-square counts | owner |
| | (2) `ENABLE_PIECE_MOBILITY` SF-style | 8101-8116 | OFF | — | correctly fenced (comment s_e.h:1536) |
| | (3) `ENABLE_MOBILITY` whole-board | 7797-7807 | OFF | attack_bitmasks | — |
| **Pawn structure** | doubled 125 (935/1072) + walls/chains (`structural_bonus`, clamped w/ positional under `PAWN_CLAMP_MID=225` 1048) | pawn eval | **LIVE** | own scans | single channel (iso/backward/majority/closedness all OFF) |
| **Passed pawns** | (1) `evaluate_passers` V3 (rank-table × realizability R) | 6312-6369 | **LIVE** | per-passer | owner (V3 removed the in-loop rank bonus 954-961, the −100 attack bonuses 1018-1022/1407-1411, and the AE passer block 5164) |
| | (2) `boost_pieces_for_supporting_passed_pawns` | 5992-6276, add 7306 | **LIVE** | pieces/attackers on the path ahead (PPS_*) | overlaps (1): blockade priced in BOTH `PPS_*_BLOCK` and V3's R |
| | (3) capgains pawn-rank credit | 8796-8799/8821-8824 | **LIVE** | pawn_rank_bonuses on captured pawn | complementary intent (capture of a passer), currency leak per A1 |
| **Threats / hanging** | (1) `threats_term_scaled` | 7300, 5647-5724 | **LIVE** | attackersMask per weak piece; `THREAT_HANGING` fires on nd==0 or na>nd | overlap ↓ |
| | (2) `capture_gains` | 7262 | **LIVE** | SEE≥0 captures — a hanging piece IS a SEE≥0 capture | REDUNDANT overlap on the hanging subset (same physical fact, different machinery) |
| **Whole-board activity imbalance** | O/D imbalance term | 7386-7400 | **LIVE** | attackingLayer sums | it is placement channel (4) and king-zone channel (3) wearing a third name |

---

## C. Capture-gains ↔ material ↔ PVB — the exact trace

1. 7245-7256: PieceVals recomputed = TRUE material (`ENABLE_MATERIAL_COUNT_FIX=true`).
2. 7264: `approximate_capture_gains` runs. Per applied simulated capture: gains accumulate into the
   return value AND `blackPieceVal -= value_gained` (8815) / `whitePieceVal -= value_gained` (8836) —
   where `value_gained` = SEE result (+ promo credit if gated on) **+ pawn_rank_bonus for pawn targets**.
3. 7273: `total += cg` (× conditioned scale) — the pending exchange is now IN the eval **once**.
4. 7279: `PIECEVAL_RECOMPUTE_LATE=false` ⇒ the mutation stays.
5. 7341→5614: `MOD_KS_REALIZ=128` damps king danger by the attacking side's **hypothetical** material
   backing.
6. 7920-7950: `piece_value_boost` = `((leader−loser)/leader) × PV_BOOST_MAG(10000)` on the
   **hypothetical** PieceVals, fired only when the trigger latched (|total| ≥ 1500 at the latch point —
   which in the midgame is BEFORE capture_gains was added: latch 7367 reads the pre-tail total; in the
   endgame branch 7645 it is after capgains/threats/passers but before AE).
7. 8133: breakdown `material` = the hypothetical diff — **this is why an equal-material midgame position
   can print `material` +5.29 while `capture_gains` prints +4.29: the same pending exchange appears in
   both fields**, and it genuinely reaches `total` twice whenever the PVB trigger is armed — once as
   `cg`, once inside the boost's matDiff. When the trigger is not armed the second copy costs nothing
   (the breakdown still shows it).

So: **PVB reads the capgain-mutated material, the `material` detail is not a term (nothing adds
`blackPieceVal−whitePieceVal` to `total` directly), and the double-count is conditional on the PVB
trigger** — which is exactly when it hurts most (already-lopsided positions get the pending capture
amplified by a 10000-magnitude ratio boost).

Falsify: set `PIECEVAL_RECOMPUTE_LATE=1` (single knob, already built) and diff evals on the 16-pawn
repro `2R3r1/8/8/k7/8/8/6p1/3K3R b` and the +5.29 equal-material position; the `material` field must
drop to the true census and PVB must stop moving with pending captures. Expect NOT byte-identical and a
required MOD_KS_REALIZ retune (search_engine.h:1738-1740).

## D. Placement — how many times one square reaches `total`

For a knight standing on (and attacking) central squares near a king, ONE physical placement fact is
credited via:

1. `whitePlacementLayer[KNIGHT-1][x][y]` → `total` (1356);
2. `attackingLayer[0|1][x][y]` occupancy + per-attacked-square → `total` (1359-1360, 1397-1398);
3. the same two table reads → `central_score` ×2/×1.5 → `total` (1368, 1405 → 7364);
4. the same attackingLayer reads → O/D accumulators → imbalance ×3 → `total` (1364-1365 → 7386-7400);
5. `square_values[sq] = |result|` → capture-selection side-channel (7044).

Channels 3 and 4 read the SAME cells as 1-2 — they are not new evidence about the position, they are
re-weightings. A joint fit over {PST scales, SCALE_ATTACK_LAYER, SCALE_CENTRAL, IMBALANCE_SCALE,
KS_ZONE_ATTACK_PCT} is fitting five coefficients over ~two independent signals; this is the concrete
mechanism behind "the corpus fit flattens eval" — the optimizer can trade these against each other
freely on the corpus and the game-relevant mixture is unidentified.

## E. Dead / gated / unwired (encountered; knob + default)

**Dead code at defaults (not just gated — unreachable/unread):**
- `get_latent_threat_score` (5726) — skipped by `ENABLE_KS_REPLACE_LT=true` (s_e.h:1160).
- `approximate_capture_gains1` (8331) + `pressure_*/support_*/num_attackers/num_supporters` +
  `update_pressure_and_support_tables` (4778) + pin variant — writers all commented, consumer
  unreferenced. Still pay per-eval `fill(0)` (6899-6905).
- Flat `BISHOP_PAIR_BONUS=300`/`KNIGHT_PAIR_BONUS=200` (7682-7698) — `ENABLE_KAUFMAN_IMBALANCE=true`.
- `g_rook_file_bonus` computed (2412/2645) but consumer OFF (`ENABLE_ROOK_TENSION_COND=false`).
- `square_values` writes for midgame pawns partially (1275) and eg pieces — consumed only by the LVA
  gather; midgame rook/pawn entries are 0 ⇒ the default LVA ranking is broken for those attackers.

**Gated OFF (byte-identical), grouped:**
- Material/realizability: `ENABLE_CAPG_REALIZ`, `REALIZ_MAT_K=0`, `REALIZ_PHASE_K=0`,
  `PIECEVAL_RECOMPUTE_LATE`, `MOD_MAT_PAWNS/OPPB=0`, `MOD_PVBOOST_COMP/MOB=0`, `PV_BOOST_PHASE_K=0`.
- KS: `MOD_KS_BACKING/CONTROL=0`, `KS_DEF_MAG=100`, `KS_LIGHT_MAG=0`, `ENABLE_KS_V2=false`,
  `KS_CONSOLIDATE=false`, `KS_MIN_ATTACKERS=0`, `KS_DEFENDER=0`, `KS_STORM=1`/`KS_OPEN_FILE=2` LIVE but
  many siblings 0 (`KS_OVERLOAD`, `KS_ZONE_NORM`, `KS_INTERACT`, `KS_ATT_PRODUCT`, `KS_DYN`,
  `ENABLE_KS_AIM`, …).
- Placement conditioners: `MOD_PIECES_LEVEL/CONTROL/DEFEND=0`, `ENABLE_NPEDGE_DAMP(_EG)`.
- New-content terms: `ENABLE_MOBILITY`, `ENABLE_PIECE_MOBILITY`, `SPACE_MAG=0`, `OUTPOST_*=0`,
  `ISOLATED/BACKWARD_PAWN_PEN=0`, `PAWN_MAJORITY_MAG_*=0`, `ENABLE_CLOSEDNESS`, `ENABLE_WINNABILITY`,
  `ENABLE_ENDGAME_SCALE`.
- Passers: `ENABLE_PASSER_DANGER`, `ENABLE_PASSER_V2`, `ENABLE_PASSER_KRACE_MG`,
  `ENABLE_PASSER_ORD_FLOOR` (check), rank floors 0.
- Capgains bundle (parked, memory-tracked): `ENABLE_CAPG_NET_SELECT`, `ENABLE_CAPG_PROMO_CREDIT`,
  `ENABLE_CAPG_LVA_STATIC`, `ENABLE_CAPG_FILE_INVARIANT_TIEBREAK`, `ENABLE_CAPG_EVADE_POLARITY_FIX`,
  `ENABLE_CAPG_TEMPO`. LIVE in capgains: `ENABLE_CAPG_PIN=true`, `ENABLE_CAPG_COND=true`,
  `ENABLE_CAPG_INVARIANT_ORDER=true`, `ENABLE_CAPGAIN_PAWN_FIX=true`, `ENABLE_SEE_FIX=true`.
- Inert-by-value: `THREATS_QUIET_PCT=100` (tension gate), `SCALE_*=100` family.

## F. Top-5 redundancies ranked by expected joint-fit distortion

Ranking principle: how freely can a fit trade this channel against its twin on a quiet-position corpus
while changing behaviour in games — collinear channels rank above merely-overlapping ones, and channels
touching MATERIAL (the largest magnitudes) rank above king-zone ones.

1. **Capture simulation double-booked through material.** `total += cg` (7273) AND the same simulated
   exchange rewrites the PieceVals (8815/8836) that `piece_value_boost` (7920-7950) and the KS damp
   (5614) consume; plus positional pawn-rank credit leaks into the material currency (8799).
   *De-dup:* flip `PIECEVAL_RECOMPUTE_LATE=true` (keep `capture_gains` as the sole owner of pending
   exchanges — it is the conditioned/tapered copy; the PVB ratio is the least-conditioned consumer),
   re-sweep `MOD_KS_REALIZ`. Falsified if the 16-pawn repro and the +5.29/+4.29 pair do NOT move.

2. **King-zone pressure: 3 live additive channels off 2 signals** (unit-KS off `attack_bitmasks`;
   attackingLayer king slice off table geometry feeding `total` directly AND O/D→imbalance ×3).
   The imbalance copy is the least conditioned (no phase taper, no floor, no realizability — one-sided
   ×3 on raw sums) while unit-KS carries taper+floor+damp. *De-dup:* sweep `KS_ZONE_ATTACK_PCT` 50→0
   (the knob's own comment, 9005-9008, says it exists precisely because "this king-zone slice…
   double-counts"), making unit-KS the owner and reducing the imbalance term to central/whole-board
   activity. Falsified if KS-theme STS/game deltas at PCT=0 + retuned KING_SAFETY_MAG cannot recover
   parity (would show the layer slice carries independent information).

3. **`central_score` is a literal re-read of the placement+attack tables** (A4) — zero new information,
   phase-stepped and clamped differently, so a fit sees an extra free coefficient on the same signal.
   *De-dup:* retire `SCALE_CENTRAL→0` and fold the intended center emphasis into the tables themselves
   (they are already center-heavy); keep the per-piece copy (it is the tapered/clamped one via
   PAWN_CLAMP etc.). Falsified if a PCT-0/SCALE-0 arm shows a move-change footprint that no PST rescale
   can reproduce.

4. **Hanging pieces priced by both `threats_term_scaled` and `capture_gains`.** `THREAT_HANGING` (+ the
   na>nd weak test, 5682) fires on exactly the pieces the capgains gather books as SEE≥0 captures
   (8552). Two live channels, different machinery, same physical fact; threats is capped (800/target)
   and scaled 75, capgains is tension-conditioned — both partially tamed, still summed. *De-dup:*
   `THREATS_STANDING_ONLY=true` (knob exists, 5682) so threats owns standing pressure on defended-but-
   pressured pieces and capgains owns volatile hanging material. Falsified if the two terms' per-position
   contributions are NOT correlated on the corpus (measure before shipping: correlate br_threats vs
   br_capture on the 23k bank).

5. **King shelter counted three times** (flat 185/75 at 3116/3119; the ×4/×2 attackingLayer shield
   multipliers in the SAME loop 3111-3132; `KS_SHIELD`+`KS_OPEN_FILE` units inside `king_safety_danger`
   5378-5389). *De-dup:* `KS_CONSOLIDATE=true` zeroes the flat constants (built), or `ENABLE_KS_V2`
   re-homes them as tunables — keep the unit-KS copy (conditioned by floor/taper/damp), retire the flat
   one (unconditioned). Falsified if shelter-theme positions regress at KS_CONSOLIDATE=1 with
   KS_SHIELD retuned upward.

**Runner-up:** passer blockade priced in both `PPS_ENEMY/OWN_BLOCK` (6020-6024) and V3's
realizability R (6324) — same stop-square facts; V3's R is the conditioned copy.

## What this map could NOT determine from reading

- Whether the O/D imbalance term carries any information beyond the placement/king channels *in
  practice* (its per-piece shift pattern differs by type, so it is a differently-weighted projection,
  not an exact scalar multiple — only an ablation-vs-PST-rescale experiment separates them).
- The actual corpus-level correlation matrix of the breakdown fields (the fit-distortion ranking above
  is structural, not measured). `fit_bench_guarded.py` on the banked breakdowns can produce it cheaply.
- `advanced_endgame_eval`'s interior (only skimmed past the mate/edge-drive and the V3-skipped passer
  block); its `total`-rewrite makes every midgame-tail term's endgame meaning conditional on it.
- Endgame per-piece evaluators were sampled, not exhaustively read; O/D shift constants there may differ.

## Corrections to the established findings quoted in the task

- "Four live king-zone channels" — confirmed, but channels (3) attackingLayer-direct and (4) O/D are the
  SAME accumulator (collinear), which is stronger than "four channels": it is 4 channels over 2 signals
  (5 counting the shelter constants' own attackingLayer multiplier).
- "`material` mutated in the capture loop" — true for live consumers, but with two refinements: the
  COUNT_FIX recompute means capgains STARTS from true material (the mutation is only downstream), and
  the mutation includes a POSITIONAL component (pawn-rank credit), not just simulated material.
- "MOD_KS_REALIZ=128 damps unit-KS" — confirmed live, and it is tuned AGAINST the mutated material
  (search_engine.h:1460: with COUNT_FIX off the same value was −196 STS), so redundancy-1's fix is a
  retune, not a toggle.
- The `pt_queens` 5 mp lead "next = xray `values[type]>>6`" — the xray term exists at 2053/2193 (bishop)
  and 3788-4638 (eg sliders); it is indeed unscaled by SCALE_ATTACK_LAYER, consistent with that memory.
