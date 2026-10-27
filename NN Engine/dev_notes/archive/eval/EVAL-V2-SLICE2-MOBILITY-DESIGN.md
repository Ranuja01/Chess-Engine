# Eval v2 — SLICE 2 DESIGN: mobility + per-piece placement + rook files

@author: Ranuja Pinnaduwage (maintained with Claude)

Status: **mobility core + rook files BUILT and GATED (2026-09-14); placement sub-terms not built; no magnitude read yet.**
Started 2026-09-14.

| gate (2026-09-14) | result |
|---|---|
| arm 0 byte-identity, WAC d10 LONG_FORMAT | ✅ `250 / 35,310,778 / 3.784` |
| v2 shipped config, slice 2 OFF (moved attack-build gate) | ✅ `246 / 63,221,361 / 4.087` — identical to 09-13 |
| detector oracle `_mobility_detector_oracle.py`, 3,004 positions × 2 sides | ✅ **0 mismatches** at `KS_V2_XRAY=1`, and at `XRAY=0 + EXCL_QUEEN=1 + EXCL_LOWRANK=1`; 99.2% of positions have differing W/B counts (non-vacuous) |
| `_eval_symmetry.py` N=800, mobility 115 + rook files 50/25 ON | ✅ colour swap **0/800**. File mirror 21/651 @ 5 mp — **identical with slice 2 OFF**, so pre-existing (queen positions; inherited PST) |
| knob provably executes (`_v2_positional_spread.py`) | ✅ knight rim-vs-centre 30 → 59 mp · open-file rook 0 → −69 · sibling std median 36 → 45 · rook-move \|dev\| 9 → 23 |
| re-verified after the KPK-bitbase rebuild (new code default off) | ✅ arm 0 `250 / 35,310,778 / 3.784` · v2 shipped `246 / 63,221,361 / 4.087` |
| tempo swing identity (`eval_symmetry.py`, 600 FENs stratified by non-pawn count) | ✅ **+0.000 / +0.000 / max 0.000** with mobility 115 + rook files ON — v2 stays exactly side-to-move-blind | Register: `EVAL-V2-CURRENT-CONFIG.md` §5 (slice 2, tested ALONE).
Kickoff checklist followed in order: `SESSION-HANDOFF-2026-09-13.md` §3.

---

## 0. WHAT THE PREWORK FOUND (checklist steps 1-4)

### 0.1 Record-check — nothing in this slice has a RESOLVED verdict either way
| concept | best standing reading | class |
|---|---|---|
| whole-board mobility (`ENABLE_MOBILITY`) | +1.2pp win%, sign-consistent on 3 corpora, ~1σ (09-10 paired nulls) | **unresolved — the best lead on record** |
| SF per-piece mobility (`ENABLE_PIECE_MOBILITY`) | STS −175 | ☠️ **closed on a PROXY**: one knob changed area + added the floor + disabled the cheap surrogates |
| cheap rook mobility / bishop complex (v1 live) | §I ablation ≈ +0.1% each | v1's SHAPES carry ~no accuracy → copying them buys nothing |
| outposts | "harmful" at ONE point (150/80) on a comparator later superseded | unresolved |
| rook files | frontier-flat on pre-08-14 STS | unresolved |
| long diagonal · minor-behind-pawn · trapped rook · weak queen · king protector | — | **never tried** |

☠️ **Mobility's STS sign once FLIPPED on cheap-rook-mobility presence (+52 vs −50).** ⇒ the first measurement of this
slice is a **2×2 mobility × rook files**, never either alone.

### 0.2 Four scans over v1 (what NOT to carry)
- v1's piece activity reaches the score through FOUR channels in the midgame: direct · per-piece clamp
  (`MG_CLAMP_*`) · saturating `central_bounded` · saturating `ovd_imbalance`. ⇒ **v2 has no shared budget and no clamp
  on any slice-2 term.**
- v1 has **no rook PST**; knights/bishops/queens switch evaluators at a phase cliff (64/65); rook file penalties are
  mostly absorbed by the 300 cap. None of this is ported.
- 🐛 v1 defects found, recorded not fixed (frozen control): `evaluate_rooks_midgame` writes
  `update_pressure_and_support_tables` for **Black only** (White twin commented out); the latent-rook diagonal-ray bug
  is live (`ENABLE_ROOK_LATENT_RAY_FIX=false`).
- Infrastructure v2 can reuse: `attacks_mask(colour, occ, sq, pt)` is PEXT, accepts ANY occupancy, reads no eval
  global · `slider_blockers` / `pin_mask` (`cpp_bitboard.h`) are pure · `PawnEntry.openFiles/halfOpen[2]` already
  exist for rook files · castling rights are in `V2Context` (trapped rook needs them).

### 0.3 v2's positional spread (🧰 `_v2_positional_spread.py`, 370 positions, quiet siblings, mp)
| median | rung 0 (PST) | + pawns | shipped (+KS) |
|---|---|---|---|
| sibling std | **11** | 19 | 36 |
| \|dev\| by moved piece N / B / R / Q | 11.4 / 8.0 / **2.9** / 10.6 | 11.7 / 9.4 / 4.8 / 11.9 | 15 / 13 / 9 / 21 |
| knight rim a3 vs centre e5 | **30** | 30 | 30 |

★ **Rook moves are ordered by 2.9 mp** and are the largest quiet-move class (29%). This slice lands exactly there.

---

## 1. FIVE-ENGINE CONTRAST (from source: SF 1.1 / 11 / 15.1 local, Ethereal + Weiss fetched)

### 1.1 Mobility
| | agreement | detail |
|---|---|---|
| per-piece COUNT → nonlinear table, mg/eg split | **5/5** | all concave; **4/5 have a negative floor** (SF1.1 is linear-capped) |
| computed in the SAME loop that fills the attack maps | **5/5** | ⇒ reuse `build_side_attacks`; no second pass |
| exclude squares attacked by enemy pawns | 4/5 | not SF1.1 |
| exclude own blocked pawns | 4/5 | SF11/15 + Ethereal + Weiss |
| exclude own king | 3/5 | SF11/15, Ethereal |
| own non-pawn pieces' squares COUNT | 4/5 | only SF1.1 excludes all own-occupied |
| exclude own queen · own low-rank pawns | 2/5 · 3/5 | SF11/15 · SF11/15 + Weiss (rank 2) |
| pinned piece restricted to its pin line | **1 lineage** (SF11/15) | |
| x-ray transparency | **all five differ** | SF11: B through all Q; R through all Q + own R |

**Ratio that governs magnitude — knight table range ÷ the SAME engine's knight PST rim-vs-centre spread (mg):**
SF1.1 **1.0×** · SF11 **1.1×** · SF15.1 **1.2×** · Weiss **3.8×** · Ethereal **9-13×** (its PST is nearly flat).
⇒ The references disagree by an order of magnitude on how much of "piece activity" sits in mobility vs placement.
The ratio is unit-free (both numbers are the same engine, same phase), so it transfers; any single one of them does
not, because it depends on how much that engine's PST carries.

### 1.2 Per-piece placement
| term | agreement | shape disagreement |
|---|---|---|
| rook on open / semi-open file | **5/5** | Weiss scores only the FORWARD part of the file; SF15.1 adds a closed-file penalty |
| outposts | 4/5 (not Weiss) | SF11 requires pawn support; Ethereal indexes BY support; SF15 adds reachable + uncontested |
| minor behind pawn | 4/5 (not SF1.1) | SF mg-heavy `S(18,3)`; Ethereal/Weiss **eg-heavy** `S(3,28)` / `S(9,32)` — opposite phase |
| bad bishop | 4/5 (not SF1.1) | three different multipliers: SF11 ×(1+blocked centre) · SF15 zero when pawn-defended · Weiss a PRODUCT with blocked centre · Ethereal rammed pawns only |
| long diagonal · king protector · trapped rook · weak queen | 3/5 each | SF lineage (+Ethereal for diagonal / pin) |
| rook on 7th | 2/5 | **dropped by SF11+ and Weiss** (carried by eg-heavy rook mobility) |
| rook on queen file · bishop x-ray pawns | 1/5 | |
⚠️ Named handlers checked: SF11 `RookOnFile` **shadows** `TrappedRook` (else-if); SF15.1 `UncontestedOutpost` overrides
`Outpost`; SF15.1 `Rook/BishopOnKingRing` fire only when the piece does NOT already attack the ring.

### 1.3 What the adoption rule makes of this ([[adopt-reference-methods-only-if-universally-superior]])
- **CORE (universal, default shape):** per-piece concave mobility with a negative floor, same loop as the attack maps,
  area excluding enemy pawn attacks + own blocked pawns + own king · rook open/semi-open file.
- **CANDIDATES (references split — ours legitimate, theirs a knob):** own-queen and low-rank-pawn exclusion ·
  pin-line restriction · outposts · minor-behind-pawn (BOTH phase profiles) · bad bishop (the multiplier forms) ·
  long diagonal · trapped rook · weak queen.
- **NOT PLANNED:** rook on 7th (2/5 and abandoned by the lineage that had it) · queen file · x-ray pawns (1/5).

---

## 2. DESIGN

### 2.1 Mobility
- **Where:** inside `build_side_attacks`, which already iterates every piece with its attack mask. It gains an optional
  accumulator: when mobility is on, each N/B/R/Q adds `table[pt][popcount(a & area)]` to its side's mg/eg sums. One
  loop, one attack computation per piece. ☠️ No second attack pass.
- **Gate:** the side-attack build moves from `KS_V2_MAX > 0` to `KS_V2_MAX > 0 || MOB_V2_MAG > 0`. KS still adds only
  when its own knob is on, so rung-1-off + mobility-on is a clean arm.
- **X-ray:** mobility uses the SAME occupancy as the KS maps (`KS_V2_XRAY`, shipped =1 = SF11's form). ⚠️ Deliberate
  coupling — the references all differ here, and a separate occupancy would force a second attack pass. Documented as
  a knob interaction, not hidden.
- **Area** (enemy side's pawn attacks come from the two pawn bitboards, ~4 shifts; no `PawnEntry` dependency):
  `~(enemy pawn attacks | own king | own blocked pawns)`, candidates `MOB_V2_EXCL_QUEEN`, `MOB_V2_EXCL_LOWRANK`.
- **Pins:** `MOB_V2_PIN` (default 0, SF lineage only) restricts a pinned piece's counted squares to its pin line via
  `slider_blockers`.
- **Tables — SHAPE from SF11, SCALE from the ladder.** SF11's four (mg, eg) tables, each leg converted by ITS OWN pawn
  (mg /128, eg /213, × 1000) so the mg:eg relationship is preserved in pawn terms — which is our flat pawn's unit of
  account — and then the WHOLE set multiplied by one factor chosen so the **knight mg table range equals
  `MOB_V2_MAG` mp**. Shape transfers, scale is measured ([[matching-a-reference-term-is-not-being-right]]).
  ⚠️ This is where the `PASSER_V2_MG_PCT` trap would bite if eg were converted by the mg pawn; recorded so it is
  checked in review.
- **Floor:** kept (4/5). The June "does charging for immobility help" question is answered here for the first time
  unbundled, because area, floor and tables are each separable by knob.

#### 2.1b MOBILITY FORM BAKE-OFF (2026-09-15, after placement E shipped) — record-check result and candidate list
Record-check (all mobility history, any name): **every candidate below is a first measurement in v2**, except
`MOB_V2_EXCL_QUEEN` (one §I point at 600: mean −5.56 vs −5.75, KS-critical worst +1.67 vs +1.91 — trims the worst case for
a small mean cost; never on regret). v1's mobility results are all invalidated or v1-regime (pre-08-21 fixed-opening gate;
STS-closed; regret read against 50; the best v1 lead is an unresolved +1.2pp null). 1000 exists on §I only; 800 never run.
| # | candidate | knob | notes |
|---|---|---|---|
| a | table SHAPE SF15.1 / Ethereal / Weiss at equal knight mg range | `MOB_V2_TABLE` 1/2/3 | tables fetched from source |
| b | area: own queen · own low-rank pawns | `MOB_V2_EXCL_QUEEN` · `MOB_V2_EXCL_LOWRANK` | built + oracle-verified |
| c | endgame share | `MOB_V2_EG_PCT` | built 09-15 |
| d | pinned piece restricted to pin line (mobility only; KS maps untouched) | `MOB_V2_PIN` | SF lineage; `BB_RAYS` gives the line |
| e | x-ray separate from KS | — | ⚠️ only if no second attack pass; else record as not built |
| f | magnitude 800 / 1000 on regret | `MOB_V2_MAG` | |
| g | ★ OURS-FIRST (v1): a square counts only if not attacked by a LOWER-VALUE enemy piece (v1 knights/queens, `cpp_bitboard.cpp:1436-1480`, `:2881-2896`) — v2/SF use enemy-pawn attacks only | TBD | needs both sides' attack maps before counting ⇒ store per-piece masks in `MobAcc` during the one build, count after both sides are built (no second `attacks_mask`) |
Not candidates: v1's linear popcounts, 225 cap, rook forward zone (subsumed by the concave table / closed), latent (measured null).
Protocol: neutrals re-measured on the shipped-E base (running) → build all knobs (defaults byte-identical to `250 / 61,352,373`)
→ oracle per form → §I single-change ladder → regret for winners vs the new neutrals → collinearity (mobility × E terms) →
one SPRT on the combined winner vs the shipped base. No winner on BOTH §I mean and worst ⇒ no games; SF11 form confirmed.

**Neutrals on the shipped-E base (`ASPIRATION_DELTA=300`, d7, JOBS=1):** `_v2` **51.1%** (3,992 changed / 33.4%, delta
−0.037; was 50.7 on the mobility-only base) · primary **49.9%** (4,782 / 31.9%, delta +0.080; was 49.8). Every
mobility-form regret read uses THESE. (A neutral is the tool's drift under a search-only change — the bar, not a score.)

**Build + gates (2026-09-15):** ✅ BUILT. Byte-identity at defaults EXACT on both arms — SHIP+E `250 / 61,352,373 / EBF 4.114`,
v1 control `250 / 35,310,778 / EBF 3.784`. ✅ Mobility oracle at the shipped form: 3,010 positions / 6,020 sides, **0 mismatches**,
99.2% non-vacuous ⇒ the `mobility_build` refactor and the SAFE mask storage are count-identical when off. `[toggles]` echoes
`MOB_V2_TABLE=0 MOB_V2_EG_PCT=100 MOB_V2_PIN=0 MOB_V2_SAFE=0` (echoed AND read — the wiring trap).
✅ Oracle `TABLE=1` (SF15.1), `TABLE=2` (Ethereal), `TABLE=3` (Weiss), `PIN=1`: 0 mismatches, 99.2% non-vacuous each.
★ Not a vacuous pass for TABLE: the oracle picks its own table from the SAME env knob, so an engine that IGNORED
`MOB_V2_TABLE` would mismatch on every raw sum. ⚠️ For PIN the same argument is WEAKER — the non-vacuity metric counts
positions where White's and Black's counts differ, not positions where a pin FIRES. Cross-check at ladder time: a `pin` arm
whose §I numbers equal `ship` exactly means the path is not firing, not that it is neutral.
✅ Oracle `SAFE=1` and `SAFE=2`: 0 mismatches (99.2% / **99.0%** non-vacuous — the shifted rate is weak evidence SAFE actually
changes counts).
✅ Combined arm `XRAY=0 SAFE=1 PIN=1 TABLE=2` (every new path at once, x-ray OFF): 0 mismatches, 99.2% non-vacuous.
✅ Placement oracle under `PIN=1 BADB_V2_FORM=1`: 0 mismatches over 30,100 term-checks, **all 10 terms fire** (badb 72.9%,
behind 57.0%, longdiag 28.5%, traprook 21.1%, reach 17.4%, latent_b 31.8%, latent_r 15.5%, weakq 11.1%, outpost_n 10.2%,
outpost_b 7.7%) ⇒ the trapped-rook reuse path is exercised under PIN, not just present.
✅ Symmetry gate, both arms (all-forms `TABLE=2 PIN=1 SAFE=1`, and `SAFE=2 EG_PCT=150`, on SHIP+E): **colour swap 0/800
violations**; file mirror 21 @ 5 mp — EXACTLY the pre-existing per-file pawn-table baseline, unchanged by the new forms.
✅ **§I LADDER (14 arms, N=2500 × 6 corpora, one change each on SHIP+E; negative = better; WORST decides):**
| arm | mean% | WORST% | read |
|---|---|---|---|
| **exlow** (`MOB_V2_EXCL_LOWRANK`) | **−1.08** | **−0.34** | ✅ better on **6/6** — the biggest clean win |
| **pin** (`MOB_V2_PIN`) | **−0.51** | **−0.06** | ✅ better on **6/6**; NOT inert (≠ ship ⇒ the path fires) |
| eg200 · eg150 | −1.52 · −0.89 | +0.74 · +0.37 | ❌ mean-only: buys general corpora, costs KS-critical |
| mag1000 · mag800 | −2.10 · −1.26 | +1.37 · +0.67 | ❌ same trade, larger — confirms the 600-vs-1000 §I/worst split |
| safe1 · safe2 (OURS) | −0.48 · −0.24 | +0.40 · +0.60 | ❌ v1's lower-value-attacker test does not clear; 2 worse than 1 |
| tab1 SF15.1 | +0.35 | +0.57 | ❌ worse, though closest to SF11 |
| tab3 Weiss · tab2 Ethereal | +1.86 · +3.15 | +3.38 · +5.86 | ❌ clearly worse — their tables assume their own PSTs |
| exq (`MOB_V2_EXCL_QUEEN`) | +0.22 | +0.45 | ❌ worse on BOTH — reverses its single 09-14 reading |
★ **SF11's table SHAPE survives the bake-off** (no reference table beats it); the two winners are AREA and PIN, not the table.
**Prediction scorecard (registered before the run):** ✅ (2) Ethereal/Weiss worse — right, and by a lot. ✅ (4) safe2 < safe1.
✅ (5) mag800 between 600 and 1000 with a worse worst case. ❌ (1) SF15.1 was NOT within ±0.15 (it is +0.35, clearly worse).
❌ (3) pin/safe1 were outside ±0.2 mean. ❌ (6) `exq` did NOT repeat its 09-14 trade — it is worse on mean AND worst, so that
single §I point at 600 did not survive re-measurement on this base. ❌ (7) TWO candidates cleared, not at most one.
**Regret gate (vs neutrals primary 49.9 / `_v2` 51.1):** `pin` on `_v2` **51.7%** (+0.6pp; 3,999 changed / 33.5%, delta −0.262,
better on opening/midgame, endgame flat). Right sign, but INSIDE the ~2-2.5pp bar ⇒ not resolvable alone — consistent with §I,
not confirmation. ⚠️ `cr4_CRITICAL` reads 44.4% on 27 positions — unreadable (n_crit), quoted so it is not mistaken for a signal.
`pin` on primary **49.8%** (−0.1pp; 4,919 changed / 32.8%, delta +0.045, endgame +0.165 the worst slice).
⇒ **`pin` is a NULL on regret** (+0.6 / −0.1, signs opposed, both inside the bar) while §I liked it 6/6. One instrument
resolves it, the other cannot — NOT the two-instrument agreement the corroboration rule needs for games on its own.
It stays a candidate only as part of the combined arm; a §I-only case does not move a shipped form that games validated at +162.
`exlow` on `_v2` **50.7%** (−0.4pp vs 51.1; 4,439 changed / **37.2%** — the largest footprint of any arm here, delta −0.189).
Opening improves clearly (−0.395) but `ps3_end_EDGE` worsens (+0.605) — the same "helps the opening, costs the endgame edge"
shape §I hinted at. Inside the bar ⇒ not resolvable.
`exlow` on primary **50.7%** (+0.8pp vs 49.9; 5,342 changed / 35.6%, delta −0.068) — its best read, still inside the bar.
⇒ `exlow` across corpora: **+0.8 / −0.4**, opposed signs, both sub-bar — a regret NULL, like `pin` (+0.6 / −0.1). Neither §I
winner is confirmed by the move-level instrument, and each has ~33-37% of moves changed, so the tool had a large footprint
to read and still saw nothing.
`pin+exlow` on `_v2` **49.3%** (**−1.8pp** vs 51.1; 4,386 changed / 36.7%, delta −0.014) — the WORST read of the bake-off and
worse than either part alone on this corpus (pin +0.6 · exlow −0.4 · combined −1.8) ⇒ the two changes INTERFERE rather than
add (both shrink the counted area; together they strip squares each was pricing). `pin+exlow` on primary **50.5%** (+0.6pp;
5,354 changed / 35.7%) ⇒ **+0.6 / −1.8 — opposed signs, cross-set replication FAILS.** The combination is a null too.

★★ **BAKE-OFF VERDICT (2026-09-15): NO CHANGE. The shipped mobility form is confirmed.**
| what was tested | outcome |
|---|---|
| table SHAPE vs SF15.1 / Ethereal / Weiss, all at equal knight mg range | ✅ SF11's shape WINS — no reference table is better on §I |
| area: `EXCL_QUEEN` · `EXCL_LOWRANK` | ❌ / §I-only (regret null) — our area stands |
| endgame share 50/150/200 · magnitude 800/1000 | ❌ mean-only; all buy general corpora and cost KS-critical |
| SF's pin-line restriction | §I-only, regret null ⇒ PARKED with a trigger (§2.1c) |
| OURS-FIRST safe-square (v1's lower-value-attacker test) | ❌ fails the worst-case rule — recorded as a measured loss |
| the two §I winners combined | ❌ worse than either alone on `_v2`; cross-set fail |
⇒ The 09-15 due-diligence gap (owner, "did mobility get the same scrutiny as placement?") is **CLOSED**: mobility now has the
alternative-forms bake-off placement got, and it survived unchanged. ☠️ **Nothing was adopted on §I alone** — corpus fit does not
move a form that games validated at +162 Elo ([[corpus-fit-is-anti-correlated-with-elo]]).
★ Cost of proving the parked candidates directly: our 1,200-game SPRT gave ±23 Elo, so resolving ±5 Elo needs ~21× more games
(~25,000 ≈ 9 days). That arithmetic — not a judgement about the ideas — is why they ride bundles or wait for a better instrument.
★ This is the [[bundling-is-refuted-components-cancel-26-percent]] shape appearing inside mobility's own area definition —
two area restrictions are NOT disjoint by construction, unlike the placement sub-terms.

#### 2.1c PARKED WITH NAMED TRIGGERS (owner, 2026-09-15) — built, gated, default OFF, NOT deleted
| item | state | why parked | ★ TRIGGER to revisit |
|---|---|---|---|
| `MOB_V2_PIN` | §I better 6/6 (−0.51 / −0.06); regret NULL (−0.1 / +0.6); SF lineage only (1/5) | our instruments average over ordinary positions where ABSOLUTE PINS ARE RARE — a null here is "unreadable", not "worthless" ([[the-eval-failure-record-is-mostly-unresolved-nulls-not-refutations]]) | **a pin-dense / tactical position class.** Owner's framing: positions where a pinned piece's apparent power is fake and the eval should say so. Needs an instrument that SELECTS such positions (e.g. a pinned-piece corpus split of the regret set, or a WAC/STS subset filtered by `king_blockers != 0`), plus a fires-rate measurement — ⚠️ we never measured HOW OFTEN pin fires, only its aggregate effect |
| `MOB_V2_EXCL_LOWRANK` | §I best clean arm (−1.08 / −0.34, 6/6); regret +0.8 / −0.4 = NULL; 3/5 references (SF11, SF15.1, Weiss rank-2 only) | too small to price alone (resolving ±5 Elo needs ~25,000 games ≈ 9 days at our rate) | **rides the next REGRESSION BUNDLE** (slice 3), where the bundle SPRT asks "does this group cost anything" and the slice-end cumulative SPRT catches drift |
| `MOB_V2_SAFE` 1/2 (ours, from v1) | §I fails the worst-case rule (+0.40 / +0.60) | measured loss, recorded honestly | a form change only — not this definition |
| `MOB_V2_TABLE` 1-3, `MOB_V2_EG_PCT`, `MOB_V2_MAG` 800/1000, `MOB_V2_EXCL_QUEEN` | all fail the both-better rule | SF11's shape + our area win the bake-off | re-open only if the §I/worst trade changes (e.g. after a KS re-tune, since the worst column IS the KS-critical corpus) |
☠️ **Nothing here is deleted.** Every knob stays wired and byte-identical at default, so any of these is one env var away from a
re-measurement the day an instrument that can resolve it exists.
⚠️ Reading so far: BOTH §I winners are regret-NULL individually (pin +0.6/−0.1, exlow −0.4/?). If the combined arm is also
null on both corpora, the honest verdict is **no change** — §I alone must not move a form that games validated at +162 Elo,
and that outcome still CLOSES the due-diligence gap (SF11's shape confirmed against four references).
☠️ First build failed: a duplicate `mob_reset` left by a comment-move edit. The `build` sub deletes the `.so` before compiling,
so a failed build leaves NO engine — never leave it unresolved.

**Build status (2026-09-15, code written):**
- `MOB_V2_TABLE` 0-3: tables in `MOB_TAB_MG/EG[table][type][count]` with per-table knight range {95, 99, 147, 98} and pawn
  pairs SF11 128/213 · SF15.1 126/208 · Ethereal 82/144 · Weiss 104/204; table 0 reproduces the shipped arithmetic exactly.
- `MOB_V2_EG_PCT`: applied after conversion, skipped at 100 (byte-identical, no overflow).
- `MOB_V2_PIN`: SF's blockers_for_king rebuilt in `mob_king_blockers` (either colour, snipers removed) — squares leave the
  area, our pinned pieces count only `ray(ksq, sq)`. Mobility only; KS maps untouched (SF also clips its attack maps — a
  deliberate difference). ⚠️ Trapped rook reuses the resulting per-rook count, so PIN changes trapped rook too (as in SF).
- `MOB_V2_SAFE` 1/2 (ours-first): per-piece masks stored in the one pass, counted after both sides' maps exist; trapped rook
  keeps the plain count.
- One setup helper `mobility_build` now serves the eval dispatch and both probes (probes exercise the search path); only the
  counters are reset per eval (the old `MobAcc{}` zeroed arrays that are read only up to their counts).
- Reference findings worth keeping: Weiss is the only engine whose AREA-filtered attacks feed KS; Ethereal feeds KS the
  unfiltered set; SF feeds `mobility` into kingDanger (our wiring test of that was null). Weiss does NOT exclude king or
  queen from the area and counts only rank-2 pawns as low-rank.
- **§I ladder (one call, arms run sequentially = one engine load), each change ALONE on SHIP+E:**
  `ship` · `tab1` `tab2` `tab3` · `eg50` `eg150` `eg200` · `pin` · `safe1` `safe2` · `exq` `exlow` · `mag800` `mag1000`.
  Rule (same as placement): replace only if better than `ship` on BOTH mean and worst; ±0.05 floor.
- **Registered predictions (2026-09-15, before any number; my magnitude calls have been optimistic 6× running, so these are
  deliberately conservative):** (1) SF15.1 within ±0.15 mean of SF11 — near-identical shape. (2) Ethereal and Weiss WORSE on
  worst — their tables were tuned against their own (flat / different) PSTs; Weiss's eg queen table is 4× steeper. (3) PIN and
  SAFE1 within ±0.2 mean — both fire on a small fraction of pieces. (4) SAFE2 worse than SAFE1 (it removes squares knights
  and bishops legitimately contest). (5) `mag800` sits between 600 and 1000 on mean with a worse KS-critical worst case.
  (6) `exq` repeats its 600 read (trims worst, costs mean). (7) At most ONE candidate clears the both-better rule.
- Oracle `_mobility_detector_oracle.py` mirrors TABLE / PIN / SAFE with hand edge cases. ⚠️ Its table digits come from the
  same fetch as the C++ — it verifies transcription + indexing, not the source digits.

### 2.2 Rook files
- `PawnEntry.openFiles` / `halfOpen[s]` (already built). Open = no pawn of either colour; semi = no OWN pawn.
- One knob pair `ROOKFILE_V2_OPEN` / `ROOKFILE_V2_SEMI` (mp, mg; eg leg by SF11's eg/mg ratio in PAWN terms, as 2.1).
- Scale: SF11 open file 47 vs its 84 knight PST spread = 0.56×; SF1.1 0.63×; SF15.1 0.58×; Weiss 1.1×; Ethereal
  2.6× ⇒ against v2's 30 mp: **~17-80 mp**. ☠️ v1 runs 250 base / 300 cap — **3-15× every reference ratio**.
- Weiss's forward-only file and SF15.1's closed-file penalty are candidates, not defaults.
- ⚠️ Needs `build_pawn_entry` when only rook files are on: gate becomes `PS || PASSER || ROOKFILE`.

### 2.3 Placement sub-terms — later in the slice, each behind its own knob, default 0
Order: outposts (needs `PawnEntry.attacks` + a span) → minor behind pawn → bad bishop → long diagonal → trapped rook
(castling rights present) → weak queen (`slider_blockers`). Each gets its own detector check before its magnitude.

#### 2.3.0 Status (2026-09-14, daytime): CODE WRITTEN, NOT YET BUILT OR GATED
All seven sub-terms are in `eval_v2.cpp` (`placement_detect` + `placement_mp`), default off, as percent knobs where
**100 = SF11's plain pawn conversion**: `OUTPOST_V2_PCT` · `REACH_V2_PCT` · `BEHIND_V2_PCT` · `BADB_V2_PCT` ·
`LONGDIAG_V2_PCT` · `TRAPROOK_V2_PCT` · `WEAKQ_V2_PCT`. Probe `placement_probe` / `ChessAI.placement_counts`;
independent oracle 🧰 `_placement_detector_oracle.py` (rebuilds backward / blocked / span / area / attacks from
python-chess; per-term non-vacuity check). Build waits for the mobility SPRT (no rebuild during a games run).
Corrections made while porting from SF11 source, against the spec table below:
- **Weak queen** counts a single blocker of EITHER colour with the snipers removed from the occupancy — so it does NOT
  reuse `cpp_bitboard.h`'s `slider_blockers`, which returns only the current side's blockers.
- **Bad bishop's** "blocked" counts own pawns blocked by ANY piece on files c-f (not only by pawns).
- **Castling rights** in search and in `ev` are the same rook-square mask (`cpp_bitboard.h:1412-1424` clears rook squares / the
  back rank on king moves), so trapped rook's ×2 means the same thing in play as in the oracle.

**Gates after the placement build (2026-09-14):**
| gate | result |
|---|---|
| v2 shipped WAC byte-identity | ✅ `246 / 63,221,361 / 4.087` — the SPRT segments stay the same arms |
| arm 0 WAC byte-identity | ✅ `250 / 35,310,778 / 3.784` |
| `_placement_detector_oracle.py`, 3,008 positions × 8 terms | ✅ **0 mismatches** at `KS_V2_XRAY=1` and `=0`; every term fires (outpost_b 7.7% … badb 75.8%) |
| `_eval_symmetry.py`, all seven at 100% | colour swap ✅ 0/800 · ☠️ file mirror **22** vs baseline 21 — one new **576 mp** violation, traced to SF11's trapped-rook side test `(kf < FILE_E) == (file_of(s) < kf)`, which is asymmetric when the rook shares the king's file. **Symmetrised** to `kf < 4 ? f <= kf : f >= kf` (identical to SF for kingside kings), oracle mirrored. ✅ **Re-gated after rebuild: file mirror back to exactly the pre-existing 21 @ 5 mp; oracle 0 mismatches (traprook fires 21.0%)** |
| knobs execute (`_v2_positional_spread.py`, all at 100%) | ✅ sibling std 36 → 126 mp; knight a3 −5 → **+136** |

☠️★ **100% (the pawn conversion) is far too large for these terms in v2, and the probe shows the mechanism:**
knight a3 reads +136 almost entirely from MINOR-BEHIND-PAWN, which (as in SF) pays the UNDEVELOPED b1/c1/f1/g1 minors
for standing behind their own start-rank pawns — so 1.Na3 LOSES ~141 mp, against v2's 30 mp knight PST spread. SF can
afford S(18,3) because its PST rewards development far more. ⇒ Every sub-term gets a §I ladder DOWN toward the
positional anchor (≈5-25%) as well as up; no sub-term ships at 100% by default.

**Added after the gates, NOT YET BUILT:** `BEHIND_V2_FORM` (0 = SF11 `S(18,3)` mg-heavy · 1 = Weiss `S(9,32)` eg-heavy, 87/157 mp) —
the references split on this term's phase shape, and the mg-heavy form is what pays undeveloped minors. Detector and
oracle are unchanged (only the value table differs), so the default-0 byte-identity is unaffected; it builds with the
next pause and joins the ladder as `behind25f1` / `behind50f1` arms.

**▶️ NEXT for the placement terms (queued behind the SPRT — games hold the RAM): the §I ladder, one call.**
```
pyrun diagnostics/_eval_accuracy_multi.py N=2500 <shipped v2 knobs> \
  'ARMS=ship:EVAL_ARM=1|outpost10:OUTPOST_V2_PCT=10|outpost25:OUTPOST_V2_PCT=25|outpost50:OUTPOST_V2_PCT=50|outpost100:OUTPOST_V2_PCT=100|
   reach10..100|behind10..100|badb10..100|longdiag10..100|traprook10..100|weakq10..100'
```
Each term alone on the shipped base, 10 / 25 / 50 / 100% (100 = pawn conversion). Then the survivors as a bundle, §I +
regret vs a same-session neutral, and a regression-bundle SPRT on top of whichever mobility setting wins its games.
**Registered predictions (2026-09-14, before any ladder number):**
1. **Minor-behind-pawn is HARMFUL at 100% on §I** and best at ≤25% or 0 — it pays undeveloped minors (the knight-a3 row).
2. **Bad bishop improves §I monotonically at low %** (it fires in 76% of positions, so it has the most signal to add) and
   degrades by 100%.
3. **Trapped rook and weak queen are near-inert on §I** at every % (fire 21% / 11%, small per-position effect).
4. **Outpost (knight ×2) helps at 25-50%**, with the worst-case column on the variant corpus.
5. ⚠️ Calibration: six consecutive magnitude predictions have been too optimistic. If these read smaller than
   predicted, that is the expected direction of my error, not a surprise.

**§I PLACEMENT LADDER — RESULT (2026-09-14, base = shipped v2 + `MOB_V2_MAG=600`, 6 corpora, worst-case decides):**

| term | 10% | 25% | 50% | 100% | verdict |
|---|---|---|---|---|---|
| **outpost** | −0.06 / −0.01 | −0.15 / −0.01 | −0.28 / −0.02 | **−0.46 / −0.03** | ✅ negative on 6/6 at EVERY level, still improving at 100 |
| reachable outpost | −0.01 / +0.03 | −0.01 / +0.07 | −0.02 / +0.14 | +0.01 / +0.34 | ❌ no gain, worst grows |
| behind (SF S(18,3)) | −0.04 / +0.03 | −0.11 / +0.07 | −0.19 / +0.15 | −0.30 / +0.44 | ⚠️ worst on variant |
| **behind (Weiss S(9,32))** | — | **−0.17 / +0.05** | −0.34 / +0.10 | — | ★ dominates SF's form at equal % on mean AND worst |
| **bad bishop** | −0.14 / −0.11 | −0.34 / −0.24 | **−0.63 / −0.37** | −1.04 / −0.29 | ✅ negative on 6/6 at every level; worst peaks at 50 |
| long diagonal | −0.07 / +0.06 | −0.16 / +0.15 | −0.28 / +0.31 | −0.37 / +0.63 | ❌ KS-critical corpus degrades at every level |
| **trapped rook** | **−0.18 / −0.01** | **−0.41 / +0.03** | −0.73 / +0.23 | −1.07 / +1.11 | ✅ largest mean mover; variant corpus turns harmful above 25 |
| weak queen | −0.05 / +0.01 | −0.12 / +0.02 | −0.21 / +0.04 | −0.33 / +0.17 | ~inert, inside the floor to 50 |

**Registered predictions scored:** (1) behind harmful at 100 — PARTLY WRONG: mean improves, worst degrades; the eg-heavy
form is better. (2) bad bishop improves low, degrades by 100 — MOSTLY RIGHT on the worst column (peaks at 50); mean still
improving. (3) trapped rook + weak queen near-inert — WRONG for trapped rook (the biggest mean mover), right for weak
queen. (4) outposts best at 25-50 with variant worst — WRONG: monotone to 100, worst ≈ 0 everywhere.
**Survivors → bundle check** (single-term §I says nothing about cancellation, which is exactly how v1's bundling failed):
outpost 100 · bad bishop 50 · trapped rook 25 · weak queen 25 · behind 25 (Weiss form). Dropped: reachable outpost, long diagonal.
**Registered before the bundle result:** sum of the five single-term means = −0.46 −0.63 −0.41 −0.12 −0.17 = **−1.79%**.
PREDICTION: bundle A reads **−1.4% to −1.8%** (mild sub-additivity; the terms touch different pieces, and mobility × rook
files was exactly additive), worst ≤ +0.1%. ☠️ **If it reads below ~−1.3% (≥ 26% cancellation, v1's measured figure),
the bundle is cancelling, and the members go to the regret gate individually rather than together.**
(Wording fix: "below ~−1.3%" means WEAKER than −1.3%, i.e. a magnitude under 1.3.)

**BUNDLE RESULT (same base):**
| arm | mean | worst | vs sum of parts |
|---|---|---|---|
| A: outpost 100 · badb 50 · traprook 25 · weakq 25 · behind 25 (Weiss) | −1.74% | **−0.69%** (6/6 better) | −1.79 predicted ⇒ **97% additive** ✅ |
| **B: A with badb 100** | **−2.11%** | **−1.14%** | best on BOTH columns |
| C: A without behind | −1.58% | −0.74% | behind is worth −0.16 mean |
| D: A with traprook 10 | −1.51% | −0.64% | traprook 25 > 10 |
| E: A + longdiag 10 | −1.81% | −0.63% | +0.07 mean for a worse worst ⇒ long diagonal stays OUT |

★★ **The placement terms ADD — and reinforce the worst case.** Every bundle is negative on all six corpora and its worst
case is better than any member's alone: the opposite of v1's measured 26% cancellation, and consistent with v2's
disjoint-term design (each term reads a different piece and a different feature). ⚠️ This is the bundling refutation's
stated BOUNDARY (it was measured on v1's degenerate terms), now tested from the other side on §I — games still decide.
★ Bad bishop's worst case peaked at 50 ALONE but improves to 100 INSIDE the bundle — a term's optimum is not portable
from solo to bundle; always re-ladder the member that sits at an interior optimum.
▶️ **Candidate: bundle B** on top of mobility 600 → d7 regret gate vs same-session neutrals on BOTH corpora (new
neutrals: nulls are eval-specific) → regression-bundle SPRT.

**d7 REGRET, bundle B vs the mobility-600 base, primary corpus (15,000):**
| split | changed | win% | delta |
|---|---|---|---|
| **all** | 5,646 (37.6%) | **50.2%** (2,594 / 2,570) | −0.016 |
| opening · midgame · endgame | 1,825 · 2,655 · 1,166 | **48.7** · 50.8 · 51.4 | +0.04 · +0.07 · −0.30 |
| n_crit | 29 | 51.9% | unreadable |
**Same-session NEUTRAL on the mobility-600 base** (`ASPIRATION_DELTA=300`): **49.8%** (4,709 changed · opening 50.9 ·
midgame 49.5 · endgame 49.2 · n_crit 24).
☠️ **PRIMARY VERDICT: the placement bundle is a regret NULL — +0.4pp over its measured neutral (~0.4σ).** No gain, no
overall harm. By phase: opening **−2.2pp** (48.7 vs 50.9, ~1.2σ, unresolved) · midgame +1.3 · endgame +2.2.
⚠️ The mobility base moved the neutral from 48.4 to 49.8 — a fresh null per base was necessary again.
▶️ Cross-set `_v2` running. If the OPENING lean replicates there, minor-behind-pawn and bad bishop (the two terms with an
opening-phase mechanism) get solo regret runs before the bundle goes to a regression SPRT.

**CROSS-SET `_v2` (11,940), bundle B vs the mobility-600 base:**
| split | changed | win% | delta |
|---|---|---|---|
| **all** | 4,574 (38.3%) | **48.7%** (2,032 / 2,140) | **+0.1402** |
| opening · midgame · endgame | 1,522 · 2,131 · 921 | 49.7 · **48.3** · **47.8** | +0.18 · +0.04 · +0.31 |
| cr2_minor · cr3_moderate · n_crit | 651 · 160 · 30 | **45.4** · 47.1 · 44.8 | +0.85 · +0.90 · +0.03 |
| ps2_mid_EDGE | 382 | **41.7** | +0.91 |
⏳ **Pending `_v2`'s own neutral on this base** (running; on the pre-mobility base it was 50.8).
⚠️ **The lean MOVED between corpora:** primary leaned negative only in the OPENING; `_v2` is fine in the opening and leans
negative in the midgame, endgame and the moderately critical strata. Inconsistent location usually means noise — but if
v2's neutral lands near 50.8, this is ~−2pp, which is a harm signal at the cross-set bar.
▶️ If harm is confirmed: the bundle does NOT go to games as-is. Owner's rule — rethink OUR implementation first; a
leave-one-out regret run per member (5 runs, one corpus) locates it. The opening-only hypothesis (behind / bad bishop) did
NOT replicate, so leave-one-out must cover all five members, not just those two.

**`_v2` NEUTRAL on the mobility-600 base: 50.7%** (3,851 changed · opening 52.3 · midgame 49.8 · endgame 50.3 ·
cr2 51.2 · `ps2_mid_EDGE` **43.0** · n_crit 29).

☠️ **CROSS-SET VERDICT — placement bundle B on the d7 regret gate:**
| corpus | neutral | bundle B | edge | ~σ |
|---|---|---|---|---|
| primary | 49.8 | 50.2 | **+0.4pp** | 0.4 |
| `_v2` | 50.7 | 48.7 | **−2.0pp** | 1.7 |
**Neither a gain nor CONFIRMED harm** — the corpora disagree in sign and the −2.0 does not replicate on primary. But it is
a harm LEAN at about the cross-set bar, set against a strongly positive §I (−2.11%, 6/6): **the instruments CONFLICT ⇒ no
games on bundle B as-is** (corroboration rule). ⚠️ `ps2_mid_EDGE` 41.7% is NOT bundle harm — the neutral itself reads 43.0
there. The v2 lean is broad (opening −2.6 · midgame −1.5 · endgame −2.5 · cr2 −5.8 at ~2σ).
▶️ Owner's rule applied — rethink the implementation before blaming concepts all references carry: leave-one-out on `_v2`
(below). d7 regret is fixed-depth, so it stays valid during the owner's games.

**Pre-registered leave-one-out plan (runs ONLY if `_v2`'s neutral confirms harm; written before any such run):**
(Update: harm is a LEAN, not confirmed — run anyway, because it is cheap, it is the rethink-first step, and it
refines the bundle either way.)

**Leave-one-out results (`_v2`, same-session neutral 50.7, full bundle B 48.7):**
| arm | changed | win% | vs full bundle | vs neutral | delta |
|---|---|---|---|---|---|
| **minus bad bishop** | 4,401 | **49.9** | **+1.2pp** | −0.8pp (in noise) | −0.08 (was +0.14) |
| **minus trapped rook** | 4,699 | **50.2** | **+1.5pp** | −0.5pp (in noise) | −0.05 |
| minus outpost | 4,521 | 49.6 | +0.9pp | −1.1pp | +0.04 |
| minus weak queen | 4,732 | 49.9 | +1.2pp | −0.8pp | −0.09 |
| minus behind | 4,667 | 50.4 | +1.7pp | −0.3pp | −0.10 |

**FINAL LEAVE-ONE-OUT VERDICT (all five in):** recoveries +0.9 (outpost) · +1.2 (bad bishop) · +1.2 (weak queen) · +1.5
(trapped rook) · +1.7 (behind) — a 0.8pp band against ~1.1pp SE per arm. **No member carries the v2 lean; it is diffuse,
at noise level, and consistent with dilution alone.** Behind-pawn is only NOMINALLY the largest (and it is the one member
whose shape was an ours-vs-references choice). Pre-registered prediction (trapped rook #1, bad bishop #2) is NOT
supported — neither stands out from the dilution band.
**BUNDLE D (outpost 100 · bad bishop 50 · trapped rook 10 · weak queen 25 · behind 25 Weiss) — d7 regret, `_v2`:**
| arm | changed | win% | edge vs `_v2` neutral (50.7) | delta | opening · midgame · endgame |
|---|---|---|---|---|---|
| bundle B | 4,574 | 48.7 | −2.0pp | +0.14 | 49.7 · 48.3 · 47.8 |
| **bundle D** | 4,620 | **50.7** | **0.0pp** | **−0.14** | 50.7 · 50.0 · 52.4 |
✅ **D is CLEAN on `_v2`**: the −2.0pp lean is gone at the lower magnitudes and the mean regret delta flips to better. vs
the neutral by phase: opening −1.6 · midgame +0.2 · endgame +2.1 — all noise, nothing consistent. §I for D: −1.51% mean,
worst −0.64%, 6/6. ⇒ "no harm", which is the small-terms policy's bar for a regression SPRT (not a gain claim).
**Primary-corpus D (15,000):** 5,570 changed · **48.6%** vs the base neutral 49.8 ⇒ **−1.2pp (~1σ)** · opening 49.1 ·
midgame 48.5 · endgame 48.3 · n_crit 24.
✅ **By the pre-registered rule (not ≥ ~2pp below neutral), D passes on BOTH corpora** (`_v2` 0.0pp, primary −1.2pp) ⇒
tonight's fallback candidate stays **bundle D**, not A.
⚠️ The lean keeps MOVING: bundle B was +0.4 primary / −2.0 `_v2`; bundle D is −1.2 primary / 0.0 `_v2`. A sign that wanders
between corpora and arms is the signature of diffuse noise, not a term effect — D averages ≈ −0.6pp from neutral across both.
▶️ Decision rule for tonight (fixed before bundle D's reads): D clean on both corpora → regression SPRT on D; D also leans
negative → the lean is inherent to the term set, not its magnitude → regression SPRT on bundle A, instruments-conflict
flagged for the owner's veto.
★ Removing bad bishop recovers ~60% of the gap to the neutral and flips the regret delta to negative — **bad bishop @100
is at least a major carrier of the v2 lean** (pre-registered suspect #2). Consistent with its SOLO worst-case optimum
being 50: it reached 100 only on bundle §I. Not closed until the other members are in.
☠️ **CORRECTED with the outpost arm (below): the next paragraph OVERCLAIMS.** Removing ANY member shrinks the bundle's
footprint, so some recovery toward the neutral is expected from DILUTION alone. Outpost — the member with the cleanest §I
record — still recovers +0.9pp when removed, which sets roughly the dilution-only floor. Bad bishop (+1.2) and trapped rook
(+1.5) exceed it by only 0.3-0.6pp, inside one arm's SE (~1.1pp). ⇒ **Leave-one-out cannot resolve a carrier; the v2 lean
is diffuse and at noise level.** "Shared by the two largest members" is NOT supported. Bundle D stays the next test as a
lower-magnitude rethink, not as a fix for a proven culprit.
✅ **CONFIRMED by the weak-queen arm:** weak queen is near-INERT on §I (−0.12% at 25), yet removing it recovers +1.2pp —
exactly what removing bad bishop recovers. All four recoveries (+0.9 · +1.2 · +1.2 · +1.5) sit within one SE of each other.
⇒ the v2 lean is carried by NO member; "shared by the two largest" is REFUTED, not merely unsupported.
★ Transferable: **a leave-one-out on a bundle needs a dilution control** — compare each removal against the recovery from
removing the most innocent member, never against the full bundle alone.
~~★★ **With trapped rook in: the lean is SHARED, not one culprit.**~~ Removing trapped rook recovers ~75% of the gap, removing
bad bishop ~60% — each alone leaves the bundle within noise of the neutral. Both pre-registered suspects carry it, and
they are the two largest-magnitude members. ⇒ **Rethought implementation: keep both, lower both** — bad bishop 50 +
trapped rook 10, which is §I bundle D (−1.51% mean, worst −0.64%, 6/6). It trades ~0.6pp of §I mean for, predicted, a
clean regret read. Queued after the remaining leave-one-out arms.
★ Transferable: §I kept raising both terms' magnitude inside the bundle (badb 50 → 100 was better on BOTH §I columns); the
move-level instrument says that is where the harm starts. **A bundle's §I optimum overshoots its move-level optimum** —
the anti-correlation family, visible term by term.
Corpus `_v2` (where the lean is). Base = mobility 600. Five candidate arms, each = bundle B minus one member:
`-outpost` · `-badb` · `-traprook` · `-weakq` · `-behind`. The member whose REMOVAL raises win% most (toward/above the neutral)
is the harm carrier. Read each against the same-session `_v2` neutral already measured.
PREDICTION (ranked): **(1) trapped rook @25** — biggest §I mean mover, variant corpus already harmful above 25, and `_v2`'s
worst stratum `ps2_mid_EDGE` (41.7%) is the mid/end boundary where rook-trapping positions cluster · **(2) bad bishop @100**
— above its SOLO worst-case optimum (50), taken to 100 only on bundle §I · (3-5) outpost / weak queen / behind, near-inert.
⚠️ If NO single removal restores the neutral, the harm is an INTERACTION, not a member — then fall back to bundle A
(bad bishop 50) and re-run the gate, rather than hunting a culprit that does not exist.
⚠️ **§I (−2.11%) and regret (≈flat) are diverging for the placement bundle**, where they agreed for mobility.
Under the small-terms policy the bundle does not need to show a GAIN, only no HARM (regression SPRT, ≥ −10 Elo passes) —
so a flat read does not disqualify it; a read clearly BELOW the neutral would.
⚠️ Watch the OPENING split (48.7): it is where minor-behind-pawn and bad bishop act most. If the neutral confirms a weak
read, those two members get their own regret runs before the bundle goes to games.

#### 2.3.2 Owner review (2026-09-14): efficiency, and best-definition-per-term — what was true, and what follows
Honest status at the owner's question ("extremely efficient, correct, not SF copies, uniquely ours unless universally backed"):
- **Correct: YES, verified** (oracle 0 mismatches both x-ray paths, symmetry clean after symmetrising trapped rook, byte-identity).
- **Efficient: NOT as written.** `placement_detect` broke the design's own "no second attack pass" rule: trapped rook recomputed
  each rook's attacks and the mobility area that the mobility loop had already built; reachable outpost recomputes knight attacks.
  ✅ **Fixed (code written, rebuild pending): `MobAcc` keeps per-rook area counts; trapped rook reads them** (fallback kept for
  mobility-off). Byte-identical by construction; proved by the bundle-D WAC fingerprint before vs after the rebuild, and the probe
  now exercises the reuse path so the oracle validates the path search runs. Reachable outpost (dropped) and weak queen's two
  empty-board lookups were left as they are.
  **Pre-fix reference, bundle D on shipped v2 (mobility 600 + KPK exact), current .so: WAC d10 `248 / 57,474,821 / EBF 4.031`**
  — the rebuilt engine must reproduce it EXACTLY. (Side reading: −4.3% nodes vs shipped v2's 60,036,572; solves in the floor.)
- **Uniquely ours: ONLY PARTLY.** Detector definitions are largely SF11's; magnitudes are ours (ladders). Minor-behind-pawn uses
  Weiss's shape (chosen by measurement). ☠️ Gaps: bad bishop used SF11's multiplier WITHOUT testing the other three reference forms
  (references split ⇒ theirs are candidates, per the adoption rule); trapped rook (3/5, SF lineage) and weak queen (3/5) are NOT
  universally backed — they entered as measured candidates.

▶️ **Owner direction: efficiency first; then pick the best definition per term from ANY engine, or ours if better.**
Plan: (1) tonight's regression SPRT runs bundle D with the efficiency fix — identical values, so it tests "is this concept set
harmless"; (2) per-term FORM knobs from source (research running: Ethereal defended-indexed outposts + rammed bad bishop + queen
relative pin, SF15.1 uncontested outpost + pawn-defended bad-bishop zeroing, Weiss product-form bad bishop, SF1.1 linear trapped
rook) laddered head-to-head on §I + regret; (3) OUR candidates alongside: v1's latent bishop/rook activity (x-ray scope through own
pieces — distinctive, no SF analogue); (4) a form replaces D's only if it beats it on both instruments → its own small bundle test.

**PER-TERM FORMS FROM SOURCE (research 2026-09-14; SF1.1/11/15.1 local, Ethereal/Weiss fetched — function names, no line numbers):**
| term | forms | one-owner / collinearity note |
|---|---|---|
| outpost | span: refined (SF11/15 exclude backward/blocked enemy pawns) vs raw (SF1.1, Ethereal) — 2 vs 2 · eligibility: defended required (SF11) / defended OR pawn-in-front (SF15.1) / defence-INDEXED `[outside][defended]` (Ethereal: `KnightOutpost {{12,-32},{40,0}},{{7,-24},{21,-3}}`, `BishopOutpost {{16,-16},{50,-3}},{{9,-9},{-4,-4}}`) / square tables + exchange bonus (SF1.1) · SF15.1 `UncontestedOutpost S(0,10)×pawns-on-half` REPLACES Outpost for side-file knights with ≤1 enemy piece on their half | overlaps knight mobility — gate it |
| behind pawn | identical predicate in 4/5 (any-colour pawn in front); only phase shape differs (SF mg-heavy vs Ethereal N `S(3,28)` / B `S(4,24)`, Weiss `S(9,32)` eg-heavy) | already laddered; Weiss shape won |
| bad bishop | SF11 `S(3,7)·N·(1+blk)` · SF15.1 file table `{S(3,8),S(3,9),S(2,7),S(3,7)}·N·(!pawnDefended+blk)` · **Weiss `S(-1,-5)·N·blk` pure product** (zero with no blocked central pawn) · Ethereal `S(-8,-17)·N_rammed` (own pawns on the bishop's colour with an ENEMY pawn in front) | ★ the Weiss and Ethereal forms fire only in CLOSED structures ⇒ far less overlap with bishop mobility than SF11's |
| trapped rook | SF1.1 linear `180−16·mob` (mg), mob ≤ 6, king on rank 1 or rook's rank, no own half-open file toward the edge, halved if can castle · SF11/15 step `S(52,10)/S(55,13)×(1+!castle)` at mob ≤ 3 · **Ethereal and Weiss: NONE — only the mobility floor** (Ethereal 0-mob rook `S(-127,-148)`) | ☠️ **DOUBLE-PAY with mobility's negative floor** — SF pays twice, 2/5 engines pay once through mobility. One-owner rule ⇒ mobility owns it; trapped rook is dropped or REDEFINED to cover only what mobility cannot see (SF1.1's king-rank + no-open-file-to-edge geometry is the candidate) |
| weak queen | SF (single blocker of either colour, snipers removed) ≈ Ethereal `QueenRelativePin S(-22,-13)` via `discoveredAttacks` (first blocker removed) — differ only on an R–R battery with no blocker | near-identical; inert on §I |
| long diagonal | SF ≡ Ethereal as detectors; Ethereal adds eg weight `S(26,20)` | stays dropped (KS-critical harm) |
| king protector | SF linear own-king distance vs Ethereal `KnightInSiberia` dead-band ≥4 from the NEARER king | not ported (overlaps KS zone) |
| bishop pair | 5/5 carry it; **not in v2 yet** | slated for slice 3 (Kaufman/pairs) |
Nothing else clears the ≥3-engine bar (bishop x-ray pawns, rook/bishop-on-king-ring are SF15.1-only; rook on 7th 2/5).

**★ COLLINEARITY GATE — FIRST RUN (2026-09-14, 🧰 `_v2_term_collinearity.py`, 10,000 positions over 4 corpora, White − Black detector values):**
| placement term | VIF | largest cross-correlation |
|---|---|---|
| outpost_n | 1.10 | mob_N +0.23 |
| outpost_b | 1.05 | mob_B +0.20 |
| reach_n | 1.15 | mob_N +0.31 |
| behind | 1.11 | ≤ 0.16 |
| **badb_units** | 1.30 | **mob_B +0.38** (largest in the set) |
| longdiag | 1.08 | mob_B +0.22 |
| **traprook_units** | 1.10 | **mob_table_eg −0.25 · mob_table_mg −0.22 · mob_R −0.20** |
| weakq | 1.00 | ≈ 0 |
✅ **PASS — no cross-subsystem pair reaches \|r\| ≥ 0.7 and no placement term reaches VIF ≥ 5.** The placement terms are
essentially independent of mobility and of each other. The only VIF flags are mobility's own counts vs its own table sums
(same detector by construction; excluded by design).
☠️ **Withdrawn: the "trapped rook double-pays mobility's negative floor" concern.** Measured r −0.25 (~6% shared variance):
SF's step test mostly fires on positions the mobility floor does not. The concern was conceptual; the data does not
support it. Bad bishop overlaps bishop mobility the most (r +0.38, ~14%), still far from collinear.
⚠️ Boundary: position-level W−B overlap, not sibling-move-difference overlap; KS and pawn structure have no count probe yet,
so overlaps with them are NOT covered. ⇒ bundle D now passes §I + regret + non-collinearity.

**FORM KNOBS — CODE WRITTEN 2026-09-14 (evening), NOT YET BUILT OR GATED.** All default 0 = the SF11 forms already gated
⇒ byte-identical by construction (every form-0 path reduces to the previous expression); proved after the rebuild by the
bundle-D WAC fingerprint `248 / 57,474,821` and the shipped `250 / 60,036,572`.
| knob | forms | constants (each leg by its own engine's pawn) |
|---|---|---|
| `OUTPOST_V2_FORM` | 0 SF11 · **1 Ethereal** raw span, `[outside][defended]` cells · **2 SF15.1** defended OR pawn in front, no knight ×2 | Eth N mg {146,488,85,256} eg {−222,0,−167,−21} · B mg {195,610,110,−49} eg {−111,−21,−63,−28} · SF15 N 429/163 · B 246/120 |
| `BADB_V2_FORM` | 0 SF11 N(1+blk) · **1 SF15.1** N(!defended+blk) by file class · **2 Weiss** N·blk · **3 Ethereal** rammed-only | SF15 mg {24,24,16,24} eg {38,43,34,34} · Weiss 10/25 · Eth 98/118 |
| `TRAPROOK_V2_FORM` | 0 SF11 step · **1 SF1.1** linear 180−16·mob, mob ≤ 6, king rank, no own open file to the edge, halved with castling | raw SF1.1 units ×1000/204, mg only |
| `LATENT_V2_PCT` | **OURS** — latent squares behind own blockers that attack an enemy pawn (bishops, rooks) | 100 = v1's 15 / 10 mp, mg only |
Both trapped-rook forms share one helper (`trap_rook_units`) on both the reuse and fallback paths, keeping the
file-symmetrised side test. The probe widens to 20 entries (latent added; Ethereal cells and SF15.1 classes bit-packed).
🧰 `_placement_detector_oracle.py` rewritten to mirror every form and the latent term independently, including the packing.
▶️ Gates after the rebuild: oracle under each form (and XRAY=0 once) · symmetry with each form on · byte-identity at defaults
· collinearity with the forms on · then the §I form ladder head-to-head.

#### 2.3.3 OVERNIGHT QUEUE (2026-09-14 night) — written BEFORE the chain starts, run unattended step by step
Each step launches when the previous one notifies; every launch uses the auto-approved runner form (no script files, no `$`).
**SHIP** = `EVAL_ARM=1 KS_V2_ZONE_SF=1 KS_V2_XRAY=1 KS_V2_COORD=256 KS_V2_WEAK=57 KS_V2_ADJ=61 KS_V2_NO_QUEEN=321 KS_V2_CHK_Q=126 KS_V2_CHK_R=122 KS_V2_CHK_B=80 KS_V2_CHK_N=152 KS_V2_MAX=4000 KS_V2_HALF=600 KS_V2_ONSET=450 PS_V2_MAG=100 PASSER_V2_MAG=60 DRAW_V2_CLASS=1 MOB_V2_MAG=600 DRAW_V2_KPK_EXACT=1`
**D** = `OUTPOST_V2_PCT=100 BADB_V2_PCT=50 TRAPROOK_V2_PCT=10 WEAKQ_V2_PCT=25 BEHIND_V2_PCT=25 BEHIND_V2_FORM=1`

| # | step | pass criterion | on failure |
|---|---|---|---|
| 1 | primary-corpus D regret finishes | not ≥ ~2pp below the 49.8 neutral | fallback candidate becomes bundle A (flagged) |
| 2 | `build` | compiles | stop; no games; note for the owner |
| 3a | WAC d10: arm 0 · SHIP · SHIP+D | `250/35,310,778` · `250/60,036,572` · `248/57,474,821` EXACTLY | any mismatch ⇒ STOP, no games (a default path changed) |
| 3b | oracles: placement at forms 0 · OUTPOST 1 · OUTPOST 2 · BADB 1 · 2 · 3 · TRAPROOK 1 · XRAY=0; mobility once (`MobAcc` changed) | 0 mismatches, non-vacuous | that form is excluded from the ladder; D unaffected |
| 3c | `_eval_symmetry.py` N=800 with each non-default form on | colour 0 violations; file mirror = the pre-existing 21 | that form excluded |
| 4 | §I form ladder, one call, base SHIP+D, one form swapped per arm: OUTPOST 1 @100/@50 · OUTPOST 2 @100/@50 · BADB 1/2/3 @50/@100 · TRAPROOK 1 @10/@25 · LATENT @100/@300/@1000 | a form replaces D's only if better on BOTH mean and worst | keep D's form |
| 5 | bundle E = D with the winners; collinearity gate with E's forms on | no \|r\| ≥ 0.7, no placement VIF ≥ 5 | drop the flagged winner, revert to D's form |
| 6 | d7 regret for E (if E ≠ D), primary + `_v2`, against the measured neutrals 49.8 / 50.7 | no lean ≥ ~2pp below neutral on either corpus | SPRT falls back to D |
| 7 | runner edit: `sprt_ab` gains an optional `elo0` arg (default 0 ⇒ every existing caller unchanged) — only when no runner job is executing | — | — |
| 8 | regression SPRT: SHIP+E (or D) vs SHIP · `elo0 -10 elo1 0` · LIGHTNING · conc 4 · `openings_uho.txt` · seed 13 · max 1200 | H1 = costs ≤ ~10 Elo (the policy bar, not a gain claim) | — |
If E = D (no form wins at step 4), skip 5-6 and run step 8 on D.

**Queue progress:**
- Step 1 ✅ primary-corpus D −1.2pp vs neutral (passes; fallback stays D).
- Step 2 ✅ build (efficiency fix + form knobs + latent term).
- Step 3a: SHIP+D **`248 / 57,474,821` EXACT** ✅ · SHIP **`250 / 60,036,572` EXACT** ✅ · arm 0 **`250 / 35,310,778` EXACT** ✅.
- Step 3b: placement oracle at default forms ✅ **0 mismatches / 30,100 checks**, every term fires incl. latent (bishops 31.8%, rooks
  15.5%) · `OUTPOST_V2_FORM=1` (Ethereal, packed cells) ✅ 0 mismatches · **all forms ✅**: OUTPOST 2 + BADB 1 (packed classes)
  + TRAPROOK 1 in one run (0 mismatches; fire 13.5/12.1% · 72.9% · 26.3%) · BADB 2 Weiss with XRAY=0 (0 mismatches; 52.7%, and the
  trapped-rook FALLBACK path 21.1%) · BADB 3 Ethereal rammed (0 mismatches; 36.1%). ⇒ **150,500 term-checks across 5 runs, 0
  mismatches, every form non-vacuous.** Mobility oracle re-check (`MobAcc` changed) ✅ 0 mismatches / 6,008 side-positions · arm 0 WAC ⏳.
- Step 3c symmetry (all placement terms at 100%, behind Weiss, latent 300%): group 1 = OUTPOST 1 + BADB 1 + TRAPROOK 1 ✅ colour
  0/800, file mirror = the pre-existing 21 @ 5 mp · group 2 (OUTPOST 2 + BADB 2) ✅ same · group 3 (BADB 3) ✅ same.
- ✅ **STEP 3 COMPLETE — every gate passed.** Byte-identity exact on all three arms; every form + latent oracle-verified (0 mismatches);
  mobility oracle re-checked; symmetry clean for every non-default form. ⇒ the form ladder is valid and bundle D remains a safe fallback.
- Step 4 ✅ **§I FORM LADDER (base = SHIP + D, one swap per arm; rule: replace only if better on BOTH mean and worst):**

| arm | mean | worst | verdict |
|---|---|---|---|
| outpost Ethereal @100 / @50 | −0.14 / +0.07 | +0.25 / +0.35 | ❌ |
| outpost SF15.1 @100 / @50 | −0.34 / −0.02 | +0.23 / +0.45 | ❌ variant corpus degrades |
| **bad bishop SF15.1 @100** | **−0.37** | **−0.12** | ✅ **WINNER — better on 6/6** |
| bad bishop SF15.1 @50 | +0.01 | +0.14 | ❌ |
| bad bishop Weiss @50 / @100 | +0.34 / +0.11 | +0.50 / +0.26 | ❌ |
| bad bishop Ethereal @50 / @100 | +0.63 / +0.70 | +1.09 / +1.38 | ❌ clearly worse |
| trapped rook SF1.1 @10 / @25 | −0.01 / −0.26 | +0.06 / +0.02 | ❌ strict rule; **@25 a near-miss inside the ±0.05 floor — candidate for a later ladder** |
| **LATENT (ours)** @100 / @300 / @1000 | +0.01 / +0.05 / +0.40 | +0.07 / +0.24 / +1.17 | ❌ **no gain; harmful as it grows** |

★ **Bundle E = D with `BADB_V2_FORM=1 BADB_V2_PCT=100`** (SF15.1 bad bishop: the `1` in the multiplier becomes 0 for a pawn-defended
bishop, plus a file-class table). ⚠️ Part of its gain may be effective MAGNITUDE rather than shape — SF11@100 was not in this ladder, and
bundle B (SF11@100) leaned on regret; E's regret gate is what decides.
☠️ **Our LATENT term does not earn a place on §I** — neutral at v1's scale, harmful above it. It stays built and OFF; recorded as a
measured null for the ours-first candidate, not buried. Weiss and Ethereal bad-bishop forms (the least mobility-correlated in theory) are
clearly worse in v2.
- Step 7 ✅ runner: `sprt_ab` takes an optional 9th arg `elo0` (default 0 ⇒ existing callers unchanged); wired into the `sprt.py` call on
  the single-line `sprt_ab` invocation only — the `gate` sub's identical `--elo0 0` is untouched. ☠️ A blind replace would have edited
  both (the first attempt refused on 2 matches).
- Step 5 ✅ **collinearity gate with `BADB_V2_FORM=1`: PASS** (script unpacks the form-packed probe fields first). SF15.1 bad bishop
  **VIF 1.25, r +0.33 with bishop mobility** — slightly LESS collinear than SF11's form (1.30 / +0.38). No cross-subsystem flags; the
  only VIF flags are mobility's counts vs its own table sums (same detector, excluded by design).
- Step 6 ✅ **E regret CLEAN on both corpora** (JOBS=1 each): primary `placeE_primary` **49.9%** vs neutral 49.8 (**+0.1pp**; 5,832
  changed / 38.9%, delta −0.022) · `_v2` `placeE_v2set` **49.2%** vs neutral 50.7 (**−1.5pp**; 4,713 / 39.5%, delta +0.103). Neither
  ≥ ~2pp below ⇒ E passes the pre-registered rule. vs D (−1.2 / 0.0): primary better, `_v2` worse — opposite signs across corpora, both
  inside the cross-set bar, so E and D are INDISTINGUISHABLE on regret; E was preferred on §I (better 6/6). `n_crit` 24 / 29 — unreadable.
  ⚠️ `_v2` `cr2_minor` 45.0% (657 changed) is the largest single-stratum lean; noted, not actionable at this n.
- Step 8 ▶️ **regression SPRT launched on SHIP + E** (2026-09-15): `sprt_ab … s2_placeE_regress 1200 0 4 openings_uho.txt 13 -10`
  (elo0 −10, elo1 0 ⇒ H1 = costs ≤ ~10 Elo).
  **RESULT — INCONCLUSIVE at the 1,200-game cap, positive lean.** Segment 1 (seed 13) **+103 −83 =36 / 222**, LLR +0.804 — killed
  03:29 by a Windows Update restart. Segment 2 (seed 14, same `.so`, written 09-14 23:53 before launch ⇒ poolable) **+396 −380 =202 / 978**,
  elo +5.7 ±25.6, LLR +1.091. **Pooled +499 −463 =238 / 1,200 = 51.5% ⇒ ≈ +10 Elo (≈ ±23), LLR ≈ +1.9** (segment sum; bound +2.94).
  The pooled LLR never went negative; never near the −2.94 "costs > 10 Elo" bound. Segment 1's +31 was small-sample; the long segment
  settled at +6. Reading: consistent with E being free to mildly positive; NOT a formal H1. Ship decision is the owner's
  (options: ship on this evidence, or a third segment ~800-1,000 games for a formal pass).
  ★★ **SEGMENT 3 (2026-09-16, seed 15) ACCEPTS H1 ON ITS OWN: `+611 −548 =296 of 1,455 (52.2%), elo +15.1 ±21.0, LLR +3.039`**
  — no pooling approximation needed. Pooled over all three segments: **+1,110 −1,011 =534 of 2,655 ≈ +13 Elo**.
  ⇒ **Placement bundle E is GAMES-CONFIRMED**, not provisionally shipped. ⚠️ The bound tested is still "does not cost ≥10 Elo"
  (elo0 −10 / elo1 0); +13 is CONSISTENT with a real gain but this design does not prove one. The 09-15 "inconclusive" reading
  is superseded: it was a max-games stop, not evidence of nothing — the same effect simply needed ~2.2× the games to resolve.
Rough timing from step 1: build + gates ~40 min · ladder ~15 · E regret ~2 h at JOBS=2 · SPRT launch ≈ 3 h after step 1.

**OURS-FIRST candidate — `LATENT_V2`, from v1's latent bishop/rook activity (read 2026-09-14,
`cpp_bitboard.cpp` `get_latent_bishop_activity_score` / `get_latent_rook_activity_score`).** No reference-engine analogue.
v1 fuses three ideas:
| part | v1 mechanics | v2 port |
|---|---|---|
| LATENT SQUARES | squares a slider would reach if its OWN blockers in its current attack set were removed (bishop: all own pieces; rook: own NON-pawn pieces), minus squares already attacked | ✅ one extra `attacks_mask` per slider with a modified occupancy |
| heat-map scoring | `attackingLayer/3` on each latent square, plus writes into v1's global off/def/central accumulators | ❌ the heat map is retired in v2 — does not port |
| LATENT PAWN PRESSURE | +15 (bishop) / +10 (rook) per latent square not ours that ATTACKS AN ENEMY PAWN, then +5 per empty square in a second-order scan from it | ✅ the count ports cheaply; the second-order scan is costly — optional, measured |
Proposed v2 form: per bishop and rook, count latent squares that attack an enemy pawn (optionally + second-order empties),
percent-scaled like the other sub-terms, same detector-oracle and gates. It is the owner-distinctive concept of
"pressure that arrives when our own pieces move", kept without the retired heat map.
🐛 v1's rook version still carries the live diagonal-ray bug in its second-order scan (`ENABLE_ROOK_LATENT_RAY_FIX=false`); v2
uses rook rays there by construction.

#### 2.3.1 Specs (prep, 2026-09-14). SF11 constants cited, SF units (pawn mg 128 / eg 213).
⚠️ **Sizing lesson from §3.3 applies here**: mobility's §I optimum sat near the PAWN conversion, not the
positional-spread one. Every sub-term therefore gets a ladder spanning BOTH anchors — the positional-spread value AND
the pawn-converted value — never one guessed point (outposts' only record is one guessed point).

| term | detector (exact) | SF11 value | pawn-converted mp (mg / eg) | agreement / shape candidates | knob (proposed) |
|---|---|---|---|---|---|
| **outpost** | minor on relative ranks 4-6, **defended by own pawn**, and NOT in `pawn_attacks_span(Them)` — ⚠️ SF's span EXCLUDES enemy pawns that are backward or blocked (`pawns.cpp:114-115`), so a square is still an outpost if the only enemy pawn able to challenge it cannot advance | `Outpost S(30,21)`, **knight ×2** (`evaluate.cpp:138,296`) | N 469/197 · B 234/99 | 4/5. Ethereal indexes by `[outside-file][defended]` instead of requiring defence; SF15.1 adds `UncontestedOutpost` override | `OUTPOST_V2_MAG` (+ `OUTPOST_V2_REQ_DEF` = SF vs Ethereal) |
| reachable outpost | knight whose attacks reach an outpost square not occupied by own piece | `ReachableOutpost S(32,10)` knight only | 250/47 | SF lineage | `OUTPOST_V2_REACH` |
| **minor behind pawn** | minor with ANY pawn (either colour) directly in front | `MinorBehindPawn S(18,3)` | 141/14 | 4/5 but **phase OPPOSITE**: SF mg-heavy, Ethereal/Weiss eg-heavy (S(3,28), S(9,32)) ⇒ phase is a knob | `BEHIND_V2_MG`, `BEHIND_V2_EG` |
| **bad bishop** | own pawns on the bishop's colour × (1 + own blocked pawns on files c-f) | `BishopPawns S(3,7)` per pawn (`:130,312-315`) | 23/33 per unit | 4/5, three multipliers: SF11 (1+blocked centre) · SF15 zero if bishop pawn-defended · Weiss product with blocked centre · Ethereal rammed pawns only | `BADB_V2_MAG`, `BADB_V2_FORM` ∈ {SF11, SF15, WEISS} |
| long diagonal | bishop seeing ≥2 of d4/e4/d5/e5 through pawns only (`attacks_bb<BISHOP>(s, pos.pieces(PAWN))`) | `LongDiagonalBishop S(45,0)` | 352/0 | 3/5 (Ethereal requires the bishop ON a long diagonal outside the centre) | `LONGDIAG_V2_MAG` |
| trapped rook | rook mobility ≤ 3, own pawn on its file, rook on the king's edge side (`(kf < FILE_E) == (file_of(s) < kf)`) — ★ **else-branch of RookOnFile** (shadowed when the file is open/semi) | `TrappedRook S(52,10) × (1 + !castling_rights(Us))` (`:148,347-351`) | 406/47 (×2 without castling rights) | 3/5 (SF lineage; SF1.1 a different linear form) | `TRAPROOK_V2_MAG` — needs the per-rook mobility COUNT, which `MobAcc` currently sums per type ⇒ expose the per-piece count first |
| weak queen | an enemy rook/bishop `slider_blockers` pin or discovered attack on our queen | `WeakQueen S(49,15)` (`:149,359-360`) | 383/70 | 3/5 (Ethereal `QueenRelativePin S(-22,-13)`) | `WEAKQ_V2_MAG` — `slider_blockers(queen_sq, …)` is pure (`cpp_bitboard.h:714`) |
| ~~rook on 7th~~ | — | dropped in SF11+ and Weiss | — | 2/5 | not planned |
| ~~king protector~~ | Chebyshev distance of N/B to own king | `KingProtector S(7,8)` per square | 55/38 | 3/5 | defer: overlaps KS zone defence |

⚠️ **Order of the checks per term** (the standing protocol): record-check (outposts HAS history: 150/80 read harmful at
one point on a superseded comparator) → detector vs an independent python-chess mirror on asymmetric positions →
byte-identity at knob 0 → symmetry → §I ladder over both anchors → STS at the §I-best and one neighbour.
⚠️ **Two known interactions to measure, not assume**: outpost × mobility (a knight on an outpost usually has high
mobility — SF pays both) and trapped rook × rook files (SF's else-if exclusivity is a DESIGN choice we could reject).

### 2.4 Contract items (all rungs)
- Pure, zero globals; one parameterised body per side; no clamp; Black-positive.
- **Byte-identity:** `MOB_V2_MAG=0` and `ROOKFILE_V2_*=0` ⇒ the shipped v2 result exactly (and arm 0 untouched).
  ⚠️ Moving the side-attack gate must not change anything when mobility is off — verified by the v2 WAC fingerprint
  `246 / 63,221,361`, not assumed.
- **Breakdown:** publish `mobility` / rook-file score and the per-side raw count sums as detector fields, with EB bits.
- **Detector oracle before any magnitude:** a C++ probe exporting per-side per-type area-filtered counts, compared to an
  independent python-chess implementation on ASYMMETRIC positions only (the 09-13 vacuous-pass lesson).

---

## 3. MEASUREMENT PLAN — predictions registered BEFORE any run

| step | instrument | arms |
|---|---|---|
| A | gates | arm-0 byte-id `250 / 35,310,778 / 3.784` · v2 knobs-off byte-id `246 / 63,221,361` · oracle 0 mismatches · `_eval_symmetry.py` clean · tempo swing still exactly 0.000 |
| B | **2×2** §I (6 corpora, worst-case column) + STS | {mob off, mob @ref} × {rookfile off, rookfile @ref} |
| C | **magnitude ladder** §I + STS | `MOB_V2_MAG` ∈ {30, 60, 115, 300} (= reference ratios 1×, 2×, 3.8×, 10× of v2's knight PST spread) |
| D | area / floor / pin knobs, one at a time at the best rung of C | |
| E | games | `sprt_ab`, node-limited, ≥2 budgets, `openings_uho.txt` + varied seed — only if B-D corroborate |

### 3.1 Step B results — the 2×2 at the reference point (mobility 115, rook files 50/25)

| arm | STS (shipped 1698) | §I mean | §I worst | corpora improved |
|---|---|---|---|---|
| mobility 115 | 1618 (**−80**) | **−1.39%** | +0.35% (KS-critical) | 5/6 |
| rook files 50/25 | 1608 (**−90**) | −0.32% | **−0.06%** | 6/6 |
| both | 1659 (**−39**) | **−1.70%** | +0.29% (KS-critical) | 5/6 |

- ★ **§I is ADDITIVE**: −1.39 + −0.32 = −1.71 vs −1.70 measured. Registered prediction 1 (negative interaction) is
  **refuted on §I**. On STS the pair reads BETTER than either alone (−39 vs −80 / −90) — the opposite sign of the
  predicted interaction — but every cell is inside the ±150 floor, so the STS interaction is UNREADABLE, not positive.
- ★ Mobility's standard-corpus gain (~−2.0% on each of four corpora) is **~2× the whole pawn layer's** (−0.97%) —
  the largest §I term since the pawn taper.
- ⚠️ The worst-case is the KS-critical corpus, +0.35% (7× the floor). Candidate mechanism, UNTESTED: mobility
  re-prices the same attacking pieces KS already counts. Check with `MOB_V2_EXCL_QUEEN` and a KS-on/off pair before
  believing it.
- ☠️ **INSTRUMENTS CONFLICT**: §I strongly positive (28× its floor), STS negative inside its ±150 floor on both
  single-term cells. Corroboration rule ⇒ **no games on this point**; the magnitude ladder decides. Registered
  prediction 3 (+60 to +150 STS at the best rung) is already under strain — the sixth consecutive optimistic
  magnitude prediction if it fails.

### 3.2 Step C results — the magnitude ladder (00:01-00:19)

| `MOB_V2_MAG` | 0 | 30 | 60 | 115 | 300 |
|---|---|---|---|---|---|
| STS (floor ±150) | 1698 | 1642 (−56) | 1617 (−81) | 1618 (−80) | 1634 (−64) |
| §I mean / worst | — | −0.38 / +0.09 | −0.75 / +0.18 | −1.39 / +0.35 | **−3.34 / +0.92** |

| rook files open/semi | 20/10 | 50/25 | 100/50 |
|---|---|---|---|
| STS | 1626 (−72) | 1608 (−90) | 1551 (**−147**) |
| §I mean / worst | −0.13 / −0.02 | −0.32 / −0.06 | −0.64 / −0.11 (all 6 corpora improve) |

**Readings.**
1. ☠️ **Mobility's STS response has NO DOSE RESPONSE**: −56 at 30 mp, −81/−80 at 60/115, −64 at 300. A 30 mp term moves
   siblings by a few mp, so a −56 there is unlikely to be evaluative. Same shape as tempo @25 (−42, "inert"):
   the signature of a PERTURBATION floor, not of the term. ⇒ Null arm `MOB_V2_MAG=3` queued to test exactly that.
2. **§I is exactly LINEAR in magnitude** for both terms (mobility −0.38 → −3.34 over 10×; rook files −0.13 → −0.64 over
   5×), with no optimum yet. The worst case (KS-critical) grows linearly with mobility, but mean gain grows ~3.6× faster.
   ⇒ §I crank at 600 / 1000 / 2000 queued (CRANK UNTIL IT BREAKS).
3. **Rook files are the only GRADED STS response, and it is harmful**: −72 → −90 → −147. §I likes them on 6/6 corpora.
   Monotone harm with size on the move-level instrument is the more credible reading here; do not raise them.
4. Registered prediction 2 (STS interior peak 60-115) **REFUTED** — STS is flat, §I monotone. Prediction 3 (+60 to
   +150 STS) **REFUTED** — the sixth consecutive optimistic magnitude prediction.

### 3.2b STS null arm + node count

| arm | STS | WAC d10 |
|---|---|---|
| `MOB_V2_MAG=3` (values ±1-3 mp — a near-null perturbation) | **1696 (−2)** | — |
| `MOB_V2_MAG=115` | 1618 (−80) | 248 / **61,675,676 (−2.4% nodes)** / EBF 4.054 |

☠️ **My perturbation-floor explanation (§3.2 reading 1) is NOT supported**: a near-null arm reads 0, so "any change
reads −60" is false. The STS response is **STEP-SHAPED**: ~0 at 3 mp, −56 at 30, then FLAT −56..−81 to 300. A step
that saturates fits threshold/margin coupling better than a graded evaluative harm — HYPOTHESIS, untested. STS @10
queued to locate the step.
Registered prediction 5 (nodes fall 5-15%): **direction right, magnitude short** (−2.4%). WAC solves 246 → 248 are inside
its ±5-6 floor and non-discriminating.

### 3.2c THIRD INSTRUMENT — d7 move-regret gate (`_ks_footprint_regret.py`), shipped v2 vs `MOB_V2_MAG=600`

Primary corpus `game_regret_set.csv`, 15,000 positions, JOBS=2, shipped v2 knobs inherited by BOTH arms:

| split | changed | win% of changes | delta |
|---|---|---|---|
| **all** | 6,740 (44.9%) | **53.3%** (3,243 / 2,847) | −0.3086 |
| opening · midgame · endgame | 1,994 · 3,096 · 1,650 | 54.7 · 53.9 · **50.1** | −0.42 · −0.42 · +0.03 |
| `cr4_CRITICAL` (**n_crit = 46**) | 46 | 59.1% | −4.71 |

**Same-session NEUTRAL arm** (`ASPIRATION_DELTA=300`, same corpus, same shipped v2 base):

| arm | changed | win% | opening · midgame · endgame | n_crit |
|---|---|---|---|---|
| neutral | 4,151 (27.7%) | **48.4%** | 48.0 · 49.1 · 47.4 | 30 |
| **mobility 600** | 6,740 (44.9%) | **53.3%** | 54.7 · 53.9 · 50.1 | 46 |

★★ **PRIMARY CORPUS: mobility 600 clears the measured null by +4.9pp** (~4.7σ combined binomial) — about 2× the
~2-2.5pp cross-set bar — and beats neutral in every phase (+6.7 · +4.8 · +2.7).
⚠️ **v2's neutral sits at 48.4, BELOW v1's 49.8-50.4 band** — reading against 50 would have understated the edge.
The null is arm- and eval-specific, exactly as the tool warns; it had to be measured here.
**CROSS-SET `game_regret_set_v2.csv` (11,940 positions, disjoint from primary):**

| arm | changed | win% | opening · midgame · endgame | cr2 · cr3 | n_crit |
|---|---|---|---|---|---|
| **mobility 600** | 5,323 (44.6%) | **53.9%** | 55.0 · 53.7 · **52.6** | 57.1 (n=748) · 55.3 (n=179) | 31 (46.7%) |
| neutral | 3,384 (28.3%) | **50.8%** | 50.4 · 50.2 · 52.2 | 50.1 · 58.5 (n=126) | 32 (50.0%) |

★★★ **REGRET GATE: MOBILITY 600 PASSES, REPLICATED.**

| corpus | neutral | mobility 600 | edge over measured null | opening · midgame · endgame edge |
|---|---|---|---|---|
| primary (15,000) | 48.4 | 53.3 | **+4.9pp** (~4.7σ) | +6.7 · +4.8 · +2.7 |
| `_v2` (11,940, disjoint) | 50.8 | 53.9 | **+3.1pp** (~2.7σ) | +4.6 · +3.5 · +0.4 |

Both clear the ~2-2.5pp cross-set bar with the same sign. The edge lives in the OPENING/MIDGAME and fades to ~neutral in
the endgame on v2 — where a mobility term should act. n_crit 46 / 31 ⇒ the critical tail is unmeasurable; this is an
aggregate result, quoted as such.
⚠️ The two neutrals differ by 2.4pp (48.4 vs 50.8) — the null is corpus-specific on v2 too; reading either corpus
against 50 would have mis-sized the edge in opposite directions.

### 3.4 ★ THE INSTRUMENT PICTURE AT MOBILITY 600 (end of 2026-09-14 overnight)
| instrument | reading | resolves? |
|---|---|---|
| §I (6 corpora) | **−5.75% mean**, worst +1.91% (KS-critical) | ✅ strongly positive, 100× floor |
| d7 regret, win% vs same-session neutral | **+4.9pp / +3.1pp**, replicated | ✅ positive, clears the cross-set bar |
| STS | 1588 (−110) — flat −37..−110 across 10-1000 mp | ❌ inside ±150, does not price the magnitude |
| WAC d10 (@115) | −2.4% nodes, solves in floor | fingerprint only |

⇒ **Two independent instruments agree; the third cannot resolve.** That is the corroboration rule's condition for
spending games. **The decision is the owner's.** Recommended if taken: `sprt_ab` shipped v2 vs `+MOB_V2_MAG=600`
(regret-validated point; §I peaks higher at 1000 but its worst case grows and regret was not run there), node-limited,
≥2 budgets, `openings_uho.txt` + varied seed. Rook files stay OFF for that run: their only graded move-level signal
(STS) was monotone harmful, and their §I gain is small.
⚠️ Open risk to name up front: the KS-critical §I worst case (+1.91% at 600). `MOB_V2_EXCL_QUEEN` trimmed it slightly.
A KS × mobility interaction is plausible and untested.
▶️ If it replicates, the instrument picture becomes **§I strongly + · regret strongly + · STS flat inside its floor** —
two agreeing, one unable to resolve — which is the corroboration rule's condition for spending GAMES (owner's call).

### 3.5 GAMES — `sprt_ab` A = shipped v2 + `MOB_V2_MAG=600`, B = shipped v2 (LIGHTNING, conc 4, `openings_uho.txt`, elo1 5)
Run in SEGMENTS, paused for builds and pooled by adding each segment's final `+W −L =D` (A's perspective). The engine
.so between segments must reproduce the shipped v2 WAC fingerprint `246 / 63,221,361`, or the segments are different arms.

| segment | seed | games | tally (A) | why it ended |
|---|---|---|---|---|
| 1 | 11 | 12 | **+6 −2 =4** | paused to build/gate the placement sub-terms |
| 2 | 12 | 307 | **+199 −64 =44 (72.0%), +164.0 ±45.7** | ★ **H1 ACCEPTED** (LLR crossed +2.960 at game 302; in-flight games closed it at +2.881) — .so re-verified `246 / 63,221,361` before it started |
| **POOLED** | 11 + 12 | **319** | **+205 −66 =48 → 71.8% → ≈ +162 Elo (±~45)** | segments share identical arms (fingerprint-verified between them) |

★★★ **SLICE 2 MOBILITY PASSES IN GAMES: ≈ +162 Elo pooled, the largest single gain of the rebuild** (KS +101, pawns
+60.4). Timed LIGHTNING games, so mobility's own compute cost is already charged. Three instruments predicted it:
§I −5.75%, d7 regret +4.9 / +3.1pp over measured neutrals, **and STS read it −110** — the corroboration rule chose to
spend the games on the two agreeing instruments and ignore the one that could not resolve, and was right.
⚠️ SPRT buys confidence, not magnitude: ±45 is wide. ✅ **SHIPPED in the v2 config (`MOB_V2_MAG=600`) on the owner's
sign-off, 2026-09-14.** New shipped v2 fingerprint: **WAC d10 `250 / 60,036,533 / 4.043` · STS 1588** — the old
`246 / 63,221,361` is now the pre-mobility config.
**WAC d10 fingerprint at `MOB_V2_MAG=600` (first measurement): 250 / 60,036,533 / EBF 4.043** — **−5.0% nodes** vs
shipped v2's 63,221,361 (registered prediction 5, "nodes fall 5-15%", lands at the edge of its range); solves are inside
WAC's floor. Shipped v2 re-verified `246 / 63,221,361` on the same build.

**TEMPO RE-TEST TRIGGER FIRED (mobility landed) — first point:** STS at `TEMPO_V2_MG=50 TEMPO_V2_EG=28` on the
mobility-600 base reads **1695 vs that base's 1588 → +107**. On the base WITHOUT mobility the same tempo read 1511 vs 1698
→ **−187**. A 294-point swing in tempo's effect between regimes — the direction the owner's "richer eval" hypothesis
predicted.
⚠️ **A lean, not a result:** +107 is inside the ±150 floor, it is ONE point (tempo's first five-point ladder was unordered;
our rule is a ladder, never a point), STS is the instrument that could not see mobility, and tempo's only demonstrated
channel was margin coupling — which mobility's −5% nodes could itself have shifted.
▶️ Next when the RAM frees: tempo 25/14 and 100/55 on the mobility base (ladder), then the node screen and §I, before any
tempo games.
**Second point:** tempo 25/14 on the mobility-600 base → **1704 (+116)** (without mobility it was 1656, −42).
| tempo | without mobility (vs 1698) | with mobility 600 (vs 1588) |
|---|---|---|
| 25/14 | −42 | **+116** |
| 50/28 | −187 | **+107** |
Both points flipped from negative to positive by a similar amount — more consistent than tempo's first, unordered ladder.
⚠️ Still a LEAN: both inside ±150, and tempo's only demonstrated channel is margin coupling, which mobility's −5% nodes could
have shifted. 100/55 completes the ladder; then the node screen; no tempo games before both.
**Ladder complete — third point 100/55 → 1709 (+121)** (without mobility: −194).
| tempo | without mobility (vs 1698) | with mobility 600 (vs 1588) |
|---|---|---|
| 25/14 | −42 | **+116** |
| 50/28 | −187 | **+107** |
| 100/55 | −194 | **+121** |
★ **Three consistent positive readings where all three were negative before mobility** — "three unresolvable signals the same
way are a result": the DIRECTION has flipped.
☠️ **But the response is FLAT across a 4× magnitude range** (+107..+121). A term with evaluative content shows a dose response;
a flat offset is the signature of threshold/margin coupling — mobility's −5% nodes moved where the fixed millipawn margins
bite, and any side-to-move constant now tips them favourably. Same shape mobility itself showed on STS.
⇒ Direction credible, mechanism unproven. Next: WAC node screen at 25 and 100 on the mobility base. If nodes also move without
dose response, tempo belongs to the MARGIN RE-SWEEP checkpoint (its named second trigger), not to a standalone term; games only
after that, and only on the owner's choice.
**Node screen, first point — tempo 100/55 on the mobility base (no KPK, matching the ladder): WAC d10 `251 / 56,974,651 / EBF 4.009`
= −5.1% nodes** vs `250 / 60,036,533`. ★ **Tempo's NODE effect has also flipped sign:** before mobility it ADDED nodes (+10.8% at
200/110, +2.8% at 800/440). Tempo has zero sibling variance, so 100% of that is threshold crossing — consistent with margin
coupling that mobility has turned favourable. ⏳ 25/14 running: a similar −5% at ¼ the magnitude would pin it on margin coupling.
**Second point — tempo 25/14: WAC d10 `253 / 57,000,644 / EBF 4.020` = −5.06% nodes.**
| tempo (mobility base) | WAC nodes | vs 60,036,533 | STS vs 1588 |
|---|---|---|---|
| 25/14 | 57,000,644 | −5.06% | +116 |
| 100/55 | 56,974,651 | −5.10% | +121 |
☠️★ **VERDICT: tempo's effect is BINARY, not evaluative.** Nodes and STS move identically across a 4× magnitude range: any
non-zero side-to-move constant shifts search thresholds by the same amount. It adds no knowledge; it acts on pruning margins
still fitted to v1's spread. ⇒ **Tempo stays PARKED and folds into the MARGIN RE-SWEEP checkpoint** (its named second trigger):
re-tuning v2's margins may capture the same −5% nodes directly, without a fake eval term. **No tempo games.**
★ Transferable: **a magnitude ladder on a zero-variance term is a coupling detector** — flat response in both nodes and
move-level score means the term is a switch on search thresholds, not an eval signal.

⚠️ No point estimate is read before the LLR is a real fraction of its bound (rung 2 read +6 at 125 games, +60 at 987).
☠️ Pausing lesson: `pkill -f <pattern>` inside `bash -lc "..."` matches ITS OWN command line and kills itself (exit 15)
before later commands run. Use bracket patterns (`[e]ngine_server`) and re-list to verify nothing survives.

### 3.3 §I crank (00:25) — where accuracy turns over

| `MOB_V2_MAG` | 300 | 600 | **1000** | 2000 | 600 + `EXCL_QUEEN` |
|---|---|---|---|---|---|
| §I mean | −3.34 | −5.75 | **−7.56** | −5.93 | −5.56 |
| §I worst (KS-critical) | +0.92 | +1.91 | +3.31 | +7.17 (UHO and variant also turn) | +1.67 |

☠️★ **The §I optimum is ~1000 mp of knight range — about where a plain PAWN conversion of SF11 lands (742), and ~33×
v2's measured knight PST spread.** The design's sizing premise ("mobility range ≈ 1-10× v2's PST spread") did NOT hold
on accuracy. Plausible reason, untested: v2's PSTs are themselves 3-10× smaller than SF11's cells (memory
`pt-star-is-material-inclusive`), so measuring against v2's own spread measured an under-scaled denominator.
⇒ Bound on [[convert-reference-constants-by-positional-scale-not-by-the-pawn]]: it was right for TEMPO (zero
variance, margin-coupled) and is contradicted, on §I, for MOBILITY (high variance, real ordering signal).
⚠️ **But this is CORPUS FIT**, which our record says is anti-correlated with Elo, and the worst case grows steeply
(+3.31% at 1000). A candidate, not a verdict. STS at 600/1000 is the next read.
**STS at the crank (same engine):** `MOB_V2_MAG=600` → **1588 (−110)**. Full STS series vs 1698: 3 → 1696 · **10 → 1661** ·
30 → 1642 · 60 → 1617 · 115 → 1618 · 300 → 1634 · 600 → 1588.
⚠️ With the 10-mp point the "STEP" reading of §3.2b is also withdrawn.
**Final STS series with the 1000 point: 3 → 1696 (−2) · 10 → 1661 (−37) · 30 → 1642 (−56) · 60 → 1617 (−81) · 115 → 1618
(−80) · 300 → 1634 (−64) · 600 → 1588 (−110) · 1000 → 1622 (−76).**
☠️★ **STS does not price mobility's MAGNITUDE.** From ~10 mp up it is FLAT and UNORDERED (−37..−110 over two orders of
magnitude, the −110 at 600 followed by −76 at 1000), every point inside ±150 — while §I improves 20× over the same
range. Neither a perturbation floor (3 mp reads 0), nor a step, nor a slide (1000 undoes 600): a small, roughly constant
negative offset once the term is on at all. ⇒ **STS cannot decide this term. A third instrument must** — the d7 move
regret gate, or games. Per the corroboration rule, games only with the owner's explicit choice. ⚠️ STS begins to fall exactly where §I approaches its optimum —
the `corpus-fit-is-anti-correlated-with-elo` shape — but −110 is still inside the ±150 floor: a lean, not a result.
`EXCL_QUEEN` trims the KS worst case (+1.91 → +1.67) for a small mean cost — consistent with the KS double-count
hypothesis, not proof of it.

**Registered predictions (2026-09-14, before any build):**
1. The 2×2 shows **negative interaction** (mob+rookfile < sum of each), because eg-heavy rook mobility and open files
   price overlapping rook activity. Confidence moderate.
2. The STS ladder **peaks interior (60-115)**, not at 30 or 300 — i.e. v2's PSTs are too small for SF's 1.1× to be
   enough, and Ethereal's 10× only works because its PST is flat. ⚠️ If it is monotone to 300, that says v2's PST is
   under-scaled, not that mobility is huge — record it that way.
3. Mobility at its best rung is the largest single STS gain since passers: **+60 to +150 STS**. ⚠️ My last five
   magnitude predictions were all wrong in one direction (too optimistic) — weight accordingly.
4. §I improves less than STS suggests (KS precedent: −1.44% §I for +101 Elo); a §I worst-case REGRESSION on the
   variant corpus would not surprise me and would not by itself block games.
5. Nodes at d10 FALL (a richer quiet-move signal sharpens ordering and cutoffs), by 5-15%.
