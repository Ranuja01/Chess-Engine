# EVAL v2 vs the giants — SMALL NUANCES audit (research pre-plan, 2026-10-09)

**Status: RESEARCH PRE-PLAN for discussion with the owner. Nothing here is a decision, a ship, or a measurement.**
Read-only audit; no engine was run, no file outside this one was touched.

**Question.** Before the final (structural) retune, which *small* reference details — conditions, gates, interactions,
special cases, scaling shapes — does v2 handle differently from SF11 / SF15.1-classical / Ethereal / Weiss, and which of
those differences are (a) never measured, (b) measured on an instrument since shown biased, or (c) genuinely closed?
Big missing features are NOT in scope (they are mapped in `EVAL-V2-GAP-AUDIT-2026-09-21.md` and the parked register).

**Sources read directly.** Ours: `eval_v2.cpp` (full), `search_engine.h` knob defaults, `cpp_bitboard.cpp:665-682`
(`passed_span_*`). References: SF11 `evaluate.cpp / pawns.cpp / material.cpp / psqt.cpp / endgame.cpp` and SF15.1
`evaluate.cpp / pawns.cpp` from the local trees (line numbers exact); Ethereal and Weiss `src/evaluate.c` at GitHub
`master` via WebFetch (⚠️ NOT the register's pinned commits @0e47e9b / @c735b8f; line numbers unavailable, function and
constant names quoted instead). Record: MEMORY.md + the six memories named in the brief, `our_eval_reference.md`,
`EVAL-V2-GAP-AUDIT-2026-09-21.md`, `EVAL-V2-CURRENT-CONFIG.md`, `EVAL-V2-PARKED-REGISTER.md`, `REFERENCE-BENCH-LADDER.md`
§3a, C3 doc §20a-§21a, `PASSER-SYSTEM-DESIGN-2026-10-03.md`, `PAWN_MODEL.md` §8a, `KING_SAFETY_MODEL.md` §4a/§5b,
`EVAL-V2-RUNG2-PAWN-DESIGN.md` (exclusivity no-op), `EVAL-V2-RUNG2B-PASSER-DESIGN.md`, `EVAL-V2-SLICE1-DRAW-DESIGN.md`.

**Marking.** **EVIDENCE** = both sides cited (our file:line AND reference file:line/URL). **SPECULATION** = an inference
about impact or mechanism, labelled as such. Impact estimates are priors, not measurements.

**Name-collision rule.** Every compared item is defined by its CONTENTS, never by its trace-row name. In particular:
SF "King safety" = shelter/storm + kingDanger + `minPawnDist` (eg king-to-own-pawn) + `PawnlessFlank` + `FlankAttacks`;
SF "Material" = piece values + PSQT; SF "Imbalance" = Kaufman census (v1's `imbalance_*` is OvD, unrelated);
SF "Initiative"/"Winnable" = complexity + the eg scale factor; SF "Passed" includes candidates and `PassedFile`.

**Units.** Ours: millipawns, pawn = 1000, flat across phase, Black-positive. SF11: mg pawn 128 / eg pawn 213 (SF11
`types.h`); SF15.1 126 / 208; Ethereal 82 / 144; Weiss 104 / 204. Material-denominated terms convert by the pawn;
positional ones by the positional spread ([[convert-reference-constants-by-positional-scale-not-by-the-pawn]]), and for
ordering terms both anchors are a LADDER, never one point. Scales never transfer; shapes and channel sets do.

**Owner's design rules applied throughout** (so nothing below proposes dropping an invention on purity grounds):
unique-where-better, never self-nerf — single-lineage ideas enter at weight 0 and the fit/games price them
([[unique-where-better-never-self-nerf]]); one owner per concept; no toe-stepping (every candidate names its owner).

---

## 0. RECORD CHECK FIRST — which past closures still stand, and which were measured on a biased instrument

The brief asks, for every item: *resolved, or merely unreadable?* The record has three instrument classes that have
since been shown unfit for specific question types. Items closed on them are re-opened here as UNREADABLE, not rejected.

| instrument | what it cannot read | items it closed that this audit treats as UNREADABLE |
|---|---|---|
| **root-Δ depth proxy** (`_revival_screen` knobs/dual; ours' = d10 − α·Δroot) | any DYNAMIC term — search resolves the root delta; the real effect is at the leaves ([[root-delta-depth-proxy-is-biased-against-dynamic-terms]], 10-07) | threats "CLOSED" (reversed 10-07: real d10 −7.8%) · PX "adds nothing at depth" (reversed 10-08: real d10 −5.56% on the ship base) · the passer path ladder "rejected 09-21" (reversed 10-08: −1.25% real d10; and ☠️ dead under any C1 table, `passer_value_mp:1565-1581` `continue`s before the ladder) · mobility cells / space / longdiag / reach / latent / KPROT "null" (C3 §20a: "needs the same real re-search") |
| **§I static accuracy alone** (positive OR negative) | move-level effect ("read KS −1.44% where games said +101"; six concepts died move-null while §I-positive) | tempo direction, rook files (later settled on move-level, genuinely closed), bishop pair (move read was an unresolved null), `KS_V2_EG_PCT` d7 (one neutral arm only; rejection rests on the WAC veto) |
| **STS alone / contamination-era harness** | anything inside ±150 | v1 convertibility scale (−3.3 STS, reverted same day — UNRESOLVED NULL), pawn_majority, `LATENT_V2_PCT` (instrument unstated) |
| **universality rule applied as PERMISSION** (the ≥3-of-5 bar) | nothing — it was never an instrument. The owner corrected this 09-27: single-lineage ideas go in at 0, they are not skipped | "castling MAX (SF-only)", "shelter→danger feedback (2/4)", `PassedFile` (2/4), `WeakLever` (2/4), `RookOnClosedFile`, `PawnDoubled2`, Weiss `PassedSquare`, Ethereal `KingDefenders` — ALL **NEVER-MEASURED**, parked ON-RULE (register §"SINGLE-LINEAGE — skip") |

Two of those "ON-RULE" counts are also WRONG as counts (record corrections, §7): shelter→danger feedback is **3/4, two
lineages** (Ethereal's `pksafety` feeds its `safety` sum — fetched and quoted below), and the castling max is **2/4 one
lineage** (SF11 + SF15.1; Ethereal confirmed absent; Weiss absent).

What IS genuinely closed and is not re-surfaced: rook files (two disjoint move-level sets, 09-20) · tempo (margin
coupling, replicated) · material/pawn taper (depth target 10-05 + v1 `EG_EXIST` interior plateau) · `MOB_V2_SAFE` /
`MOB_V2_EXCL_QUEEN` (§I worst-case + reversal) · flat bishop pair (owned by Kaufman now) · Kaufman verbatim tables
(fitted cells shipped instead) · tier-2b/KR-vs-minor as *Elo* work (tablebase ground truth) · the additive winnability
form (−23..−25 twice; multiplicative shipped).

---

## 1. PAWN STRUCTURE (SF `pawns.cpp` evaluate(); Ethereal `evaluatePawns`; Weiss pawn block)

**Contents compared:** isolated · doubled · backward · connected (phalanx/support × rank × opposed) · weak-unopposed ·
weak-lever · blocked-pawn bonuses. Ours: `build_pawn_entry` (`eval_v2.cpp:1099-1221`) + `pawn_structure_mp`
(`:1267-1388`); shipped `PS_V2_MAG=100`, connected `CONN_MAG 21 / SUPPORT 99 / EG_RATIO 101`, C1 tables for
doubled/isolated/backward/WU when `C1_V2_FIT` (ship uses constants: doubled 86/263, iso file table ×0/×100, backward
0/113, WU 0/0).

| # | reference detail (file:line) | ours (file:line or absent) | likely impact (reason) | already tried? | suggested test | effort |
|---|---|---|---|---|---|---|
| P1 | **Doubled penalty is GATED on `!support`** — SF11 `pawns.cpp:149-151` `if (!support) score -= Doubled*doubled + WeakLever*more_than_one(lever)`; SF15.1 `pawns.cpp:189-191` identical. The flagged pawn is the FRONT one (`doubled = ourPawns & (s - Up)`, `:103`), and it is the front pawn's support that waives the charge. Ethereal's `PawnStacked[flag][file]` is likewise condition-indexed (unstack-able vs frozen). Weiss: unconditional `PawnDoubled` (+ `PawnDoubled2` for a one-square gap). ⇒ **3/4, two lineages condition it.** | `e.doubled = own & (own >> 8)` flags the REAR pawn (`:1118`) and `side_* -= nd * DOUBLED_*` is unconditional (`:1344-1345`; C1 path `:1330-1331` also a flat count). **No support gate; no front/rear distinction.** EVIDENCE. | Medium-low. The eg leg is 263 mp per pair — a supported doubled pawn (e.g. d4/d3 with c3) pays the full eg liability in ours and nothing in SF. SPECULATION: the C1 fit absorbed the average, so the unconditional count is right on average and wrong on the supported/unsupported split — exactly the "narrow, not wrong" pattern. | v1-era note only (`passer-doubled-hce-comparison-2026-07-22.md:33,54,308`: "recondition `!support`/phase before shipping" — never done in v2). NEVER-MEASURED in v2. | Structural → split `doubled` into two count columns (front-pawn supported / unsupported) in `v2_features`, fit on the depth target + static component, gate per part. Static check first: fire rates of the two classes on the labelled rows. | Low (one AND per side + 2 columns) |
| P2 | **`WeakLever` S(0,56)** — unsupported pawn attacked by TWO enemy pawns, eg-only, same `!support` gate (SF11 `:151`; SF15.1 `:191` S(2,57)). | Absent. `e.attacks2[]` (`:1106-1107`) and `e.supported[]` already exist, so the detector is `own & ~supported & enemy_attacks2`. | Low (register: "3.59% firing, pure endgame"). SPECULATION: endgame-only and cheap; the retune's eg lane is exactly where the record says the residual is. | Register: "weak-lever absent … later" (2/4, ON-RULE). NEVER-MEASURED. | Structural → one count column at 0 in the retune; static check of fire rate on endgame rows. | Trivial |
| P3 | **Isolated/backward pawn on the EDGE file**: SF15.1 `pawns.cpp:187` pays `WeakUnopposed` on a backward pawn only off the a/h files (`bool(~(FileABB|FileHBB) & s)`); SF11 pays it everywhere. **SF-lineage splits internally.** | WU block `:1369-1373` (knobs 0/0, move-null 09-20) and the C1 `nwu` column (`:1339-1341`) count every file. | Negligible alone (WU itself was move-null at 49.3/49.3/50.1). Listed for completeness only. | WU measured MOVE-NULL 09-20 (fair instrument). | Fold into the WU column's definition if WU is ever re-priced; otherwise nothing. | Trivial |
| P4 | **Isolated-but-doubled special case** (SF15.1 `pawns.cpp:176-179`): an isolated pawn that is `opposed`, has an own pawn BEHIND on its file, and no enemy pawn on adjacent files is charged `Doubled` INSTEAD of `Isolated+WeakUnopposed`. | Both charges stack (`:1326-1354`). | Low; a rare class. SPECULATION. | Never considered. | Static: count the class on the labelled rows before deciding anything. | Trivial |
| P5 | **`BlockedPawn[2]` = S(−19,−8) rank 5 / S(−7,+3) rank 6** (SF15.1 `pawns.cpp:43, :193-194`): an own pawn BLOCKED by an enemy pawn on its 5th/6th rank. Single lineage (SF15.1 only). | `e.blocked[]` is built (`:1131`) and consumed only by bad-bishop form 3 (not shipped). PX has blocked cells for PASSERS only (`:3575-3579`). | Low-medium. SPECULATION: this is the "advanced but stuck" class the record ties to the K+P loss (a doubled/blocked g-pawn read as an asset). | Register S1/S2 lists blockedness as an INPUT to closedness; the per-pawn bonus never. NEVER-MEASURED. | Structural → 2 count columns at 0 (blocked on rel. rank 5 / 6). | Trivial |
| P6 | **`DoubledEarly` S(17,7)** (SF15.1 `:37, :131-136`): extra doubled charge while no own pawn is fixed against an enemy pawn or its attacks (i.e. the structure is still fluid). | Absent. | Low. Single lineage. | Never considered. | Count column at 0; probably leave. | Trivial |
| P7 | Connected pawns: SF `(2 + phalanx − opposed)` modifier, `+21/22·popcount(support)`, eg = v·(r−2)/4 (SF11 `:135-138`; SF15.1 `:168-171`). | `:1290-1322` identical shape; `PS_V2_EG_RATIO` carries the mg/eg unit difference explicitly. EVIDENCE: faithful. | — (shipped 10-04, +14.5 ± 7.7). | — | — | — |
| P8 | SF11 `Connected`/`Isolated`/`Backward` are an **else-if chain** (`:133-147`), as is Ethereal's; `PS_V2_CONN_EXCL` exists "because the references disagree" (`:1281-1282`). | ☠️ The knob is **dead by construction**: `backward` needs no neighbour at rank ≤ own (`rear_nb`, `:1138`), while `supported`/`phalanx` ARE such neighbours, so `backward ∩ connected = ∅` and `isolated ∩ connected = ∅`. Recorded as a NO-OP in `EVAL-V2-RUNG2-PAWN-DESIGN.md:556-565`. EVIDENCE. | None — but the code comment at `:1279-1282` ("SF stacks them") is also WRONG about SF11: SF11's chain excludes too. The only stacking that exists is `isolated + backward` and `doubled + anything` (both engines stack those). | Measured no-op 09-12. | No test. Cleanup candidate: delete the knob or re-point `CONN_EXCL=2` ("also exclude passers") which is the only live exclusivity question (RUNG2B §Q2). | Trivial |

---

## 2. PASSED PAWNS (SF `Evaluation::passed()`; Ethereal `evaluatePassed`; Weiss passed block)

**Contents compared:** rank table · candidate detection and halving · king proximity (both kings; second push) · path /
stop-square safety ladder · pieces behind · file · square rule · double-passer handling. Ours: `passer_value_mp`
(`:1540-1660`, shipped `PASSER_V2_MAG=100`, eg leg only, `KING_THEM/US` 2228/938 = SF's 19/4 : 2) · the ladder at
`PATH_PCT=0` · PX cells (`px_counts :3535-3592`, built at 0, 51 cells) · `PS_V2_REAR_DOUBLED=2`.

| # | reference detail | ours | likely impact | already tried? | suggested test | effort |
|---|---|---|---|---|---|---|
| X1 | **Second-push king term**: SF11 `evaluate.cpp:608-609` / SF15.1 `:783-784` `if (r != RANK_7) bonus −= S(0, king_proximity(Us, blockSq+Up)·w)` — our king's distance to the square AFTER the stop, eg, rank-weighted. SF lineage only (Ethereal/Weiss measure distance to the pawn/stop once). | Shipped scorer: absent. PX cells 44-46 (`PX_KD2_US`, `:3590`) carry it at 0. EVIDENCE. | Low-medium; it is the one king-escort detail the shipped scorer lacks, and king escort is where SF11's static edge on passers sits (PASSER-SYSTEM §7: eg ranks 5-6, ≈0.6-0.75pp). | PX fit (10-04) priced it inside "ESCORT" (−0.16% alone) — on the PROXY. The 10-08 REAL d10 of PX cells + joint re-price read **−5.56%** on the ship base (C3 §20a) → gated in queue #42. | Already in the PX gate; no separate action. Mark the proxy-era "adds nothing" as superseded. | — |
| X2 | **`PassedFile`**: SF11 `:644` `− S(11,8)·map_to_queenside(file)`, SF15.1 `:815` `− S(13,8)·edge_distance` — **edge passers worth MORE** than central ones. SF lineage (2/4). | PX cell 49 (`PX_FILE`, `:3554`) at 0. | Low. SPECULATION: in pawn endings an outside passer is a textbook winner; the record says pawn endings persist at d10 (−9.3) but mostly as a SEARCH gap (§3a). | Register: ON-RULE skip; PX built it → gate pending. | Inside the PX gate. | — |
| X3 | **Square rule** (Weiss `PassedSquare S(−26,422)`: `nonPawnCount[!color]==0 && Distance(sq,promo) < Distance(kingSq(!color),promo) − ((!color)==sideToMove)`). Note Weiss uses side-to-move; ours is deliberately stm-free. | PX cell 50 (`:3562-3570`): defender's king more than ONE step outside the square (`kd − 1 > steps`), stm-free, with the 2nd-rank double-push. EVIDENCE: our form is strictly more conservative (it needs a 2-tempo margin) so it never needs `turn`. | Low-medium in pawn endings. ⚠️ The record (§3a) attributes the pawn-ending d10 gap to SEARCH (SF11 d10 −1.0 vs ours −8.5), so a static square rule is a static-only gain; still the owner's POT-endgame design names it. | PX gate. | Inside the PX gate; afterwards, if kept, verify against KPK-exact on the 1-pawn subset (must agree 100% where both fire). | — |
| X4 | **Candidate halving also halves a TRUE passer with ANY pawn directly in front** (SF11 `:640-642` `!pawn_passed(s+Up) \|\| (pieces(PAWN) & (s+Up))`) — i.e. a rear-doubled passer is paid HALF in SF; SF15.1 `pawns.cpp:158` instead un-flags it (`passed &= !(forward_file_bb & ourPawns)`); Ethereal `continue`s on the distance bonuses only. | `PS_V2_REAR_DOUBLED=2` drops it entirely (`:1167-1173`); mode 1 (demote to candidate = half) built, untested. EVIDENCE. | Negligible (3.4% fire). Shipped on correctness. | Mode 2 shipped (Bundle A); mode 1 untested. | None needed. | — |
| X5 | **Candidate condition (c)**: SF `shift<Up>(support) & ~(theirPawns \| doubleAttackThem)` (SF11 `pawns.cpp:125`) — the pushed supporter must land on a square not DOUBLE-attacked by enemy pawns. | `sps = (sup << 8) & ~enemy` (`:1190`, `:1198`) omits `doubleAttackThem`. EVIDENCE. | Negligible; a detector refinement affecting rare rank-5+ candidates. | `_pawn_term_overlap.py` oracle mirrors OUR predicate, so it would not catch this. | Static: count positions where the two predicates differ (one probe run). Fix if non-zero and free. | Trivial |
| X6 | **SF15.1 removes BLOCKED candidates without pawn help** (`evaluate.cpp:750-761`: `helpers = shift<Up>(ourPawns) & ~theirPieces & (~attackedBy2[Them] \| attackedBy[Us])`), and its ladder counts enemy PIECES as unsafe squares (`:795`) with a 5th rung "all unsafe squares are attacked by our pawns" (36/30/17/7/0, `:801-805`). | Ladder is SF11's 4-rung form (`:1628-1631`), dead under C1 anyway. PX cell 20-23 (`PX_PATH_FREE`) uses pieces+attacks (`:3583`), PX stop-state cells split blocker TYPE (owner's invention). EVIDENCE. | The 10-08 real d10 says the path/stop dynamics ARE live (−1.25% ladder, −5.56% PX). Which FORM wins is open: SF11 4-rung · SF15.1 5-rung · Ethereal 2×2 · ours PX cells. | PX vs ladder re-measured 10-08; queue #42 gates PX. | If PX gates in, the SF15.1 "attacked-by-our-pawns" rung is one extra cell at 0 for the retune; otherwise nothing. | Low |
| X7 | Both kings' proximity in SF is to the BLOCK square capped at 5 (`:579-581`); Ethereal to the PAWN, per-rank tables for BOTH kings; Weiss to `forward` with `(rank−3)` weight on the enemy king only. | Stop square, `ps_kdist` cap 5, `w=5r−13`, eg only (`:1588-1596`); PX cells 36-43 per rank both legs. EVIDENCE: SF-faithful; owner chose mg legs via PX (PASSER-SYSTEM §6). | — | PX. | — | — |

---

## 3. KING SAFETY & SHELTER (SF `king()` + `Pawns::do_king_safety`; Ethereal `evaluateKings`+`evaluateKingsPawns`; Weiss)

**Contents compared:** attack units (count × weight, weak squares, checks, flank, blockers, no-queen, knight defender) ·
output transform · shelter/storm · castling max · `minPawnDist` · `PawnlessFlank`. Ours: `ks_channels :802-957`,
`ks_units :966-1026`, `ks_danger_mp :1041-1047` (Hill curve, `KS_V2_EG_PCT=66`), KS-B `ksb_cells :3346-3380` (56 cells
compiled), KFL (`:3446-3467`, gated, not shipped), KPROT (`:3479-3491`, gated).

| # | reference detail | ours | likely impact | already tried? | suggested test | effort |
|---|---|---|---|---|---|---|
| K1 | **Shelter at the CASTLING target squares**: SF11 `pawns.cpp:233-237` / SF15.1 `:281-285` take `max(shelter(ksq), shelter(G1), shelter(C1))` by MG value when the right exists. Ethereal: **absent** (verified by fetch, §7). Weiss: absent. ⇒ SF lineage only (2/4, ONE lineage). | `KSB_V2_CASTLE` built (`ksb_side :3395-3415`, max by blended value) but **0 in the ship, and the feature extractor flags mode 1 as unmodelled (`:3733`) ⇒ it was never fitted and never gamed.** EVIDENCE. | Medium. SPECULATION: without it, an uncastled king with intact castling rights is scored on its CURRENT (usually bare) files, so the eval over-penalises not-yet-castled kings and under-rewards keeping the right — a plausible opening-move-ordering bias (STS "AKPC" was a weak theme for v1). It is a MECHANISM (which square the cells read), not a weight. | Register: "castling MAX (SF-only)" DEAD / "2/4" skip — **NEVER-MEASURED, ON-RULE.** The REBUILD-LOG `:2628,2642` explicitly chose "No castling MAX … on universality grounds", which the owner's 09-27 rule overrides. | Dynamic-ish (king placement choice) → real d10 re-search with `KSB_V2_CASTLE=1` on the shipped KS-B cells vs an identical-conditions ship re-run; static pre-check: fire rate (positions with a castling right where a target square scores higher) and the mg-only max (SF's `compare`) vs our blended max. | Low (knob exists; needs the extractor to model it before any refit) |
| K2 | **Shelter/storm FEEDS the danger sum** — SF11 `evaluate.cpp:455` `− 6·mg(score)/8` (score = shelter) inside `kingDanger`; SF15.1 `:609` same (annotated ~8 Elo); **Ethereal: `safety += … + ei->pksafety[US]`** where `evaluateKingsPawns` writes `pksafety += SafetyShelter[…]` and `SafetyStorm[…]` beside the direct `pkeval` cells (fetched, quoted §7). ⇒ **3/4, TWO lineages** — not 2/4. | KS-B is a DIRECT score only (`:4279-4281`); the comment at `:4272` names "a shelter→danger coupling as ONE scalar" as a later test. C3 §3 item 5: "not built; testable as a fit arm from the C3 cells". EVIDENCE. | Medium. This is the interaction the owner's KS philosophy wants ("fire only on real danger"): a bare king makes the SAME attackers more dangerous. The gap audit called SF's double-wiring "choose one channel or both"; Ethereal choosing BOTH independently upgrades it from a design choice to a convergent one. | Register DEAD list: "shelter→danger feedback (2/4)" — ON-RULE, count WRONG, NEVER-MEASURED in v2 (v1's `KS_SHIELD` is a different mechanism). | KS is judged on near-equal discrimination + quiet fire rate by phase ([[ks-recall-is-structural-and-firing-does-not-discriminate]]) → add `u −= k·shelter_mg_units` as one knob at 0; fit k jointly with the KS attack knobs on the depth target (the 10-03 method); then both instruments. | Low-medium (one scalar; KS-B's mg leg already computed per king) |
| K3 | **Safe-check de-duplication and the own-queen exclusion**: SF11 `:408-412` `queenChecks = (b1\|b2) & attackedBy[Them][QUEEN] & safe & ~attackedBy[Us][QUEEN] & ~rookChecks`; `:419-422` `bishopChecks = b2 & … & ~queenChecks`. "Rook checks are more valuable" — a square giving both pays ROOK only; a queen check on a square our own queen covers does not count (queen trade). SF15.1 `:569-577` identical + a single/multiple multiplicity table. Ethereal: per-type popcounts, no de-dup. Weiss: per-piece `checks` counts. | `chk_q = (rookRays\|bishopRays) & safe & att.by[QUEEN]` (`:935`), `chk_b = bishopRays & safe & att.by[BISHOP]` (`:936`) — **no `~rookChecks`, no `~queenChecks`, no `~def.by[QUEEN]`**; firing is boolean per type (`KS_V2_CHK_COUNT=0`). EVIDENCE. | Low-medium. With boolean firing, a Q+R battery on one open file pays CHK_R + CHK_Q (249+247 fitted units) where SF pays one. SPECULATION: the joint KS fit (10-03) priced CHK_Q at 247 (SF 780) partly BECAUSE of this double fire — the constants absorbed the shape. A safe queen check covered by our queen is also over-counted. | Never discussed in the KS design (`EVAL-V2-RUNG1-KS-DESIGN.md` only lists `CHK_COUNT=1` as "built, untested"). NEVER-MEASURED. | KS discrimination metric: add `KS_V2_CHK_DEDUP=1` (SF's two exclusions) and re-fit the four CHK cells jointly; compare near-equal discrimination and quiet fire rate; then the usual two instruments. Pair with `CHK_COUNT=1` so the form question is settled once. | Low |
| K4 | **Pawn attacks SEED the attacker count** — SF11 `evaluate.cpp:243` / SF15.1 `:378` `kingAttackersCount[Them] = popcount(kingRing & pawn_attacks(Them))` (a COUNT of ring squares, feeding the count×weight product). Ethereal/Weiss: absent (verified). SF lineage only. | `KS_V2_PAWN_ATT` seeds 0/1 (`:911-919`), ships 0. EVIDENCE: shape differs (bool vs count) AND off. | Low. SF-only; and our coordination is a continuous `COORD` knob rather than SF's product, so the seed's leverage differs. | Built, swept in the K1/K2 fits? Not in the shipped block ⇒ at 0; C3 doc does not list a result. Treat as NEVER-MEASURED in its count form. | One arm in the next KS joint fit (count form), judged on discrimination. Low prior. | Trivial |
| K5 | **Shelter pawn attacked by an enemy pawn**: SF15.1 `pawns.cpp:236` excludes it (`ourPawns = … & ~pawnAttacks[Them]`); SF11 `:191` does NOT; Ethereal does not; Weiss does (`Shelter` counts allied pawns not attacked). 2/4, two lineages, and the SF lineage splits. | KS-B gives it its OWN cell (state 5, `:3366`) — strictly more expressive than either. EVIDENCE: superset. | — | Shipped inside KS-B. | — | — |
| K6 | **`KingOnFile[semiopen us][semiopen them]`** (SF15.1 `pawns.cpp:75-76, :260`): S(−18,11) king on a fully open file, S(−6,−3) semi-open ours, S(5,−4) semi-open theirs. Single lineage. Weiss `KingLineDanger[count]` (open queen-rays from the king through own men) is the independent cousin (1/4). | KS-B's "no pawn" state is the pinned reference per file class, so (no own, no enemy) on a file = 0 by construction; (own, none) and (none, enemy) have cells. The 2×2 is therefore EXPRESSIBLE except that the open-file case is the zero and cannot carry an eg-positive value (SF's +11 eg: an open file is GOOD for the king in the endgame). EVIDENCE (`:3340-3341` "none … pinned to 0"). | Low. The eg sign flip (open file good in eg) is the only thing the current cell basis cannot say. | Register: `KingLineDanger` 1/4 skip. NEVER-MEASURED. | Count column at 0 for the retune: "king's own file has no pawn of either colour" (mg,eg). | Trivial |
| K7 | SF11/15.1 `kingDanger += mg(mobility[Them] − mobility[Us])` (`:452`/`:606`). | Not in KS-A. v1's `KS_MOB_EDGE` port measured null across 4× (KING_SAFETY_MODEL §4a). | Low; v1's null was on a sound instrument class (move regret with neutral band). Prior low; SF annotates it ~0.5 Elo. | v1 refuted; v2 never. | Skip unless the KS fit wants a channel. | — |
| K8 | `minPawnDist` eg −16·d (SF11 `pawns.cpp:240-248`; SF15.1 caps at 6) + `PawnlessFlank` S(17,95) (`evaluate.cpp:464-465`). | KFL cells (own/enemy split, flank states) — gated, real d10 −2.78% on the ship base but instruments split (SF18 −3 ± 8 / self-play +15). EVIDENCE. | Known. | Measured 10-05/10-08; → final retune. | — | — |
| K9 | Ethereal `KingDefenders[count]` (own pawns + minors in the king area) and `SafetyAdjustment` (1/4). | KS-A has `KNIGHT_DEF` (SF) and OUR contest channels (`CONTEST_EXCESS/SQ`), which count defence per square. EVIDENCE: our contest channels are a finer defender model (owner's invention, shipped). | — | Shipped. | — | — |

---

## 4. MOBILITY (SF `pieces()` + `mobilityArea`; Ethereal; Weiss)

**Contents compared:** area definition · pin handling · x-ray occupancy · table shape · phase legs. Ours: `mob_area
:1684-1695`, `build_side_attacks :647-707`, `mobility_mp :1794-1823`; shipped `MOB_V2_MAG=600`, `EG_PCT=125`, `PIN=1`,
`EXCL_LOWRANK=1`, `EXCL_QUEEN=0`, `KS_V2_XRAY=1`.

| # | reference detail | ours | likely impact | already tried? | suggested test | effort |
|---|---|---|---|---|---|---|
| M1 | Every reference exclusion exists as a knob; the gap audit closed the subsystem structurally (`EVAL-V2-GAP-AUDIT §MOBILITY`). SF excludes the OWN QUEEN's square (`evaluate.cpp:230`); Weiss/Ethereal do not. | `EXCL_QUEEN=0` (its single 09-14 §I point reversed). | — | Measured. | — | — |
| M2 | **Mobility legs**: SF11 pieces' eg tables rise steeply for rooks (−76 → 171) — "rook activity is an endgame property" — and SF's eg pawn is 1.66× its mg pawn, so the eg leg is already relatively LARGER in pawn terms than the table suggests. | `MOB_V2_EG_PCT=125` (one ratio for all four piece types). The 10-07 real d10 found "mobility EG LEG ONLY −3.2/−4.2/−0.3" but mob cells gave 0 in games. | Known; → the retune's split legs (owner 10-07: every term's mg and eg fitted separately). | Measured. | — | — |
| M3 | Trapped rook reads the SAME area mobility as SF (`else if (mob <= 3)` after the semi-open test, SF11 `:343-351`); SF15.1 adds `RookOnClosedFile` S(10,5) when our pawn on the rook's file is BLOCKED (`:499-504`). | Trapped rook shipped @10, file-symmetrised. `RookOnClosedFile` absent (register: 1/4, "no record hit"). | Low. SPECULATION: a rook behind a blocked own pawn is dead weight; the term is a tiny per-rook penalty. | NEVER-MEASURED. | Count column at 0 (`rooks & own & file of a blocked own pawn`) in the retune. | Trivial |

---

## 5. PLACEMENT / PIECES (SF `pieces()` non-mobility terms; Ethereal piece functions; Weiss)

**Contents compared:** outposts (+reachable, +uncontested) · minor behind pawn · bad bishop · long diagonal · trapped
rook · weak queen · KingProtector · rook/bishop on king ring · rook on queen file · bishop x-ray pawns · rook on 7th.
Ours: `placement_detect :2400-2528`, `placement_mp :2532-2592`; shipped bundle E (outpost SF11 @100, bad bishop SF15.1
@100, trapped rook @10, weak queen @25, minor-behind Weiss @25); KPROT gated.

| # | reference detail | ours | likely impact | already tried? | suggested test | effort |
|---|---|---|---|---|---|---|
| L1 | **`UncontestedOutpost`** (SF15.1 `:441-445`): a KNIGHT on a SIDE outpost (off c-f) with no attacks on enemy non-pawns and ≤1 enemy piece on that wing gets S(0,10)·(pawns on that wing) INSTEAD of the outpost bonus — i.e. "a knight on an outpost with nothing to do is not worth the outpost". Single lineage (SF15.1). | SF11 form (`:2427-2439`): every outpost pays. | Low-medium in closed/locked positions (exactly the `centre_locked` class where space was the only live signal). SPECULATION. | Never considered. | Static: how often an outpost knight has zero enemy non-pawn targets on the labelled rows; if >5%, one count column at 0. | Low |
| L2 | **`RookOnKingRing` S(16,0) / `BishopOnKingRing` S(24,0)** (SF15.1 `:424-428`): a rook on a file intersecting the enemy king ring, or a bishop whose pawn-only-occupancy diagonal hits it, that is NOT already a king attacker — LATENT king pressure, scored in pieces() not in danger. Single lineage. | KS-A counts attackers through the x-ray maps (`KS_V2_XRAY`, `:666-669`) which already see sliders THROUGH queens/own rooks but not through pawns. `LATENT_V2_PCT` (ours, pawn-targets) is a different concept and read null. EVIDENCE: absent. | Low-medium. ⚠️ Owner-concept adjacency: this is "pressure that is not yet kinetic", which is POT's middlegame idea. One owner per concept — it belongs to POT mg (or to KS-A as a channel), not to placement. | Never. | Flag for the POT mg design; do NOT add to placement. | — |
| L3 | **`BishopXRayPawns` S(4,5)·(enemy pawns on the bishop's empty-board diagonals)** (SF15.1 `:469`). Single lineage. | Absent. Bad bishop (own pawns on the colour) shipped. | Low. | Never. | Count column at 0; low prior. | Trivial |
| L4 | **KingProtector per TYPE** (SF15.1 `:236` N S(9,9) / B S(7,9); SF11 one value `:306`); Ethereal `KnightInSiberia` (knights only, dead band ≥4, to the NEARER king). | KPROT: 12 cells (type × distance 1..6+) — superset. Real d10 0.0/−1.3/+3.5; instruments split (SF +16 four-seed / self-play −2). | Known → retune. | Measured. | — | — |
| L5 | **`RookOnQueenFile` S(7,6)** (SF11 `:339-340` only; SF15.1 dropped it). | Absent. | Negligible; the SF lineage itself dropped it. | Register 1/4. | Skip. | — |
| L6 | Ethereal `RookOnSeventh` is GATED on the enemy king being on its 7th/8th rank; register's "rook on 7th (2/5) DEAD" reflects v1's UNgated form. | Absent in v2 (SF11+ dropped it: "eg-heavy rook mobility carries it", `:2277-2279`). | Negligible; the gate is the only new information. | v1 form closed. | Skip unless the retune's rook eg mobility is found to under-pay 7th-rank rooks. | — |
| L7 | Minor-behind-pawn: SF uses a pawn of EITHER colour directly in front (`:302`); Weiss `NBBehindPawn` only OWN pawns (`ShiftBB(pieceBB(PAWN), down)` — all pawns, actually, per the fetch); Ethereal `KnightBehindPawn` uses `pawnAdvance(pieces[PAWN])` = all pawns. | `pawn_in_front = c.pawns` either colour (`:2422`). EVIDENCE: faithful. | — | Shipped @25 Weiss form. | — | — |

---

## 6. THREATS (SF `threats()`; Ethereal `evaluateThreats`; Weiss)

Threats is NOT shipping (10-08: real d10 −7.8% and self-play +32 ± 9 but SF18 @1000 −5 ± 9 — a genuine instrument
split) and has moved to the **search transition** (handoff 10-07 "ORDER RESHAPED"). The nuances below are recorded for
the dynamic-lane retune that follows the search work, not for action now.

| # | reference detail | ours | likely impact | already tried? | suggested test | effort |
|---|---|---|---|---|---|---|
| T1 | **Threats on the enemy QUEEN**: SF11 `:549-562` `KnightOnQueen` S(16,12) / `SliderOnQueen` S(59,18) on squares in OUR mobility area not strongly protected (slider needs `attackedBy2[Us]`); SF15.1 `:706-723` doubles both when it is the ONLY queen (`queenImbalance`); Ethereal `ThreatQueenAttackedByOne`; Weiss's victim tables include the queen. ⇒ 3/4 by concept, 2 lineages. | Victim tables index 4 = queen (`TH_MINOR_*[4]`, `TH_ROOK_*[4]`) so a DIRECT attack on the queen is priced; the NEXT-MOVE (reachable-square) threat is absent. Config: "the unbuilt SF legs". EVIDENCE. | Medium when threats returns: it is the term the record ties to "queen vs minor compensation" errors (SF15.1's `queenImbalance` doubling is specifically about queen-vs-no-queen, memory [[v2-overvalues-queen-vs-minor-compensation]]). | Gap audit T1 "inputs exist"; NEVER-BUILT. | Dynamic → when threats re-enters (search lane): one leg at 0, real d10 re-search. | Low |
| T2 | `ThreatByKing` is a BOOLEAN in SF (`:514-515` `if (weak & attackedBy[Us][KING])`) and `weak` INCLUDES pawns; Ethereal uses popcount over MINORS/ROOKS only (fetched); Weiss removes the king. | `popcount(weak & us.by[KING] & ~c.pawns)` (`:1910`) — Ethereal's count × SF's victim set MINUS pawns: a third form. EVIDENCE. | Low. | Leg fitted as one multiplier (10-05). | Settle the form when the leg is re-priced. | Trivial |
| T3 | **`WeakQueenProtection` S(14,0)** (SF15.1 `:677`): extra on a weak piece whose only protector is the queen. Single lineage. | Absent. | Low. | Config lists it unbuilt. | Leg at 0 later. | Trivial |
| T4 | Ethereal `ThreatOverloadedPieces` (attacked once, defended once) and Weiss/Ethereal `ThreatWeakPawn`. | Absent / pawn targets via `THREAT_V2_PAWN_TARGETS`. | Low. | Register single-lineage. | Later. | — |

---

## 7. MATERIAL / IMBALANCE / PHASE (SF `material.cpp`, `psqt.cpp`; Ethereal `evaluateClosedness`; Weiss)

**Contents compared:** piece values and their taper · PSQT · Kaufman census (incl. the bishop-pair pseudo-piece) ·
closedness-conditioned values · phase formula. Ours: `v2_piece_value :199-216` (flat), `build_context :262-287`
(phase 15800/61700 of npm = SF's 23.6%/91.9% of start npm — EVIDENCE, locked), Fit-A tapered PST
(`rung0_tapered_pst :398-424`), Kaufman FORM 3 fitted cells (`kaufman_mp :2173-2205`), MCL classes (`:2145-2171`, 0).

| # | reference detail | ours | likely impact | already tried? | suggested test | effort |
|---|---|---|---|---|---|---|
| I1 | Piece values taper in every reference (SF11 N 781→854, pawn 128→213; Ethereal/Weiss likewise). | Flat; `EVAL_V2_PIECE_MG_PCT=100`, `EVAL_V2_PAWN_MG=1000`. Material taper null on the depth target (10-05); v1's `EG_EXIST` ladder peaked at the shipped value (PAWN_MODEL §8a). | Closed on fair instruments. ⚠️ The owner's 10-07 rule ("fit every term's mg and eg legs separately") reopens the QUESTION as a free pair of columns in the retune, which is the right place — not as a standalone ladder again. | Measured twice. | Fold into the giant retune as free legs; nothing standalone. | — |
| I2 | SF applies the imbalance's bishop-pair pseudo-piece with `pieceCount[Us][pt1]` gating (`material.cpp:92-104`) and divides by 16 (`:215`). | FORM 3 fitted cells, `KAUF_V2_PAIR=1`, divisor 1000 (cells already mp). EVIDENCE: arithmetic matches. | — | Shipped 10-03. | — | — |
| I3 | Ethereal `evaluateClosedness`: `closedness = clamp((pawns + 3·rammed − 4·openFiles)/3, 0, 8)` → `ClosednessKnightAdjustment[c]`, `ClosednessRookAdjustment[c]` per knight/rook DIFFERENCE. Single lineage. v1 built it (dead code). | Absent. `e.blocked[]` and `e.openFiles` exist (`:1131, :1218`). | Low-medium. SPECULATION: a 9-cell table per piece type is cheap and it is the one census term that sees STRUCTURE; Kaufman cannot. ⚠️ One-owner: it re-prices N/R values by structure, so it must be fitted WITH the Kaufman cells, never beside them. | Register S1/S2 (1/4 explicit form). NEVER-MEASURED in v2. | Structural → 2 × 9 count columns at 0 in the retune, fitted jointly with Kaufman + piece values (the owner's rule for census terms). | Low |
| I4 | SF15.1 space weight adds `min(blocked_count, 9)` (`:856`) — closedness again, inside space. | Space parked (`SPACE_V2_*`), weight = (pieces−1)². | Low. | Space's named trigger (closed-centre corpus) unfired. | Only with space. | — |

---

## 8. ENDGAME SCALING / DRAWISHNESS (SF `scale_factor` + `material.cpp:198-204` + `endgame.cpp`; SF15.1 `winnable`;
Ethereal `evaluateScaleFactor` + `evaluateComplexity`; Weiss `ScaleFactor`)

**Contents compared:** the generic eg scale (pawn count, OCB, one flank), specialised scales (pawnless leader, lone minor,
queen vs none, rook endings), rule-50 damping, and the initiative/complexity correction. Ours: `win_scale_adjust
:3664-3685` (shipped `POT_V2_WIN=1 BASE −37 SP 34 OCB −80`; `ONEFLANK`/`PASSED` at 0), `draw_class :3149-3251`,
`tier2b_value_mp :3283-3309` (0), `win_adjust :3635-3651` (additive form, 0).

Reference shapes side by side (fraction of the eg lead retained vs the LEADER's pawn count p; EVIDENCE from source):

| p | SF11 `min(64, 36+7p)/64` (`:755`) | SF15.1 same −4·!bothFlanks (`:943-946`) | Ethereal `min(128, 96+8p)/128` | Weiss `(128−(8−p)²)/128`, −20 if one flank | **ours** `clamp(27+34p, 0, 64)/64` (`:3678-3681`) |
|---|---|---|---|---|---|
| 0 | 0.56 | 0.50 (one flank) | 0.75 | 0.50 | **0.42** |
| 1 | 0.67 | 0.61 | 0.81 | 0.62 | **0.95** |
| 2 | 0.78 | 0.72 | 0.875 | 0.72 | **1.00** |
| 3 | 0.89 | 0.83 | 0.94 | 0.80 | 1.00 |
| 4 | 1.00 | 0.94 | 1.00 | 0.875 | 1.00 |
| 6 | 1.00 | 1.00 | 1.00 | 0.97 | 1.00 |

| # | reference detail | ours | likely impact | already tried? | suggested test | effort |
|---|---|---|---|---|---|---|
| E1 | **Pawn-count ramp SHAPE**: all four references scale a leader with 2-3 pawns (SF 0.78/0.89, Ethereal 0.875/0.94, Weiss 0.72/0.80); none saturates before 4 pawns. | Saturates at **2 pawns** (table above); the C3 doc already notes "the scale is inert at ≥2 leader pawns, so the 10-04 K+P loss (5 pawns) is out of its reach BY FORM". EVIDENCE. | Medium. SPECULATION: the fit set SP=34 on static labels and confirmed it on the depth target (+0.04%), so the LINEAR form's optimum is genuine — but a linear clamp cannot express Weiss's concave ramp or SF's 4-pawn saturation. The question is form, not magnitude. | Winnability knobs re-fit 10-05 (depth target, confirmed); form never varied beyond linear. | Structural → replace `SP·sp` by a 5-cell table (p = 0,1,2,3,≥4) fitted on the endgame depth rows (`_win_depth_fit.py` already exists); compare val and the pawn-ending bias (`_eg_leg_inspect.py`). ⚠️ Winnability is a scale on the summed total, hence near move-neutral ([[winnability-moves-evals-but-not-moves]]) — judge by games, expect a small effect. | Low |
| E2 | **OCB with OTHER pieces**: SF11 `:755` uses the 2-per-pawn slope whenever `opposite_bishops()` (any other material); SF15.1 `:923-924` `22 + 3·pieces(strong)`; Ethereal `SCALE_OCB_ONE_KNIGHT` / `SCALE_OCB_ONE_ROOK`; Weiss OCB when `nonPawnCount ≤ 2` each and equal (one extra piece each). ⇒ **4/4 scale impure OCB.** | `ocb` requires `!(knights\|rooks\|queens)` — PURE OCB only (`:3673`). EVIDENCE: a 4/4-universal CONDITION we restrict to its narrowest case. | Medium. The 10-07 bench puts 38% of the endgame excess in rook+minor and 35% in mixed endings (§3a); OCB+rooks/OCB+knights live there. SPECULATION. | OCB cell fitted (−80) on pure OCB only; impure never measured. | Structural → add `POT_V2_WIN_OCB2` (OCB + exactly one extra equal piece pair, or SF15.1's `22+3·pieces` form) at 0; endgame depth rows; gate. | Low |
| E3 | **Pawnless leader rule** (SF11 `material.cpp:198-204`): leader with NO pawns and ≤ a bishop ahead in npm → `factor` 0 if its npm < rook (K+minor vs K+pawns, KBB? no — KmmKm is 4 or 14), else 4/64 (weak side ≤ bishop, e.g. KR vs KB) or 14/64 (e.g. KRB vs KR). Ethereal `SCALE_DRAW` when the strong side is K + one minor. Weiss: pawnScale only (0.50). | `draw_class` zeroes only exact cases (KBvK, KNvK, KBvKB, KNvKN, KBvKN, KNNvK, fortress KBPsK, KPK exact); **K+minor vs K+pawns is NOT in it** (n_nk = 2 but `nk != bishops`), so the minor side, if the eval says it leads, keeps 27/64 = 42% of a ~2.5-pawn lead ≈ +1 pawn. Tier-2b (KR vs minor technique value) parked at 0. EVIDENCE. | Medium for the pure-minor class the bench flags (pure minor −4.5 at d10 persists, §3a) — IF those rows are "minor vs pawns" cases. ⚠️ Tension with the register: "tier-2b … DEAD — tablebase ground truth, zero headroom (5 failures/315)" was a MOVE-failure count on a suite; the 10-07 reading is a win%-BIAS at depth on labelled rows. Different instruments; the DEAD verdict covers KR-vs-minor, not K+minor-vs-pawns. | NEVER-MEASURED for K+minor vs pawns. | Two paths, owner's choice: (a) a `POT_V2_WIN_LONEMINOR` cell at 0 (leader has no pawns and exactly one minor) → fit; or (b) a CLASSIFICATION: a side with K+B or K+N alone can never deliver mate except with the defender's own pawn boxing its king (mate-in-1-class, which the DTM-weighted gate tolerates and search finds) → oracle sweep (`_draw_oracle.py`) then `draw_class` one-sided zero. (b) follows the owner's "classifications ship on oracle proof" rule. | Low |
| E4 | **Queen vs no queen scale** (SF15.1 `:937-939`): `sf = 37 + 3·minors(side without the queen)` → 0.58-0.72 of the eg lead. Single lineage, but it targets the EXACT live error: queen-imbalance misjudgement persists at depth (−5.7 → −4.7pp after Kaufman; MCL narrow classes NEUTRAL as additive mg-weighted terms). | No queen-vs-none input in the scale (`:3669-3679`). MCL (`:2154-2155`) is ADDITIVE and mg-weighted (fades to 0 in the eg — the OPPOSITE phase of SF15.1's rule). EVIDENCE. | Medium-high relative to its cost: the one documented persistent material error, attacked so far only in the additive/mg form. SPECULATION: a multiplicative eg form is what the references use for "hard to convert". | MCL additive form measured neutral (10-01). The SCALE form never. | Structural → `POT_V2_WIN_QVN` (+ `_QVN_MINOR` per minor of the queenless side) at 0; `_win_depth_fit.py` on the endgame rows; check the queen-class bias in `_eg_leg_inspect.py`; gate on both instruments. | Low |
| E5 | **Rook-ending drawishness** (SF15.1 `:929-934`): R vs R, leader ≤ +1 pawn, leader's pawns on ONE flank, weak king adjacent to its own pawn → 36/64. Single lineage; SF11 instead has the KRPKR Philidor family (`endgame.cpp:431-522`) as specialised SCALES. | Nothing rook-specific; pure rook endings −4.4 at d10 persist (§3a). EVIDENCE. | Medium-low. The persistence at d10 is the signal; the record calls the pawn-ending part a SEARCH gap and is silent on rook endings. | Never. | A `POT_V2_WIN_ROOKEND` input at 0 (SF15.1's four conditions as one bit) on endgame rows — cheap probe of whether the class is eval-expressible at all before any design. | Low |
| E6 | **Rule-50 damping**: SF11 `:757` `sf −= (rule50 − 12)/4`. Single lineage (SF; Ethereal/Weiss absent). | No `rule50` input to the eval at all (signature `:4046`); SLICE1-DRAW §6: needs plumbing AND the eval-cache key. EVIDENCE. | Low as Elo; the owner named "avoid drifting into drawn positions" as a motivation. | Flagged, not buildable without plumbing. | Out of scope for the retune; note for search v2 (the hash key change belongs there). | Medium (plumbing) |
| E7 | **Initiative/complexity MG leg**: SF11 `:730` `u = sign(mg)·max(min(complexity+50, 0), −\|mg\|)` — a NEGATIVE-ONLY mg correction when complexity is very low (few pawns, one flank); SF15.1 `:901` same. Ethereal's complexity is eg-only; Weiss none. | Winnability is eg-weighted only (`win_scale_adjust` uses `(256−phase)`; `win_adjust` likewise). EVIDENCE. | Low (fires only when complexity < −50, i.e. nearly pawnless middlegames). | Never. | Skip unless the retune exposes an mg bias in sparse positions. | — |
| E8 | SF applies the scale to the EG LEG and picks the strong side by eg sign (`:745, :818`); ours scales the blended total and picks by total sign (`:3667, :3683`) — exact only when mg = eg or phase = 0. `EVAL_V2_PAIR=1` would make it exact. EVIDENCE. | Known (comment at `:3658-3660`). | Low (endgames have phase ≈ 0). | Pair mode measured (behaviour change, not flipped). | Only when pair mode ships. | — |
| E9 | SF15.1 pure-OCB scale rises with the leader's PASSERS (`18 + 4·passers`, `:920`); SF11 fixed 22. | `POT_V2_WIN_PASSED` exists at 0; the 10-05 re-fit read −0.33% (below the 0.5% bar → final retune). EVIDENCE. | Known. | Measured. | Retune. | — |
| E10 | Ethereal `SCALE_LARGE_PAWN_ADV` (no queens, ≤1 piece each, leader +3 pawns or more) scales the eg lead UP (above normal). Single lineage; the only reference that ever scales > 1. | `f` is clamped at 64 (`:3680`) — cannot scale up. | Low; interesting as a form question (can a scale ever exceed 1?). | Never. | Skip; mention to the owner as a design option only. | — |

---

## 9. PST / PHASE HANDLING / TEMPO

| # | reference detail | ours | likely impact | already tried? | suggested test | effort |
|---|---|---|---|---|---|---|
| S1 | Phase: SF from npm (`material.cpp:132-135`, limits 23.6%/91.9% of start npm); Ethereal/Weiss from piece COUNTS (Q4 R2 minor1, 0-24 / 0-128). | npm-based, same limits as SF (`:265-270`, `EVAL_V2_MG_LIMIT 61700 / EG_LIMIT 15800`). EVIDENCE: SF-faithful, LOCKED. | — | Locked by design. | — | — |
| S2 | SF11 pawn PSQT (`psqt.cpp:93-102`) is file-ASYMMETRIC by design ("asymmetric distribution"); all piece tables are file-mirror symmetric via `map_to_queenside`. | Fit A tables are file-mirror TIED (384 half-board cells). EVIDENCE: ours is symmetric by construction; the symmetry gate (`_eval_symmetry.py`) is a ship gate. | Noted only: SF's pawn-table asymmetry is a deliberate chess-knowledge choice (kingside pawns differ) that our gate forbids. Owner's call whether file-mirror symmetry is a design axiom or a convenience. | — | None. | — |
| S3 | Tempo: SF11 `Eval::Tempo` 28 added after the blend (`:833`); SF15.1 classical has NO tempo (`:1037-1042`, only `(v/16)·16` grain); Ethereal 20; Weiss 18. | `TEMPO_V2_*` 0, parked on margin coupling (replicated), = the STM nuisance in the fits (C3 §20). EVIDENCE. | Closed. ⚠️ The SF lineage itself DROPPED tempo by 15.1, so the 2/4 count is "two lineages keep it, one dropped it". | Measured. | Search lane (stand-pat/side-to-move), as the handoff says. | — |
| S4 | SF15.1 eval GRAIN `(v/16)·16` (`:1037`) — a deliberate 16-unit quantisation (~0.13 pawn) to reduce TT/search noise. Single lineage. | None. | Search-side; not eval knowledge. | Never. | Note for search v2 only. | — |

---

## 10. RANKED TOP-10 — nuances worth a test BEFORE or INSIDE the final retune

Ranking = (prior impact) × (cheapness) × (how badly the record mis-filed it). Every item respects one-owner and names its
lane (STRUCTURAL → depth target + static component, confirmed by a real d10 re-search; DYNAMIC → real d10 first; CLASS →
oracle). None is a ship recommendation.

1. **E4 — queen-vs-no-queen as an EG SCALE input** (SF15.1 `:937-939`). The one persistent material misjudgement
   (−4.7pp at depth) has only been attacked additively/mg-weighted; the references' form for "hard to convert" is a
   multiplicative eg scale. Owner: POT winnability. Lane: structural (scale), `_win_depth_fit.py` on endgame rows.
2. **E2 — OCB with other pieces** (4/4 universal CONDITION; ours restricts to pure OCB). Owner: POT winnability. Rows:
   rook+minor and mixed endings carry 73% of the endgame excess (§3a).
3. **K2 — shelter → danger coupling** (3/4, TWO lineages — register says 2/4). One scalar `u −= k·shelter_mg` fitted
   jointly with the KS attack knobs (the 10-03 method). Owner: KS-A (KS-B stays the direct score). Judge on near-equal
   discrimination + quiet fire rate by phase, never raw fire rate.
4. **K1 — shelter at the castling-target squares** (`KSB_V2_CASTLE` exists, 0, never fitted because the extractor flags
   it unmodelled). SF lineage only, but it is a MECHANISM about which square the shipped cells read. Lane: real d10
   re-search vs identical-conditions ship re-run; static pre-check of fire rate. Fix the extractor first.
5. **P1 — doubled penalty gated on the FRONT pawn's support** (SF11+SF15.1 `!support`; Ethereal's stacked flag is
   support-aware; 3/4, two lineages). Two count columns in the retune. Owner: pawn structure.
6. **E1 — pawn-count ramp as a 5-cell table instead of a linear clamp** (ours is the only one of five that saturates at
   2 leader pawns). Owner: POT winnability. Expect small (near move-neutral form); cheap because the fitter exists.
7. **E3 — K + lone minor as leader cannot win** (SF `material.cpp:198-200` one-sided 0; Ethereal `SCALE_DRAW`).
   Either a winnability cell or an oracle-proved one-sided classification. Owner: draw_class / POT. Check first
   whether the bench's "pure minor −4.5 at d10" rows are this class (one query on the bench dumps).
8. **K3 — safe-check de-duplication + own-queen exclusion** (SF11 `:408-422`; SF15.1 `:569-577`). One knob, refit the
   four CHK cells jointly; pair with `CHK_COUNT=1` to settle the check FORM in one pass. Owner: KS-A.
9. **I3 — closedness-conditioned knight/rook values** (Ethereal; v1 had it dead). 2 × 9 count columns fitted WITH
   Kaufman + piece values (never beside them). Owner: material census.
10. **P2 + P5 + M3 + K6 — four trivial count columns at 0 for the retune** (`WeakLever`, `BlockedPawn` r5/r6,
    `RookOnClosedFile`, king-on-open-file eg sign). Each is single- or two-lineage, each costs one AND, and the retune
    can price or zero them without a design session. Owner: pawn structure / placement / KS-B.

Deliberately NOT in the top-10: everything in the threats table (search lane by owner decision 10-08); the passer
items X1-X3 (already inside the PX gate, queue #42); E6 rule-50 (plumbing + hash key → search v2); L2 king-ring latent
pressure (belongs to the POT mg design, not placement).

---

## 11. OPEN QUESTIONS FOR THE OWNER

1. **Which lane for the winnability shape items (E1/E2/E4/E5)?** They are scale inputs on the summed total, so they
   move evals far more than moves ([[winnability-moves-evals-but-not-moves]]). Fit them on the endgame depth rows as
   structural, but gate by GAMES only with the expectation that the signal is small — or fold them into the POT endgame
   design session rather than the retune?
2. **E3: cell or classification?** "K + lone minor cannot win" is provable up to mate-in-1-class positions, which the
   DTM-weighted gate already tolerates for KBvKB/KNvKN. Do you want it as an oracle-gated one-sided zero (your June
   rule), or priced as a cell and left to the fit?
3. **K2 ownership.** The shelter→danger coupling makes KS-B's output an INPUT to KS-A. Does that violate one-owner
   (KS-B owns shelter, KS-A owns attack units), or is "the same detector feeding two transformations" acceptable as it
   was for the pawn masks feeding passers and placement?
4. **K1 and the universality rule.** The castling max and the shelter→danger feedback were skipped ON-RULE in
   September (REBUILD-LOG `:2628,2642`). Your 09-27 correction says single-lineage ideas enter at 0. Confirm that the
   September "skip" decisions are void and these two are candidates.
5. **S2: is file-mirror symmetry a design axiom?** SF's pawn PSQT is deliberately asymmetric. Our symmetry gate forbids
   it. Keep the axiom (my recommendation: yes — the gate has caught three real defects), or allow a fitted asymmetric
   pawn table as a late experiment?
6. **E10: may a scale ever exceed 1?** Ethereal scales UP a large pawn advantage with few pieces. Our `f ≤ 64` cannot.
   A design choice, not a defect — do you want the option in the POT design?
7. **Threats' nuances (T1-T4)** are parked with threats itself. Do you want them written into the search-transition
   plan now so they are not re-derived later?

---

## 12. RECORD CORRECTIONS SURFACED BY THIS AUDIT

- ☠️ **"shelter→danger feedback (2/4)"** (`EVAL-V2-PARKED-REGISTER.md:80`) is **3/4, two lineages**: Ethereal's
  `evaluateKingsPawns` writes `ei->pksafety[US] += SafetyShelter[…]` / `SafetyStorm[…]` and `evaluateKings` adds
  `ei->pksafety[US]` into `safety` before `-mg·MAX(0,mg)/720` (fetched 2026-10-09, master). SF11 `:455`, SF15.1 `:609`.
- ☠️ **"castling MAX (SF-only)"** is correct as a lineage count (SF11 `pawns.cpp:233-237`, SF15.1 `:281-285`; Ethereal
  absent by fetch; Weiss absent) — but it is NEVER-MEASURED, not DEAD, and the knob already exists (`KSB_V2_CASTLE`).
- ☠️ `eval_v2.cpp:1279-1282` says "SF stacks them [connected and backward]" — SF11 `pawns.cpp:133-147` is an else-if
  chain exactly like Ethereal's; the knob `PS_V2_CONN_EXCL` is vacuous by construction either way (already recorded in
  `EVAL-V2-RUNG2-PAWN-DESIGN.md:556-565`).
- ⚠️ The gap audit's "P3 rook-behind-passer IS CLOSED BY the ladder's rejection" rests on the 09-21 d7 regret read; the
  ladder itself REVERSED on a real d10 re-search 10-08 (C3 §20a) and PX carries own/enemy R/Q-behind as cells 47-48.
  P3 is therefore OPEN inside the PX gate, not closed.
- ⚠️ The register's tier-2b "DEAD — zero headroom" (move-failure count on a suite) and the 10-07 bench's "pure minor
  −4.5 / pure rook −4.4 persist at d10" (win% bias on labelled rows) are different instruments on overlapping classes.
  Neither invalidates the other; §8 E3/E5 name the cheapest way to tell which class carries the bias.
- ⚠️ Tempo's reference count: SF15.1 classical has NO tempo (`evaluate.cpp:1037-1042`); the SF lineage dropped it.
- ⚠️ `EVAL-V2-GAP-AUDIT` K5/K4 single-lineage labels stand; but `KS_V2_UNSAFE` (SF's 148×unsafeChecks) and
  `KS_V2_BLOCKERS` (98×blockers) are now SHIPPED non-zero (30 / −20) via the 10-03 joint fit — the audit's "cheap,
  single-lineage, inputs computed" items were in fact cashed. The register's "SINGLE-LINEAGE — skip" list should drop
  them.

---

## 13. WHAT THIS AUDIT DID NOT DO (so the next reader does not assume it)

- No static counts were run (read-only session with an overnight gate in flight). Every "fire rate" in the test column
  is the FIRST step, not a known number.
- Ethereal and Weiss were read at `master`, not the pinned commits; constant VALUES there may differ from the
  register's pins, and line numbers are not given. Every Ethereal/Weiss claim above is a condition or a shape, not a
  magnitude.
- SF1.1 was not re-read (its term set matches ours and its constants are irrelevant under the shape-not-scale rule).
- `ship_tables_v2.h` (the compiled KS-B / Kaufman / PST cells) was not opened; where a fitted value is quoted it comes
  from `EVAL-V2-CURRENT-CONFIG.md` §1.
