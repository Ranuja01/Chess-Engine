# POT (Potential) — RESEARCH PRE-PLAN for the owner design discussion (2026-10-08/09)

**Status: INPUT FOR A DISCUSSION, NOT A DECISION.** Read-only research; nothing built, run or edited. Every claim is tagged
**[EVIDENCE: file:line | URL]** or **[SPECULATION]**. Where a past closure is quoted, it is classified **RESOLVED** (the
instrument was fair for the question) or **UNREADABLE** (biased instrument / different form / never fired) per the
"fifteen nulls" rule (`EVAL-V2-INVENTORY-2026-09-25.md`).

**Lineage (owner's request, 2026-09-29):** POT (Potential) = "OvD reworked" — the owner's v1 long-term-pressure invention
(credited with taking the engine from struggling vs 1600 bots to competing with 2000s on chess.com), redesigned king-free
[EVIDENCE: memory `ovd-is-the-owners-long-term-pressure-concept.md:11-13, 42-45`; `eval_v2.cpp:57-63`].

## 0. The binding definition (owner, 09-30 / 10-05 / 10-07) and what it rules in and out

- POT = the POTENTIAL for either side (tempo to the mover) to STRUCTURALLY transform the position so that another subsystem
  later "shines through"; it speaks only where that subsystem is UNRESOLVED, judged from STRUCTURE (can the owner's INPUTS
  still change?), **never from the owner's output**; it recedes as the position becomes kinetic
  [EVIDENCE: `TEXEL-C3-DETECTORS-DESIGN-2026-09-27.md:693-729` (§17); memory `ovd-…:51-59, 114-123`].
- It FEEDS / MODULATES existing terms; no re-scoring of an owned concept; one owner per concept; collinearity check before
  code; few features, each with its own signal [EVIDENCE: memory `ovd-…:76-81`; C3 §14 `:611-615`].
- Winnability (endgame convertibility, multiplicative scale) is a separate, SHIPPED owner (`POT_V2_WIN*`,
  `win_scale_adjust`) [EVIDENCE: `eval_v2.cpp:3653-3685, 4349-4356`; C3 §16 `:667-691`; gate +20.2 Elo, memory `ovd-…:70-75`].
- Timing-tempo belongs to POT; a flat side-to-move bonus does not (closed on MECHANISM: a constant shared by every sibling
  cannot reorder moves) [EVIDENCE: `EVAL-V2-SLICE1-TEMPO-DESIGN.md:236-237, 263`; C3 §8 `:269-272`].
- Must be cheap: v2 eval is ~11.5% of node cost; pawn-only features live in the pawn cache and are ~free per node
  [EVIDENCE: memory `MEMORY.md` → `the-nps-gap-is-mostly-not-eval`; C3 §8 `:278`].
- The 10-01 middlegame screen tested POT types as ADDED SCORES and was null; the feeder/modulator form is UNTESTED
  [EVIDENCE: `POT-TYPE-DEFINITIONS-2026-09-30.md:163-182`; memory `ovd-…:120-123`].
- Order (owner 10-08): known structural elements (KFL, passers — queue #39-#43, the gate running now) → **POT design (mg =
  structural, design + fit now; eg races/tempo/entry = design now, FIT LATER)** → structural retune → search transition
  (SF11 pawn-ending rules, corr hist, threats) → dynamic-lane retune incl. "eg-POT dynamic parts"
  [EVIDENCE: `SESSION-HANDOFF-2026-10-07.md:122-130`].

### 0a. Two data facts that reshape the endgame side (added 10-07 evening)
1. The pawn-ending persistence at d10 (−9.3 win% pts) is mostly **OUR SEARCH**, not the eval: SF11 d10 bias −1.0 vs ours
   −8.5 on the same 313 pawn-only rows (ours d14 −5.8). SF11 switches off null move and all shallow pruning when the mover has
   only pawns and extends passed-pawn pushes [EVIDENCE: `REFERENCE-BENCH-LADDER.md:364-373` (§3a); SF11 `search.cpp:846, 998,
   1079`; memory `the-remaining-eval-gap-is-endgame-structural.md:19-25`]. Pawn endings are statically NOT worse than SF11
   (0.87× on all 26 own-play rows; the 1.57× rested on a tiny val slice) [EVIDENCE: same §3a].
2. No static eval (ours / SF11 / SF15c) sees the tempo / race / opposition verdicts in the 9 owner-reviewed pawn endings
   [EVIDENCE: `REFERENCE-BENCH-LADDER.md:372`; cases in `diagnostics/ks_sets/pe_cases_fens.txt:2-10`]. ⇒ eg-POT "dynamic
   potential" must not claim what the search rules will fix; its honest value is **static discrimination** (pruning /
   stand-pat / ordering read statics mid-search — owner 10-07) [EVIDENCE: memory `static-discrimination-matters-…:11-25`].
   The one eval-side K+P fact that search cannot rescue: **~1/3 of SF-DRAWN K+P positions read decisive (|cp| > 150) by
   our d10 SEARCH** (both arms of the connected gate) — the owner's 10-04 loss at scale; the shipped scale is INERT at ≥ 2
   leader pawns BY FORM [EVIDENCE: C3 §19f `:994-996`; §20a `:1071-1072`; `OWNER-GAMES-ANALYSIS-2026-10-01.md:52-61`].

## 1. Record check — what was already tried, and whether each closure is RESOLVED or UNREADABLE

| item | what was measured | verdict | class |
|---|---|---|---|
| POT mg types T1/T3/T4/T5 as ADDED gate scores vs the depth residual (SF18 d14 − our d10), 8,128 rows, beyond 68 KS channels + 184 C1/C3 + stm | all \|σ\| < 2.2 raw, < 1.5 beyond controls; castling-race KS feeder 1.2σ at depth | NULL, PARKED | **RESOLVED for that form and that label**; UNREADABLE for (a) the feeder/modulator form, (b) potential beyond ~14 plies (invisible to the SF18 d14 label — the stop rule's own caveat) [EVIDENCE: `POT-TYPE-DEFINITIONS…:163-182`] |
| §14 mg screen: lever_now / tension / mobile_majority / projected-winnability inputs vs SF18 search labels | all ≤ 1.1σ except "passed" −4.3σ (owned by passers) | NULL | **RESOLVED for ungated COUNT features on the STATIC residual**; UNREADABLE for gated / outcome-resolved forms (the knowledge doc's own reading) [EVIDENCE: C3 `:627-634`; `POT-TRANSFORMATION-KNOWLEDGE…:86-91`] |
| T4 flank-majority precursor | event lift **1.65× (+18.8σ)** in our d6 games; T1/T3 lever-reach lifts ≤ 1 | the only demonstrated precursor | RESOLVED-POSITIVE as a PREDICTOR of passers; its VALUE beyond passers + search read +0.9σ (added-score form) [EVIDENCE: `POT-TYPE-DEFINITIONS…:147-161, 172`] |
| §8 "LEVER OUTCOME" feature (resolve each lever with v2's own pawn scorer; best Δ per side discounted by moves needed) | listed as first-cut feature 1; the §11/§14 pilots tested lever COUNTS (`lever_now` fires 3.5%), never the outcome-resolved Δ | — | **NEVER BUILT** [EVIDENCE: C3 §8 `:279-281` vs §11 `:550-552`, §14 `:628`] |
| §8a move-class screen: transformation-critical positions 1.6-2.5%, traps ~3× commoner; `transform_critical.csv` (382) / `transform_trap.csv` (1,146) built; our engine passes at d8/d14 "needs engine slots" | the outcome fit is POT's instrument; the move test is the harm check | sets exist, **never engine-passed** | NEVER RUN [EVIDENCE: C3 `:292-313`] |
| v1 pawn-majority term (06-25) | median \|our−SF\| at d10 flat/worse; static attribution said placement/capgains drove the error, pawn terms negligible | PARKED | **UNREADABLE for v2**: contamination-era harness, d10 search proxy, v1's capgains/placement terms are gone in v2 [EVIDENCE: memory `pawn-majority-term.md:19-21`; `EVAL-V2-PARKED-REGISTER.md:101`] |
| Winnability additive form (Fit W, Fit W-SF) | −24.9 / −22.9 Elo on fresh seeds | NEGATIVE | **RESOLVED — the FORM** (sign(T)·C discontinuous at 0); the scale-factor form then passed +20.2 [EVIDENCE: C3 `:591-595, 635-644, 688-691`] |
| v1 winnability "moves 81% of evals, changes 0.5% of meaningful moves" | a near-monotone scale on the SUMMED TOTAL cannot reorder siblings | mechanism | RESOLVED, and it CONSTRAINS DESIGN: a total-scale cannot be a discriminator; a feeder into ONE term's leg can [EVIDENCE: memory `winnability-moves-evals-but-not-moves.md:19-26, 78-79`] |
| Flat tempo (TEMPO_V2_MG/EG) | identity gate exact; node effect flips sign with margins | PARKED at 0 | RESOLVED on mechanism; TIMING tempo untested [EVIDENCE: `EVAL-V2-SLICE1-TEMPO-DESIGN.md:236-237, 263-290`] |
| PX passer system (51 cells incl. king escort 36-46, square rule 50) | 10-04 "adds nothing at depth" (root-Δ proxy) → REVERSED by real d10 re-search: **−5.56% vs ship** (#40/#41); gate = queue #42/#43 (RUNNING, 4 slots) | pending | the proxy-era closure was UNREADABLE; the real read is in; **do not touch until the gate reports** [EVIDENCE: C3 `:1058-1067`; memory `root-delta-depth-proxy-…:12-20`] |
| KFL (king–pawn distance, pawnless flank) | real d10 −2.78% vs ship; costs PX's endgame when stacked | pending gate | same as above [EVIDENCE: C3 `:1063-1065`; `eval_v2.cpp:3446-3467`] |
| Winnability knobs re-fit on the depth target | shipped knobs confirmed (+0.04%); +PASSED −0.33% | below bar | RESOLVED: the scale as formed is priced; its INPUT SET is the open question [EVIDENCE: C3 `:1068-1072`] |
| Tier-2b K+R vs K+minor technique gradient | 200-position TB suite: base engine 0 blunders in 33 opportunities | MOVE-NULL, parked | RESOLVED for play; the accuracy case survives (AUC 0.48, +2.1 pawns over-read) for the NNUE teacher [EVIDENCE: memory `kr-vs-kminor-…:11-32`] |
| Binary draw classes + exact KPK bitbase | `draw_class` NO-false-positive set; `DRAW_V2_KPK_EXACT` shipped | shipped | RESOLVED; anything "usually drawn, sometimes won" is a MAGNITUDE → scale, never a bool [EVIDENCE: `eval_v2.cpp:3130-3251`; memory `endgame-draw-detection.md:70-87`] |
| "Eval payoff is opening/midgame, not deep endgame" (09-07) vs "the gap is endgames" (10-07) | both measured | both stand | NOT a contradiction: 09-07 = how much a BETTER eval buys regret per phase (mg 3× eg); 10-07 = where OUR DEFICIT vs SF11 sits (endgames). ⇒ eg-POT closes a known deficit; mg-POT, if it works, pays where eval matters most [EVIDENCE: memory `eval-payoff-…:14-23`; C3 §21 `:1134-1139`] |
| KS-B storm vs POT storm ownership | owner 09-30 | KS-B owns king-directed storms; POT's storm = non-king lever REACH as a precursor | RESOLVED [EVIDENCE: `POT-TYPE-DEFINITIONS…:102-106`] |
| Heat map / v1 OvD form | retired decomposed, mostly KS | — | RESOLVED; do not re-port [EVIDENCE: `eval_v2.cpp:46-60`; memory `ovd-…:33-34`] |

**Reading:** the middlegame concept has two genuinely untested forms (feeder/modulator; outcome-resolved lever Δ) and one
untested instrument (a label with a longer horizon than SF18 d14). The endgame side has one item with hard evidence waiting
(passer-creation potential as a winnability input) and a set of "dynamic" items whose value will only be legible AFTER the
pawn-ending search rules (owner's order: design now, fit later).

## 2. Chess-knowledge distillation (what GMs / endgame theory say; sources)

The 09-30 knowledge doc already condensed the strategic literature (Steinitz accumulation/discharge, Soviet "transformation of
advantages", Nimzowitsch, Kmoch, Silman, Dvoretsky/Yusupov/Aagaard prophylaxis, Shereshevsky, Flores Rios/Soltis structure
families) [EVIDENCE: `POT-TRANSFORMATION-KNOWLEDGE-2026-09-30.md:9-36`]. This section adds what that pass lacked — the
ENDGAME dynamics and web-checkable citations — and extracts the mechanisable statement of each idea.

### 2a. Middlegame transformation potential
- **Kmoch, *Pawn Power in Chess* (1959)** — the mechanisable vocabulary: lever (pawns in mutual capture contact — "the agent
  of change"), ram (mutual blockade — "freezes the structure"), duo, candidate/sentries/helpers, majority, frontspan/rearspan
  [EVIDENCE: https://www.chess.com/blog/SonofPearl/pawn-power- ; https://chessprogramming.org/Hans_Kmoch]. ★ `passer_potential`
  already implements Kmoch's CLASSIC candidate (helpers ≥ sentries, no own pawn ahead) [EVIDENCE: `eval_v2.cpp:2841-2881`].
- **Nimzowitsch, *My System*** — attack a chain at its BASE; a flank attack is sound only with a stable/closed centre;
  restrain–blockade–destroy; prophylaxis. **Carlsbad minority attack**: White's numerical MINORITY is a "qualitative majority"
  because its pawns are free to advance to attack the base (b4-b5 vs c6) [EVIDENCE:
  https://userpages.cs.umbc.edu/sherman/Chess/masterprep/lectures/fall95/lesson12.html ;
  https://thechessworld.com/articles/openings/the-queens-gambit-the-minority-attack-in-the-carlsbad-pawn-structure/].
  Mechanisable statement: the VALUE of a lever is the structure it leaves behind (a fixed backward pawn on a half-open file),
  realisable only if the lever can still be played — [SPECULATION] exactly the §8 "lever outcome" feature.
- **Suba, *Dynamic Chess Strategy* (1991, BCF Book of the Year)** — "the accumulation of potential"; a flexible structure holds
  latent energy from breaks AVAILABLE BUT NOT PLAYED; committing spends it [EVIDENCE: https://www.newinchess.com/dynamic-chess-strategy ;
  https://www.chess.com/article/view/my-bookshelf]. This is the literary twin of the owner's potential/kinetic split
  [SPECULATION as to equivalence].
- **Closed centre + king in the centre** — the human test is not "is the king central?" but (1) can the centre be opened
  (lever / supported lever-push), (2) are heavy pieces behind the break, (3) tempi until the defender castles
  [EVIDENCE: `POT-TRANSFORMATION-KNOWLEDGE…:33-36` (the 09-30 distillation; book-sourced, unverified by URL)].
- **Shereshevsky, *Endgame Strategy*** — "do not hurry", centralise the king, fight for the initiative, exploit TWO WEAKNESSES,
  exchange the right pieces at the right moment; think schematically [EVIDENCE: https://www.newinchess.com/chess-endgame-strategy-shereshevsky].
  Mechanisable: "two weaknesses" = the defender's king cannot cover two distant targets ⇒ file DISTANCE between targets is the
  structural quantity [SPECULATION].

### 2b. Endgame dynamic potential (the part the 09-30 pass did not cover)
- **Key squares / critical squares (K+P vs K)** — pawn on ranks 2-4: three key squares two ranks ahead; ranks 5-6: six; 7th:
  adjacent 7th/8th-rank squares; rook-pawn exception (only two; defender draws by reaching the c/f file). "Opposition is a
  MEANS to an end; the end is PENETRATION to a key square" (Averbakh). Win if two of {king in front, opposition, king on 6th}
  [EVIDENCE: https://en.wikipedia.org/wiki/King_and_pawn_versus_king_endgame ; https://en.wikipedia.org/wiki/Opposition_(chess)].
  Books: Müller & Lamprecht *Secrets of Pawn Endings* (2007), *Fundamental Chess Endings* (2001); Fine & Benko *Basic Chess
  Endings*; Averbakh *Chess Endgames: Essential Knowledge*; Dvoretsky *Endgame Manual*.
- **Rule of the square** — the defending king catches an unassisted pawn iff it can enter the square whose side is the pawn's
  distance to promotion; the side to move shifts the boundary by one [EVIDENCE: same Wikipedia page; engine form at
  https://www.chessprogramming.org/Unstoppable_Passer — Fruit's `king_passer`: king outside the square OR own king controls the
  whole frontspan]. ★ Our PX cell 50 is the stm-FREE conservative version (defender more than ONE step outside) and the exact
  KPK bitbase owns K+P vs K [EVIDENCE: `eval_v2.cpp:3525-3526, 3562-3570`; `:3219`].
- **Opposition (direct / distant / diagonal) = a special zugzwang**; distant opposition with an odd number of squares between
  kings belongs to the side NOT to move [EVIDENCE: https://en.wikipedia.org/wiki/Opposition_(chess)].
- **Corresponding squares** — squares of reciprocal zugzwang; generalises opposition to blocked positions (Lasker–Reichhelm
  1901; Halberstadt & Duchamp 1932; Grigoriev; Dvoretsky; Mednis; Müller & Lamprecht)
  [EVIDENCE: https://en.wikipedia.org/wiki/Corresponding_squares]. Not statically computable in general (it IS a retrograde
  search) — the honest cheap proxy is parity of tempi + who is to move [SPECULATION].
- **Reserve (spare) tempi** — a pawn move that does not change the structure but passes the move; the side with MORE spare
  tempi wins the zugzwang fight ("White has two spare tempi f3/h3, Black one …f6 ⇒ 1.h3 f6 2.f3 wins"). Dvoretsky's Manual has
  a chapter "Reserve Tempi" (exploiting them; both sides having them); Müller & Lamprecht §"Reserve Tempi"
  [EVIDENCE: https://en.wikipedia.org/wiki/Tempo_(chess) ; https://shop.chess-tigers.de/cdn/shop/files/dem5_excerpt.pdf (contents)].
- **Triangulation** — losing a move with the king to hand the zugzwang back; needs a spare triangle of squares the defender's
  king cannot mirror [EVIDENCE: https://en.wikipedia.org/wiki/Opposition_(chess) (section on triangulation)].
- **Outside passed pawn = DECOY**: "separated by several files from the rest of the pawns", it deflects the defending king from
  the other pawns (Fischer–Larsen 1971); protected passers and connected passers ("steamroller") are the other named classes
  [EVIDENCE: https://en.wikipedia.org/wiki/Passed_pawn].
- **King races** — both kings running (to a passer, to the enemy pawns): a Chebyshev-distance comparison plus side-to-move;
  pe_cases #3/#6 are of this class [EVIDENCE: `pe_cases_fens.txt:4, 7`].
- **Engine practice for pawn endings**: opposition / corresponding squares "start to dominate" in closed structures; the
  recommended engine answer is SEARCH-side (switch off null move; Rebel extended 3 plies on entering a pawn ending)
  [EVIDENCE: https://www.chessprogramming.org/Pawn_Endgame]. This matches §0a-1: the pawn-ending verdict belongs to search;
  the static side's job is not to mislead the pruning.
- **AI side**: AlphaZero "dynamic concepts" are learned from the contrast of the chosen line vs rollouts and were teachable to
  four top GMs (Schut, Tomašev, McGrath, Hassabis, Paquet, Kim — arXiv 2310.16410; PNAS 2025)
  [EVIDENCE: https://arxiv.org/abs/2310.16410v1 ; https://oatml.cs.ox.ac.uk/publications/202504_Schut_AZConcepts.html].
  Relevance: the concepts a strong searcher knows and a static eval lacks are PLAN-shaped; labels for them need a horizon
  longer than d14 or ground truth (tablebases) [SPECULATION as to transfer].

## 3. What the reference engines actually encode — and what NONE of them do

### 3a. Stockfish 11 (classical) [EVIDENCE: local source]
- `initiative()` — complexity = 9·passed + 11·pawns + 9·outflanking (king file-dist − rank-dist) + 12·infiltration (a king past
  mid-board) + 21·pawnsOnBothFlanks + 51·pure-pawn-ending − 43·almostUnwinnable − 100; applied to the (mg,eg) PAIR, sign-capped
  (never flips) [`evaluate.cpp:699-737`]; applied AFTER a lazy exit ⇒ only near-balanced positions [`:790-793, 813`].
- `scale_factor()` — OCB with only bishops: 22; else min(sf, 36 + (OCB ? 2 : 7)·strong-side pawns); rule50 decay
  [`:743-761`]; material.cpp pawnless rule: leader without pawns and ≤ a bishop ahead ⇒ DRAW / 4 / 14 [`material.cpp:196-204`].
- `endgame.cpp` named recognisers: value fns KXK, KBNK, **KPK (bitbase)**, KRKP, KRKB, KRKN, KQKP, KQKR, KNNKP, KNNK(=draw);
  scaling fns KBPsK (wrong-bishop rook pawn FORTRESS), KQKRPs, KRPKR (Philidor/Lucena-type rules), KRPKB, KRPPKRP, KPsK,
  KBPKB, KBPPKB, KBPKN, KNPK, KNPKB, **KPKP** (remove the weak pawn, probe the bitbase; exception for a pawn on the 5th+)
  [`endgame.cpp:118-806`].
- `passed()` — rank table; for r > 3: both kings' distance to the stop (capped 5, weight 5r−13) + own king's distance to the
  square after the stop; unsafe-squares ladder 35/20/9 (+5 if stop defended); candidates scaled [`evaluate.cpp:574-636`].
- Search: null move and ALL shallow pruning gated on the mover having non-pawn material; passed-pawn push extension for the
  killer; qsearch futility skips advanced pawn pushes [`search.cpp:846, 998, 1079, 1475`].
- Pawns: `BlockedStorm` — a rammed storm pawn is no storm (the one mechanical resolved/unresolved split)
  [EVIDENCE: C3 §8 `:265-267` cites `pawns.cpp:186-215`; not re-read here].

### 3b. Stockfish 15.1 classical [EVIDENCE: local source]
- `winnable()` — same inputs, outflanking now SIGNED (file-dist + rank difference), infiltration 24, −110 [`evaluate.cpp:871-903`].
- `scale_factor` — richer: pure OCB 18 + 4·strong passers; other OCB 22 + 3·strong pieces; **rook endgame, ≤ 1 pawn up, strong
  pawns on ONE flank, weak king touching its pawns ⇒ 36**; lone queen 37 + 3·(weak side's minors); else 36 + 7·pawns −4·oneflank
  (twice) [`:907-947`].
- `space()` — weight = pieces − 3 + min(blocked_count, 9): the ONLY reference encoding of "closed centre", and it REWARDS space in
  closed positions (a kinetic read, not a potential) [`:831-857`].

### 3c. Ethereal (master, fetched; line numbers unverified) [EVIDENCE: https://raw.githubusercontent.com/AndyGrant/Ethereal/master/src/evaluate.c]
- `evaluateComplexity` — eg-only: 8·pawns + 82·bothFlanks + 76·pawnEndgame − 157; sign-capped; **no king inputs**.
- `evaluateScaleFactor` — OCB variants (bishops only / +1 knight each / +1 rook each), lone queen vs pieces, strong side =
  K + one minor ⇒ DRAW, large pawn advantage, else min(normal, 96 + 8·strong pawns).
- `evaluatePassed` — table[canAdvance][safeAdvance][rank], per-rank friendly/enemy king distance, safe-promotion-path S(−49,57).
- Absent: dynamic tempo (flat `Tempo` only), opposition, levers/breaks, closed centre.

### 3d. Weiss (master, fetched; unverified) [EVIDENCE: https://raw.githubusercontent.com/TerjeKir/weiss/master/src/evaluate.c]
- `ScaleFactor` — pawnScale = 128 − (8 − strong pawns)²; −20 if pawns on one flank only; OCB 64/96. **No complexity term.**
- Passers: `PassedDistUs[r]`, `PassedDistThem·(rank−3)`, defended, blocked, free advance, rook behind, **`PassedSquare`**
  (square rule, defender without pieces; S(−26,422) per our passer design doc [EVIDENCE: `PASSER-SYSTEM-DESIGN-2026-10-03.md:30`]).
- Flat `Tempo = 18`. Absent: levers, majorities, opposition, initiative.

### 3e. What NONE of the four encode (= the room for a unique-where-better feature)
[EVIDENCE: absence across 3a-3d; SPECULATION where marked]
1. **Opposition / corresponding squares / zugzwang parity** beyond the KPK bitbase (SF only) — no reference has a tempo-parity
   or reserve-tempi count.
2. **"Can the leader EVER create a passer?"** — every complexity term counts passers THAT EXIST (SF `passed_count` includes its
   narrow one-exchange candidate) and pawns; none asks whether the leader's pawns are all opposed / rear-doubled. Our
   `passer_potential` (Kmoch classic candidate) is the missing input [`eval_v2.cpp:2841-2881`].
3. **Outside-passer decoy geometry** as a distance between the passer (or would-be passer) and the DEFENDER'S OTHER DUTIES;
   references price king-to-passer distance only.
4. **A forward-looking value for levers / breaks** — SF's `lever`/`leverPush` exist only to DETECT connected/passed pawns;
   no reference scores what a playable break would leave behind (Kmoch's lever as "agent of change").
5. **Closed-centre gating of flank play** (Nimzowitsch) — only `BlockedStorm`'s local predicate; SF15's blocked_count rewards
   space, which is the kinetic reading.
6. **Majority → passer potential before any candidate exists** (T4) — Ethereal's "large pawn advantage" scale is a count, not
   a majority on a flank.
7. **Timing tempo** (moves needed to execute vs the opponent's) — all four use a flat side-to-move constant.
8. **King-entry ROUTE** — SF's `infiltration` is a rank bool; none ask whether the pawn structure admits an entry (holes).
Items 1-3 are endgame; 4-7 middlegame; 8 both. All are single-lineage-or-less, so under the owner's rule they enter at weight 0
and are priced by the fit and games, never adopted on authority [EVIDENCE: memory `unique-where-better-never-self-nerf.md:11-17`].

## 4. Candidate catalogue — FEEDER / MODULATOR form only (nothing added as a stand-alone score)

Conventions: **Owner** = the shipped term whose leg/scale the candidate feeds. **U** = the structural "unresolved" detector
(must be computable without reading the owner's output). **Cost class**: P = pawn-only (lives in the pawn cache, ~free per
node); P+K = pawn + king squares (cheap, but the king is not in the pawn hash — a few popcounts per eval); A = needs the shared
attack maps (already built for mobility/PX; a few extra ANDs). **Measure** always = (i) Python prototype on existing dumps
(`fitC_{mg,eg}_ours1004_d10_s*of4.csv`, `kp_stress_*`, `pe_rows_*`) with controls + the no-overlap check (`_pot_mg_screen.py`
style, max |corr| vs the 106 + C3 + PX features) → (ii) C++ at 0, byte-identical fingerprints (254 / 50,622,239 / 4.029),
closure + colour/file symmetry → (iii) **REAL d10 re-search** (`_revival_screen.py MODE=dual VALOUT` → `_depth_residual_pass.py`
4 shards → `MODE=dualread`) against an IDENTICAL-CONDITIONS ship re-run (harness null −0.10%) → (iv) K+P stress held-out
check (`_kp_stress_check.py`, a SEED never used for fitting) → (v) SF18 @1000 + self-play, fresh seeds, pooled, ~2σ combined
[EVIDENCE: memory `root-delta-depth-proxy-…:18-20`; `SESSION-HANDOFF-2026-10-07.md:145-148, 214-217`; `_kp_stress_check.py:2-11`].

### 4a. Endgame POT (eg) — "dynamic potential"; hands over to winnability

**E1. Passer-creation potential → WINNABILITY input** (owner's 10-04/10-07 item) — ★ highest evidence
- Owner/consumer: the shipped scale `win_scale_adjust` (`POT_V2_WIN*`), i.e. the `f` factor. Form: `f -= NOPOT ·
  [passer_potential(leader) == 0]`, optionally graded by the defender's material (pawn-only strongest, fades with their npm)
  and by leader pawn count (the owner's case had 5 pawns; today's SP cell is inert there) [EVIDENCE: `eval_v2.cpp:3664-3685`;
  C3 `:1071-1072`].
- U (structure): every leader pawn is opposed or rear-doubled, and no classic candidate exists ⇒ the structure alone cannot
  produce a passer; a king raid could (that is the king-activity input's job, E2) [EVIDENCE: `passer_potential` header `:2851-2853`].
  Kinetic limit: the moment a candidate/passer appears the mask is non-empty and E1 is silent — PX/passers own it.
- Cheap proxy: already built, pawn-only, exported as probe slots 27-28 [`eval_v2.cpp:2896, 2930`]. Cost class P (cache the two
  masks in `PawnEntry`).
- Overlap: PX cells price TRUE passers (disjoint when the mask is empty); winnability SP/ONEFLANK/PASSED cells (PASSED fits at
  +5 for −0.33%: the "passers exist" direction — E1 is the complementary "cannot exist" direction, so check collinearity with
  PASSED explicitly); Layer A `candidate` (SF's narrow one-exchange test; a strict subset) [EVIDENCE: `:2844-2849`; C3 `:1069-1070`].
- Measure: target = the "~1/3 of SF-drawn K+P read decisive" share (`kp_stress_sf18.csv` W/D/L agreement) and win% MSE on
  pawn-only + minor-only eg rows; pe_cases #9 (truth 0.00; ours +0.55…+0.88 static) as the hand check; the protocol above.
  ★ For ≤ 7-man rows, label with the **Lichess tablebase** (`_draw_oracle.py` already queries it) — exact WDL, no horizon — a
  genuinely new label for the endgame side [EVIDENCE: memory `endgame-draw-detection.md:28-30`; SPECULATION on row counts].
- Risks: it is still a scale on the total (move-neutral by the §1 law) — its payoff is CALIBRATION (refusing to over-press,
  accepting repetition) and static discrimination for pruning, not move choice; the self-play instrument shares the blind spot
  (both arms over-press), so expect SF18 to read it and self-play to read ~0 [SPECULATION].

**E2. King-entry / infiltration potential → modulator for KFL and the PX king-escort cells** (design now, fit after the
search rules)
- Owner/consumer: KFL (own/enemy king–pawn distance cells, `kfl_cells`) and PX_KD_* (king distance to the stop). Both say where
  the king IS; neither asks whether the structure ADMITS the king: a locked chain with no hole is sealed (kinetic); a chain with
  an unguarded entry square is live [EVIDENCE: `eval_v2.cpp:3446-3467, 3522, 3588-3590`].
- U: pawn-only "entry map" = squares on ranks 5-7 (relative) not in the enemy pawns' attack span and not blocked, reachable
  from the king's current region; U = 1 if an entry square exists within k king steps, 0 if none. Cheap proxy: the enemy pawn
  attack-span fill ∧ ¬pawn occupancy (pawn-only, class P for the map; P+K for the distance). pe_cases #1/#2/#4 are this class.
- Overlap: KFL (direct), SF-style `infiltration` bool (in `win_inputs` in[3], not in the shipped scale), KS-A in middlegames
  (keep eg-weighted). Fit AFTER the pawn-ending search rules so that the static read is not confounded by what search then
  fixes [EVIDENCE: `SESSION-HANDOFF-2026-10-07.md:126-130`].
- Measure: static sibling-ordering on the 313 pawn rows + TB labels for ≤ 7 men; real d10 re-search on eg rows.

**E3. Reserve-tempi / zugzwang parity → the TIMING tempo POT is allowed to own** (design now, fit later)
- Owner/consumer: none exists — this is POT's own leg by the owner's rule ("tempo returns only as timing"). Consumer options:
  (a) a tiny SIGNED eg term gated to LOCKED pawn endings only; (b) a modulator on the eg scale (`f`) when the mover has no
  spare tempo and the position is locked (likely drawn/kinetic) [EVIDENCE: C3 §8 `:269-272`].
- U: pawn-only board or pawns + symmetric single minor; rams present (locked); kings in the pawn zone. Proxy: per side,
  count pawns whose push square is empty, not attacked by an enemy pawn, and does not create a lever (a "free pawn move");
  parity = (spare_us − spare_them) with the side to move; this is the second legitimate reader of `c.turn` (the KPK rule
  already reads it for the same reason) [EVIDENCE: `eval_v2.cpp:3235-3240, 4310-4311`]. Cost class P for the counts; the
  parity application is one compare.
- Overlap: nothing reads tempo today; `TEMPO_V2_*` at 0. Risk: zugzwang truth is retrograde (corresponding squares) — the proxy
  is a coarse majority-of-cases rule; a WRONG parity read in a locked ending is an over-confidence error of the worst kind ⇒
  gate on TB labels with NO-false-decisive as the acceptance (the owner's asymmetric rule for draws) [EVIDENCE: `eval_v2.cpp:3138-3139`].
- Measure: TB-labelled locked K+P rows (pe_cases #1, #4, #5); `_kp_stress` dense rows exceed 7 men → SF18 d14 only.

**E4. Outside-passer decoy geometry → PX / passer_potential** (measure only AFTER the PX gate)
- Owner: PX (cell 49 file-from-edge, KD_THEM) already prices part of it for TRUE passers. POT's addition would be the SAME
  geometry for the classic-candidate (not yet passed) pawn on the far flank, discounted. Overlap HIGH ⇒ not a first candidate;
  if PX ships, fold as a `passer_potential`-masked PX-style cell rather than a POT feature [EVIDENCE: `eval_v2.cpp:3525, 3554`].

**E5. Drawishness in pure rook / pure minor endings (d10 persists −4.4 / −4.5)** — this is WINNABILITY's input set, not POT,
by the owner's definition; listed because the owner asked. SF15.1 has three single-lineage inputs we lack: rook ending ≤ 1
pawn up + one flank + weak king touching its pawns ⇒ 36/64; lone-queen vs minors; OCB graded by passers / pieces
[`stockfish_15.1 evaluate.cpp:914-943`]. Enter as cells at 0 in `win_scale_adjust`; fit with E1 on the eg depth rows by type
(`_endgame_types.py` strata). Not a POT candidate — a winnability extension.

### 4b. Middlegame POT (mg) — "transformation potential"; structural, fit now

**M1. Lever-outcome feeder → PAWN STRUCTURE legs** (the §8 feature never built; T3 + T5 in one mechanism)
- Owner/consumer: pawn structure (isolated / doubled / backward / weak-unopposed) and connected pawns. Form: for each lever a
  side can play now or by one supported push, resolve the exchange on the PAWN bitboards only and re-score the resulting
  structure with `build_pawn_entry` + the structure scorer; Δ_k = (score after) − (score now) for the side holding the lever;
  POT_mg = Σ_sides best Δ_k · P_k · U_k with P_k discounted by moves-to-execute (the timing tempo, side to move first)
  [EVIDENCE: C3 §8 `:279-281`]. U_k = 1 while the lever is unplayed, 0 once the exchange has happened (the structure term then
  owns the result).
- Cheap proxy: pawn-only ⇒ pawn-hash resident. Cost: one extra `build_pawn_entry` + structure score per playable lever on a
  pawn-hash MISS only (typically 0-4 levers); [SPECULATION] < 2% NPS at a normal hit rate; measure.
- Overlap: the owner terms score the RESULT only (silent now); `lever`/`blocked` detectors exist (probe slots 12-15); KS-B
  storm (exclude levers on the shelter files of a castled king — king-directed storms are KS-B's); mobility/space (a freeing
  break's value is mostly space — T6 deferred). Collinearity check vs the 106 C1 features + KS-B cells + PX before C++.
- Measure: (i) depth residual on mg rows beyond controls (the 10-01 screen's exact instrument, now with an outcome-resolved
  feature instead of a gate count); (ii) ★ the SIBLING-ORDERING metric the owner asked for — tension resolution is a
  sibling-ordering question by nature (T13), so build `_move_match_arms.py`-style ordering agreement vs SF18 on the
  `transform_critical.csv` (382) / `transform_trap.csv` (1,146) sets (harm check: no rise in premature transformations)
  [EVIDENCE: C3 `:300-313`; memory `static-discrimination-…:23-24`]; (iii) real d10 re-search; (iv) games.
- Predictions (registered): raw depth-residual r beyond controls 1-3σ (the 10-01 counts read ≤ 1.5σ; an outcome-resolved Δ
  should be sharper or the concept is dead on this label); ordering agreement on `transform_critical` +2…+6 pp; no change on
  `transform_trap`; NPS ≥ −2%; games 0…+10 Elo on both instruments. If (i) < 1σ beyond controls AND (ii) flat ⇒ the mg lever
  form is closed on a fair test (feeder form + ordering metric).

**M2. Heavy pieces behind a CLOSED file × openability → ROOK-FILE / mobility correction** (T8', has a ready target)
- Owner/consumer: rook files (`ROOKFILE_V2_OPEN/SEMI`, built at 0, −0.21% on the depth target) and mobility. §18a found we
  ALREADY over-credit heavy pieces on closed king files (−5.0σ beyond controls): an existing term scores closed-file pressure as
  kinetic. Form: credit on a CLOSED file ∝ P(open) where P(open) ∈ {0: rammed, no lever reach; ½: lever reachable; 1: lever
  now}; this is a CORRECTION shape (it can only reduce today's over-credit), so it is also the first POT item with a negative
  prior on magnitude [EVIDENCE: `POT-TYPE-DEFINITIONS…:113-117`; C3 `:1017`].
- Cheap proxy: per-file ram/lever/lever-reach flags from the pawn entry (class P) × rook/queen file occupancy (one AND).
- Overlap: rook files term itself; KS attack channels (rook on a king file); PX_RQB (rook behind a passer). Run the collinearity
  check vs KS channels first — §18a's −5.0σ was measured beyond those controls, so it is presumably not KS [EVIDENCE: same].
- Measure: the −5.0σ finding re-run with the P(open) feature as the candidate; real d10 re-search; prediction: beyond-controls
  signal 2-4σ; fit −0.2…−0.6%; games ≤ +8.

**M3. Majority → passer potential (T4) → PASSER mg leg feeder** (owner: "IS POT")
- Owner/consumer: passers/candidates (`passer_value_mp` reads `passed | candidate`, eg leg only; mg legs re-priced alone HURT:
  +4.06% real d10, −22 Elo in 10-04 games) [EVIDENCE: `POT-TYPE-DEFINITIONS…:59-66`; C3 `:969-971, 1059`]. Form: a SMALL mg
  credit per `passer_potential` pawn that is NOT yet a Layer-A candidate/passer (U = 1), scaled by P(passer) (the 1.65× lift),
  silent once a candidate qualifies. The classic-candidate mask is sharper than "flank majority" (it requires helpers ≥
  sentries) — the 10-01 gate used the loose majority.
- Cheap: already built (class P). Overlap: candidates (eg), Kaufman cells (pawn count × material), §14's "we OVER-rate the
  leader's mg edge when passers exist" lead points the other way ⇒ sign is for the fit.
- Measure: depth residual beyond controls with the sharper mask; prediction: 1-2σ raw (T4 gate read +0.9σ with the loose
  gate); honest prior: LOW-MODERATE; cheap to screen in Python (slots 27-28 are exported) before any C++.

**M4. Closed centre × stuck king → KS modulator (T1)**; **M5. Closed centre → KS-B storm modulator (T2)**
- Owner: KS / KS-B. Form: KS danger (or storm cells) × (1 + α·P_open(centre)) for a king on d/e with castling pending; storm
  cells × [centre closed] (Nimzowitsch) [EVIDENCE: `POT-TYPE-DEFINITIONS…:27-47`].
- Record: the T1 gate read −2.2σ RAW (wrong sign) and the castling-race feeder faded at depth (1.2σ); T2's content is largely
  `BlockedStorm`-shaped and KS-B shipped jointly 10-03 [EVIDENCE: `:169, 176-177`; C3 §20 ship notes]. Prior: LOW for both.
  Cheap to screen on existing dumps (central ram / lever-reach features exist in `_pot_coverage.py`); do NOT build C++ unless
  the multiplier form shows ≥ 2σ beyond the KS/KS-B cells.

**Deliberately NOT proposed:** a flat tempo at any magnitude (closed on mechanism); re-porting the heat map (retired
decomposed); "projected winnability" (overlaps winnability + structure by construction, null in §14); an ungated mobile
majority (owned); a K+R-vs-minor technique gradient (move-null on a TB suite); a 5th passer VALUATION mechanism (v1
graveyard; v2's PX is the detection-side answer) [EVIDENCE: `EVAL-V2-SLICE1-TEMPO-DESIGN.md:263`; `eval_v2.cpp:46-52`;
C3 `:614-615`; memory `kr-vs-kminor-…:11`; memory `passer-law-…:16-17`].

## 5. Recommended first three prototypes, measurement plan, predictions

**Order of work (respecting the 10-08 order):** all three can be PROTOTYPED IN PYTHON NOW on existing dumps while the PX/KFL
gate occupies the engine slots (no engine, no build, no ChessAI import needed for the pawn-only features if the Python mirror
in `_pawn_term_overlap.py` is extended with `passer_potential` — it is validated 8/8 against the C++ probe
[EVIDENCE: `eval_v2.cpp:2883-2905`]). C++ only after the gate finishes and the owner approves.

1. **E1 — passer-creation potential as a winnability input** (eg; fit now — it is structural, not a race).
   - Why first: the owner's own 10-04/10-05 item; hard evidence (1/3 of SF-drawn K+P read decisive; the scale is inert at ≥ 2
     pawns by form); detector built and probe-exported; cost ~0; owner = shipped winnability ⇒ no new subsystem.
   - Plan: (a) Python: extend the pawn mirror with the classic-candidate rule; compute the leader-mask-empty flag on
     `kp_stress_sf18.csv` + `fitC_eg_ours1004_d10` pawn-only/minor-only rows; fit `f` with the new cell(s) in `_win_depth_fit.py`
     (depth target) and read the W/D/L agreement; TB-label the ≤ 7-man subset (exact truth). (b) C++: `POT_V2_WIN_NOPOT` (and
     a graded-by-npm variant) at 0, byte-identical; closure; symmetry. (c) real d10 re-search vs identical-conditions ship.
     (d) held-out K+P stress with a NEW SEED. (e) SF18 @1000 + self-play, fresh seeds.
   - Predictions: SF-drawn-read-decisive share on `kp_stress` 33% → 20-25% (both arms re-run); win% MSE on pawn-only eg
     rows −3…−8%, on ALL eg rows −0.1…−0.5% (rare class); games: SF18 +3…+12, self-play −3…+5 (shared blind spot ⇒ the
     instruments MAY split — pre-registered); NPS within noise. Failure mode to watch: a false "cannot create a passer" when a
     king raid can (E2's job) — read the TB-labelled false-decisive→false-draw flips, must be ~0.

2. **M1 — lever-outcome feeder into pawn structure + the sibling-ordering metric** (mg; structural; fit now).
   - Why second: the one untested FORM of the middlegame concept with a stated owner, a cheap pawn-hash implementation, and an
     instrument that matches the owner's 10-07 principle (ordering) — plus two ready-made test sets never engine-passed.
   - Plan: Python resolution of levers with the existing pawn mirror and v2's structure constants (`ship_tables_v2.h`) on
     `fitC_mg_ours1004_d10` rows; depth residual beyond controls; ordering agreement on `transform_critical/trap` (needs engine
     slots — AFTER the gate); then C++ at 0 if ≥ 2σ beyond controls OR ordering +2 pp.
   - Predictions: see M1 above. Stop rule: < 1σ AND ordering flat ⇒ close the mg lever form on a fair test; the remaining
     mg hypothesis would then be the LABEL HORIZON only (SF d20+ labels on a 2k-row sample).

3. **M2 — heavy-piece-behind-closed-file openability correction** (mg; has a −5.0σ target already on record).
   - Plan: reproduce §18a's −5.0σ with the current ship's dumps; add P(open) as the candidate; collinearity vs KS channels and
     rook-file cells; real d10 re-search; games if the fit ≥ 0.3%.
   - Predictions: beyond-controls 2-4σ; fit −0.2…−0.6%; NPS 0; games ≤ +8 (a correction, not a new signal).

**Designed now, fitted later (after SF11's pawn-ending search rules land):** E2 (king entry) and E3 (reserve tempi) — write
the detectors and the TB-label harness now so the measurement is one command when the search arc reaches pawn endings.
**Measured only after the PX gate:** E4, M3 (both overlap PX / candidates). **Screen-only, low prior:** M4, M5.

## 6. Open questions for the owner

1. **E1 form:** should "leader cannot create a passer" act only on pawn-only boards (strongest, safest) or grade with the
   defender's non-pawn material (a knight can still blockade; a bishop pair cannot stop a king raid)? And graded by leader pawn
   count, or a flat cell?
2. **Labels:** may eg-POT use tablebase WDL (≤ 7 men, Lichess endpoint) as the ground truth instead of SF18 d14 for pawn
   endings? It removes the horizon caveat that parked mg-POT, at the cost of mixing two label sources in one fit.
3. **Timing tempo (E3):** is a second legitimate reader of `c.turn` acceptable in the eval (the KPK rule is the precedent), given
   the "eval is side-to-move-blind" symmetry gate (`_eval_symmetry.py` mirrors `turn`, so it cannot check it)?
4. **M1 scope:** should lever outcomes include levers against a castled king's shelter files (KS-B's storm territory) or exclude
   them by construction? My proposal: exclude (king-free POT).
5. **The ordering metric:** build it as an extension of `_move_match_arms.py` (per owner "don't grow the pile") — agreed? Which
   SF18 depth defines the reference ordering (d14 as in the labels, or deeper)?
6. **Pairing with the retune:** POT cells are to be fitted BEFORE the giant retune so it prices them (owner 10-07) — but E2/E3
   are "fit later". Do they enter the retune as frozen zeros, or does the retune wait?
7. **Horizon test for mg-POT:** if M1/M2 are null on SF18 d14 labels, is a one-off SF18 d20+ labelling of ~2k mg rows worth the
   engine time to settle the "potential beyond 14 plies" caveat, or is mg-POT then closed?
8. **Name/credit:** confirm the shipped-block naming "POT (Potential) — OvD reworked" and the OvD credit line for the canonical
   docs when any POT cell ships (your 09-29 request).
9. **E5:** treat the SF15.1 rook/queen/OCB scale inputs as a winnability extension inside the eg retune (not POT) — agreed?
10. Anything from your own games that is a TRANSFORMATION we have not listed (the Bg7/Ba3 diagonal question from 09-30 is
    still open)?

## 7. Cross-reference index (files read for this pre-plan)
Memory: `ovd-is-the-owners-long-term-pressure-concept`, `the-remaining-eval-gap-is-endgame-structural`,
`root-delta-depth-proxy-is-biased-against-dynamic-terms`, `unique-where-better-never-self-nerf`,
`static-discrimination-matters-even-when-search-fixes-the-verdict`, `winnability-moves-evals-but-not-moves`,
`pawn-majority-term`, `endgame-draw-detection`, `kr-vs-kminor-…`, `eval-payoff-is-opening-midgame-not-endgame`,
`passer-law-…-graveyard`, `passer-defect-is-blockade-cost-blindness`, `ks-b-shelter-was-deferred-not-rejected`,
`realizability-conditioning-architecture`. dev_notes: `POT-TYPE-DEFINITIONS-2026-09-30.md`,
`POT-TRANSFORMATION-KNOWLEDGE-2026-09-30.md`, `TEXEL-C3-DETECTORS-DESIGN-2026-09-27.md` §8, §11-21a,
`SESSION-HANDOFF-2026-10-07.md`, `REFERENCE-BENCH-LADDER.md` §3a, `OWNER-GAMES-ANALYSIS-2026-10-01.md`,
`PASSER-SYSTEM-DESIGN-2026-10-03.md`, `EVAL-V2-PARKED-REGISTER.md`, `EVAL-V2-SLICE1-TEMPO-DESIGN.md`, `DIAGNOSTICS-TOOLKIT.md`,
`INSTRUMENT-MAP.md` (headers). Code: `eval_v2.cpp` (`passer_potential`, `pawn_entry_probe`, `kfl_cells`, `px_counts`,
`win_inputs`/`win_adjust`, `win_scale_adjust`, `draw_class`, tempo block, winnability application); diagnostics
`_static_vs_search_cases.py`, `_pe_search_bias.py`, `_kp_stress_check.py`, `gen_kp_fens.py`; `ks_sets/pe_cases_fens.txt`.
References: SF11 `evaluate.cpp`, `material.cpp`, `endgame.cpp`, `search.cpp`; SF15.1 `evaluate.cpp`; Ethereal and Weiss
`evaluate.c` (GitHub master, fetched; line numbers unverified); chessprogramming.org (Pawn Endgame, Unstoppable Passer);
Wikipedia (K+P vs K, Opposition, Corresponding squares, Passed pawn, Tempo); arXiv 2310.16410.

---
## Framing notes from the owner conversation (2026-10-09, before the POT discussion)
Owner: winnability is now STANDALONE (not POT's endgame half); POT = intuitively capturing POTENTIAL ("not yet kinetic").
Proposed three-quantity split (for discussion):
- **Winnability** — can the edge that exists NOW be converted? (multiplicative scale; material config, pawn count, OCB, "leader
  can ever create a passer", fortress/blockade, material realizability, queen-vs-no-queen scale).
- **POT magnitude = complexity** — how much can the position still change (direction-free). SF11's "Initiative" (SF12+
  "winnable") and Ethereal's complexity are THIS, lumped with convertibility — which is why SF files *infiltration* there.
- **POT direction = potential** — WHO benefits from what is unresolved; TIMING TEMPO decides races/entry/zugzwang, so it lives here.
  The flat side-to-move bonus stays a search margin; "initiative" in the human (forcing-play) sense is kinetic → threats/search.
- Ownership rule: decides whether an EXISTING edge converts → winnability; describes an edge that does not exist yet but could →
  POT. Owner's distinction applied: infiltration ALREADY happened = kinetic; the POSSIBILITY (entry squares, defender too slow)
  = POT ("entry potential"). Meeting point to settle: "no potential left ⇒ drawish" (the 10-04 game) — one owner only.
- Also: reserve-tempi / zugzwang parity (encoded by no reference) · tablebase WDL labels as the horizon-free instrument for ≤7-man
  eg-POT · later (search v2): a cheap "how unresolved" signal could drive LMR/time management (unique-where-better candidate).
- ☠️ **NAMING (10-09): our "winnability" ≠ SF's "winnable".** SF11 = `initiative()` (additive complexity; mg can only DAMPEN —
  `min(complexity + 50, 0)` — eg either way) + `scale_factor()` (multiplicative eg scale). SF12+/15.1 renamed initiative →
  `winnable()` AND folded the scale factor into it. Ethereal = `evaluateComplexity()` (king-free) + `evaluateScaleFactor()`.
  Weiss = scale factor only. OURS: "winnability" (`POT_V2_WIN`) ≈ SF's `scale_factor()` only; the old additive `WIN_V2` ≈ SF11
  initiative (failed on form). In our scheme: complexity → POT magnitude; scale factor → winnability.
- **POT's two halves exist PER PHASE** (owner): mg complexity (tension, levers, centre, unresolved KS) / mg potential (whose
  transformation works) · eg complexity (pending races/entries, reserve tempi) / eg potential (who wins the race/entry; tempo).
  Separate mg/eg legs like every v2 term. SF11's asymmetry is a design hint: mg complexity mostly = "less certain" (dampen),
  eg complexity can mean "more winning" too.
- **CONSOLIDATION NEEDED (owner 10-09).** The giants run TWO mechanisms here; we have one: (a) the multiplicative eg SCALE —
  shipped, but missing SF15.1's queen-vs-no-queen, OCB-with-pieces, rook-ending and lone-minor-leader rules (EVAL-NUANCES doc) →
  structural, do them with the known elements before the structural retune; (b) COMPLEXITY — absent (WIN_V2 failed on its
  additive form; a continuous form is untested) → POT's magnitude half. Draw/repetition behaviour (10-04 game) re-checked in the
  search transition where eval meets repetition.
- **mg POT must push EITHER WAY (owner):** its POTENTIAL half is directional — "this trade gives us a good pawn structure / an
  isolated passer" favours one side. SF's "mg only dampens" applies to its COMPLEXITY only (SF has no potential term). So: mg
  complexity mostly dampens · mg potential both ways · eg both halves both ways. Guard rails: score the OPTION not the execution
  (once the trade happens, search and the pawn/passer terms own the result); POT feeds/modulates the owner term (likelihood of
  the structure), never prices the structure itself.
