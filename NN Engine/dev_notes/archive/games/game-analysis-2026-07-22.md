# Game analysis + eval-bug findings (2026-07-22)

Four Chessiverse games (user vs "IM John Bartholomew" personality bot, 2434) + one engine-vs-SF18(0.0005s)
self-play, all analyzed with SF18-search depth 18-20 (`diagnostics/analyze_game.py`, moves in
`ks_sets/game_moves.txt`). Chessiverse "personality bots" imitate HUMAN BLITZ incl. human blunders → a nominal
2434 is NOT tactically flawless.

## The chessiverse games (opponent behaviour)
- **G1 (user White, win):** dead equal until **11...Rxh2?? (−4.33)** — bot grabbed h2 while b7/a8 hung; 13.Qxb7
  →15.Qxa8 won a rook. Clean conversion. One-move gift.
- **G2 (user Black, win):** user was actually LOSING — own inaccuracies (11...Na5, 12...c6, **14...Nb7 → +4.1**)
  let the bot build a WINNING attack; the **Greek-gift 15.Bxh7+ was SOUND**. Bot then blundered the follow-up
  **17.Rh4?? (−8.86, best Qd3)** → 17...Bxg5! 18.Qh5 Bxh4 winning. Saved by the bot's conversion error.
- **G3 (user Black, win — the earned one):** equal queenless middlegame; user built a real initiative
  (12...Nb3, 14...Be3 dark-square infiltration vs a stuck Kd1); bot cracked in a CLUSTER (13.Ra2? missing
  axb4, 14.h3?) and user harvested with an accurate knight raid (Nxd2/Nxf1+/Ng3+/Nxh1 winning both rooks).
  A genuine positional squeeze + clean tactics, not a gift.
- Pattern: these bots build real attacks but misfire on precise CONVERSION — human-blitz "collapse" behaviour.

## Engine vs SF18@0.0005s (OUR ENGINE = Black, won by mate)
NOT a strength signal (SF18 lobotomised → near-random); a QUALITY-AT-BLITZ-DEPTH sample. Mutual blunderfest;
value = the specific eval/search mistakes. Classified via `diagnostics/blunder_probe.py` (replay to pre-blunder
ply, ask our engine at deep depth 14: still plays it = EVAL bug; avoids it = SEARCH/depth).

- **28...c5?? = EVAL BUG (deep-14 still plays c5).** FEN `5q1k/7p/2ppRp2/p5p1/2P3P1/Q7/PP3PPK/3r4 b`. Our total
  **−3.21 vs SF11 −0.38** (~2.8p over-read for Black). **`term_dump_fen.py` localises it: Imbalance/OvD −1.57
  vs SF11 0.00** (+ Space −0.49) — our OvD term credits Black ~1.5p for an ILLUSORY attack on White's exposed
  h2 king (Q+R aimed at it) that SF sees as not real. **KingSafety actually UNDER-reads here (−1.38 vs −2.62)**
  → this is an **OvD/imbalance over-read, NOT a KS bug.** Ties to the triple-count (king-zone attacker−defender)
  and the realizability theme: OvD credits an unrealizable attack. Candidate for the OvD side of the joint fit
  (position-specific; blanket IMBALANCE_SCALE↓ was already debunked as a general lever — needs realizability
  conditioning, not a flat cut).
- **44...Ke6?? = SEARCH/depth** (deep-14 plays the correct c4). A blitz-depth artifact, not an eval flaw.

## Tooling added
`analyze_game.py` (SF18 per-move swings + blunder flags; reads `ks_sets/game_moves.txt`), `blunder_probe.py`
(eval-vs-search classifier), `term_dump_fen.py` (per-term ours-vs-SF11 on one FEN).

## Backlog: phase-conditioned capgains REALIZABILITY (user idea, 2026-07-22)
Capgains over-reads (#2 capg+6.0 refuted by a discovered check; #6 capg+0.70) skew worse in sharp middlegames
than quiet endgames. We already have conditioning infra (`ENABLE_CAPG_COND` tension-conditioning,
`SCALE_CAPTURE_GAINS`, realizability-style scaling). IDEA: add a PHASE taper to the capgains realizability
scale. CATCH: need a corpus where capgains actually FIRES (small set). TRACTABLE because capg is a computed
term — MINE positions with |capture_gains contribution| >= ~1.5p from the bank + collapse dataset, cross-ref SF
disagreement → split into "fires-and-misleads" (failure set to shrink) vs "fires-and-helps" (guard). Seed
exemplars already: FEN2 `r1bq1rk1/4npb1/6pp/1pppp3/QP1nP1P1/P1NPB3/3N1PBP/R3K2R w`, FEN6
`4B3/8/P3k3/2p5/2P1pp1P/1P2P3/3r4/1K6 w`. Capgains is NOT wrong to flag these (SF just lacks the term) — goal =
calibrate HOW MUCH to trust it WHEN, not remove it. Folds into the joint fit as a capg-phase knob.

## KEY NUANCE re passers (2026-07-22, verified via raw SF11 tables + SF18-search):
SF is NOT directionally-right because of a better PASSER term — on FEN6 SF's Passed is +1.70 (favors White) the
WRONG way (Black wins, SF18 −3.33); what saves SF is offsetting King safety −1.74 + Pawns/Rooks/Mobility. SF's
passer bonus AGGRESSIVELY credits advanced pawns (h6 +2.76 on FEN5 despite g7; a6 +3.21 on FEN6) and can
over-read; balance + search correct it. LESSON: the fix is JOINT (passer + offsetting KS/threats/mobility for
the defender), NOT a passer boost in isolation (which would blow passers up — the user's standing caution).
Our failure = we under-fire passers AND over-credit material+placement AND lack the offsets. `raw_sf11_dump.py`,
`sf11_breakdown_fens.py`, `positional_collapse_dossier.py`.

## Takeaways
- The OvD over-read of unrealizable king attacks (c5 dossier) is a concrete, SF-grounded eval bug worth a
  realizability-conditioned fix on the OvD term — distinct from the KS work.
- Endgame technique is loose at blitz DEPTH (search), but pure K+P conversion was accurate.
