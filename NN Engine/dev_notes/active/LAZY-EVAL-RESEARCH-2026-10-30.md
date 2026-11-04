# Lazy / cheap evaluation — research for a dedicated transition item (2026-10-30)

Owner (10-30): a dedicated transition item, researched and discussed during idle time, slotted in once the eval is settled
(the eval items done, optimised and retuned). It helps speed in general and gives a cheap static value that other static-eval
consumers can use. Two open questions from the owner, answered from the reference code below:
1. Does Stockfish have a cheap eval AND a lazy eval, or is the lazy exit simply returning the cheap part?
2. How does a lazy exit avoid misjudging sacrifices and positional compensation, which can outweigh material?

Two separate things share the name (EVAL-TRANSITION-ITEMS-2026-10-30.md §2d): (A) a lazy exit inside the eval — this item;
(B) a staged node-entry eval for search consumers (`improving`, null eval gate, corrhist) — the search arc.

## 1. What the reference engines do (read from source)

**SF11** (`stockfish_11_linux/.../src/evaluate.cpp`). The eval is built from cheapest to most expensive, and the lazy exit
returns the cheap part itself — there is no separate cheap function:
- `score` = material + piece-square (updated incrementally on make / unmake) + material imbalance (material hash) + pawn
  structure (pawn hash). All nearly free.
- `:790-793`: `v = (mg + eg) / 2; if (|v| > LazyThreshold + non_pawn_material / 64) return v;` with `LazyThreshold = 1400`
  (`:77`). SF11 pawn ≈ 128 mg / 213 eg, so the threshold is roughly 8 pawns of cheap-part imbalance, plus ≈ 1.5 pawns with
  full material on the board (the npm term grows with material).
- Only past the exit: pieces, mobility, king safety, threats, passed pawns, space, then initiative and the scale factor. So
  initiative and scaling only ever act in the near-balanced band (memory `winnability-moves-evals-but-not-moves`: our v1 lacked
  this domain gate — a correctness fault, not only speed).

**SF15.1 classical** (`stockfish_15_linux/.../src/evaluate.cpp`). Two checkpoints:
- `:991-995`: `lazy_skip(T)` = `|mg + eg| > T + |thread bestValue| · 5/4 + non_pawn_material / 32`.
- Checkpoint 1 (`:997`, T = 3631) after material / PST / imbalance / pawns; checkpoint 2 (`:1017`, T = 2084) after pieces,
  mobility, king safety and passers — before threats, space and winnable().
- The `|bestValue| · 5/4` term: when the game itself is already lopsided (root score large), the threshold rises, so the exit
  does not swallow every node of a won game.
- `:1059`: in the hybrid, the CLASSICAL eval is used instead of NNUE when more than 7 pieces remain and |psq| > 1760 — the
  cheaper evaluator for decided positions ("~3 Elo").

**SF18** (`stockfish_18_linux/src/evaluate.cpp`). A true cheap evaluator, compensation-aware:
- `:43-49`: `simple_eval` = material only; `use_smallnet` when |simple_eval| > 962 (≈ 4.6 pawns).
- `:61-70`: the small net evaluates; **if its result comes back near level (|nnue| < 277 ≈ 1.3 pawns), re-evaluate with the
  big net** — "when higher eval accuracy is worth the time spent". Material says lopsided, the cheap evaluator says close ⇒ that
  is compensation, and the full evaluator is called.

**The rest of the field (engine-contrast 10-30; read: SF16.0 / 16.1 / 17.0 / 17.1 / 18 / 19, Ethereal, Weiss, Xiphos, Laser,
Berserk; not read: Obsidian, Stormphrax, Koivisto, Gyatso):**
- **No other HCE engine read has a lazy exit** — Ethereal, Weiss, Xiphos, Laser evaluate fully (special-endgame early-outs only).
  They save time by conditional skips instead: Ethereal scores king safety only when `kingAttackersCount > 1 − enemy queens`,
  and its centre-space term only with enough heavy pieces; Weiss accumulates attack units only when attacks or checks exist.
- **Material / PST are incremental in every reference read** (updated on make / unmake) — their cheap part is free.
- **No alpha / beta-aware lazy eval anywhere**; the only search-aware threshold is SF15.1's `bestValue` term.
- **SF lineage**: SF16.0 classical fallback when |material + PST| > ≈ 9.8 pawns → SF16.1 small net at |material| > ≈ 5 pawns,
  no re-check → SF17.0 adds the big-net re-check when the small net says near level (or disagrees in sign) → SF17.1 / SF18
  thresholds 236 / 277 → **SF19 removes the small net entirely** (one network; `stockfish_19_linux/src/evaluate.cpp`).
- **Correction history × a cheap evaluator** (SF17.1-19): the search calls one `evaluate()` and corrects whatever answered —
  same tables, same keys, no guard; the TT stores the uncorrected value. The evaluator choice is a pure function of the position,
  so one position never mixes the two, and the material keys already separate the lopsided regime.

**Our side (verified):** material and PST are recomputed in every eval (`eval_v2.cpp` PST loop, `build_context` census). Code
order today: material / PST / Kaufman → the attack build (KS units, mobility, threats; unconditional) → pawn entry → passers and
placement (both read the attack maps). The search already computes a node-entry static eval at null-window nodes with 1-6 plies
left, for reverse futility and the null eval gate (`search_engine.cpp` `rfp_want_eval`), and that is the value corrhist corrects
(both corrhist forms are off at defaults).

**Design implications:** (1) make material / PST incremental first — exact and value-neutral, and the precondition for a cheap
part that is actually cheap; (2) use a position-only threshold (SF11's material term), not SF15.1's `bestValue` term, which would
break the eval cache; (3) reorder so the pawn entry and pawn structure come before the attack build, and put a second checkpoint
after king safety (SF15.1's shape) so a king attack is always counted — the HCE answer to compensation; (4) the HCE references'
alternative or complement is conditional skips, e.g. king-safety units only when attackers exist (result-identical if our units
are exactly 0 then — to check).

## 2. Sacrifices and compensation — how they are kept safe
1. **The threshold is huge.** SF11 ≈ 8-10 pawns of cheap-part imbalance; SF15.1's first checkpoint larger still. Ordinary
   sacrifices (a pawn, the exchange, even a piece ≈ 3 pawns) sit far below it, so the full eval runs on them.
2. **It grows with material on the board** (npm / 64 or / 32): more pieces = more scope for an attack = harder to fire.
3. **It grows with the game's own score** (SF15.1 `|bestValue|·5/4`).
4. **Sacrifices are found by the search, not by the static eval.** A working sacrifice pays off inside the horizon (mate,
   material regained); the eval at the moment of the sacrifice does not need to know. What a lazy exit risks is long-term
   compensation the search cannot resolve — exactly what the large threshold keeps out.
5. **SF18 checks the cheap answer** and pays for the full eval when the cheap evaluator itself says "not so lopsided".
6. A second checkpoint after king safety (SF15.1) means an attack on the king is always counted before the late exit fires.

## 3. What it would mean for us (to verify before building)
- Our cheap part would be material + PST + Kaufman imbalance + pawn structure (the pawn hash, `PawnEntry`). Check whether our
  material / PST are incremental (SF's are) or recomputed per eval — it decides how cheap "cheap" is.
- Expensive part: the attack build (KS-A units, threats, mobility, placement), KS-B / KFL / KPROT cells, the POT scale.
  The 10-22 profile put eval at ~21-27% of search time; gen 6.4 added threats.
- A lazy exit changes evals only in lopsided positions ⇒ not result-identical: symmetry gate + games, like any eval change.
- Our eval is cached (`EVAL_CACHE_FORM 2`); a lazy value is a pure function of the position, so caching stays consistent as
  long as the threshold does not read search state (SF15.1's `bestValue` term does — a choice to make).

## 4. Measurement plan (before any games)
1. Counter (behind the diagnostic guard): the fraction of eval calls in real search where |cheap part| exceeds candidate
   thresholds (6 / 8 / 10 pawns), by phase.
2. Time: the share of eval time spent in the terms a lazy exit would skip (part of the gen 6.4 re-profile).
3. Error where it fires: |full − cheap| on those positions, and how many change side of level or cross a likely α/β window
   (the compensation cases) — on the WDL corpus and on positions sampled from game trees.
4. Then games at equal time, with the margins left as shipped; the margins re-sweep stays LAST.

## 5. Placement
**Owner 11-04:** in the transition this is an exploration — how much of (A) we can get done and prove so far, not a finished
system; it carries on into the search arc, where the other NPS and EBF helpers and the search features that read the static
eval (part B) should make it matter more.
After the eval items and their retune (the owner's condition), before the final margins re-sweep. Separately, the damp-only
drawishness threshold being built now can carry its own near-balance gate (as SF's initiative effectively does), without
waiting for the lazy exit.
