# POT — knowledge distillation: positional TRANSFORMATIONS (2026-09-30)

Research agent report (Opus, from its own knowledge — NO web lookups this pass; chapter numbers approximate, the
Pálsson & Björnsson citation and any pawn-structure-clustering literature UNVERIFIED). [INF] = the agent's inference.
Purpose: the owner's "work backwards" method (C3 doc §17) — transformation type → end result → precursors → early signs →
owning subsystem, before any detector is designed. POT = "OvD reworked" (the owner's v1 invention; lineage in §13).

## 1. Human theory (condensed)
- **Steinitz** (Lasker's Manual, Book V): the side with the advantage must attack or it evaporates; small advantages
  ACCUMULATE and are DISCHARGED at a transformation point — the origin of "potential". King may stay central in closed
  positions IF the opponent cannot open it.
- **Soviet school — transformation of advantages:** Pachman *Modern Chess Strategy* (permanent vs temporary; advantages
  exchanged for others); Kotov *Think Like a Grandmaster* (temporary elements must be converted to permanent before they
  fade); Watson *Secrets of Modern Chess Strategy* (exchange of advantages, dynamic equilibrium); **Suba *Dynamic Chess
  Strategy* — "dynamic potential" almost literally: a flexible structure holds latent energy from breaks AVAILABLE BUT NOT
  PLAYED; keeping tension keeps it, committing spends it** — the closest literary match to POT's potential/kinetic split.
- **Nimzowitsch *My System*:** restrain–blockade–destroy; attack a pawn chain at its BASE (the lever is the only way to
  transform a locked chain; play on the side the chain points to); open file = result, rook behind the future break =
  precursor; prophylaxis / over-protection; a flank attack needs a closed/stable centre.
- **Kmoch *Pawn Power in Chess*** — the most mechanisable: ram, LEVER (pawn contact), duo, sweeper, candidate, majority,
  cram, holes. **Levers are the agent of change; rams freeze the structure.** [INF] ⇒ the balance of existing + reachable
  levers is the natural "unresolved structural energy" variable.
- **Silman** (imbalances; create the position where your imbalance dominates; dynamic → static).
- **Dvoretsky / Yusupov / Aagaard:** prophylaxis — the opponent's POTENTIAL (a freeing break) judged by what it allows;
  "don't hurry"; keep vs release tension.
- **Petrosian / Karpov** (Marin *Learn from the Legends*): exchange sac = material → structure; prophylaxis kills breaks
  before they are playable (removes the opponent's potential).
- **Shereshevsky *Endgame Strategy*:** two weaknesses; do not hurry; TRANSITION into the favourable endgame (trades
  change which advantages count).
- **Flores Rios *Chess Structures* / Soltis *Pawn Structure Chess*:** a structure family DEFINES its breaks, and the
  breaks ARE the transformation menu (Carlsbad minority / e4; IQP d5 + attack vs blockade; hanging pawns; Hedgehog …b5/…d5;
  closed KID c5 vs …f5-f4-g5; French …c5/…f6 vs d4; Stonewall; Benoni; Najdorf; Caro/Slav). [INF] detectable from ram/lever
  patterns on files c-f.
- **Closed centre + king in the centre** — the human test is NOT "is the king central?" but: (1) can the centre be opened
  (central lever / supported lever-push)? (2) are the attacker's heavy pieces behind the break? (3) how many tempi until
  the defender can castle? King on e1/e8 behind a d/e ram with no central lever = safe for now.

## 2. Taxonomy
| Type | End result | Precursors (detectable) | Early signs | Owning subsystem |
|---|---|---|---|---|
| T1 Central opening vs uncastled king | open d/e files, exposed king | king on d/e, castling pending/lost; central lever or supported lever-push; development lead; heavy pieces on d/e | "centre opens before he castles" | KS |
| T2 Flank storm vs castled king | open g/h (b/c) files, broken shelter | closed/stable centre; unblocked storm pawns; hook pawn; opposite castling | Nimzowitsch's centre rule; the race | KS (shelter/storm) |
| T3 Chain-base attack | chain dissolved / base fixed, half-open file | ram + chain; base lever reachable; pieces on the base | "attack the base" (KID, French) | pawn structure, files |
| T4 Majority → passer | passed pawn | healthy majority, candidate; fewer pieces; far king | "majority + trades" | passers, endgame |
| T5 Minority attack / fix weakness | fixed backward/isolated pawn on half-open file | minority vs majority; half-open file; lever reachable | Carlsbad | pawn structure, rook files |
| T6 Freeing break | cramp relieved / space converted | space imbalance; freeing lever; pieces behind it | "if he gets …d5 in, he's fine" | mobility / space |
| T7 Trade-down conversion | favourable endgame | a static plus that grows in the endgame; trades available | Shereshevsky's transition | endgame scaling (winnability) |
| T8 Good vs bad minor / outpost | locked bad bishop, permanent outpost | rams on one colour; unguardable hole; knight route | Silman minor-piece imbalance | placement, pawn structure |
| T9 Second weakness | overload, material | one fixed weakness + a lever on the other flank | two weaknesses | material (after) |
| T10 Dynamic → static | initiative cashed | development lead, open lines, tension that must resolve | Steinitz, Kotov | material/structure — KINETIC, mostly not POT |
| T11 Exchange sac | material for blockade/domination | strong minor square; closed; rooks lack files | Petrosian, Karpov | material vs placement |
| T12 Blockade / restraint | opponent's potential cancelled | blockade square held by a minor; over-protected point | Nimzowitsch; prophylaxis | passers (blockade), mobility |
| T13 Tension resolution / capture choice | new structure family | mutual levers | recapture judgement | pawn structure |

**Priority (frequent AND statically detectable) [INF]:** T1 · T2 (POT's addition = the closed-centre gating; storm itself is
owned by KS-B) · T4 · T3 · T5 · T7 (= the winnability leg) · T8.
**Common thread:** Kmoch lever balance (existing + reachable), gated by rams and by whether the RESULT already exists.

## 3. Computer / AI studies
- **McGrath et al. 2022, PNAS "Acquisition of chess knowledge in AlphaZero":** linear probes decode SF8 eval terms and
  hand concepts; ALL probed concepts are STATIC.
- **Schut et al. 2023 (arXiv 2310.16410), "Bridging the Human–AI Knowledge Gap":** "DYNAMIC concepts" learned from the
  contrast of the chosen MCTS line vs sub-optimal rollouts — temporal, plan-like (often prophylactic quiet ideas);
  taught to GMs. The nearest analogue to POT.
- **Jenner et al. 2024 (NeurIPS), learned look-ahead in Lc0:** the policy net represents moves 2 plies ahead (tactical scale).
- **Lc0 WDL draw mass + moves-left head; community "sharpness" from WDL** [INF]: learned proxies for "how much can still happen".
- **Stockfish classical:** `initiative` → `winnable` (static proxy for future winning potential = our eg leg);
  Kmoch terms in `pawns.cpp` (`lever`, `leverPush`, `phalanx`, `blocked`); **`BlockedStorm`** (rammed storm pawn = no
  threat — a mechanical resolved/unresolved split); space weighted by `blockedCount` (SF12+).
- **Guid & Bratko 2006:** complexity = how often the best move changes across depths (search-derived "unresolved").
- **No published study** predicting WHEN files open / kings get exposed from static features [INF] — an open,
  measurable question (label from game sequences: open k plies later?).

## 4. Unresolved vs resolved (structure, never score — owner rule, C3 §17)
- **Unresolved (potential live):** levers (central tension); reachable lever-pushes (supported); central rams WITH a
  base lever available; king on c-f with castling pending while the centre has break potential; candidate passers /
  unfixed majority; heavy pieces behind a break; high piece count (trade-down potential).
- **Kinetic (owner term has it):** open/half-open files in the king zone; passer created; shelter already gone; tension
  resolved.
- **Resolved AND quiet (the key case):** castled king, intact shelter, no enemy lever reachable on the shelter files, no
  semi-open file toward it; a fully rammed lever-free structure; symmetric, no majority. ⇒ owner term ≈ 0 AND potential ≈ 0.
- **Gating rule [INF]:** POT speaks for type k when k's PRECURSORS are present AND its RESULT is absent; silent when the
  result appears (owner takes over) or the precursors die (lever gone, break permanently blocked, king castled into an
  intact shelter).

## 5. What this means against our record (my reading, 2026-09-30)
- The §14 null tested lever/tension/majority features UNGATED (no precursor-present ∧ result-absent gate, no owner-state
  gate) and scored them as a pawn-STRUCTURE delta, not as the owner subsystem's FUTURE. The literature's variable is the
  same; the gating and the target are what was missing ⇒ the null does NOT refute the gated form.
- Owners already built in v2: KS (+ KS-B shelter/storm at 0), pawn structure, passers/candidates, mobility, placement,
  winnability (§16). POT must carry only their LATENT future.
