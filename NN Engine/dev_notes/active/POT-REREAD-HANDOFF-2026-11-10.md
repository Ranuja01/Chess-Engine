# POT reread: a one-time analysis brief (2026-11-10)

**For:** a one-time analysis by another model (Fable 5.2), handed over by the owner. **Not** a build request: read,
analyse and propose. Nothing here asks for code changes.

## 1. The engine and where it stands

AlphaUX is a single-threaded, hand-written-eval (HCE) C++ chess engine. Production is gen 7.1 (11-09). The eval
arc and the transition arc are closed. The next arc is **search v2**, a rebuild of the search into the modern
structure: negamax, node-type templates and a per-ply stack, a node-entry static eval feeding the standard pruning
consumers, then the standard pruning, reduction and extension set. The owner's own ideas are kept beside the standard
parts behind switches, and games decide. Plan: `dev_notes/active/SEARCH-V2-PLAN-2026-11-10.md`.
Units are millipawns (P 1000, N 3250, B 3450, R 5000, Q 10000).

Measured gap to Stockfish 11 (11-10, plan §6.1): SF11 reaches depth 13 in 1/24 of our time, about 80% of that from
tree size. Our branching factor is flat at about 1.9 from depth 6 to 13; SF11's falls to 1.46.

## 2. POT: the owner's concept

POT (Potential) is the owner's own invention, reworked from **OvD**, his v1 "pressure vs defence" term. He credits OvD
with taking the engine from 1600-level to 2000-level bots on chess.com. The concept, in his words and decisions
(memory `ovd-is-the-owners-long-term-pressure-concept`; `dev_notes/archive/eval/TEXEL-C3-DETECTORS-DESIGN-2026-09-27.md`
§8 and §17):

- POT = the **potential** for either side to **structurally transform** the position so that another subsystem later
  "shines through". Example: a closed centre with a king stuck in it. King safety says nothing yet; POT says "this
  can blow up".
- **Potential vs kinetic.** POT speaks only where a subsystem is still **unresolved** (cannot yet judge reliably), and
  goes silent once it can ("the potential is now kinetic"). Whether a subsystem is unresolved is judged from
  **structure** (can its inputs still change?), never from its own output: a quiet king-safety term may be kinetic
  with nothing to say.
- Two halves: **magnitude** (complexity: how unresolved) and **direction** (potential: who benefits; timing / tempo
  lives here). In the endgame it hands over to **winnability**, which is now a separate shipped term.
- Method the owner set: catalogue the transformation types and their end results, work **backwards** to their
  structural precursors (GM intuition), detect "unresolved" with high precision, then tune.
- Constraints: every POT feature needs an **owner** distinct from existing terms (no re-describing king safety,
  passers or pawn structure; "we may end up with a mess like v1 again"); a collinearity check before code; incremental
  value on top of everything else; **efficient for NPS**; and it must "holistically bring us closer to SF18 search
  without collapsing genuine decision making".
- The owner considers POT may simply not work, and wants stopping points rather than a sunk cost (memory
  `pot-may-not-work-use-stopping-points`).
- **POT-as-search** (10-19 transition list): the search-side use. Three parts, none built:
  1. the race term together with the pruning margins;
  2. an "unresolved" signal for reductions (LMR) and time;
  3. pending-break protection: owner game 7, where a pending pawn break against our king was pruned or not reached.

## 3. What was built and how each piece was judged

| piece | form tested | instrument | result | source |
|---|---|---|---|---|
| Winnability (endgame hand-over) | additive `sign(T)·C` | games vs SF18 | −25 / −23 (discontinuous at 0) | ovd memory |
| Winnability, rebuilt | multiplicative scale factor (reference form) | games vs SF18 @250k | **+20.2** | same |
| C64 class cells (pure-pawn / one-minor confidence) | eval cells | games | **shipped in gen 4** | POT brief §5f |
| mg POT types as added scores (10-01) | additive | depth screen, target SF18 − our d10 | null, all \|σ\| < 2.2 | `POT-TYPE-DEFINITIONS-2026-09-30.md` last section |
| Endgame king-path race | additive | move test on critical pairs (SF18 multi-PV) | z +0.63 (needs +2); straight-line race z +1.57 | POT brief §5i |
| Complexity scaler | SF12+'s bounded edge leg on the d12 score | d12 endgame win% MSE vs SF18 label | −0.41% (needs ≥ 2%) | §5k |
| Middlegame POT (pending breaks, tension, half-open pressure, majority, storm levers, space; each × "unresolved") | linear, nested on the d12 score | win% MSE vs SF18 d14 and realised t+16 / t+24; critical-pair move test | −0.46% beyond a calibration control; move test z −0.77; pending-break features ≈ 0 weight | §5l-§5m |
| Ceiling test (boosted trees, owner idea) | gradient-boosted trees on d12 + rich inputs | win% MSE | −10% at first, then **withdrawn**: a label artefact (below) | §5n |

**Never tested** (the preplan's own list, POT brief §5n): E1 passer-creation potential → winnability (rated the highest
evidence); M1 in **feeder** form on `transform_critical` / `transform_trap`; M2 closed-file openability correction
(−5.0σ on record); M3 majority → passer as a middlegame feeder. Also the owner's 10-04 design item: "the leader cannot
create a passer" as a winnability input. Preplan: `dev_notes/active/research-2026-10-08/POT-DESIGN-PREPLAN.md`.

## 4. Instrument facts learned since (they bear on how to read §3)

- **The SF centipawn label was distorted.** SF 15.1+ scales its cp output by material, and our fixed cp → win% formula
  ignores that. On 3,931 rows the old label and SF's own WDL model differ by 15.9 win% points on average (memory
  `sf-cp-labels-are-material-scaled-use-wdl`, found 10-19 at the end of the POT work). The race, complexity and
  middlegame fits above were scored on the old label. The boosted-tree "signal" did not survive SF's WDL.
- **A static depth proxy is biased against dynamic terms.** Adding a term's root static delta to a stored d10 score
  reads dynamic terms backwards: threats read +12.6% worse by proxy, −7.8% better in a real d10 re-search (memory
  `root-delta-depth-proxy-is-biased-against-dynamic-terms`). POT is a dynamic idea by definition.
- **Form matters.** Winnability failed additive and passed multiplicative. The owner's rule says POT is a
  feeder / modulator, while the 10-01 screen and the middlegame fit tested added scores.
- **Accuracy on a corpus is not Elo.** Several real accuracy gains did not become Elo, and the most accurate retune
  arm lost games (memories `corpus-fit-is-anti-correlated-with-elo`, `real-d10-accuracy-gains-did-not-become-elo`).
  Games decide.
- **An eval gain pays through pruning.** Threats in eval alone ≈ 0; with `RFP_MARGIN` 600, +66 (memory
  `eval-accuracy-payoff-is-pruning`).
- **Most of the owner's game losses are search-side.** Triage of 17 owner-game positions (10-18): eval 6, depth 7,
  pruning 2, ok 2. Game 7's unseen g-file break was re-read as a depth / pruning case. Tool
  `diagnostics/_case_triage.py`, cases `diagnostics/ks_sets/pot_owner_cases.txt`, POT brief §5j.

## 5. Files to read

- Concept: `dev_notes/archive/eval/TEXEL-C3-DETECTORS-DESIGN-2026-09-27.md` §8, §17;
  `dev_notes/archive/eval/POT-TYPE-DEFINITIONS-2026-09-30.md`; `dev_notes/archive/eval/POT-TRANSFORMATION-KNOWLEDGE-2026-09-30.md`.
- Plan and record: `dev_notes/active/research-2026-10-08/POT-DESIGN-PREPLAN.md`;
  `dev_notes/archive/eval/POT-DESIGN-BRIEF-2026-10-13.md` (§5 onward).
- Fits and probes: `diagnostics/_pot_mg_fit.py`, `_pot_complexity_fit.py`, `_pot_depth_screen.py`, `_pot_gbm_ceiling.py`,
  `_pot_race_proto.py`, `_pot_mg_screen.py`, `_case_triage.py`; cases `diagnostics/ks_sets/pot_owner_cases.txt`.
- Parked state: `dev_notes/current/EVAL-V2-PARKED-REGISTER.md` (rows for the complexity scaler and middlegame POT).
- Where it could live: `dev_notes/active/SEARCH-V2-PLAN-2026-11-10.md` (§3 stages S4-S5);
  `dev_notes/reference/reckless-search-brief-2026-11-06.md` (candidates 2 and 3: |correction| as an uncertainty signal,
  line-based reduction signals).

## 6. Questions

1. **Which of the §3 verdicts are refutations, and which are only unreadable** on the instrument that produced them
   (the cp label, the static proxy, the additive form)? For each, say what the cleanest re-test would be.
2. **Is there a search-side form of POT** that fits the concept better than an eval term? Candidates to assess, not
   to assume: "how unresolved" as a modulator of reductions, pruning margins, extensions or time; protection of
   pending-break / lever moves from pruning and reduction; a transformation-move class in move ordering. For each:
   what it reads, what it changes, why it is not already covered by standard search parts (check extensions,
   history, singular extensions, improving, |correction| uncertainty), and its NPS cost.
3. **Where in search v2's stages** would each candidate be testable, and what does it depend on (the per-ply stack, the
   node-entry static eval, the history redesign)?
4. **Which untested items (E1, M1-M3, the "leader cannot create a passer" input)** still look worth a test, in eval
   or in search?
5. **The concept itself:** is "potential vs kinetic, judged from structure" implementable cheaply enough to run at every
   node, or only at the root and PV nodes? Where would the precision of "unresolved" come from?

## 7. What a useful answer looks like

- Separate **facts verified in the code or the record** (cite the file and section, or the stable function name) from
  **reasoning**, and mark anything taken from general chess-engine knowledge rather than these files.
- Each proposal comes with: what it changes; the instrument that would judge it and how that instrument can mislead
  (`dev_notes/current/INSTRUMENT-MAP.md`); a keep criterion stated before any run; a rough cost in box time; and where it
  sits in the search v2 stages.
- Say plainly if the answer is "this part of POT is unlikely to work". The owner asked for a fair shot, not a rescue.
- The answer is reviewed by Claude Code and the owner against the code before anything is built, as with the 11-10
  speed ideas (`owner-cpp-ideas.md` §14-15, reviewed in the search v2 plan §7).

## 8. One prior reading, to challenge rather than adopt (Claude Code, 11-10)

Most POT verdicts are less settled than "parked" suggests. They used the distorted cp label, a static or additive form,
or a d12 static-accuracy target, which are the three instruments that misled us elsewhere. The one consistent signal
is that the owner's concrete failure (game 7) was a depth / pruning miss. That points to POT's natural home as a
search signal, beside reductions and extensions, more than as an eval term. This reading may be wrong; test it against
the files.
