# ⚠️ SUPERSEDED — see [`PAWN_MODEL.md`](PAWN_MODEL.md)

**This brief's two central premises were measured WRONG on the same day it was written.** Kept for the
record because the QUESTION SET below was good and drove the whole investigation; the proposed answers were
not.
- ☠️ **"Index on file × rank"** — file is the WEAKEST of the four factors measured (~18 cp, and only in the
  median). Obstruction and rank dominate. A rank term on the STRUCTURAL bonuses is worth having; a
  file × rank *table* as the primary index is not what the data asks for.
- ☠️ **"Raise the per-pawn clamp (within reason)"** — the clamp is correctly sized. Raising
  `PAWN_CLAMP_MID` to 300 measurably WORSENS win%-error (279.51 → 281.59), and 141 of the mean 211 raw is
  placement/attacking layers rather than structure, so extra headroom mostly feeds terms nobody wanted to grow.
- ✅ **What the questions produced instead**: the level error (~57 cp per pawn), the first ever FITTED pawn
  tables, `ISOLATED`/`BACKWARD` switched on at 80/120 — and the finding that a −14 corpus gain with all
  guards improving still returned **elo ~−15**.

---

# Pawn representation redesign — the brief (owner's direction, 2026-08-05)

Not a plan yet. This is the QUESTION SET that has to be answered with data before any of it is built,
plus the constraints the answers have to respect. Written because the current pawn evaluation is a pile of
independently-tuned terms with no shared definition of what makes a pawn good or bad.

## THE GOAL
Fold the structural and positional pawn signals into ONE coherent scheme where penalties, boosts and passer
value live together, indexed by **file AND rank**, with realizability scaling passers up or down — and
raise the per-pawn clamp (within reason) so the richer signal is not immediately truncated.

## ⚠️ THE CONSTRAINT THAT KILLS NAIVE VERSIONS
Every added bonus competes for the same clamp: `total -= std::min(225, structural + positional)` in the
midgame path, `std::min(175, structural)` in the endgame. Adding terms under a binding clamp does nothing.
**Instrument how often the clamp binds BEFORE designing anything that adds to it** — if it saturates in
most positions, granularity is wasted and the clamp is the actual lever.
⚠️ Owner's second constraint: a file×rank scheme has more ways to go wrong than a file-only one. The risk is
**overvaluing early pawn pushes** — rank bonuses already exist, so a rank-aware chain bonus stacks on top of
an existing rank signal. Any scheme must be checked against "does this make us push pawns instead of play".

## ❓ THE QUESTIONS TO ANSWER WITH DATA (owner's framing — answer these FIRST)
These are not rhetorical; they decide the shape of the table:
1. Is a **"weak" pawn on the 6th** better than a **"strong" pawn on the 5th**?
2. Does that one-rank difference change **depending on which ranks** are being compared (2↔3 vs 6↔7)?
3. Are they roughly equal — i.e. does the strong/weak axis cancel one rank of advancement?
4. If it depends on other factors, **which** factors? (blockade, opposition, supporting pieces, phase?)
▶️ Method: build small scripts that EMULATE the candidate scoring in isolation and fit them against data,
**across many position sets** so the answer is not an artifact of one corpus
([[corpus-composition-decides-the-optimum]] — a 29%→56% general-play change INVERTED a knob ranking).
▶️ The deliverable is a written justification we can both state plainly, THEN an implementation — cheap.

## WHAT WE ALREADY HAVE (verified 2026-08-05, do not re-derive)
| concept | where | note |
|---|---|---|
| doubled | L874 mid / L3097 eg | 125 / 150 mp per pawn on a shared file, FLAT |
| rear doubled = not a passer | L8655 | routed to DEFAULT rank table (90 mp vs 1085 at rank 6) — the largest pawn penalty |
| phalanx (same-rank neighbour) | `left`/`right` → `pawn_wall_file_bonus[x]` | FILE-scaled (mid), flat 100 (eg) |
| support (diagonal behind) | `sw`/`se` → `pawn_chain_file_bonus[x] + 15` | FILE-scaled (mid), flat 135 (eg) |
| latent support | +30 mid / +50 eg | potential support, no SF analogue |
| defending another pawn | in the attack loop | `pawn_chain_file_bonus[x]` (mid) / 115 (eg) |
| rank advancement | `default_` / `passed_` / `endgame_pawn_rank_bonus` | phase-blended in `evaluate_passers` |
| passer realizability | `passer_realizability_R` → `mag × R/256` | ⚠️ `R` internally clamps to [0,384] |
| isolated / backward | L7614 | **BUILT, DEFAULT 0, measured worthless at 200/100 on 4 venues** |

**Missing vs SF:** `WeakUnopposed`, `BlockedPawn[]` (5th/6th), phalanx/opposed modulation, and SF's richer
passer DETECTION (lever / leverPush / phalanx counts / rank-5 support) — ours is occupancy-based.

## 🚨 THE FINDING THAT SHOULD SHAPE THE WHOLE EFFORT
[[most-eval-error-is-move-neutral]]: on 60 collapse decision positions, **52% have a large eval error yet we
play a fine move**, and only 18% involve a bad move at all. So "make the pawn eval more accurate" is NOT
automatically a strength strategy. **Target the errors that CHANGE MOVES.** Before investing, audit whether
`decision_fen` is even the mistake ply (score every ply of a collapsed game and find where move_loss spikes).

## VENUES
- **KP diagnostic** (`kpgauntlet` + `gen_kp_fens.py`): 300 games in ~12 min. Baselines KP mixed **67.3%**,
  KP dense **68.8%** (n400, seed 0). ⚠️ DIAGNOSTIC ONLY — never ship on it.
- **General SPRT** decides ships. ⚖️ Only game changes expected to clear the ~20-40 Elo floor; pawn terms
  are individually small, so expect to bundle.
- ⚠️ Neither STS nor WAC has ever rewarded a pawn-structure change: `iso200/bwd100` read **+58 STS** and
  ~0 Elo on four venues.
