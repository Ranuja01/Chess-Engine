# Kaufman quadratic material-imbalance term — built + deterministically verified (2026-07-20)

## Provenance (NOT copied from SF)
The *model* is Larry Kaufman's published quadratic material-imbalance (1999; Chess Programming Wiki
"Material imbalance") — general chess-programming knowledge, independent of Stockfish. We implement the public
mechanism; **our coefficients are FIT FROM OUR OWN DATA** (`diagnostics/kaufman_fit.py`), not SF's constants.

## Why (diagnosis chain)
Hypothesis test (`sf11_depth_test.py`) proved the eval over-read is EVAL not depth (SF11 at our leaf depth 12
evaluates fen3/P2/P3 correctly; we don't). SF11 study showed SF re-prices material by census (Kaufman) — a
dimension we entirely lacked (our `imbalance_white/black` is an unrelated offense-vs-defense proxy). Primary aim:
material-based endgame collapses (pawns-vs-pieces); secondary: general eval strength (the ~590-Elo equal-depth
gap). **Caveat found by the fit:** material imbalance is only ~0.07-0.12p of our ~1.28p eval gap vs SF11 — the
gap is DOMINATED by KING SAFETY (mean|our_KS − SF11_KS|=0.63p; 34/400 positions SF sees >1p KS, we read <half).
So Kaufman is a small general lever; KS is the big one (next lane). BUT the material-COLLAPSE subset still
benefits measurably (below), which is the targeted win.

## The fit (linear-in-coefficients → ridge least squares)
`diagnostics/kaufman_fit.py N lambda`: our eval with pair bonuses OFF (Kaufman will own them), target =
`(SF11_static − our_eval)` residual in milli-pawns, features = piece-count PRODUCTS (White−Black difference
form), ridge with a fixed-seed train/held-out split. Chosen fit = **N=800, λ=50** (cleanest chess-correct signs).
Held-out mean|gap| 1.190 → 1.119 (train 1.192 → 1.122 = no overfit). λ sweep: held-out reduction flat ~0.066p
across λ (material-census signal is small); imbalance-subset (55% of held-out) 1.386 → 1.268.

**The data independently rediscovered Kaufman's known chess** (validates the mechanism):
`knight×pawn = +178` (knights love pawns), `rook×rook = −90` (rook redundancy), `knight×knight = −28` (knight
redundancy), bishop-pair positive. Compiled tables (milli-pawn) live in `cpp_bitboard.cpp` at the term.

## Implementation (gated, byte-identical default)
- `search_engine.h`: `ENABLE_KAUFMAN_IMBALANCE=false`, `KAUFMAN_SCALE=100` (+ env wiring / toggles dump).
- `cpp_bitboard.cpp` `placement_and_piece_eval` cold tail (after the pair block): `static constexpr`
  KAUFMAN_OURS/THEIRS[6][6]; compute 10 counts inline (idx 0=bishop-pair pseudo, 1=P..5=Q); scalar =
  Σ_{pt2≤pt1} OURS*(cw1·cw2−cb1·cb2) + THEIRS*(cw1·cb2−cb1·cw2); `total -= scalar*KAUFMAN_SCALE/100`
  (White-POV milli → Black-positive). `br_kaufman` breakdown slot.
- **Double-count reconciliation:** the flat `BISHOP_PAIR_BONUS`/`KNIGHT_PAIR_BONUS` block is SKIPPED when Kaufman
  is on (Kaufman owns material-combo). Fit was done pairs-OFF, so this is consistent.
- Byte-identical at default confirmed: WAC 243/300, NODES 39,914,378, EBF 3.655 (= shipped).

## Deterministic verification (Kaufman ON) — all gates pass
| gate | base | Kaufman | note |
|---|---|---|---|
| eval-gap vs SF11 mean|gap| | 1.285 | **1.223** | drops (more SF-like) |
| STS /3000 | 1606 | 1606 | identical, no positional decay |
| WAC /300 | 243 | 240 | −3 (minor) |
| passers.csv 150 | ~74 | 76 | neutral (search-time noise) |
| fen3 total (SF ~0) | +5.58 | +4.90 | deflates |
| P2 total (SF −1.15) | +4.91 | **+1.74** | big deflation toward SF |
| P3 keep-out (SF +4.84) | +9.72 | +9.07 | holds (winning) |

The material-collapse subset (fen3/P2) genuinely deflates while P3 holds and general benches don't decay.

## Game gate — 2-seed POSITIVE (confirmation running)
vs the existing `ab_base_s0/s1` baseline (engine byte-identical at default):
| config | s0 | s1 | avg score | avg collapses |
|---|---|---|---|---|
| base | 42.0/75 | 38.0/90 | 40.0% | 82.5 |
| Kaufman | 42.2/78 | 43.8/77 | **43.0% (+3.0)** | **77.5 (−5)** |
Beats base on BOTH seeds; score UP + collapses DOWN — passes the user's two-sided guard (general play did NOT
decay). First game-positive eval lever this session. Caveat: 2 seeds, +3% ≈ 1 SE ⇒ 4-seed confirmation running
(`ab_base_s2/s3` + `ab_kauf_s2/s3`). If it holds → ship candidate (KAUFMAN_SCALE=100, the N800/λ50 β).

## Next (independent of this verdict)
KING SAFETY is the dominant eval-gap lever (0.63p, vs Kaufman's ~0.07p) and our proven shipped lane — the
highest-value eval direction after this. Also pending: fen3-class pawn-structure/rank-boost realizability
(separate per-pawn axis).
