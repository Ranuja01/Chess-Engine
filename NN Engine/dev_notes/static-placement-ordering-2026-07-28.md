# Static placement ordering (PST tiebreaker) — NO-GO, 2026-07-28

## The idea
Quiets with zero history all score identically, so their relative order is whatever move generation
produced — and LMP prunes / LMR reduces by that arbitrary index, which is also where wrong reductions
cluster (L6 4.12%, L8 6.25% vs ~1%). Break those ties with the eval's own placement layer.

Built as `ENABLE_STATIC_ORDER` + `STATIC_ORDER_{MODE,WEIGHT,HIST_MAX,PIECES,KING_EG_ONLY}`, via one shared
helper `staticPlacementScore()` in `move_gen.h` called from BOTH quiet scorers (the `score_move` lambda and
`score_quiet`, the lazy re-sort path — they already carry a "keep in sync" warning).

## Result: NEUTRAL, then disproven by sweep

| config | solves | nodes | FMC (m0/total) |
|---|---|---|---|
| base | 249 | 38,840,709 | **87.55%** |
| v1 delta (all pieces) | 246 | 40,105,056 | 86.81% |
| v1 destination | 244 | 38,719,634 | 87.39% |
| **v2 delta (corrected)** | 249 | 39,180,884 | **87.57%** |
| v2 destination | 244 | 39,132,359 | 87.28% |

**Fire counter kills the "no opportunities" explanation:** `eligible=40,834,495 fires=23,075,064 (56.5%)` —
roughly ONE eligible quiet per node searched. The term speaks constantly and changes nothing.

**Weight sweep kills the "correct but swamped" explanation** (delta mode):

| weight | 30 | 100 | 300 | 1000 |
|---|---|---|---|---|
| solves | 245 | 249 | 244 | 248 |
| FMC | 87.53% | 87.57% | 87.23% | 87.44% |

A 33× range produces no monotone response and no plateau. ⇒ **the placement signal does not predict cutoffs
among history-thin quiets.**

## ★ THE REAL FINDING: ordering is NOT our bottleneck
**FMC is already 87.55%** — cutoffs land on the first move seven times in eight, leaving ~12.5% total
headroom, most of it presumably genuinely hard nodes. A cheap static prior has almost nowhere to work.
⇒ Lower the priority of the whole ordering program, including L1 (attack maps) and L2 (king zone): they are
the same hypothesis with strictly more expensive inputs. **Measure FMC headroom before building an ordering
feature.**

## ★ Two eval facts found on the way (independent of the ordering result)
From `rebuild_scaled_placement` (`cpp_bitboard.cpp`): *"Rook PST is dead code (never read) -> scale 100.
King placement is endgame-only (SCALE_PLACE_KING_EG)."*
- **The rook PST is never read by the eval.** Rooks get no placement term at all; their live signal is
  open-file logic (`ROOK_OPEN_BASE`/`ROOK_SEMI`/`ROOK_7TH`).
- **The king PST is endgame-only.** Midgame king placement is handled by king safety, not a table.
  ⇒ We are not missing an ENDGAME king table; we lack a MIDGAME one, and the eval covers that elsewhere.

Both are deliberate (a rook's worth is contextual, not square-intrinsic), so they are notes about eval
design, not bugs.

## ⚠️ Method notes
- **Ordering need NOT agree with the eval.** I designed around "the tiebreaker must use the eval's tables",
  and excluded rooks by default on that basis. That principle is wrong: eval answers *how good is this
  position*, ordering answers *which move causes a cutoff*. A table the eval ignores can still be a fine
  ordering prior. Rook exclusion should have been measured, not assumed.
- **v2 changed TWO variables at once** (rook mask + king phase gate), so its improvement over v1 is
  unattributed. The 2×2 was never run — if this line is ever revived, run it.
- The fire counter should have been in the FIRST round: a neutral bench with an unknown firing rate cannot
  distinguish "useless signal" from "no opportunities". See [[counter-on-taken-branch]].

## Status
All knobs gated default-off; **byte-id verified 249 / 38,840,709 after every build**. Nothing committed.
