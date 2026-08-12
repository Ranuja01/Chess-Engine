# Fable consult — the collapse is a COUNTERPLAY/initiative problem, not static magnitude. Build a new term, activate parked machinery, or reshape? (2026-07-11)

*Self-contained; repo access + pointers in §7. Follows the fantasy-realizability consult — that damp was
built, gauntletted, and FAILED, and the failure taught us the real shape of the problem.*

## 0. Engine + venue (unchanged)
Custom C++ **HCE** engine, pre-NNUE by choice, non-negamax, absolute Black-positive eval, millipawns.
Refs: **SF11** (classical, labelled terms) + **SF18** (NNUE). **Venue trust:** external **gauntlet** (our
engine vs throttled native SF18, ≥3 seeds) = TRUTH; lightning self-play + a retired cploss compass
ANTI-PREDICT. We also have a fast offline **move-selection lens** (per-stratum move-flip vs SF, below)
used for MECHANISM, not ship decisions.

## 1. What just happened — three static-eval damps failed, and the third failure was diagnostic
We chased a proven CONDITIONAL static over-read (+368cp on collapse positions; calibrated elsewhere; SF11
reads them ~0). Three attempts to DAMP it:
- **l1a (broad optimism damp):** collapse rate halved, **−80 Elo** (gauntlet). Optimism is load-bearing.
- **npedge damp (surgical, Fable-designed, offline-screen-clean):** the fantasy-vs-real discriminator was
  clean (holdout AUC 0.85, `npedge`=piece-material backing), the damp targeted only the pawn-placement
  credit keyed on unbacked×midgame, offline slices all passed (fantasy −59cp, real-wins held ±20, endgame
  gated). **Gauntlet: net −1.8% over 3 paired seeds (s0 +2.0 / s1 −8.4 / s2 +1.0) = NO-GO.**
- **npedge + tactical-tension quiet-gate:** fixed the tactical leak but destroyed its own collapse benefit.

## 2. ⭐ The mechanism (move-selection lens) — a static positional damp SCATTERS TACTICS
We built `moves_dump`+`move_flip_report`: for an 824-position stratified corpus (collapse / sts / neutral /
game) it plays OUR move at fixed depth and scores it by SF18 WDL-cploss, per stratum. The npedge damp:

| stratum (n)   | Δ move-selection loss (− = damp improves) |
|---------------|-------------------------------------------|
| collapse (24) | −0.65 (better)                            |
| game (244)    | −0.25 (better)                            |
| neutral (200) | −0.06                                     |
| **sts (350)** | **+0.35 (WORSE — tactical stratum)**      |
| ALL (818)     | +0.04 (net flat)                          |

The damp improves POSITIONAL move choice but WORSENS TACTICAL move choice. Net cploss ≈0 — a cancellation.
But a tactical error loses a whole game while a quiet gain rarely swings one, so flat-mean = net-negative
games (seed1's sharp openings ate the −8.4). Root: **any eval change that moves values reshapes the
search's move ordering / pruning bounds — in a tactical position that scatters the tactical resolution**
(also why the damp cost +7% nodes and −3 WAC). Confirmed: `tension_by_stratum.py` shows the collapse
positions are TACTICAL — mean 5.6 SEE-captures, 92% ≥2 — statistically identical to `sts`(5.7/98%) and
real `game`(5.4/92%), far above `neutral`(3.1/76%). So a tension gate that spares tactics also switches
the damp off on the collapses (they share the regime). **⇒ static positional damping is the WRONG TOOL:
you can't lower the over-read without disturbing tactics in exactly the positions that matter.**

## 3. Diagnosing the actual LOSING moves (`diagnose_collapse_moves.py`, 142 losing collapse decisions, SF d14)
| category | % | meaning |
|----------|---|---------|
| downstream | 36% | we played SF's OWN move at the eval peak → over-read didn't change our move; loss is later |
| over-push  | 27% | we play an aggressive capture/push and get hit (~40% walk into a capture/check refutation) |
| other quiet error | 25% | quiet move, big loss |
| minor | 13% | our move only slightly worse |

So the over-read either **doesn't change our move** (36%) or changes it **in a tactical position where a
damp backfires** (27%). The largest *fixable-at-the-decision* slice is the **~27% over-push into
counterplay = counterplay-blindness**: we push our nominal advantage while ignoring the opponent's
counterplay. This wants a COHERENT ADDED danger/initiative signal that favors the consolidating move (SF's
choice), NOT a damp that fights the search.

## 4. The code landscape (what already exists — we don't want to duplicate/re-tread)
- **`latent_threat` IS our LIVE king-safety term** (king-zone pressure via `attack_bitmasks`, added by
  default). A higher-DOF twin `king_safety_score` (SF11 attack-units → nonlinear table → phase taper) is
  PARKED (`KING_SAFETY_MAG=0`). Prior "REPLACE latent_threat with KS" = −47 Elo (its non-KS content was
  load-bearing) — but that was LIGHTNING (anti-predictive), never gauntletted.
- **`get_static_threats_score`** (`ENABLE_THREATS=false`) = whole-board loose/hanging/weak-piece threats,
  `threats_by(Black) − threats_by(White)`. The non-king threat representation we lack — BUILT, never
  gauntlet-tested.
- **`MOD_PVBOOST_COMP` / `MOD_PIECES_DEFEND`** already damp a lead by (opp offense − our defense) — crude
  counterplay damps, part of the failed l1a / neutral DEFEND-only. The "counterplay damps the leader" idea
  has been tried as a MULTIPLIER; the untried form is ADDITIVE.
- **`imbalance_white/black`** = OvD (offense-vs-defense edge) — closest existing analog to an initiative term.
- **The eval is deliberately COLOR-SYMMETRIC with NO tempo/initiative** (mirror residual 0.000); `turn`
  only affects `capture_gains` (SEE first-mover) + two endgame king-races. Asymmetry everywhere else is
  gated on material/placement SIGN, never `turn`. **SF's "Initiative" (sign-preserving damp toward 0 on
  complexity) is a known, explicitly-unbuilt gap.**
- Untapped signals for a whole-board term (the king terms never read these): `white/blackOffensiveScore`,
  `white/blackDefensiveScore`, `pressure_*/support_*`, plus the threats term.
- Insertion window is clean: at `cpp_bitboard.cpp:6401–6451` all board-control accumulators are COMPLETE
  and stable, and this is BEFORE every downstream optimism term (latent_threat, king_safety, imbalance,
  MOD_PIECES_*, piece_value_boost) — so a term here can both READ the full picture and FEED downstream.

## 5. What we propose to build (pressure-test this)
A **whole-board counterplay/initiative term**, computed at the mid-window (6401–6451), reading the untapped
board-control accumulators + whole-board threats (NOT re-scanning the king zone — avoids triple-counting
latent_threat/KS). Form: **ADDITIVE, SF-Initiative-shaped** (sign-preserving nudge toward 0 / a danger
subtraction from the leader) so it favors the consolidating move instead of damping a positive term.
Asymmetry via material/placement SIGN (the TRAILING side's activity edge over the leader's defense), not
`turn`. Screened with the move-selection lens: must lift the over-push/collapse decisions WITHOUT
scattering `sts` (the exact failure mode of the damp), then gauntlet ≥3 seeds, SCORE primary.

## 6. Questions
1. **New term vs activate existing?** Given latent_threat (live KS), king_safety (parked twin), COMP/DEFEND
   (tried counterplay damps), `get_static_threats_score` (parked whole-board threats), and imbalance (OvD)
   — is a NEW term justified, or is the highest-value move to ACTIVATE/reshape one of these (e.g. turn on
   the whole-board threats term, or recast COMP as additive)? Which existing piece is the right foundation?
2. **Additive vs multiplier.** Given the 3× damp-fails-because-it-scatters-tactics finding, is an ADDITIVE
   SF-Initiative shape (sign-preserving damp toward 0, or a danger subtraction) the right form — and does
   additive have its own tactical-scatter risk (it still changes eval values → move ordering)? Is there a
   form that changes the leaf VALUE without disturbing the ordering/pruning that resolves tactics?
3. **Scheduling.** Mid (feed downstream realizability) vs a pure additive leaf. Given the damp failures,
   is fueling downstream realizability worth the re-entanglement risk, or keep it a leaf?
4. **The initiative asymmetry (the user's key point).** The eval is fully symmetric with no tempo; a naive
   `our_attack − their_attack` won't credit realizing threats first, and counterplay exists for BOTH sides.
   Should the counterplay term stay symmetric (sign-gated on who's ahead), or introduce a real side-to-move
   initiative asymmetry? How do you handle "both sides have counterplay" so it doesn't wrongly cancel — and
   does the answer depend on the eval being called at alternating side-to-move leaf nodes?
5. **The accurate cheap DETECTOR.** What actually separates "real convertible advantage" from "advantage
   with live counterplay against us"? Offense/defense edge? whole-board threats (loose pieces)? mobility
   imbalance? passed-pawn races / king exposure? A composite — and what's the minimal set + weighting? We
   have SF11 per-term available to fit against, and the move-selection lens to screen.

Also: any failure mode we're not seeing, or a fundamentally different framing (e.g. this is really a SEARCH
/ counterplay-visibility problem and no static term will fix the 27% cleanly)?

## 7. Repo pointers
- Data/tools: `diagnostics/moves_dump.py` + `move_flip_report.py` (the move-selection lens),
  `tension_by_stratum.py`, `diagnose_collapse_moves.py`, `screen_slices.py`; corpus
  `selfplay/tune_data/cploss_corpus.csv` (+ d12 SF cache). Dev note `dev_notes/npedge-damp-build-2026-07-10.md`.
  Memory `[[collapse-is-tactical-not-static]]`, `[[collapse-fix-load-bearing-optimism]]`,
  `[[eval-accuracy-payoff-is-pruning]]`, `[[realizability-conditioning-architecture]]`,
  `[[capg-tension-conditioning]]`, `[[ks-detection-rebuild]]`, `[[external-gauntlet-calibrated]]`.
- Code: `cpp_bitboard.cpp` — `get_latent_threat_score` (:5179), `king_safety_score`/`king_safety_danger`
  (:4967/:5114), `get_static_threats_score` (:5137), `imbalance` (:6516), offense/defense accumulators
  (:71, populated in the per-piece evaluators), insertion window (:6401–6451), MOD_PVBOOST_COMP (:6907).
  Knobs in `search_engine.h`: `ENABLE_THREATS`/`SCALE_THREATS`, `KING_SAFETY_MAG`/`KS_*`,
  `MOD_PVBOOST_COMP`, `MOD_PIECES_DEFEND`.
